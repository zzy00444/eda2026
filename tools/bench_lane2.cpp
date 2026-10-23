// Runtime prototype for the lane-level model: SIMD-assisted parsing, a batched
// prefetch pipeline, and coarse-select / full-evaluate lane reduction.
//
// Model (offline-fitted; tables synthetic here, shapes/access are real):
//   delay = gap(x0,y0,x1,y1)
//         + min_{k in shortlist[src][tgt]} ( G(src,k,cell) + B[k][tgt] )
// The shortlist has ~13 lanes; a rank-8 int8 coarse pass over those lanes selects
// the best two, which are then evaluated at full rank (int16).
//
// Modes isolate the cost components:
//   copy      read + copy line + ",100"        (I/O floor)
//   readonly  read + parse + checksum           (parse only)
//   parse     read + parse + write              (parse + format)
//   full      read + parse + model + write      (real runtime)
//   fullnoprefetch  same as full without the prefetch pass (pipeline benefit)
//
// Usage:
//   bench_lane2 <ports.txt> <gap.txt> <request.csv> <result.csv> <mode>
//               [rank=256] [coarse_rank=8] [lanes=13] [select=2] [block=64]
//
// Compile: g++ -O3 -mavx2 -std=c++17 -o build/bench_lane2 tools/bench_lane2.cpp

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <string>
#include <vector>

namespace {

constexpr int kNumSource = 208;
constexpr int kNumTarget = 288;
constexpr int kNumLane = 160;
constexpr int kNumCell = 60200;
constexpr int kWidth = 120;
constexpr int kHeight = 550;
constexpr int kShortlist = 13;

struct PortEntry {
    char name[24];
    int16_t source;
    int16_t target;
};

uint32_t gKey[1 << 16];
int16_t gVal[1 << 16];

inline uint32_t hashName(const char* p, int len) {
    uint64_t w = 0;
    const int n1 = len < 8 ? len : 8;
    memcpy(&w, p, static_cast<size_t>(n1));
    uint64_t h = w * 0x9E3779B97F4A7C15ull;
    if (len > 8) {
        uint64_t w2 = 0;
        memcpy(&w2, p + 8, static_cast<size_t>(len - 8));
        h ^= (w2 + 0x94D049BB133111EBull) * 0xBF58476D1CE4E5B9ull;
    }
    h ^= static_cast<uint64_t>(len) * 0x2545F4914F6CDD1Dull;
    h ^= h >> 29;
    return static_cast<uint32_t>(h);
}

inline int16_t fastPortId(const char* p, int len) {
    const uint32_t h = hashName(p, len);
    uint32_t slot = h & 0xffffu;
    while (true) {
        const int16_t v = gVal[slot];
        if (v == -1) return -1;
        if (gKey[slot] == h) return v;
        slot = (slot + 1) & 0xffffu;
    }
}

PortEntry gPorts[512];
int gRank = 256;
int gCoarseRank = 8;
int gLanes = kShortlist;
int gSelect = 2;

int16_t* gCoef = nullptr;      // [cell][rank]
int16_t* gBasis = nullptr;     // [src*160+lane][rank]
int8_t* gCoefS = nullptr;      // [cell][coarse]
int8_t* gBasisS = nullptr;     // [src*160+lane][coarse]
int16_t* gB = nullptr;         // [lane][target]
uint8_t* gShort = nullptr;     // [src*target][lanes]

int32_t gVp[kWidth + 1];
int32_t gHp[kHeight + 1];

uint64_t gRng = 88172645463325252ull;
inline uint64_t xs() { gRng ^= gRng << 13; gRng ^= gRng >> 7; gRng ^= gRng << 17; return gRng; }

void fillTables() {
    gCoef = static_cast<int16_t*>(aligned_alloc(64, static_cast<size_t>(kNumCell) * gRank * 2));
    gBasis = static_cast<int16_t*>(aligned_alloc(64, static_cast<size_t>(kNumSource) * kNumLane * gRank * 2));
    gCoefS = static_cast<int8_t*>(aligned_alloc(64, static_cast<size_t>(kNumCell) * gCoarseRank));
    gBasisS = static_cast<int8_t*>(aligned_alloc(64, static_cast<size_t>(kNumSource) * kNumLane * gCoarseRank));
    gB = static_cast<int16_t*>(aligned_alloc(64, static_cast<size_t>(kNumLane) * kNumTarget * 2));
    gShort = static_cast<uint8_t*>(aligned_alloc(64, static_cast<size_t>(kNumSource) * kNumTarget * gLanes));
    for (size_t i = 0; i < static_cast<size_t>(kNumCell) * gRank; ++i) gCoef[i] = static_cast<int16_t>(xs() & 0x1ff) - 255;
    for (size_t i = 0; i < static_cast<size_t>(kNumSource) * kNumLane * gRank; ++i) gBasis[i] = static_cast<int16_t>(xs() & 0x1ff) - 255;
    for (size_t i = 0; i < static_cast<size_t>(kNumCell) * gCoarseRank; ++i) gCoefS[i] = static_cast<int8_t>(xs() & 0x7f) - 63;
    for (size_t i = 0; i < static_cast<size_t>(kNumSource) * kNumLane * gCoarseRank; ++i) gBasisS[i] = static_cast<int8_t>(xs() & 0x7f) - 63;
    for (size_t i = 0; i < static_cast<size_t>(kNumLane) * kNumTarget; ++i) gB[i] = static_cast<int16_t>(xs() % 3000);
    for (size_t i = 0; i < static_cast<size_t>(kNumSource) * kNumTarget * gLanes; ++i)
        gShort[i] = static_cast<uint8_t>(xs() % kNumLane);
}

inline int32_t dot16(const int16_t* a, const int16_t* b, int n) {
    __m256i acc = _mm256_setzero_si256();
    int j = 0;
    for (; j + 16 <= n; j += 16) {
        const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j));
        const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + j));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(va, vb));
    }
    __m128i lo = _mm256_castsi256_si128(acc);
    __m128i hi = _mm256_extracti128_si256(acc, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    int32_t out = _mm_cvtsi128_si32(s);
    for (; j < n; ++j) out += static_cast<int32_t>(a[j]) * static_cast<int32_t>(b[j]);
    return out;
}

inline int32_t dot8(const int8_t* a, const int8_t* b, int n) {
    int32_t acc = 0;
    for (int j = 0; j < n; ++j) acc += static_cast<int32_t>(a[j]) * static_cast<int32_t>(b[j]);
    return acc;
}

struct Row {
    const char* start;
    int len;
    int16_t src;
    int16_t tgt;
    int32_t cell;
    int32_t delay;
};

constexpr int kBlock = 64;

}  // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr,
                "usage: bench_lane2 <ports> <gap> <request> <result> <mode> [rank=256] [coarse=8] [lanes=13] [select=2]\n");
        return 2;
    }
    const std::string mode = argv[5];
    if (argc > 6) gRank = atoi(argv[6]);
    if (argc > 7) gCoarseRank = atoi(argv[7]);
    if (argc > 8) gLanes = atoi(argv[8]);
    if (argc > 9) gSelect = atoi(argv[9]);
    if (gLanes > kShortlist) gLanes = kShortlist;

    int nPorts = 0;
    {
        FILE* f = fopen(argv[1], "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
        char line[128];
        while (fgets(line, sizeof line, f)) {
            char name[32]; int s, t;
            if (sscanf(line, "%31s %d %d", name, &s, &t) != 3) continue;
            std::strncpy(gPorts[nPorts].name, name, 23);
            gPorts[nPorts].name[23] = 0;
            gPorts[nPorts].source = static_cast<int16_t>(s);
            gPorts[nPorts].target = static_cast<int16_t>(t);
            ++nPorts;
        }
        fclose(f);
    }
    for (int i = 0; i < (1 << 16); ++i) gVal[i] = -1;
    for (int i = 0; i < nPorts; ++i) {
        const char* nm = gPorts[i].name;
        const int len = static_cast<int>(strlen(nm));
        const uint32_t h = hashName(nm, len);
        uint32_t slot = h & 0xffffu;
        while (gVal[slot] != -1) slot = (slot + 1) & 0xffffu;
        gKey[slot] = h;
        gVal[slot] = static_cast<int16_t>(i);
    }
    {
        FILE* f = fopen(argv[2], "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 2; }
        int w, h;
        if (fscanf(f, "%d %d", &w, &h) != 2) return 2;
        for (int i = 0; i <= w; ++i) if (fscanf(f, "%d", &gVp[i]) != 1) return 2;
        for (int i = 0; i <= h; ++i) if (fscanf(f, "%d", &gHp[i]) != 1) return 2;
        fclose(f);
    }
    fillTables();

    FILE* in = fopen(argv[3], "rb");
    FILE* out = fopen(argv[4], "wb");
    if (!in || !out) { fprintf(stderr, "cannot open request/result\n"); return 2; }

    const bool doCompute = (mode == "full" || mode == "fullnoprefetch");
    const bool doWrite = (mode != "readonly");
    const bool doPrefetch = (mode == "full");

    constexpr size_t kIn = 1 << 22;
    constexpr size_t kOut = 1 << 23;
    std::vector<char> inbuf(kIn + 512);
    std::vector<char> outbuf(kOut + 4096);
    size_t outLen = 0;
    if (doWrite) { memcpy(outbuf.data(), "From,To,delay\n", 14); outLen = 14; }

    long rows = 0;
    long long checksum = 0;
    size_t carry = 0;
    char* base = inbuf.data();

    while (true) {
        const size_t got = fread(base + carry, 1, kIn - carry, in);
        const size_t total = carry + got;
        if (total == 0) break;
        size_t pos = 0;
        while (pos < total) {
            Row block[kBlock];
            int n = 0;
            // ---- parse a block ------------------------------------------------
            while (n < kBlock && pos < total) {
                char* lineStart = base + pos;
                char* nl = static_cast<char*>(memchr(lineStart, '\n', total - pos));
                if (!nl) break;
                char* lineEnd = nl;
                if (lineEnd > lineStart && lineEnd[-1] == '\r') --lineEnd;
                pos = static_cast<size_t>(nl - base) + 1;
                if (lineEnd - lineStart < 3) continue;
                if (lineStart[0] == 'F' && lineStart[1] == 'r' && lineStart[2] == 'o') continue;
                char* comma = static_cast<char*>(memchr(lineStart, ',', lineEnd - lineStart));
                if (!comma) continue;
                int sx = 0, sy = 0, tx = 0, ty = 0;
                char* p = lineStart + 4;
                while (p < comma && *p >= '0' && *p <= '9') sx = sx * 10 + (*p++ - '0');
                ++p;  // '_'
                while (p < comma && *p >= '0' && *p <= '9') sy = sy * 10 + (*p++ - '0');
                ++p;  // '/'
                const int sp = fastPortId(p, static_cast<int>(comma - p));
                p = comma + 1 + 4;
                while (p < lineEnd && *p >= '0' && *p <= '9') tx = tx * 10 + (*p++ - '0');
                ++p;  // '_'
                while (p < lineEnd && *p >= '0' && *p <= '9') ty = ty * 10 + (*p++ - '0');
                ++p;  // '/'
                const int tp = fastPortId(p, static_cast<int>(lineEnd - p));
                if (sp < 0 || tp < 0) continue;
                block[n].start = lineStart;
                block[n].len = static_cast<int>(lineEnd - lineStart);
                block[n].src = gPorts[sp].source;
                block[n].tgt = gPorts[tp].target;
                const int dx = tx - sx, dy = ty - sy;
                block[n].cell = ((dx + 119) * 1100 + (dy + 549)) % kNumCell;
                block[n].delay = 0;
                block[n].src = block[n].src >= 0 ? block[n].src : 0;
                block[n].tgt = block[n].tgt >= 0 ? block[n].tgt : 0;
                ++n;
            }
            if (n == 0) break;
            // ---- prefetch -----------------------------------------------------
            if (doPrefetch) {
                for (int i = 0; i < n; ++i) {
                    const char* c = reinterpret_cast<const char*>(gCoef + static_cast<size_t>(block[i].cell) * gRank);
                    for (int off = 0; off < gRank * 2; off += 64) __builtin_prefetch(c + off);
                    const uint8_t* lanesP = gShort + (static_cast<size_t>(block[i].src) * kNumTarget + block[i].tgt) * gLanes;
                    __builtin_prefetch(lanesP);
                    for (int l = 0; l < gSelect && l < gLanes; ++l) {
                        const int k = lanesP[l] & 0x7f;
                        const char* br = reinterpret_cast<const char*>(gBasis + (static_cast<size_t>(block[i].src) * kNumLane + k) * gRank);
                        for (int off = 0; off < gRank * 2; off += 64) __builtin_prefetch(br + off);
                    }
                }
            }
            // ---- compute ------------------------------------------------------
            for (int i = 0; i < n; ++i) {
                if (!doCompute) { block[i].delay = (mode == "copy") ? 100 : (block[i].cell & 0x1ff); continue; }
                const uint8_t* lanesP = gShort + (static_cast<size_t>(block[i].src) * kNumTarget + block[i].tgt) * gLanes;
                const int16_t* coefRow = gCoef + static_cast<size_t>(block[i].cell) * gRank;
                const int8_t* coefRowS = gCoefS + static_cast<size_t>(block[i].cell) * gCoarseRank;
                int bestK[4] = {0, 0, 0, 0};
                int32_t bestV[4] = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
                // coarse selection over the shortlist
                for (int l = 0; l < gLanes; ++l) {
                    const int k = lanesP[l] & 0x7f;
                    const int8_t* bS = gBasisS + (static_cast<size_t>(block[i].src) * kNumLane + k) * gCoarseRank;
                    const int32_t v = dot8(bS, coefRowS, gCoarseRank) + (gB[k * kNumTarget + block[i].tgt] << 6);
                    for (int q = 0; q < gSelect; ++q) {
                        if (v < bestV[q]) {
                            for (int r = gSelect - 1; r > q; --r) { bestV[r] = bestV[r - 1]; bestK[r] = bestK[r - 1]; }
                            bestV[q] = v; bestK[q] = k;
                            break;
                        }
                    }
                }
                int32_t best = 0x7fffffff;
                for (int q = 0; q < gSelect; ++q) {
                    const int k = bestK[q];
                    const int16_t* b = gBasis + (static_cast<size_t>(block[i].src) * kNumLane + k) * gRank;
                    const int32_t v = (dot16(b, coefRow, gRank) >> 6) + gB[k * kNumTarget + block[i].tgt];
                    if (v < best) best = v;
                }
                block[i].delay = best < 0 ? 0 : best;
            }
            // ---- write --------------------------------------------------------
            rows += n;
            for (int i = 0; i < n; ++i) {
                checksum += block[i].delay;
                if (!doWrite) continue;
                if (outLen + block[i].len + 16 > kOut) { fwrite(outbuf.data(), 1, outLen, out); outLen = 0; }
                memcpy(outbuf.data() + outLen, block[i].start, static_cast<size_t>(block[i].len));
                outLen += static_cast<size_t>(block[i].len);
                char* w = outbuf.data() + outLen;
                *w++ = ',';
                int32_t v = block[i].delay;
                if (v == 0) *w++ = '0';
                else { char t[12]; int m = 0; while (v > 0) { t[m++] = static_cast<char>('0' + v % 10); v /= 10; } while (m > 0) *w++ = t[--m]; }
                *w++ = '\n';
                outLen = static_cast<size_t>(w - outbuf.data());
            }
        }
        carry = total - pos;
        if (carry) memmove(base, base + pos, carry);
        if (got == 0) break;
    }
    if (doWrite && outLen) fwrite(outbuf.data(), 1, outLen, out);
    fclose(in);
    fclose(out);
    fprintf(stderr, "rows=%ld checksum=%lld mode=%s rank=%d coarse=%d lanes=%d select=%d\n",
            rows, checksum, mode.c_str(), gRank, gCoarseRank, gLanes, gSelect);
    return 0;
}
