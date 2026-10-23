// Micro-benchmark for the lane-level model runtime.
//
// Measures the real per-row cost of the shape
//
//     delay = gap(x0,y0,x1,y1) + min_{k in shortlist[s][t]} ( dot(basis[s][k], coef[cell]) + B[k][t] )
//
// with realistic parsing (port-name hash, coordinate decode), realistic table
// shapes and access patterns (coef 60200 x rank int16, basis 33280 x rank int16,
// shortlist 208*288 x lanes uint8, B 160 x 288 int16), and a buffered CSV writer.
//
// Three modes isolate the cost components:
//     copy   : copy the input line and append ",100"          (I/O floor)
//     parse  : parse pins, compute the cell, write a checksum (parse cost)
//     full   : parse + lane dot products + exact gap             (real runtime)
//
// Usage:
//     bench_lane <ports.txt> <gap.txt> <request.csv> <result.csv> <mode> [rank] [lanes]
//                 [coef_bytes=2] [basis_bytes=2]
//
// Compile: g++ -O3 -mavx2 -std=c++17 -o build/bench_lane tools/bench_lane.cpp

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kNumSource = 208;
constexpr int kNumTarget = 288;
constexpr int kNumLane = 160;
constexpr int kNumCell = 60200;
constexpr int kWidth = 120;
constexpr int kHeight = 550;

struct PortEntry {
    char name[24];
    int16_t source;
    int16_t target;
};

uint64_t hashName(const char* s, int len) {
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < len; ++i) {
        h ^= static_cast<uint8_t>(s[i]);
        h *= 1099511628211ull;
    }
    return h;
}

struct HashMap {
    // simple open addressing; -1 empty
    std::vector<uint64_t> keys;
    std::vector<int16_t> vals;
    uint32_t mask = 0;
    void init(int n) {
        uint32_t cap = 1;
        while (cap < static_cast<uint32_t>(n) * 4) cap <<= 1;
        keys.assign(cap, 0);
        vals.assign(cap, -1);
        mask = cap - 1;
    }
    void put(uint64_t k, int16_t v) {
        uint32_t i = static_cast<uint32_t>(k) & mask;
        while (vals[i] != -1) i = (i + 1) & mask;
        keys[i] = k;
        vals[i] = v;
    }
    inline int16_t get(uint64_t k) const {
        uint32_t i = static_cast<uint32_t>(k) & mask;
        while (true) {
            if (vals[i] == -1) return -1;
            if (keys[i] == k) return vals[i];
            i = (i + 1) & mask;
        }
    }
};

int16_t* gCoef = nullptr;      // [kNumCell][rank]
int16_t* gBasis = nullptr;     // [kNumSource*kNumLane][rank]
int16_t* gB = nullptr;         // [kNumLane][kNumTarget]
uint8_t* gShort = nullptr;     // [kNumSource*kNumTarget][lanes]
int gRank = 256;
int gLanes = 13;

int32_t gVp[kWidth + 1];
int32_t gHp[kHeight + 1];

PortEntry gPorts[512];
HashMap gMap;
uint32_t gKey[1 << 16];
int16_t gVal[1 << 16];

inline uint32_t hash32(const char* s, int len) {
    // 64-bit load + multiply-shift; port names are <= 12 bytes so two loads cover them.
    uint64_t w = 0;
    const int n1 = len < 8 ? len : 8;
    memcpy(&w, s, static_cast<size_t>(n1));
    uint64_t h = w * 0x9E3779B97F4A7C15ull;
    if (len > 8) {
        uint64_t w2 = 0;
        memcpy(&w2, s + 8, static_cast<size_t>(len - 8));
        h ^= (w2 + 0x94D049BB133111EBull) * 0xBF58476D1CE4E5B9ull;
    }
    h ^= static_cast<uint64_t>(len) * 0x2545F4914F6CDD1Dull;
    h ^= h >> 29;
    return static_cast<uint32_t>(h);
}
void buildIdTable(int n) {
    for (int i = 0; i < (1 << 16); ++i) gVal[i] = -1;
    for (int i = 0; i < n; ++i) {
        const uint32_t h = hash32(gPorts[i].name, static_cast<int>(strlen(gPorts[i].name)));
        uint32_t slot = h & 0xffffu;
        while (gVal[slot] != -1) slot = (slot + 1) & 0xffffu;
        gKey[slot] = h;
        gVal[slot] = static_cast<int16_t>(i);
    }
}
inline int16_t fastPortId(const char* p, int len) {
    const uint32_t h = hash32(p, len);
    uint32_t slot = h & 0xffffu;
    while (true) {
        const int16_t v = gVal[slot];
        if (v == -1) return -1;
        if (gKey[slot] == h) return v;
        slot = (slot + 1) & 0xffffu;
    }
}

inline int16_t lookupPort(const char* p, int len) {
    return fastPortId(p, len);
}

int parsePin(const char* s, const char* end, int& x, int& y, int& port) {
    // "SRB_<x>_<y>/<port>"
    const char* u1 = nullptr; const char* u2 = nullptr; const char* sl = nullptr;
    for (const char* p = s; p < end; ++p) {
        if (*p == '_' && !u1) { u1 = p; continue; }
        if (*p == '_' && u1 && !u2) { u2 = p; continue; }
        if (*p == '/') { sl = p; break; }
    }
    if (!u1 || !u2 || !sl) return 0;
    x = 0; for (const char* q = u1 + 1; *q >= '0' && *q <= '9'; ++q) x = x * 10 + (*q - '0');
    y = 0; for (const char* q = u2 + 1; *q >= '0' && *q <= '9'; ++q) y = y * 10 + (*q - '0');
    port = lookupPort(sl + 1, static_cast<int>(end - sl - 1));
    return port >= 0;
}

uint64_t rngState = 88172645463325252ull;
inline uint64_t xorshift() {
    rngState ^= rngState << 13;
    rngState ^= rngState >> 7;
    rngState ^= rngState << 17;
    return rngState;
}

void fillTables(int rank, int lanes, int coefBytes) {
    gRank = rank;
    gLanes = lanes;
    gCoef = static_cast<int16_t*>(aligned_alloc(64, static_cast<size_t>(kNumCell) * rank * coefBytes));
    gBasis = static_cast<int16_t*>(aligned_alloc(64, static_cast<size_t>(kNumSource) * kNumLane * rank * 2));
    gB = static_cast<int16_t*>(aligned_alloc(64, static_cast<size_t>(kNumLane) * kNumTarget * 2));
    gShort = static_cast<uint8_t*>(aligned_alloc(64, static_cast<size_t>(kNumSource) * kNumTarget * lanes));
    if (coefBytes == 1) {
        int8_t* c = reinterpret_cast<int8_t*>(gCoef);
        for (size_t i = 0; i < static_cast<size_t>(kNumCell) * rank; ++i) c[i] = static_cast<int8_t>(xorshift() & 0x7f) - 63;
    } else {
        for (size_t i = 0; i < static_cast<size_t>(kNumCell) * rank; ++i) gCoef[i] = static_cast<int16_t>(xorshift() & 0x1ff) - 255;
    }
    for (size_t i = 0; i < static_cast<size_t>(kNumSource) * kNumLane * rank; ++i)
        gBasis[i] = static_cast<int16_t>(xorshift() & 0x1ff) - 255;
    for (size_t i = 0; i < static_cast<size_t>(kNumLane) * kNumTarget; ++i)
        gB[i] = static_cast<int16_t>(xorshift() % 3000);
    for (size_t i = 0; i < static_cast<size_t>(kNumSource) * kNumTarget * lanes; ++i)
        gShort[i] = static_cast<uint8_t>(xorshift() % kNumLane);
}

inline int32_t dot(const int16_t* a, const int16_t* b, int rank) {
    int32_t acc = 0;
    for (int j = 0; j < rank; ++j) acc += static_cast<int32_t>(a[j]) * static_cast<int32_t>(b[j]);
    return acc;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: bench_lane <ports> <gap> <request> <result> <copy|parse|full> [rank] [lanes] [coef_bytes]\n");
        return 2;
    }
    const int rank = argc > 6 ? atoi(argv[6]) : 256;
    const int lanes = argc > 7 ? atoi(argv[7]) : 13;
    const int coefBytes = argc > 8 ? atoi(argv[8]) : 2;
    const std::string mode = argv[5];

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
    buildIdTable(nPorts);

    {
        FILE* f = fopen(argv[2], "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 2; }
        int w, h;
        if (fscanf(f, "%d %d", &w, &h) != 2) return 2;
        for (int i = 0; i <= w; ++i) if (fscanf(f, "%d", &gVp[i]) != 1) return 2;
        for (int i = 0; i <= h; ++i) if (fscanf(f, "%d", &gHp[i]) != 1) return 2;
        fclose(f);
    }

    fillTables(rank, lanes, coefBytes);

    FILE* in = fopen(argv[3], "rb");
    FILE* out = fopen(argv[4], "wb");
    if (!in || !out) { fprintf(stderr, "cannot open request/result\n"); return 2; }

    constexpr size_t kIn = 1 << 22;
    constexpr size_t kOut = 1 << 23;
    std::vector<char> inbuf(kIn);
    std::vector<char> outbuf(kOut + 64);
    size_t outLen = 0;
    std::memcpy(outbuf.data(), "From,To,delay\n", 14);
    outLen = 14;

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
            const char* lineStart = base + pos;
            const char* nl = static_cast<const char*>(memchr(lineStart, '\n', total - pos));
            if (!nl) break;
            const char* lineEnd = nl;   // exclusive of '\n'
            pos = static_cast<size_t>(nl - base) + 1;
            if (lineEnd > lineStart && lineEnd[-1] == '\r') --lineEnd;
            if (lineEnd - lineStart < 3) continue;
            if (lineStart[0] == 'F' && lineStart[1] == 'r' && lineStart[2] == 'o') continue;  // header
            // find the comma separating From and To
            const char* comma = static_cast<const char*>(memchr(lineStart, ',', lineEnd - lineStart));
            if (!comma) continue;

            int sx = 0, sy = 0, sp = -1, tx = 0, ty = 0, tp = -1;
            if (!parsePin(lineStart, comma, sx, sy, sp)) continue;
            if (!parsePin(comma + 1, lineEnd, tx, ty, tp)) continue;

            int delay = 0;
            if (mode == "copy") {
                delay = 100;
            } else {
                const int dx = tx - sx, dy = ty - sy;
                const int cell = ((dx + 119) * 1100 + (dy + 549)) % kNumCell;
                if (mode == "parse") {
                    delay = cell & 0x1ff;
                } else {
                    const int src = gPorts[sp].source;
                    const int tgt = gPorts[sp].target;   // unused; target from tp below
                    (void)tgt;
                    const int ts = gPorts[tp].target;
                    const int srcIdx = src >= 0 ? src : 0;
                    int32_t best = 0x7fffffff;
                    const uint8_t* lanesP = gShort + (static_cast<size_t>(srcIdx) * kNumTarget + (ts >= 0 ? ts : 0)) * gLanes;
                    const int16_t* coefRow = gCoef + static_cast<size_t>(cell) * gRank;
                    for (int l = 0; l < gLanes; ++l) {
                        const int k = lanesP[l] & 0x7f;
                        const int16_t* basisRow = gBasis + (static_cast<size_t>(srcIdx) * kNumLane + k) * gRank;
                        int32_t s2 = 0;
                        for (int j = 0; j < gRank; ++j) s2 += static_cast<int32_t>(basisRow[j]) * static_cast<int32_t>(coefRow[j]);
                        const int32_t v = (s2 >> 8) + gB[k * kNumTarget + (ts >= 0 ? ts : 0)];
                        if (v < best) best = v;
                    }
                    const int32_t gap = std::abs(gVp[tx] - gVp[sx]) + std::abs(gHp[ty] - gHp[sy]);
                    delay = best + gap;
                    if (delay < 0) delay = 0;
                }
            }
            checksum += delay;

            const size_t lineLen = static_cast<size_t>(lineEnd - lineStart);
            if (outLen + lineLen + 16 > kOut) {
                fwrite(outbuf.data(), 1, outLen, out);
                outLen = 0;
            }
            std::memcpy(outbuf.data() + outLen, lineStart, lineLen);
            outLen += lineLen;
            char* w = outbuf.data() + outLen;
            *w++ = ',';
            if (delay == 0) { *w++ = '0'; }
            else {
                char tmp[12];
                int n = 0;
                while (delay > 0) { tmp[n++] = static_cast<char>('0' + delay % 10); delay /= 10; }
                while (n > 0) *w++ = tmp[--n];
            }
            *w++ = '\n';
            outLen = static_cast<size_t>(w - outbuf.data());
            ++rows;
        }
        carry = total - pos;
        if (carry) std::memmove(base, base + pos, carry);
        if (got == 0) break;
    }
    if (outLen) fwrite(outbuf.data(), 1, outLen, out);
    fclose(in);
    fclose(out);
    fprintf(stderr, "rows=%ld checksum=%lld rank=%d lanes=%d mode=%s\n", rows, checksum, rank, lanes, mode.c_str());
    return 0;
}
