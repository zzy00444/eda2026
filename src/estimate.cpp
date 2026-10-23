// Lean estimate runtime (Plan A).
//
//   delay = gap_telescope(x0,y0,x1,y1)
//         + A[pair] + Bx[pair]*dx + By[pair]*dy
//         + su * scale[cell] * dot(U[pair], C[cell])
//
// The model blob is appended to the executable by tools/pack_lean.py; a 24-byte
// trailer ("LEANTAIL" + offset + size) at the end locates it.  The program reads
// its own executable via /proc/self/exe and mmaps the blob.
//
// Single-threaded streaming: 4 MB input buffer, 64-row double-buffered blocks so
// the model rows of block j+1 are prefetched while block j is computed.
//
// Compile: g++ -O3 -mavx2 -std=c++17 -o build/estimate src/estimate.cpp

#include <immintrin.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr int kBlock = 64;
constexpr int kText = 56;
constexpr uint32_t kMaxPorts = 512;

struct Header {
    char magic[8];
    uint32_t version, rank, nx, ny, ns, nt, nports;
    float su;
    uint32_t flags;
};

struct PortEntry {
    char name[24];
    int16_t src, tgt;
};

int gRank = 0, gNx = 0, gNy = 0, gNs = 0, gNt = 0, gNports = 0;
int gDbgCount = 0;
float gSu = 0.f;
bool gU8 = true, gC8 = false;

const float* gA = nullptr;
const float* gBx = nullptr;
const float* gBy = nullptr;
const void* gU = nullptr;
const void* gC = nullptr;
const float* gScale = nullptr;
const int32_t* gVp = nullptr;
const int32_t* gHp = nullptr;
const int8_t* gSrcVx = nullptr;
const int8_t* gSrcVy = nullptr;
const PortEntry* gPorts = nullptr;

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

inline int16_t portId(const char* p, int len) {
    const uint32_t h = hashName(p, len);
    uint32_t slot = h & 0xffffu;
    while (true) {
        const int16_t v = gVal[slot];
        if (v == -1) return -1;
        if (gKey[slot] == h) return v;
        slot = (slot + 1) & 0xffffu;
    }
}

inline int32_t dotRow(const void* cRow, const void* uRow, int rank) {
    int32_t acc = 0;
    if (gU8) {
        const int8_t* c = static_cast<const int8_t*>(cRow);
        const int8_t* u = static_cast<const int8_t*>(uRow);
        for (int j = 0; j < rank; ++j) acc += static_cast<int32_t>(c[j]) * static_cast<int32_t>(u[j]);
    } else {
        const int16_t* c = static_cast<const int16_t*>(cRow);
        const int16_t* u = static_cast<const int16_t*>(uRow);
        __m256i a = _mm256_setzero_si256();
        int j = 0;
        for (; j + 16 <= rank; j += 16) {
            a = _mm256_add_epi32(a, _mm256_madd_epi16(
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c + j)),
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(u + j))));
        }
        __m128i s = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
        s = _mm_hadd_epi32(s, s);
        s = _mm_hadd_epi32(s, s);
        acc = _mm_cvtsi128_si32(s);
        for (; j < rank; ++j) acc += static_cast<int32_t>(c[j]) * static_cast<int32_t>(u[j]);
    }
    return acc;
}

struct Row {
    char text[kText];
    uint8_t len;
    int16_t src, tgt;
    int16_t sx, sy, tx, ty;
    int32_t cell;
    int32_t delay;
};

int parseBlock(char* base, size_t total, size_t& pos, Row* out) {
    int n = 0;
    while (n < kBlock && pos < total) {
        char* lineStart = base + pos;
        char* nl = static_cast<char*>(memchr(lineStart, '\n', total - pos));
        if (!nl) break;
        char* lineEnd = nl;
        if (lineEnd > lineStart && lineEnd[-1] == '\r') --lineEnd;
        pos = static_cast<size_t>(nl - base) + 1;
        const int lineLen = static_cast<int>(lineEnd - lineStart);
        if (lineLen < 3 || lineLen >= kText) continue;
        if (lineStart[0] == 'F' && lineStart[1] == 'r' && lineStart[2] == 'o') continue;
        char* comma = static_cast<char*>(memchr(lineStart, ',', lineLen));
        if (!comma) continue;
        int sx = 0, sy = 0, tx = 0, ty = 0;
        char* p = lineStart + 4;
        while (p < comma && *p >= '0' && *p <= '9') sx = sx * 10 + (*p++ - '0');
        ++p;
        while (p < comma && *p >= '0' && *p <= '9') sy = sy * 10 + (*p++ - '0');
        ++p;
        const int sp = portId(p, static_cast<int>(comma - p));
        p = comma + 1 + 4;
        while (p < lineEnd && *p >= '0' && *p <= '9') tx = tx * 10 + (*p++ - '0');
        ++p;
        while (p < lineEnd && *p >= '0' && *p <= '9') ty = ty * 10 + (*p++ - '0');
        ++p;
        const int tp = portId(p, static_cast<int>(lineEnd - p));
        if (getenv("LEAN_DEBUG") && n < 5) fprintf(stderr, "line[%.*s] sp=%d tp=%d\\n", lineLen, lineStart, sp, tp);
        if (sp < 0 || tp < 0) continue;
        Row& r = out[n];
        memcpy(r.text, lineStart, static_cast<size_t>(lineLen));
        r.len = static_cast<uint8_t>(lineLen);
        r.src = gPorts[sp].src >= 0 ? gPorts[sp].src : 0;
        r.tgt = gPorts[tp].tgt >= 0 ? gPorts[tp].tgt : 0;
        const int dx = tx - sx, dy = ty - sy;
        r.sx = static_cast<int16_t>(sx); r.sy = static_cast<int16_t>(sy);
        r.tx = static_cast<int16_t>(tx); r.ty = static_cast<int16_t>(ty);
        r.cell = (dx + 119) * gNy + (dy + 549);
        r.delay = 0;
        ++n;
    }
    return n;
}

inline void computeRow(Row& r) {
    const int32_t pair = static_cast<int32_t>(r.src) * gNt + r.tgt;
    const int16_t vx = gSrcVx[r.src], vy = gSrcVy[r.src];
    const int fx = std::min(119, std::max(0, r.sx + vx));   // same clamp as the fit
    const int fy = std::min(549, std::max(0, r.sy + vy));
    const int32_t gap = std::abs(gVp[fx] - gVp[r.sx]) + std::abs(gVp[r.tx] - gVp[fx]) +
                        std::abs(gHp[fy] - gHp[r.sy]) + std::abs(gHp[r.ty] - gHp[fy]);
    const int cell = r.cell;
    const void* cRow = static_cast<const char*>(gC) + static_cast<size_t>(cell) * gRank * (gC8 ? 1 : 2);
    const void* uRow = static_cast<const char*>(gU) + static_cast<size_t>(pair) * gRank * (gU8 ? 1 : 2);
    const int32_t dot = dotRow(cRow, uRow, gRank);
    const float lin = gA[pair] + gBx[pair] * (r.tx - r.sx) + gBy[pair] * (r.ty - r.sy);
    const double v = static_cast<double>(lin) + static_cast<double>(dot) * gScale[cell] * gSu + gap;
    if (getenv("LEAN_DEBUG1") && gDbgCount++ < 1) {
        const int16_t* cc = static_cast<const int16_t*>(cRow);
        const int16_t* uu = static_cast<const int16_t*>(uRow);
        fprintf(stderr, "c0..3=%d,%d,%d,%d u0..3=%d,%d,%d,%d rank=%d\\n", cc[0], cc[1], cc[2], cc[3], uu[0], uu[1], uu[2], uu[3], gRank);
    }
    if (getenv("LEAN_DEBUG1") && gDbgCount < 2) {
        fprintf(stderr, "pair=%d cell=%d sx=%d sy=%d tx=%d ty=%d gap=%d lin=%.1f dot=%d sc=%.4f su=%.3g v=%.1f\\n",
                pair, cell, r.sx, r.sy, r.tx, r.ty, gap, lin, dot, gScale[cell], gSu, v);
    }
    r.delay = static_cast<int32_t>(v < 0 ? 0 : v + 0.5);
}

}  // namespace

int main(int argc, char** argv) {
    const char* inPath = nullptr;
    const char* outPath = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-in") && i + 1 < argc) inPath = argv[++i];
        else if (!strcmp(argv[i], "-out") && i + 1 < argc) outPath = argv[++i];
    }
    if (!inPath || !outPath) {
        fprintf(stderr, "usage: estimate -in <request.csv> -out <result.csv>\n");
        return 2;
    }

    // ---- locate and map the appended model blob ----
    char exePath[512];
    ssize_t n = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
    if (n <= 0) { fprintf(stderr, "cannot resolve /proc/self/exe\n"); return 2; }
    exePath[n] = 0;
    int fd = open(exePath, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "cannot open executable\n"); return 2; }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 32) { fprintf(stderr, "bad executable\n"); return 2; }
    uint64_t trailer[2];
    if (pread(fd, trailer, 16, st.st_size - 16) != 16) { fprintf(stderr, "no trailer\n"); return 2; }
    char magic[8];
    if (pread(fd, magic, 8, st.st_size - 24) != 8 || memcmp(magic, "LEANTAIL", 8) != 0) {
        fprintf(stderr, "model blob missing (append it with tools/pack_lean.py)\n");
        return 2;
    }
    void* map = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); return 2; }
    const char* blob = static_cast<const char*>(map) + trailer[0];
    const uint64_t blobSize = trailer[1];
    Header h;
    memcpy(&h, blob, sizeof(Header));
    if (memcmp(h.magic, "LEAN0001", 8) != 0) { fprintf(stderr, "bad model magic\n"); return 2; }
    gRank = h.rank; gNx = h.nx; gNy = h.ny; gNs = h.ns; gNt = h.nt; gNports = h.nports;
    gSu = h.su; gU8 = (h.flags & 1) != 0; gC8 = (h.flags & 2) != 0;

    // section table lives in the footer at the end of the blob
    const char* footer = blob + blobSize - 188;
    if (memcmp(footer, "LEANIDX1", 8) != 0) { fprintf(stderr, "bad section footer\n"); return 2; }
    const uint32_t nsect = *reinterpret_cast<const uint32_t*>(footer + 8);
    if (nsect != 11) { fprintf(stderr, "unexpected section count %u\n", nsect); return 2; }
    const uint64_t* sec = reinterpret_cast<const uint64_t*>(footer + 12);
    auto at = [&](int i) { return blob + sec[2 * i]; };
    gA = reinterpret_cast<const float*>(at(0));
    gBx = reinterpret_cast<const float*>(at(1));
    gBy = reinterpret_cast<const float*>(at(2));
    gU = at(3);
    gC = at(4);
    gScale = reinterpret_cast<const float*>(at(5));
    gVp = reinterpret_cast<const int32_t*>(at(6));
    gHp = reinterpret_cast<const int32_t*>(at(7));
    gSrcVx = reinterpret_cast<const int8_t*>(at(8));
    gSrcVy = reinterpret_cast<const int8_t*>(at(9));
    gPorts = reinterpret_cast<const PortEntry*>(at(10));

    if (getenv("LEAN_DEBUG1")) fprintf(stderr, "stage: footer ok rank=%d nports=%d name0=[%.8s]\n", gRank, gNports, gPorts[0].name);
    for (int i = 0; i < (1 << 16); ++i) gVal[i] = -1;
    if (getenv("LEAN_DEBUG1")) fprintf(stderr, "stage: table zeroed\\n");
    for (int i = 0; i < gNports; ++i) {
        const uint32_t hh = hashName(gPorts[i].name, static_cast<int>(strlen(gPorts[i].name)));
        uint32_t slot = hh & 0xffffu;
        while (gVal[slot] != -1) slot = (slot + 1) & 0xffffu;
        gKey[slot] = hh;
        gVal[slot] = static_cast<int16_t>(i);
    }

    if (getenv("LEAN_DEBUG1")) fprintf(stderr, "stage: hash built, port0=%.24s\\n", gPorts[0].name);
    FILE* in = fopen(inPath, "rb");
    FILE* out = fopen(outPath, "wb");
    if (!in || !out) { fprintf(stderr, "cannot open -in/-out files\n"); return 2; }

    constexpr size_t kIn = 1 << 22;
    constexpr size_t kOut = 1 << 23;
    std::vector<char> inbuf(kIn + 512);
    std::vector<char> outbuf(kOut + 4096);
    size_t outLen = 14;
    memcpy(outbuf.data(), "From,To,delay\n", 14);

    Row blocks[2][kBlock];
    int counts[2] = {0, 0};
    int cur = 0;
    size_t pos = 0, total = fread(inbuf.data(), 1, kIn, in);
    char* base = inbuf.data();
    counts[0] = total ? parseBlock(base, total, pos, blocks[0]) : 0;

    while (counts[cur] > 0) {
        const int nxt = 1 - cur;
        int got = parseBlock(base, total, pos, blocks[nxt]);
        if (got == 0) {
            const size_t carry = total - pos;
            if (carry) memmove(base, base + pos, carry);
            pos = 0;
            const size_t r = fread(base + carry, 1, kIn - carry, in);
            total = carry + r;
            if (r > 0) got = parseBlock(base, total, pos, blocks[nxt]);
        }
        counts[nxt] = got;
        // prefetch the next block's model rows
        for (int i = 0; i < counts[nxt]; ++i) {
            const int32_t pair = static_cast<int32_t>(blocks[nxt][i].src) * gNt + blocks[nxt][i].tgt;
            const char* c = static_cast<const char*>(gC) + static_cast<size_t>(blocks[nxt][i].cell) * gRank * (gC8 ? 1 : 2);
            const char* u = static_cast<const char*>(gU) + static_cast<size_t>(pair) * gRank * (gU8 ? 1 : 2);
            const int bytes = gRank * ((gC8 || gU8) ? 1 : 2);
            for (int off = 0; off < bytes; off += 64) __builtin_prefetch(c + off, 0, 3);
            for (int off = 0; off < bytes; off += 64) __builtin_prefetch(u + off, 0, 3);
            __builtin_prefetch(&gScale[blocks[nxt][i].cell], 0, 3);
        }
        for (int i = 0; i < counts[cur]; ++i) computeRow(blocks[cur][i]);
        for (int i = 0; i < counts[cur]; ++i) {
            const Row& r = blocks[cur][i];
            if (outLen + r.len + 16 > kOut) { fwrite(outbuf.data(), 1, outLen, out); outLen = 0; }
            memcpy(outbuf.data() + outLen, r.text, r.len);
            outLen += r.len;
            char* w = outbuf.data() + outLen;
            *w++ = ',';
            int32_t v = r.delay;
            if (v == 0) *w++ = '0';
            else { char t[12]; int m = 0; while (v > 0) { t[m++] = static_cast<char>('0' + v % 10); v /= 10; } while (m > 0) *w++ = t[--m]; }
            *w++ = '\n';
            outLen = static_cast<size_t>(w - outbuf.data());
        }
        cur = nxt;
    }
    if (outLen) fwrite(outbuf.data(), 1, outLen, out);
    fclose(in);
    fclose(out);
    return 0;
}
