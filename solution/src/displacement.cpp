// Pango SRB net-delay estimator -- submission build.
//   usage: estimate -in request.csv -out estimate_result.csv
//
// Model:
//   D(a,b,delta) = mean(a,b) + sum_r U_r(a,b) * V_r(delta)
//                  + wall crossing penalty (292 ps) + gap-line telescope
//
// Self contained: the tables (vt / U / M / ids / arch) are appended to this
// executable by tools/pack_sub.py and located through /proc/self/exe, so the
// file runs on its own.  The same tables are still accepted as separate files
// next to the executable (development convenience).
//
// Memory: the request file is streamed in 16 MB blocks and rows are handled in
// three phases per block, so peak RSS is bounded by the tables (~100 MB) plus
// the block buffers (~150 MB for a 16 MB block), far below 2 GB, regardless of
// the input size.
//
// The low-rank estimate and geometric penalties are unchanged. A small
// embedded target-port / predicted-delay table applies a fitted multiplicative
// residual correction after the original integer estimate is formed.
//   * the rank-128 dot product runs in an AVX2+FMA kernel selected at run time
//     through __builtin_cpu_support("avx2") && ("fma"); a scalar kernel is
//     always kept, so a CPU without AVX2 can never SIGILL.  The whole file is
//     still compiled without -mavx2: only the kernel is target-attributed.
//   * block-wise locality (see the phase comments in main): the block is parsed
//     first (phase A), the (row, candidate) dot jobs are radix sorted by Vt row
//     index (phase B) so the 68 MB Vt table is walked in ascending order --
//     sequential and TLB friendly -- while the U rows come from a much smaller
//     working set (~8k distinct rows / 4 MB on the benchmark data).  Phase C
//     scatters the results back into input order and emits the output.
//   * each dot job carries its own U offset / M lookup / penalty, so phase B
//     never touches the record array: it is a pure sequential read of 16-byte
//     jobs, and the 11-bit radix passes keep the scatter streams cache resident.
//   * pin names are packed into one 64-bit word and resolved by a single
//     multiply-shift perfect hash read (seed searched at start-up); the entry
//     carries port id + source index + target index in one load.
//   * line parsing is SWAR: separators are found 8 bytes at a time and the
//     coordinate fields are decoded from one 8-byte load each.
//   * phase C copies each key straight out of the block buffer with three
//     overlapping 16-byte stores (no memcpy call, no staging buffer) and the
//     delay is formatted from a two-digit table (3-4 digits are one divide).
#include <bits/stdc++.h>
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#define EST_X86 1
#else
#define EST_X86 0
#endif
#ifdef _WIN32
// Declared by hand so that <windows.h> (which defines a conflicting `Arc`) is
// not pulled in; kernel32 is linked by default.
extern "C" __declspec(dllimport) unsigned long __stdcall GetModuleFileNameA(void*, char*, unsigned long);
#endif
using namespace std;

struct SRBArc { int to; int w; };
static int NP = 0, W = 120, H = 550;
static vector<string> portName;
static vector<vector<SRBArc>> arcsFrom;
static int nSrc = 0, nTgt = 0, RANK = 0, NC = 0, NCX = 0, NCY = 0, CX0 = 0, CY0 = 0, XSTEP = 1, YSTEP = 1;
static vector<int16_t> Vt;
static vector<float> U;
static vector<int16_t> M;
static vector<float> Calib;
// Must match the bins used by tools/fit_calibration.py.
static const int CALIB_BINS[4] = {700, 1000, 2000, 4000};
static vector<int> srcPort, tgtPort, srcIdxOf, tgtIdxOf, netdx, netdy;
static vector<int> vpre, hpre, doorpre;
static vector<char> doorY;

// ---------------------------------------------------------------- blob access
// Footer appended at the very end of this executable (112 bytes, little endian):
//   magic "PANGDL02", uint64 offset[6], uint64 size[6], uint32 count=6, version=2
static const int NTAB = 6;
static unsigned long long TB_OFF[NTAB];
static bool TB_EMB = false;
static string TB_SELF;

static void blobInit(const char* argv0) {
    string self;
    char buf[4096];
#ifdef _WIN32
    unsigned long wn = GetModuleFileNameA(NULL, buf, (unsigned long)sizeof(buf));
    if (wn > 0 && wn < (unsigned long)sizeof(buf)) self.assign(buf, (size_t)wn);
#else
    ssize_t rn = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (rn > 0) { buf[rn] = 0; self.assign(buf, (size_t)rn); }
#endif
    if (self.empty() && argv0) {
        FILE* t = fopen(argv0, "rb");
        if (t) { fclose(t); self = argv0; }
    }
    if (self.empty()) return;
    TB_SELF = self;
    FILE* f = fopen(TB_SELF.c_str(), "rb");
    if (!f) return;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return; }
    long long sz = ftell(f);
    if (sz < 112) { fclose(f); return; }
    unsigned char foot[112];
    if (fseek(f, (long)(sz - 112), SEEK_SET) != 0) { fclose(f); return; }
    if (fread(foot, 1, 112, f) != 112) { fclose(f); return; }
    fclose(f);
    if (memcmp(foot, "PANGDL02", 8) != 0) return;
    unsigned int nTabs = 0, ver = 0;
    memcpy(&nTabs, foot + 104, 4);
    memcpy(&ver, foot + 108, 4);
    if (nTabs != (unsigned)NTAB || ver != 2) return;
    for (int i = 0; i < NTAB; i++) {
        unsigned long long o = 0;
        memcpy(&o, foot + 8 + 8 * i, 8);
        TB_OFF[i] = o;
    }
    TB_EMB = true;
}

static FILE* openBlob(int idx) {
    if (!TB_EMB) return NULL;
    FILE* f = fopen(TB_SELF.c_str(), "rb");
    if (!f) return NULL;
    if (fseek(f, (long)TB_OFF[idx], SEEK_SET) != 0) { fclose(f); return NULL; }
    return f;
}

static string exeDir(const char* argv0) {
    string s = argv0 ? argv0 : "";
    size_t p = s.find_last_of("/\\");
    return (p == string::npos) ? string("") : s.substr(0, p + 1);
}

// ------------------------------------------------------------ dot product
// s = sum_{r<R} u[r] * (float)v[r].  R is RANK, known only at run time, so the
// 8-wide body is followed by a fully generic scalar tail.
static float dotScalar(const float* __restrict u, const int16_t* __restrict v, int R) {
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    int r = 0;
    for (; r + 4 <= R; r += 4) {
        s0 += u[r] * (float)v[r];
        s1 += u[r + 1] * (float)v[r + 1];
        s2 += u[r + 2] * (float)v[r + 2];
        s3 += u[r + 3] * (float)v[r + 3];
    }
    float s = (s0 + s1) + (s2 + s3);
    for (; r < R; r++) s += u[r] * (float)v[r];
    return s;
}

#if EST_X86
__attribute__((target("avx2,fma")))
static float dotAvx2(const float* __restrict u, const int16_t* __restrict v, int R) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    int r = 0;
    for (; r + 16 <= R; r += 16) {
        __m256i v0 = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)(v + r)));
        __m256i v1 = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)(v + r + 8)));
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(u + r), _mm256_cvtepi32_ps(v0), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(u + r + 8), _mm256_cvtepi32_ps(v1), a1);
    }
    for (; r + 8 <= R; r += 8) {
        __m256i v0 = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)(v + r)));
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(u + r), _mm256_cvtepi32_ps(v0), a0);
    }
    a0 = _mm256_add_ps(a0, a1);
    __m128 lo = _mm256_castps256_ps128(a0), hi = _mm256_extractf128_ps(a0, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    s4 = _mm_add_ss(s4, _mm_shuffle_ps(s4, s4, 1));
    float s = _mm_cvtss_f32(s4);
    for (; r < R; r++) s += u[r] * (float)v[r];
    return s;
}
#endif

#if EST_X86
static bool gDotAvx2 = false;
#endif

static inline float dotProduct(const float* __restrict u, const int16_t* __restrict v, int R) {
#if EST_X86
    if (gDotAvx2) return dotAvx2(u, v, R);
#endif
    return dotScalar(u, v, R);
}

static inline void pfT0(const void* p) {
#if EST_X86
    _mm_prefetch((const char*)p, _MM_HINT_T0);
#else
    (void)p;
#endif
}


// ------------------------------------------------------------ byte scanning
// The request lines are short (~35 bytes) and have a fixed shape, so the field
// separators are located 8 bytes at a time instead of byte by byte.
static inline uint64_t zbyte(uint64_t x) {          // 0x80 in the bytes that are zero
    return (x - 0x0101010101010101ull) & ~x & 0x8080808080808080ull;
}
static inline uint64_t load64(const void* p) { uint64_t v; memcpy(&v, p, 8); return v; }

// first occurrence of ch in [p, end); returns end if absent
static inline const char* findByte(const char* p, const char* end, char ch) {
    const uint64_t pat = 0x0101010101010101ull * (uint64_t)(unsigned char)ch;
    while (p + 8 <= end) {
        uint64_t x = load64(p) ^ pat;
        uint64_t m = (x - 0x0101010101010101ull) & ~x & 0x8080808080808080ull;
        if (m) return p + (__builtin_ctzll(m) >> 3);
        p += 8;
    }
    while (p < end && *p != ch) p++;
    return p;
}

// 0..3 decimal digits without a branch; the loop tail keeps longer runs (and
// non-digit bytes, which the original folded in as *p - '0' too) bit identical.
static inline int parseDigits(const char* p, int n) {
    if (n > 3) {
        int v = 0;
        for (int i = 0; i < n; i++) v = v * 10 + (p[i] - '0');
        return v;
    }
    int d0 = p[0] - '0';
    int d1 = p[1] - '0';
    int d2 = p[2] - '0';
    int v2 = d0 * 10 + d1;
    int v3 = v2 * 10 + d2;
    int v = (n == 3) ? v3 : ((n == 2) ? v2 : d0);
    return (n <= 0) ? 0 : v;
}

// One "SRB_<x>_<y>/<pin>" field, bounded by `end`.
// Returns false exactly when the original inline scan would `continue`.
static inline bool parseField(const char* s, const char* end, int& x, int& y,
                              const char*& pin, int& pinLen) {
    const char* q = s + 4;
    if (q >= end) return false;
    if (q + 8 <= end) {
        uint64_t w = load64(q);
        uint64_t mu = zbyte(w ^ 0x5F5F5F5F5F5F5F5Full);
        uint64_t ms = zbyte(w ^ 0x2F2F2F2F2F2F2F2Full);
        if (mu && ms) {
            int iu = (int)(__builtin_ctzll(mu) >> 3);
            int is = (int)(__builtin_ctzll(ms) >> 3);
            if (is > iu) {
                x = parseDigits(q, iu);
                y = parseDigits(q + iu + 1, is - iu - 1);
                pin = q + is + 1;
                pinLen = (int)(end - pin);
                return true;
            }
        }
    }
    const char* u = q;
    while (u < end && *u != '_') u++;
    if (u >= end) return false;
    const char* sl = u + 1;
    while (sl < end && *sl != '/') sl++;
    if (sl >= end) return false;
    x = parseDigits(q, (int)(u - q));
    y = parseDigits(u + 1, (int)(sl - u - 1));
    pin = sl + 1;
    pinLen = (int)(end - pin);
    return true;
}

// ------------------------------------------------------------ pin name hash
// 4 bytes at a time (the pin names average 5.6 characters), length folded in.
// The table build probes to a free slot from this index, exactly as before, so
// the hash stays collision free over the 496 names.
static inline uint32_t loadU32(const char* p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}
static inline uint32_t nhash(const char* s, int n) {
    uint32_t h = 0x811C9DC5u ^ ((uint32_t)n * 0x9E3779B1u);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        h = (h ^ loadU32(s + i)) * 0x85EBCA6Bu;
        h ^= h >> 13;
    }
    uint32_t t = 0;
    for (; i < n; i++) t = (t << 8) | (uint32_t)(unsigned char)s[i];
    h = (h ^ t) * 0xC2B2AE35u;
    h ^= h >> 15;
    return h & 0xFFFFu;
}

// ------------------------------------------------------------ fast digits
// Two decimal digits per lookup; identical bytes to the old divide loop.
static const char DD2[201] =
    "00010203040506070809101112131415161718192021222324252627282930313233343536373839"
    "40414243444546474849505152535455565758596061626364656667686970717273747576777879"
    "8081828384858687888990919293949596979899";

// writes the decimal digits of v (v > 0) without leading zeros, returns length
static inline int fmtPos(char* op, uint32_t v) {
    // Delays in this data set are 3-4 digits; one divide handles up to 9999.
    if (v < 10000u) {
        uint32_t hi = v / 100u, lo = v % 100u;
        if (v >= 1000u) {
            op[0] = DD2[2 * hi]; op[1] = DD2[2 * hi + 1];
            op[2] = DD2[2 * lo]; op[3] = DD2[2 * lo + 1];
            return 4;
        }
        if (v >= 100u) {
            op[0] = DD2[2 * hi + 1];
            op[1] = DD2[2 * lo]; op[2] = DD2[2 * lo + 1];
            return 3;
        }
        if (v >= 10u) { op[0] = DD2[2 * v]; op[1] = DD2[2 * v + 1]; return 2; }
        op[0] = (char)('0' + v);
        return 1;
    }
    char t[12];
    int m = 0;
    while (v >= 100) {
        uint32_t r = v % 100;
        v /= 100;
        t[m++] = DD2[2 * r + 1];
        t[m++] = DD2[2 * r];
    }
    if (v >= 10) { t[m++] = DD2[2 * v + 1]; t[m++] = DD2[2 * v]; }
    else t[m++] = (char)('0' + v);
    for (int i = 0; i < m; i++) op[i] = t[m - 1 - i];
    return m;
}

// ------------------------------------------------------------ radix sort
// Each (row, candidate) dot job carries everything the kernel needs, so the
// sorted pass is a pure sequential read: no random lookups into the record
// array.  16 bytes per job.
//
//   key  = Vt row index * 2 + candidate (0 = vt0/cpen0, 1 = vt1/cpen1)
//   idx  = record index inside the block (result slot)
//   uoff = offset of the U vector inside U (element index)
//   mrow = M lookup, cpen = wall-crossing penalty of this candidate
struct Job {
    uint32_t key;
    uint32_t idx;
    uint32_t uoff;
    int16_t mrow;
    int16_t cpen;
};

// 11-bit radix passes: 2048 buckets keep the number of concurrent scatter
// streams (and therefore the write-combining working set) small, which is a
// lot faster than the 64K-bucket variant on this table size.
static size_t RS_CNT[2048];

// returns the buffer holding the sorted result
static Job* radixSortJobs(Job* a, Job* b, size_t n, uint32_t maxKey) {
    memset(RS_CNT, 0, sizeof(RS_CNT));
    for (size_t i = 0; i < n; i++) RS_CNT[a[i].key & 2047u]++;
    {
        size_t s = 0;
        for (int k = 0; k < 2048; k++) { size_t c = RS_CNT[k]; RS_CNT[k] = s; s += c; }
    }
    for (size_t i = 0; i < n; i++) {
        Job v = a[i];
        b[RS_CNT[v.key & 2047u]++] = v;
    }
    if (maxKey < 2048u) {
        memcpy(a, b, n * sizeof(Job));
        return a;
    }
    memset(RS_CNT, 0, sizeof(RS_CNT));
    for (size_t i = 0; i < n; i++) RS_CNT[(b[i].key >> 11) & 2047u]++;
    {
        size_t s = 0;
        for (int k = 0; k < 2048; k++) { size_t c = RS_CNT[k]; RS_CNT[k] = s; s += c; }
    }
    for (size_t i = 0; i < n; i++) {
        Job v = b[i];
        a[RS_CNT[(v.key >> 11) & 2047u]++] = v;
    }
    return a;
}


// One row of the block, for the output pass.  keyOff / keyLen address the
// "From,To" text inside the block buffer, so the output line is built from the
// input bytes directly (no staging copy).  The dot-only fields (U offset, Vt
// rows, M lookup) travel in the Job stream instead. The target index selects
// one row in the small embedded residual calibration table.
struct Rec {
    int32_t keyOff;
    uint32_t keyLen;
    int32_t lgap;       // gap-line telescope term
    int16_t netWall;    // 0 or 292
    int16_t cpen0, cpen1;
    uint8_t ncd;        // candidates 0..2
    uint8_t same;       // self query
    uint16_t tgtIdx;    // target-port index for residual calibration
};

#ifndef EST_BLOCK_MB
#define EST_BLOCK_MB 16
#endif
#ifndef EST_PF
#define EST_PF 12
#endif
#ifndef EST_PFU
#define EST_PFU 4
#endif
int main(int argc, char** argv) {
    const char* inPath = NULL; const char* outPath = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-in") && i + 1 < argc) inPath = argv[++i];
        else if (!strcmp(argv[i], "-out") && i + 1 < argc) outPath = argv[++i];
    }
    if (!inPath || !outPath) {
        fprintf(stderr, "usage: estimate -in <input_file> -out <output_file>\n");
        return 1;
    }
    string D = exeDir(argv[0]);
    blobInit(argv[0]);
#if EST_X86
    __builtin_cpu_init();
    gDotAvx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif

    auto openAny = [&](const string& base) -> FILE* {
        string p = D + base; FILE* f = fopen(p.c_str(), "rb"); if (f) return f;
        f = fopen(base.c_str(), "rb"); if (f) return f;
        p = "/w/" + base; return fopen(p.c_str(), "rb");
    };
    auto openT = [&](int idx, const char* base) -> FILE* {
        FILE* f = openBlob(idx);
        if (f) return f;
        return openAny(base);
    };

    // ---- architecture (line sites, wall doors, port names) ----
    {
        FILE* af = openT(4, "oracle_arch.txt");
        if (!af) af = openAny("arch.txt");
        if (!af) { fprintf(stderr, "cannot open arch file\n"); return 1; }
        string archText;
        {
            char tmp[1 << 16]; size_t k;
            while ((k = fread(tmp, 1, sizeof(tmp), af)) > 0) archText.append(tmp, k);
        }
        fclose(af);
        // Hand rolled whitespace tokenizer: same tokens as `in >> t` on this file
        // (space separated ASCII), but ~6x faster than istringstream extraction.
        static vector<int> vs, vd, hs, hd;
        struct Blk { int lo, hi, lf, rt; };
        vector<Blk> blks;
        const char* cur = archText.data();
        const char* const aend = cur + archText.size();
        auto nextTok = [&](const char*& s, int& len) -> bool {
            while (cur < aend && (unsigned char)*cur <= ' ') cur++;
            if (cur >= aend) return false;
            s = cur;
            while (cur < aend && (unsigned char)*cur > ' ') cur++;
            len = (int)(cur - s);
            return true;
        };
        auto skipTok = [&]() { const char* s = 0; int n = 0; nextTok(s, n); };
        auto nextInt = [&]() -> int {
            const char* s = 0; int len = 0;
            if (!nextTok(s, len)) return 0;
            int v = 0, i = 0;
            bool neg = false;
            if (i < len && s[i] == '-') { neg = true; i++; }
            for (; i < len; i++) v = v * 10 + (s[i] - '0');
            return neg ? -v : v;
        };
        auto tokEq = [](const char* s, int n, const char* k) {
            return (int)strlen(k) == n && memcmp(s, k, (size_t)n) == 0;
        };
        const char* ts = 0; int tl = 0;
        while (nextTok(ts, tl)) {
            if (tokEq(ts, tl, "NPORTS")) { NP = nextInt(); portName.resize(NP); arcsFrom.resize(NP); for (int i = 0; i < NP; i++) { const char* c = 0; const char* n = 0; const char* d = 0; int cl = 0, nl = 0, dl = 0; nextTok(c, cl); nextTok(n, nl); nextTok(d, dl); portName[i].assign(n, (size_t)nl); } }
            else if (tokEq(ts, tl, "NARCS")) { int n = nextInt(); for (int i = 0; i < n; i++) { int a, b, d; skipTok(); a = nextInt(); b = nextInt(); d = nextInt(); arcsFrom[a].push_back({b, d}); } }
            else if (tokEq(ts, tl, "NNETS")) { int n = nextInt(); netdx.assign(NP, 0); netdy.assign(NP, 0); for (int i = 0; i < n; i++) { int a, b, dx, dy; skipTok(); a = nextInt(); b = nextInt(); dx = nextInt(); dy = nextInt(); if (a >= 0 && a < NP) { netdx[a] = dx; netdy[a] = dy; } (void)b; } }
            else if (tokEq(ts, tl, "NSRB")) { nextInt(); }
            else if (tokEq(ts, tl, "GRID")) { W = nextInt(); H = nextInt(); }
            else if (tokEq(ts, tl, "NLINES")) { int n = nextInt(); for (int i = 0; i < n; i++) { int s, dl; const char* d = 0; int dl2 = 0; skipTok(); nextTok(d, dl2); s = nextInt(); dl = nextInt(); if (dl2 == 8 && memcmp(d, "vertical", 8) == 0) { vs.push_back(s); vd.push_back(dl); } else { hs.push_back(s); hd.push_back(dl); } } }
            else if (tokEq(ts, tl, "NBLOCKS")) { int n = nextInt(); for (int i = 0; i < n; i++) { Blk b; int vc, hc, vdd, hdd; skipTok(); b.lo = nextInt(); b.hi = nextInt(); b.lf = nextInt(); b.rt = nextInt(); vc = nextInt(); vdd = nextInt(); hc = nextInt(); hdd = nextInt(); (void)vc; (void)vdd; (void)hc; (void)hdd; blks.push_back(b); } }
        }
        vpre.assign(W + 2, 0); hpre.assign(H + 2, 0); doorpre.assign(H + 2, 0); doorY.assign(H, 0);
        for (size_t i = 0; i < vs.size(); i++) if (vs[i] + 1 < (int)vpre.size()) vpre[vs[i] + 1] += vd[i];
        for (int i = 1; i < (int)vpre.size(); i++) vpre[i] += vpre[i - 1];
        for (size_t i = 0; i < hs.size(); i++) if (hs[i] + 1 < (int)hpre.size()) hpre[hs[i] + 1] += hd[i];
        for (int i = 1; i < (int)hpre.size(); i++) hpre[i] += hpre[i - 1];
        for (int y = 0; y < H; y++) { bool blk = false; for (size_t i = 0; i < blks.size(); i++) { const Blk& b = blks[i]; if (80 >= b.lf && 80 <= b.rt && y >= b.lo && y <= b.hi) blk = true; } doorY[y] = blk ? 0 : 1; }
        for (int y = 0; y < H; y++) doorpre[y + 1] = doorpre[y] + (doorY[y] ? 1 : 0);
    }
    // ---- basis ----
    {
        FILE* f = openT(0, "oracle_vt.bin");
        if (!f) { fprintf(stderr, "cannot open oracle_vt.bin\n"); return 1; }
        int h[6]; if (fread(h, 4, 6, f) != 6) return 1;
        RANK = h[0]; NC = h[1]; NCX = h[2]; NCY = h[3]; CX0 = h[4]; CY0 = h[5];
        vector<float> sc(RANK);
        if (fread(sc.data(), 4, RANK, f) != (size_t)RANK) return 1;
        Vt.resize((size_t)NC * RANK);
        if (fread(Vt.data(), 2, Vt.size(), f) != Vt.size()) return 1;
        fclose(f);
    }
    {
        FILE* f = openT(1, "oracle_U.bin");
        if (!f) { fprintf(stderr, "cannot open oracle_U.bin\n"); return 1; }
        int h[6]; if (fread(h, 4, 6, f) != 6) return 1;
        nSrc = h[0]; nTgt = h[1]; int R2 = h[2]; XSTEP = h[3]; YSTEP = h[4];
        if (R2 != RANK) { fprintf(stderr, "rank mismatch\n"); return 1; }
        U.resize((size_t)nSrc * nTgt * RANK);
        if (fread(U.data(), 4, U.size(), f) != U.size()) return 1;
        fclose(f);
    }
    {
        FILE* f = openT(2, "oracle_M.bin");
        if (!f) { fprintf(stderr, "cannot open oracle_M.bin\n"); return 1; }
        int h[3]; if (fread(h, 4, 3, f) != 3) return 1;
        M.resize((size_t)nSrc * nTgt);
        if (fread(M.data(), 2, M.size(), f) != M.size()) return 1;
        fclose(f);
    }
    {
        FILE* f = openT(3, "oracle_ids.txt");
        if (!f) { fprintf(stderr, "cannot open oracle_ids.txt\n"); return 1; }
        srcPort.resize(nSrc);
        for (int i = 0; i < nSrc; i++) if (fscanf(f, "%d", &srcPort[i]) != 1) return 1;
        int v; while (fscanf(f, "%d", &v) == 1) tgtPort.push_back(v);
        fclose(f);
    }
    // fold the basis scale into U
    {
        FILE* f = openT(0, "oracle_vt.bin");
        if (!f) return 1;
        int h[6]; if (fread(h, 4, 6, f) != 6) { fclose(f); return 1; }
        vector<float> sc(RANK); if (fread(sc.data(), 4, RANK, f) != (size_t)RANK) { fclose(f); return 1; }
        fclose(f);
        for (size_t i = 0; i < (size_t)nSrc * nTgt; i++) for (int r = 0; r < RANK; r++) U[i * RANK + r] *= sc[r];
    }
    tgtIdxOf.assign(NP, -1); for (int i = 0; i < (int)tgtPort.size(); i++) tgtIdxOf[tgtPort[i]] = i;
    srcIdxOf.assign(NP, -1); for (int i = 0; i < nSrc; i++) srcIdxOf[srcPort[i]] = i;

    Calib.assign((size_t)nTgt * 5, 1.f);
    {
        FILE* f = openT(5, "oracle_calibration.bin");
        if (f) {
            size_t need = Calib.size();
            if (fread(Calib.data(), sizeof(float), need, f) != need) {
                fprintf(stderr, "invalid calibration table\n"); return 1;
            }
            fclose(f);
        }
    }

    // ---- perfect hash over port names ----
    // Pin names are at most 8 characters long in the arch data, so a name fits
    // in one zero padded 64-bit word (names never contain a NUL byte, so the
    // packing is unique).  A multiply-shift seed that maps all NP names to
    // distinct 16-bit slots is searched once at start-up; a lookup is then one
    // multiply, one shift and one table read - no probe loop, no memcmp, no
    // pointer chase into the std::string objects.  If a name were longer than
    // 8 bytes the old probing hash is used instead.
    vector<int16_t> ht(1 << 16, -1);
    for (int i = 0; i < NP; i++) { uint32_t h = nhash(portName[i].data(), (int)portName[i].size()); while (ht[h] >= 0) h = (h + 1) & 0xFFFF; ht[h] = (int16_t)i; }
    // One table read returns everything a line needs about a pin: the port id
    // (for netdx/netdy) plus its source and target index, all offset by one so
    // that a zero entry means "not a known pin".
    //   bits 0..9  = port id + 1
    //   bits 10..19 = srcIdxOf[port] + 1
    //   bits 20..29 = tgtIdxOf[port] + 1
    static uint32_t PIN[1 << 16];
    vector<uint32_t> pval(NP);
    vector<uint64_t> pkey(NP);
    int maxNameLen = 0;
    for (int i = 0; i < NP; i++) {
        int L = (int)portName[i].size();
        if (L > maxNameLen) maxNameLen = L;
        uint64_t k = 0;
        for (int c = 0; c < L && c < 8; c++) k |= (uint64_t)(unsigned char)portName[i][c] << (8 * c);
        pkey[i] = k;
        pval[i] = (uint32_t)(i + 1)
                | ((uint32_t)(srcIdxOf[i] + 1) << 10)
                | ((uint32_t)(tgtIdxOf[i] + 1) << 20);
    }
    uint64_t pinSeed = 0;
    bool pinFast = (maxNameLen <= 8 && NP <= 60000);
    if (pinFast) {
        uint64_t seed = 0x9E3779B97F4A7C15ull;
        for (int attempt = 0; attempt < 4096; attempt++) {
            memset(PIN, 0, sizeof(PIN));
            bool ok = true;
            for (int i = 0; i < NP; i++) {
                uint32_t h = (uint32_t)((pkey[i] * seed) >> 48);
                if (PIN[h] != 0) { ok = false; break; }
                PIN[h] = pval[i];
            }
            if (ok) { pinSeed = seed; break; }
            seed = seed * 0x2545F4914F6CDD1Dull + 0x9E3779B97F4A7C15ull;
        }
        pinFast = (pinSeed != 0);
    }
    // Packed lookup: 0 when the name is not a known pin.
    auto pinLookup = [&](const char* s, int n) -> uint32_t {
        if (n <= 0) return 0;
        if (pinFast) {
            if (n > 8) return 0;                  // no arch pin name that long
            uint64_t k = load64(s);
            if (n < 8) k &= ((uint64_t)1 << (8 * n)) - 1;
            return PIN[(uint32_t)((k * pinSeed) >> 48)];
        }
        uint32_t h = nhash(s, n);
        while (ht[h] >= 0) {
            int id = ht[h];
            if ((int)portName[id].size() == n && memcmp(portName[id].data(), s, n) == 0) return pval[id];
            h = (h + 1) & 0xFFFF;
        }
        return 0;
    };

    fprintf(stderr, "oracle: nSrc=%d nTgt=%d rank=%d NC=%d ports=%d embedded=%d avx2=%d\n",
            nSrc, nTgt, RANK, NC, NP, (int)TB_EMB,
#if EST_X86
            (int)gDotAvx2
#else
            0
#endif
            );

    // ---- streaming I/O ----
    FILE* fi = fopen(inPath, "rb");
    if (!fi) { fprintf(stderr, "cannot open %s\n", inPath); return 1; }
    FILE* fo = fopen(outPath, "wb");
    if (!fo) { fclose(fi); fprintf(stderr, "cannot write %s\n", outPath); return 1; }
    { const char* hdr = "From,To,delay\n"; size_t hl = strlen(hdr); if (fwrite(hdr, 1, hl, fo) != hl) { fclose(fo); fclose(fi); return 1; } }

    // ---- block buffers ----
    // Buffers are allocated uninitialised (new T[n] leaves PODs alone): touching
    // them costs page faults once, and zeroing them would be pure overhead.
    const size_t BLOCKB = (size_t)EST_BLOCK_MB << 20;   // input bytes per block
    const size_t PAD = 512;                             // slack for unaligned reads
    // one record needs at least 16 input bytes; the flush path below keeps this
    // safe for shorter lines too, so a generous bound is enough.
    const size_t CAP = BLOCKB / 16 + 1024;
    char* const dat = new char[BLOCKB + PAD + 64];
    Rec* const RC = new Rec[CAP];
    Job* const JA = new Job[CAP * 2];
    Job* const JB = new Job[CAP * 2];
    long* const RA = new long[CAP];
    long* const RB = new long[CAP];
    const size_t OUTCAP = 1u << 20;
    char* const outBuf = new char[OUTCAP + 4096];
    const float* const UP = U.data();

    size_t have = 0;
    bool eof = false;
    size_t nr = 0, nj = 0;
    uint32_t maxJobKey = 0;

    // Finish the current chunk: order the dot jobs for locality, run them, then
    // emit every line of the chunk in input order.
    auto flushChunk = [&]() {
        // ---------------- phase B: dot products, ordered by Vt row ----------
        const Job* J = JA;
        if (nj > 1) J = radixSortJobs(JA, JB, nj, maxJobKey);
        // Sorted by Vt row: the 68 MB Vt table is walked in ascending order, so
        // it streams (and stays TLB friendly); the U rows come from a much
        // smaller working set and are prefetched a few jobs ahead.
        {
            const size_t PFD = (size_t)EST_PF;
            size_t k = 0;
            const size_t kend = (nj > PFD) ? nj - PFD : 0;
            for (; k < kend; k++) {
                const Job& j2 = J[k + PFD];
                const float* up = UP + j2.uoff;
                pfT0(&Vt[(size_t)(j2.key >> 1) * RANK]);
                pfT0(up);
#if EST_PFU > 1
                pfT0(up + 16);
#endif
#if EST_PFU > 2
                pfT0(up + 32);
#endif
#if EST_PFU > 3
                pfT0(up + 48);
#endif
                const Job& j = J[k];
                const int16_t* vr = &Vt[(size_t)(j.key >> 1) * RANK];
                float s = dotProduct(UP + j.uoff, vr, RANK);
                long r = llround((double)j.mrow + (double)s);
                (j.key & 1u ? RB : RA)[j.idx] = r;
            }
            for (; k < nj; k++) {
                const Job& j = J[k];
                const int16_t* vr = &Vt[(size_t)(j.key >> 1) * RANK];
                float s = dotProduct(UP + j.uoff, vr, RANK);
                long r = llround((double)j.mrow + (double)s);
                (j.key & 1u ? RB : RA)[j.idx] = r;
            }
        }

        // ---------------- phase C: emit in input order ----------------
        {
        char* op = outBuf;
        const char* const oe = op + OUTCAP;
        for (size_t i = 0; i < nr; i++) {
            const Rec& R = RC[i];
            if (op + R.keyLen + 64 > oe) {
                fwrite(outBuf, 1, (size_t)(op - outBuf), fo);
                op = outBuf;
            }
            // Copy the key with three overlapping 16-byte stores: the bytes past
            // keyLen are straight away overwritten by the delay field, and both
            // buffers keep >= 512 bytes of slack past the last key, so no tail
            // loop and no memcpy call are needed.
            const char* kp = dat + R.keyOff;
            uint32_t kl = R.keyLen;
#if EST_X86
            if (kl <= 48) {
                __m128i a = _mm_loadu_si128((const __m128i*)kp);
                __m128i b = _mm_loadu_si128((const __m128i*)(kp + 16));
                __m128i c = _mm_loadu_si128((const __m128i*)(kp + 32));
                _mm_storeu_si128((__m128i*)op, a);
                _mm_storeu_si128((__m128i*)(op + 16), b);
                _mm_storeu_si128((__m128i*)(op + 32), c);
            } else
#endif
            {
                memcpy(op, kp, kl);
            }
            op += kl;
            op[0] = ',';
            long pred;
            if (!R.ncd) pred = 0;
            else {
                long c0 = (long)RA[i] + (long)R.cpen0;
                if (R.ncd > 1) {
                    long c1 = (long)RB[i] + (long)R.cpen1;
                    if (c1 < c0) c0 = c1;
                }
                pred = c0 + (long)R.lgap + (long)R.netWall;
                if (R.same) pred = 0;
            }
            if (pred > 0 && R.tgtIdx < nTgt) {
                int bucket = pred >= CALIB_BINS[3] ? 4 :
                             pred >= CALIB_BINS[2] ? 3 :
                             pred >= CALIB_BINS[1] ? 2 :
                             pred >= CALIB_BINS[0] ? 1 : 0;
                pred = llround((double)pred * Calib[(size_t)R.tgtIdx * 5 + bucket]);
            }
            if (pred > 0) {
                int m = fmtPos(op + 1, (uint32_t)pred);
                op += m + 1;
                *op++ = '\n';
            } else {
                op[1] = '0'; op[2] = '\n';
                op += 3;
            }
        }
        if (op != outBuf) fwrite(outBuf, 1, (size_t)(op - outBuf), fo);
        }
        nr = 0; nj = 0; maxJobKey = 0;
    };

    // ---------------- block loop ----------------
    for (;;) {
        while (!eof && have < BLOCKB) {
            size_t got = fread(dat + have, 1, BLOCKB - have, fi);
            if (got) have += got;
            else { eof = true; break; }
        }
        // Only complete lines are consumed: the tail after the last '\n' waits
        // for the next block (a line may straddle a block boundary).
        size_t proc;
        if (eof) {
            if (have > 0 && dat[have - 1] != '\n') dat[have++] = '\n';
            proc = have;
        } else {
            size_t q = have;
            while (q > 0 && dat[q - 1] != '\n') q--;
            if (q == 0) { dat[have++] = '\n'; proc = have; }   // pathological long line
            else proc = q;
        }
        if (proc == 0) break;

        // ---------------- phase A: parse, look up pins, pick candidates ---------
        {
            char* p = dat;
            char* const lim = dat + proc;
            while (p < lim) {
                if (p[0] != 'S' || p + 1 >= lim || p[1] != 'R') {
                    // header / blank / malformed: skip to the next newline
                    char* nl0 = (char*)memchr(p, '\n', (size_t)(lim - p));
                    if (!nl0) break;
                    p = nl0 + 1;
                    continue;
                }
                char* nl = (char*)memchr(p, '\n', (size_t)(lim - p));
                if (!nl) break;                            // should not happen
                const char* base = p;
                p = nl + 1;
                {
                    const char* eol = nl;
                    if (eol[-1] == '\r' || eol[-1] == ' ' || eol[-1] == '\t') {
                        do { eol--; } while (eol > base && (eol[-1] == '\r' || eol[-1] == ' ' || eol[-1] == '\t'));
                    }
                    const char* c1 = findByte(base, eol, ',');
                    if (c1 >= eol) continue;
                    const char* c2 = findByte(c1 + 1, eol, ',');
                    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
                    const char* pin0 = base;
                    const char* pin1 = base;
                    int plen0 = 0, plen1 = 0;
                    if (!parseField(base, c1, x0, y0, pin0, plen0)) continue;
                    if (!parseField(c1 + 1, c2, x1, y1, pin1, plen1)) continue;
                    if (nr == CAP) flushChunk();               // pathological input guard
                    Rec& R = RC[nr];
                    R.keyOff = (int32_t)(base - dat);
                    R.keyLen = (uint32_t)(c2 - base);
                    R.ncd = 0; R.same = 0; R.lgap = 0; R.netWall = 0; R.tgtIdx = 0;
                    R.cpen0 = 0; R.cpen1 = 0;
                    uint32_t ridx = (uint32_t)nr;
                    nr++;
                    uint32_t v0 = pinLookup(pin0, plen0);
                    uint32_t v1 = pinLookup(pin1, plen1);
                    int p0 = (int)(v0 & 0x3FFu) - 1;
                    int si = (int)((v0 >> 10) & 0x3FFu) - 1;
                    int ti = (int)((v1 >> 20) & 0x3FFu) - 1;
                    if (si < 0 || ti < 0) continue;
                    R.tgtIdx = (uint16_t)ti;
                    bool cross = (x0 <= 75 && x1 >= 90) || (x1 <= 75 && x0 >= 90);
                    int dx = x1 - x0, dy = y1 - y0;
                    int loY = min(y0, y1), hiY = max(y0, y1);
                    int dxv = cross ? ((x1 >= 90 ? x1 - 14 : x1) - (x0 >= 90 ? x0 - 14 : x0)) : dx;
                    bool bt = (doorpre[hiY + 1] - doorpre[loY]) > 0;
                    int cdxx[2], cpen[2], ncd = 0;
                    if (!cross) { cdxx[ncd] = dx; cpen[ncd] = 0; ncd++; }
                    else {
                        cdxx[ncd] = dxv; cpen[ncd] = 292; ncd++;
                        if (bt) { cdxx[ncd] = dx; cpen[ncd] = 0; ncd++; }
                    }
                    int ndx = netdx[p0], ndy = netdy[p0];
                    int netWall = 0;
                    if (ndx != 0) {
                        int xl = x0 + ndx;
                        if (xl >= 76 && xl <= 89) {
                            int lo2 = min(y0, y0 + ndy), hi2 = max(y0, y0 + ndy);
                            if (doorpre[hi2 + 1] - doorpre[lo2] == 0) netWall = 292;
                        }
                    }
                    int sx = x0 + ndx, sy = y0 + ndy;
                    long lgap = vpre[max(x0, sx)] - vpre[min(x0, sx)] + hpre[max(y0, sy)] - hpre[min(y0, sy)]
                              + vpre[max(sx, x1)] - vpre[min(sx, x1)] + hpre[max(sy, y1)] - hpre[min(sy, y1)];
                    uint32_t uoff = (uint32_t)(((size_t)si * nTgt + ti) * RANK);
                    int16_t mrow = M[(size_t)si * nTgt + ti];
                    int nvt = 0;
                    for (int ci = 0; ci < ncd; ci++) {
                        int dxx = cdxx[ci];
                        if (dxx < -120 || dxx > 120 || dy < -552 || dy > 552) continue;
                        int ix = dxx - CX0, iy = dy - CY0;
                        if (ix < 0 || ix >= NCX || iy < 0 || iy >= NCY) continue;
                        int32_t vtOff = (int32_t)(((size_t)ix * NCY + iy) * RANK);
                        if (vtOff < 0 || (size_t)vtOff + (size_t)RANK > Vt.size()) continue;
                        uint32_t key = (uint32_t)(vtOff / RANK) * 2u + (uint32_t)(ci != 0);
                        Job& j = JA[nj++];
                        j.key = key; j.idx = ridx; j.uoff = uoff;
                        j.mrow = mrow; j.cpen = (int16_t)cpen[ci];
                        if (key > maxJobKey) maxJobKey = key;
                        if (nvt == 0) R.cpen0 = (int16_t)cpen[ci];
                        else R.cpen1 = (int16_t)cpen[ci];
                        nvt++;
                    }
                    R.ncd = (uint8_t)nvt;
                    if (nvt > 0) {
                        R.lgap = (int32_t)lgap; R.netWall = (int16_t)netWall;
                        if (x0 == x1 && y0 == y1 && p0 == (int)(v1 & 0x3FFu) - 1) R.same = 1;
                    }
                }
            }
        }
        if (nr) flushChunk();

        size_t rem = have - proc;
        if (rem) memmove(dat, dat + proc, rem);
        have = rem;
        if (eof) break;
    }
    fclose(fo);
    fclose(fi);
    return 0;
}
