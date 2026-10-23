// Exact delay solver for the Pango SRB delay-estimation task.
//
// Builds the full (SRB, port) graph from arch.txt (exported by tools/arch.py)
// and runs Dial's-algorithm Dijkstra.
//
//   oracle queries <arch.txt> <request.csv> <result.csv> [limit]
//       Exact delay for every "From,To" row (output "From,To,delay").
//
//   oracle field <arch.txt> "<SRB_x_y/port>" <out.bin>
//       Full distance field from one pin, restricted to the architecturally valid
//       target ports: uint16[NSRB][NTARGET], row-major by SRB index.
//
// Graph semantics (validated against the 8 official check paths):
//   node      = (SRB, port)
//   arc edge  = (SRB, from) -> (SRB, to)        cost = arc delay
//   net edge  = (SRB, Z)    -> (target SRB, I)  cost = gap lines + crossable blocks
// A net landing on a block cell is shifted to the far side of the block counting
// only non-block cells, or broken if the block is not crossable in that direction.
// Net base delay is zero; a gap line at `site` is crossed when min < = site < max.
//
// Compile: g++ -O3 -std=c++17 -o oracle oracle.cpp

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

namespace {

constexpr int kBuckets = 4096;
constexpr uint32_t kInf = 0xFFFFFFFFu;

struct Arc { int to; int w; };
struct Net { int to; int dx, dy; bool horiz; };
struct Line { int site; int delay; };
struct Block {
    int lower, upper, left, right;
    bool vcross, hcross;
    int vdelay, hdelay;
};

int W = 0, H = 0, NP = 0, NSRB = 0;

std::vector<std::string> portName;
std::vector<std::vector<Arc>> arcsFrom;
std::vector<std::vector<Net>> netsFrom;
std::vector<int> laneOffset;   // first net lane of each port
std::vector<Line> vlines, hlines;
std::vector<Block> blocks;
std::vector<int> xy2srb;
std::vector<int> srbX, srbY;
std::vector<int> targetPorts;

std::vector<int> netTarget;    // [NSRB * NL]
std::vector<uint16_t> netCost; // [NSRB * NL]

inline int nodeOf(int srb, int port) { return srb * NP + port; }

int findBlock(int x, int y) {
    for (size_t i = 0; i < blocks.size(); ++i) {
        const Block& b = blocks[i];
        if (x >= b.left && x <= b.right && y >= b.lower && y <= b.upper) return static_cast<int>(i);
    }
    return -1;
}

struct Reader {
    FILE* f;
    explicit Reader(FILE* file) : f(file) {}
    bool next(std::string& s) {
        s.clear();
        int c;
        do { c = fgetc(f); } while (c != EOF && isspace(c));
        if (c == EOF) return false;
        while (c != EOF && !isspace(c)) { s.push_back(static_cast<char>(c)); c = fgetc(f); }
        return true;
    }
    int nextInt() { std::string s; next(s); return atoi(s.c_str()); }
    bool eof() const { return false; }
};

void loadArch(const char* fn) {
    FILE* f = fopen(fn, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", fn); exit(2); }
    Reader in(f);
    std::string tok;
    while (in.next(tok)) {
        if (tok == "GRID") {
            W = in.nextInt(); H = in.nextInt();
        } else if (tok == "PORTS") {
            NP = in.nextInt();
            portName.assign(NP, std::string());
            arcsFrom.assign(NP, {});
            netsFrom.assign(NP, {});
            for (int i = 0; i < NP; ++i) {
                int idx = in.nextInt();
                std::string name; in.next(name);
                portName[idx] = name;
            }
        } else if (tok == "ARCS") {
            int n = in.nextInt();
            for (int k = 0; k < n; ++k) {
                int a = in.nextInt(), b = in.nextInt(), d = in.nextInt();
                arcsFrom[a].push_back({b, d});
            }
        } else if (tok == "NETS") {
            int n = in.nextInt();
            for (int k = 0; k < n; ++k) {
                int a = in.nextInt(), b = in.nextInt(), dx = in.nextInt(), dy = in.nextInt();
                netsFrom[a].push_back({b, dx, dy, dx != 0});
            }
        } else if (tok == "SRBS") {
            NSRB = in.nextInt();
            xy2srb.assign(static_cast<size_t>(W) * H, -1);
            srbX.assign(NSRB, 0);
            srbY.assign(NSRB, 0);
            for (int i = 0; i < NSRB; ++i) {
                int x = in.nextInt(), y = in.nextInt();
                xy2srb[static_cast<size_t>(y) * W + x] = i;
                srbX[i] = x;
                srbY[i] = y;
            }
        } else if (tok == "LINES") {
            int n = in.nextInt();
            for (int k = 0; k < n; ++k) {
                int vert = in.nextInt(), site = in.nextInt(), delay = in.nextInt();
                (vert ? vlines : hlines).push_back({site, delay});
            }
        } else if (tok == "BLOCKS") {
            int n = in.nextInt();
            for (int k = 0; k < n; ++k) {
                Block b;
                b.lower = in.nextInt(); b.upper = in.nextInt();
                b.left = in.nextInt();  b.right = in.nextInt();
                int vc = in.nextInt(); b.vdelay = in.nextInt();
                int hc = in.nextInt(); b.hdelay = in.nextInt();
                b.vcross = vc != 0; b.hcross = hc != 0;
                blocks.push_back(b);
            }
        } else if (tok == "TARGETS") {
            int n = in.nextInt();
            targetPorts.resize(n);
            for (int i = 0; i < n; ++i) targetPorts[i] = in.nextInt();
        } else if (tok == "SOURCES") {
            int n = in.nextInt();
            for (int i = 0; i < n; ++i) (void)in.nextInt();
        } else if (tok == "END") {
            break;
        } else {
            fprintf(stderr, "unknown token %s\n", tok.c_str());
            exit(2);
        }
    }
    fclose(f);
    laneOffset.assign(NP + 1, 0);
    for (int p = 0; p < NP; ++p) laneOffset[p + 1] = laneOffset[p] + static_cast<int>(netsFrom[p].size());
    if (targetPorts.empty()) {
        std::vector<char> isTarget(NP, 0);
        for (int p = 0; p < NP; ++p)
            for (const Arc& a : arcsFrom[p]) isTarget[a.to] = 1;
        for (int p = 0; p < NP; ++p) if (isTarget[p]) targetPorts.push_back(p);
    }
}

void precomputeNets() {
    const int NL = laneOffset[NP];
    netTarget.assign(static_cast<size_t>(NSRB) * NL, -1);
    netCost.assign(static_cast<size_t>(NSRB) * NL, 0);
    for (int s = 0; s < NSRB; ++s) {
        const int x = srbX[s], y = srbY[s];
        for (int p = 0; p < NP; ++p) {
            const int lane0 = laneOffset[p];
            for (size_t k = 0; k < netsFrom[p].size(); ++k) {
                const Net& n = netsFrom[p][k];
                const size_t flat = static_cast<size_t>(s) * NL + lane0 + k;
                int nx = x + n.dx, ny = y + n.dy;
                if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
                int target = xy2srb[static_cast<size_t>(ny) * W + nx];
                if (target < 0) {
                    int bi = findBlock(nx, ny);
                    if (bi < 0) continue;
                    const Block& b = blocks[bi];
                    if (n.horiz ? !b.hcross : !b.vcross) continue;
                    const int stepX = n.dx == 0 ? 0 : (n.dx > 0 ? 1 : -1);
                    const int stepY = n.dy == 0 ? 0 : (n.dy > 0 ? 1 : -1);
                    const int need = n.horiz ? std::abs(n.dx) : std::abs(n.dy);
                    int got = 0, cx = x, cy = y;
                    while (got < need) {
                        cx += stepX; cy += stepY;
                        if (cx < 0 || cx >= W || cy < 0 || cy >= H) { got = -1; break; }
                        if (xy2srb[static_cast<size_t>(cy) * W + cx] >= 0) ++got;
                    }
                    if (got < 0) continue;
                    nx = cx; ny = cy;
                    target = xy2srb[static_cast<size_t>(ny) * W + nx];
                    if (target < 0) continue;
                }
                int cost = 0;
                if (x != nx) {
                    const int lo = std::min(x, nx), hi = std::max(x, nx);
                    for (const Line& l : vlines) if (l.site >= lo && l.site < hi) cost += l.delay;
                }
                if (y != ny) {
                    const int lo = std::min(y, ny), hi = std::max(y, ny);
                    for (const Line& l : hlines) if (l.site >= lo && l.site < hi) cost += l.delay;
                }
                for (const Block& b : blocks) {
                    if (n.horiz) {
                        if (!b.hcross) continue;
                        if (y >= b.lower && y <= b.upper) {
                            const int lo = std::min(x, nx), hi = std::max(x, nx);
                            if (!(hi < b.left || lo > b.right)) cost += b.hdelay;
                        }
                    } else {
                        if (!b.vcross) continue;
                        if (x >= b.left && x <= b.right) {
                            const int lo = std::min(y, ny), hi = std::max(y, ny);
                            if (!(hi < b.lower || lo > b.upper)) cost += b.vdelay;
                        }
                    }
                }
                netTarget[flat] = target;
                netCost[flat] = static_cast<uint16_t>(cost);
            }
        }
    }
}

struct Dijkstra {
    std::vector<uint32_t> dist;
    std::vector<uint8_t> settled;
    std::vector<std::vector<uint32_t>> bucket;
    std::vector<uint32_t> touched;
    uint64_t remaining = 0;
    uint32_t cur = 0;
    long long pops = 0;

    Dijkstra()
        : dist(static_cast<size_t>(NSRB) * NP, kInf),
          settled(static_cast<size_t>(NSRB) * NP, 0),
          bucket(kBuckets) {}

    void reset() {
        for (uint32_t v : touched) { dist[v] = kInf; settled[v] = 0; }
        touched.clear();
        for (auto& b : bucket) b.clear();
        remaining = 0;
        cur = 0;
        pops = 0;
    }

    uint32_t run(int src, int tgt) {
        dist[src] = 0;
        touched.push_back(src);
        bucket[0].push_back(src);
        ++remaining;
        const int NL = laneOffset[NP];
        while (remaining > 0) {
            std::vector<uint32_t>& b = bucket[cur % kBuckets];
            if (b.empty()) { ++cur; continue; }
            const uint32_t u = b.back(); b.pop_back(); --remaining;
            if (settled[u] || dist[u] != cur) continue;
            settled[u] = 1;
            ++pops;
            if (static_cast<int>(u) == tgt) return dist[u];
            const uint32_t du = dist[u];
            const int srb = static_cast<int>(u) / NP;
            const int port = static_cast<int>(u) % NP;
            for (const Arc& a : arcsFrom[port]) {
                const uint32_t v = static_cast<uint32_t>(nodeOf(srb, a.to));
                const uint32_t nd = du + static_cast<uint32_t>(a.w);
                if (nd < dist[v]) { dist[v] = nd; touched.push_back(v); bucket[nd % kBuckets].push_back(v); ++remaining; }
            }
            if (!netsFrom[port].empty()) {
                const int lane0 = laneOffset[port];
                const size_t base = static_cast<size_t>(srb) * NL + lane0;
                for (size_t k = 0; k < netsFrom[port].size(); ++k) {
                    const int ns = netTarget[base + k];
                    if (ns < 0) continue;
                    const uint32_t v = static_cast<uint32_t>(nodeOf(ns, netsFrom[port][k].to));
                    const uint32_t nd = du + netCost[base + k];
                    if (nd < dist[v]) { dist[v] = nd; touched.push_back(v); bucket[nd % kBuckets].push_back(v); ++remaining; }
                }
            }
        }
        return kInf;
    }
};

bool parsePin(const char* s, int& x, int& y, int& p) {
    const char* u1 = strchr(s, '_');
    const char* u2 = u1 ? strchr(u1 + 1, '_') : nullptr;
    const char* sl = u2 ? strchr(u2 + 1, '/') : nullptr;
    if (!u1 || !u2 || !sl) return false;
    x = atoi(u1 + 1);
    y = atoi(u2 + 1);
    std::string name(sl + 1);
    while (!name.empty() && (name.back() == '\r' || name.back() == '\n')) name.pop_back();
    p = -1;
    for (int i = 0; i < NP; ++i) if (portName[i] == name) { p = i; break; }
    return p >= 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage:\n"
                "  oracle queries <arch.txt> <request.csv> <result.csv> [limit]\n"
                "  oracle field   <arch.txt> <SRB_x_y/port> <out.bin>\n");
        return 2;
    }
    const std::string mode = argv[1];
    loadArch(argv[2]);
    precomputeNets();

    if (mode == "queries") {
        if (argc < 5) { fprintf(stderr, "queries needs <request> <result>\n"); return 2; }
        FILE* in = fopen(argv[3], "rb");
        if (!in) { fprintf(stderr, "cannot open %s\n", argv[3]); return 2; }
        FILE* out = fopen(argv[4], "wb");
        if (!out) { fprintf(stderr, "cannot open %s\n", argv[4]); return 2; }
        const long limit = argc > 5 ? atol(argv[5]) : -1;
        Dijkstra dj;
        fprintf(out, "From,To,delay\n");
        char buf[1024];
        long rows = 0;
        while (fgets(buf, sizeof buf, in)) {
            if (strncmp(buf, "From,", 5) == 0) continue;
            if (buf[0] == '\r' || buf[0] == '\n' || buf[0] == 0) continue;
            char* comma = strchr(buf, ',');
            if (!comma) continue;
            *comma = 0;
            char* to = comma + 1;
            char* nl = strpbrk(to, "\r\n");
            if (nl) *nl = 0;
            int sx, sy, sp, tx, ty, tp;
            if (!parsePin(buf, sx, sy, sp) || !parsePin(to, tx, ty, tp)) {
                fprintf(stderr, "bad pin: %s -> %s\n", buf, to);
                return 3;
            }
            const int s0 = xy2srb[static_cast<size_t>(sy) * W + sx];
            const int s1 = xy2srb[static_cast<size_t>(ty) * W + tx];
            if (s0 < 0 || s1 < 0) { fprintf(stderr, "pin on absent SRB\n"); return 3; }
            uint32_t ans;
            const int src = nodeOf(s0, sp), tgt = nodeOf(s1, tp);
            if (src == tgt) {
                ans = 0;
            } else {
                dj.reset();
                ans = dj.run(src, tgt);
            }
            if (ans == kInf) { fprintf(stderr, "unreachable: %s -> %s\n", buf, to); ans = 65535; }
            fprintf(out, "%s,%s,%u\n", buf, to, ans);
            if (limit > 0 && ++rows >= limit) break;
        }
        fclose(in);
        fclose(out);
        return 0;
    }

    if (mode == "field") {
        if (argc < 5) { fprintf(stderr, "field needs <pin> <out.bin>\n"); return 2; }
        int sx, sy, sp;
        if (!parsePin(argv[3], sx, sy, sp)) { fprintf(stderr, "bad pin %s\n", argv[3]); return 3; }
        const int s0 = xy2srb[static_cast<size_t>(sy) * W + sx];
        if (s0 < 0) { fprintf(stderr, "pin on absent SRB\n"); return 3; }
        Dijkstra dj;
        const uint32_t d = dj.run(nodeOf(s0, sp), -1);
        (void)d;
        const int ntg = static_cast<int>(targetPorts.size());
        std::vector<uint16_t> field(static_cast<size_t>(NSRB) * ntg);
        for (int s = 0; s < NSRB; ++s) {
            for (int k = 0; k < ntg; ++k) {
                const uint32_t dist = dj.dist[nodeOf(s, targetPorts[k])];
                field[static_cast<size_t>(s) * ntg + k] =
                    dist >= 65535u ? 65535u : static_cast<uint16_t>(dist);
            }
        }
        FILE* fo = fopen(argv[4], "wb");
        if (!fo) { fprintf(stderr, "cannot open %s\n", argv[4]); return 2; }
        fwrite(field.data(), sizeof(uint16_t), field.size(), fo);
        fclose(fo);
        fprintf(stderr, "field: srbs=%d ports=%d targets=%d pops=%lld\n", NSRB, NP, ntg, dj.pops);
        return 0;
    }

    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
}
