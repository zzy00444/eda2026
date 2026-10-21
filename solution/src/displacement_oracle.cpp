// Exact delay solver for the Pango SRB delay-estimation problem.
// Full (SRB, port) graph + Dial's-algorithm Dijkstra from the source pin, early exit at the target.
#include <bits/stdc++.h>
using namespace std;

struct Arc { int to; int w; };
struct Net { int to; int dx, dy; bool horiz; };
struct Block { int lower, upper, left, right; bool vcross, hcross; int vdelay, hdelay; };
struct Line { bool horiz; int site, delay; };

int W = 120, H = 550;
int NP = 0, NSRB = 0;
vector<string> portName, portDir;
vector<vector<Arc>> arcsFrom;
vector<vector<Net>> netsFrom;
vector<Block> blocks;
vector<Line> lines;
vector<int> xy2srb;
vector<int> sx, sy;
vector<int> vlineSite, vlineDelay, hlineSite, hlineDelay;

static inline int nodeOf(int srb, int p) { return srb * NP + p; }

int findBlock(int x, int y) {
    for (size_t i = 0; i < blocks.size(); i++) {
        const Block& b = blocks[i];
        if (x >= b.left && x <= b.right && y >= b.lower && y <= b.upper) return (int)i;
    }
    return -1;
}

bool netTarget(const Net& n, int x, int y, int& outSrb, int& outCost) {
    int nx = x + n.dx, ny = y + n.dy;
    if (nx < 0 || nx >= W || ny < 0 || ny >= H) return false;
    if (xy2srb[ny * W + nx] < 0) {
        int bi = findBlock(nx, ny);
        if (bi < 0) return false;
        bool ok = n.horiz ? blocks[bi].hcross : blocks[bi].vcross;
        if (!ok) return false;
        int step = n.horiz ? (n.dx > 0 ? 1 : -1) : (n.dy > 0 ? 1 : -1);
        int need = abs(n.horiz ? n.dx : n.dy), got = 0;
        int cx = x, cy = y;
        while (got < need) {
            if (n.horiz) cx += step; else cy += step;
            if (cx < 0 || cx >= W || cy < 0 || cy >= H) return false;
            if (xy2srb[cy * W + cx] >= 0) got++;
        }
        nx = cx; ny = cy;
    }
    int cost = 0;
    if (x != nx) {
        int lo = min(x, nx), hi = max(x, nx);
        for (size_t i = 0; i < vlineSite.size(); i++)
            if (vlineSite[i] >= lo && vlineSite[i] < hi) cost += vlineDelay[i];
    }
    if (y != ny) {
        int lo = min(y, ny), hi = max(y, ny);
        for (size_t i = 0; i < hlineSite.size(); i++)
            if (hlineSite[i] >= lo && hlineSite[i] < hi) cost += hlineDelay[i];
    }
    for (const Block& b : blocks) {
        if (n.horiz) {
            if (!b.hcross) continue;
            if (y >= b.lower && y <= b.upper) {
                int lo = min(x, nx), hi = max(x, nx);
                if (!(hi < b.left || lo > b.right)) cost += b.hdelay;
            }
        } else {
            if (!b.vcross) continue;
            if (x >= b.left && x <= b.right) {
                int lo = min(y, ny), hi = max(y, ny);
                if (!(hi < b.lower || lo > b.upper)) cost += b.vdelay;
            }
        }
    }
    outSrb = xy2srb[ny * W + nx];
    outCost = cost;
    return true;
}

void loadArch(const string& fn) {
    ifstream in(fn);
    string tag;
    while (in >> tag) {
        if (tag == "NPORTS") {
            in >> NP; portName.resize(NP); portDir.resize(NP); arcsFrom.resize(NP); netsFrom.resize(NP);
            for (int i = 0; i < NP; i++) { string c; in >> c >> portName[i] >> portDir[i]; }
        } else if (tag == "NARCS") {
            int n; in >> n;
            for (int i = 0; i < n; i++) { int a, b, d; char c; in >> c >> a >> b >> d; arcsFrom[a].push_back({b, d}); }
        } else if (tag == "NNETS") {
            int n; in >> n;
            for (int i = 0; i < n; i++) {
                int a, b, dx, dy; char c; in >> c >> a >> b >> dx >> dy;
                netsFrom[a].push_back({b, dx, dy, dx != 0});
            }
        } else if (tag == "NSRB") {
            int n; in >> n; NSRB = n;
            xy2srb.assign(W * H, -1);
            for (int i = 0; i < n; i++) { int x, y; char c; in >> c >> x >> y; xy2srb[y * W + x] = i; }
        } else if (tag == "GRID") {
            in >> W >> H;
        } else if (tag == "NLINES") {
            int n; in >> n;
            for (int i = 0; i < n; i++) {
                string dir; int site, delay; char c; in >> c >> dir >> site >> delay;
                if (dir == "vertical") { vlineSite.push_back(site); vlineDelay.push_back(delay); }
                else { hlineSite.push_back(site); hlineDelay.push_back(delay); }
            }
        } else if (tag == "NBLOCKS") {
            int n; in >> n;
            for (int i = 0; i < n; i++) {
                Block b; int vc, hc; char c;
                in >> c >> b.lower >> b.upper >> b.left >> b.right >> vc >> b.vdelay >> hc >> b.hdelay;
                b.vcross = vc; b.hcross = hc; blocks.push_back(b);
            }
        }
    }
    sx.assign(NSRB, 0); sy.assign(NSRB, 0);
    for (int i = 0; i < W * H; i++) if (xy2srb[i] >= 0) { int s = xy2srb[i]; sx[s] = i % W; sy[s] = i / W; }
}

int main(int argc, char** argv) {
    string archFn = argc > 1 ? argv[1] : "arch.txt";
    string queryFn = argc > 2 ? argv[2] : "check.txt";
    loadArch(archFn);
    fprintf(stderr, "loaded: ports=%d srbs=%d blocks=%zu vlines=%zu hlines=%zu\n",
            NP, NSRB, blocks.size(), vlineSite.size(), hlineSite.size());

    unordered_map<string, int> pid;
    for (int i = 0; i < NP; i++) pid[portName[i]] = i;

    struct Q { string from, to; int golden; };
    vector<Q> qs;
    {
        ifstream in(queryFn);
        string line;
        while (getline(in, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
            if (line.empty()) continue;
            size_t a = line.find(','), b = line.find(',', a + 1);
            if (a == string::npos) continue;
            Q q; q.from = line.substr(0, a);
            q.to = (b == string::npos) ? line.substr(a + 1) : line.substr(a + 1, b - a - 1);
            q.golden = (b == string::npos) ? -1 : atoi(line.c_str() + b + 1);
            qs.push_back(q);
        }
    }
    auto parsePin = [&](const string& s, int& x, int& y, int& p) {
        int u1 = s.find('_'), u2 = s.find('_', u1 + 1), sl = s.find('/');
        x = atoi(s.substr(u1 + 1, u2 - u1 - 1).c_str());
        y = atoi(s.substr(u2 + 1, sl - u2 - 1).c_str());
        auto it = pid.find(s.substr(sl + 1));
        p = (it == pid.end()) ? -1 : it->second;
    };

    const int NB = 2048;
    const uint16_t INF = 65535;
    size_t NN = (size_t)NSRB * NP;
    vector<uint16_t> dist(NN, INF);
    vector<uint8_t> settled(NN, 0);
    vector<vector<int>> bucket(NB);
    vector<int> touched;
    touched.reserve(1 << 20);
    long long cur = 0;

    // ---- dump mode: exact full distance field from one source pin ----
    if (argc > 3 && string(argv[3]) == "dump") {
        int x0, y0, p0;
        parsePin(qs[0].from, x0, y0, p0);
        int src = nodeOf(xy2srb[y0 * W + x0], p0);
        cur = 0; touched.clear();
        dist[src] = 0; touched.push_back(src); bucket[0].push_back(src);
        long long remaining = 1, pops = 0;
        while (remaining > 0) {
            vector<int>& b = bucket[cur % NB];
            if (b.empty()) { cur++; continue; }
            int u = b.back(); b.pop_back(); remaining--;
            if (settled[u]) continue;
            settled[u] = 1; pops++;
            uint16_t du = dist[u];
            int srb = u / NP, p = u % NP;
            int x = sx[srb], y = sy[srb];
            for (const Arc& a : arcsFrom[p]) {
                int v = nodeOf(srb, a.to);
                uint16_t nd = du + a.w;
                if (nd < dist[v]) { dist[v] = nd; settled[v] = 0; bucket[nd % NB].push_back(v); remaining++; }
            }
            for (const Net& n : netsFrom[p]) {
                int ns, gc;
                if (!netTarget(n, x, y, ns, gc)) continue;
                int v = nodeOf(ns, n.to);
                uint16_t nd = du + gc;
                if (nd < dist[v]) { dist[v] = nd; settled[v] = 0; bucket[nd % NB].push_back(v); remaining++; }
            }
        }
        FILE* fo = fopen(argv[4], "wb");
        int hdr[3] = { NSRB, NP, (int)sizeof(uint16_t) };
        fwrite(hdr, sizeof(int), 3, fo);
        fwrite(dist.data(), sizeof(uint16_t), NN, fo);
        fclose(fo);
        fprintf(stderr, "dumped %zu nodes from %s (pops=%lld)\n", NN, qs[0].from.c_str(), pops);
        return 0;
    }
    double totalTime = 0; long long totalPops = 0;
    double sumErr = 0; int nz = 0; long long nbad = 0;

    for (size_t qi = 0; qi < qs.size(); qi++) {
        int x0, y0, p0, x1, y1, p1;
        parsePin(qs[qi].from, x0, y0, p0);
        parsePin(qs[qi].to, x1, y1, p1);
        int s0 = xy2srb[y0 * W + x0], s1 = xy2srb[y1 * W + x1];
        int src = nodeOf(s0, p0), tgt = nodeOf(s1, p1);
        clock_t t0 = clock();
        cur = 0; touched.clear();
        dist[src] = 0; touched.push_back(src);
        bucket[0].push_back(src);
        long long remaining = 1, pops = 0;
        int ans = INF;
        while (remaining > 0) {
            vector<int>& b = bucket[cur % NB];
            if (b.empty()) { cur++; continue; }
            int u = b.back(); b.pop_back(); remaining--;
            if (settled[u]) continue;
            settled[u] = 1; pops++;
            uint16_t du = dist[u];
            if (u == tgt) { ans = du; break; }
            int srb = u / NP, p = u % NP;
            int x = sx[srb], y = sy[srb];
            for (const Arc& a : arcsFrom[p]) {
                int v = nodeOf(srb, a.to);
                uint16_t nd = du + a.w;
                if (nd < dist[v]) {
                    if (dist[v] == INF) touched.push_back(v);
                    dist[v] = nd; settled[v] = 0;
                    bucket[nd % NB].push_back(v); remaining++;
                }
            }
            for (const Net& n : netsFrom[p]) {
                int ns, gc;
                if (!netTarget(n, x, y, ns, gc)) continue;
                int v = nodeOf(ns, n.to);
                uint16_t nd = du + gc;
                if (nd < dist[v]) {
                    if (dist[v] == INF) touched.push_back(v);
                    dist[v] = nd; settled[v] = 0;
                    bucket[nd % NB].push_back(v); remaining++;
                }
            }
        }
        double dt = double(clock() - t0) / CLOCKS_PER_SEC;
        totalTime += dt; totalPops += pops;
        printf("%s,%s,%d,%d,%lld\n", qs[qi].from.c_str(), qs[qi].to.c_str(), qs[qi].golden, ans, pops);
        fflush(stdout);
        if (qs[qi].golden >= 0) {
            if (ans == qs[qi].golden) nz++;
            else { nbad++; fprintf(stderr, "MISMATCH %s -> %s golden=%d exact=%d\n", qs[qi].from.c_str(), qs[qi].to.c_str(), qs[qi].golden, ans); }
            sumErr += fabs((double)ans - qs[qi].golden) / max(1, qs[qi].golden);
        }
        fprintf(stderr, "q%zu golden=%d exact=%d pops=%lld t=%.2fs\n", qi, qs[qi].golden, ans, pops, dt);
        for (int v : touched) { dist[v] = INF; settled[v] = 0; }
        for (auto& v : bucket) v.clear();
    }
    fprintf(stderr, "TOTAL time=%.2fs pops=%lld  exact-match=%lld mismatch=%lld  meanRelErr=%.6f\n",
            totalTime, totalPops, (long long)nz, (long long)nbad, qs.empty() ? 0.0 : sumErr / (double)qs.size());
    return 0;
}
