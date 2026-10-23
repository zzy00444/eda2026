"""Plan A: fit and evaluate the lean candidate model from exact fields.

Model (per query):

    delay = gap_exact(x0,y0,x1,y1)
          + A[pair] + Bx[pair]*dx + By[pair]*dy          # per-pair linear detrend
          + su * scale[cell] * dot(U8[pair], C8[cell])   # rank-r residual

  * pair = src*288 + tgt (59,904);
  * cell = (dx+119)*1099 + (dy+549);
  * the detrend removes the dominant linear-in-displacement component so the
    rank-r basis only has to represent the residual;
  * U8 is int8/int16 with a global scale; C8 is int16/int8 with a per-cell scale;
  * cells not realised by any dumped source position are filled from the mean
    coefficient of the covered cells in the same residue class.

Usage
    python tools/fit_lean.py --rank 128 --sample-cells 5000 --public-rows 100000
    python tools/fit_lean.py --eval-only --model build/lean_model.npz --public-rows 1000000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from arch import load_architecture  # noqa: E402

FIELDS = ROOT / "build" / "fields"
PUBLIC = Path("F:/EDA/EDA-PANGO-SRB-Arch-Delay/EDA-PANGO-SRB-Arch-Delay/data/delay_estimate_ans.csv")
NX, NY = 239, 1099
CELLS = NX * NY


def gap_arrays(arch):
    vp = np.zeros(arch.width + 1, dtype=np.int32)
    for line in arch.vertical_lines:
        vp[line.site + 1:] += line.delay
    hp = np.zeros(arch.height + 1, dtype=np.int32)
    for line in arch.horizontal_lines:
        hp[line.site + 1:] += line.delay
    return vp, hp


def model_predict(model, cell, pair, dx, dy, gaps):
    U8, su = model["U8"], float(model["su"])
    A, Bx, By = model["A"], model["Bx"], model["By"]
    C, scale = model["C"], model["scale"]
    pred = np.zeros(len(cell))
    for st in range(0, len(cell), 200_000):
        sl = slice(st, min(st + 200_000, len(cell)))
        c = C[:, cell[sl]].astype(np.int32).T
        u = U8[pair[sl]].astype(np.int32)
        dot = (c * u).sum(axis=1)
        pred[sl] = dot * (scale[cell[sl]] * su)
    return np.rint(pred + gaps + A[pair] + Bx[pair] * dx + By[pair] * dy)


def load_public(args, arch):
    import pandas as pd

    src = [arch.ports[i] for i in arch.query_sources]
    sm = {p: i for i, p in enumerate(src)}
    tm = {arch.ports[t]: i for i, t in enumerate(arch.targets)}
    df = pd.read_csv(PUBLIC, nrows=args.public_rows if args.public_rows else None)
    sp = df["From"].str.extract(r"^SRB_(\d+)_(\d+)/(.+)$")
    tp = df["To"].str.extract(r"^SRB_(\d+)_(\d+)/(.+)$")
    si = sp[2].map(sm).to_numpy(np.int64); ti = tp[2].map(tm).to_numpy(np.int64)
    sx = sp[0].to_numpy(np.int32); sy = sp[1].to_numpy(np.int32)
    tx = tp[0].to_numpy(np.int32); ty = tp[1].to_numpy(np.int32)
    ok = (si >= 0) & (ti >= 0)
    return df, np.where(ok, si, 0), np.where(ok, ti, 0), sx, sy, tx, ty, ok


def source_nets(arch):
    """Net delta of every query source port (0,0 for logic drivers whose first hop is an arc)."""
    vx = np.zeros(len(arch.query_sources), dtype=np.int32)
    vy = np.zeros(len(arch.query_sources), dtype=np.int32)
    nets_from = {}
    for n in arch.nets:
        nets_from[n.src] = (n.dx, n.dy)
    for i, p in enumerate(arch.query_sources):
        if p in nets_from:
            vx[i], vy[i] = nets_from[p]
    return vx, vy


def telescope_gap(vp, hp, vx, vy, sx, sy, tx, ty):
    """Gap crossings including the source net hop: pin -> net far end -> target."""
    fx = np.clip(sx + vx, 0, 119)
    fy = np.clip(sy + vy, 0, 549)
    return (np.abs(vp[fx] - vp[sx]) + np.abs(vp[tx] - vp[fx]) +
            np.abs(hp[fy] - hp[sy]) + np.abs(hp[ty] - hp[fy])).astype(np.float64)


def evaluate_public(args, arch, model, vp, hp) -> dict:
    df, si, ti, sx, sy, tx, ty, ok = load_public(args, arch)
    nt = int(model["meta"][3])
    cell = (tx - sx + 119) * NY + (ty - sy + 549)
    pair = si * nt + ti
    dx = (tx - sx).astype(np.float32); dy = (ty - sy).astype(np.float32)
    vxs, vys = source_nets(arch)
    gaps = telescope_gap(vp, hp, vxs[si], vys[si], sx, sy, tx, ty)
    p = model_predict(model, cell, pair, dx, dy, gaps)
    truth = df["delay"].to_numpy(dtype=np.int64)
    pts = 100.0 * (1.0 - np.tanh(4.0 * np.abs(p - truth) / np.maximum(truth, 1)))
    pts[truth == 0] = 100.0 * (p[truth == 0] == 0)
    report = {
        "public_rows": int(len(df)),
        "rank": int(model["rank"]),
        "accuracy": float(pts.mean()),
        "mae_ps": float(np.abs(p - truth).mean()),
        "p50_p90_p99": [float(np.percentile(np.abs(p - truth), q)) for q in (50, 90, 99)],
        "unmapped_rows": int((~ok).sum()),
    }
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--sample-cells", type=int, default=5000)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--c-dtype", choices=["int8", "int16"], default="int16")
    parser.add_argument("--u-dtype", choices=["int8", "int16"], default="int16")
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "lean_model.npz")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--public-rows", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    arch = load_architecture()
    vp, hp = gap_arrays(arch)
    insts = np.array(arch.insts, dtype=np.int32)
    npos = len(insts)
    pos_of = {int(x) * 10000 + int(y): i for i, (x, y) in enumerate(insts)}
    src_names = [arch.ports[i] for i in arch.query_sources]
    ns, nt = len(src_names), len(arch.targets)
    pairs = ns * nt

    if args.eval_only:
        data = np.load(args.model or args.out)
        evaluate_public(args, arch, {k: data[k] for k in data.files}, vp, hp)
        return

    index = json.loads((FIELDS / "index.json").read_text(encoding="utf-8"))["fields"]
    by_pos: dict[tuple[int, int], dict[str, dict]] = {}
    for entry in index:
        by_pos.setdefault((entry["x"], entry["y"]), {})[entry["port"]] = entry
    positions = sorted(p for p, d in by_pos.items() if all(n in d for n in src_names))
    print(f"positions with full dumps: {positions}", flush=True)

    dxs = np.arange(-119, 120, dtype=np.int32)
    dys = np.arange(-549, 550, dtype=np.int32)
    occ = np.zeros((550, 120), dtype=bool)
    for (x, y) in arch.insts:
        occ[y, x] = True
    cover_pos = np.full((NX, NY), -1, dtype=np.int16)
    for pi, (px, py) in enumerate(positions):
        xr = dxs[:, None] + px
        yr = dys[None, :] + py
        inside = (xr >= 0) & (xr < 120) & (yr >= 0) & (yr < 550)
        ok = inside & occ[np.clip(yr, 0, 549), np.clip(xr, 0, 119)]
        cover_pos[ok & (cover_pos < 0)] = pi
    covered = cover_pos >= 0
    print(f"coverage {covered.mean():.4f} ({int(covered.sum())}/{CELLS})", flush=True)

    fields = {}
    for (px, py) in positions:
        fields[(px, py)] = [np.memmap(FIELDS / by_pos[(px, py)][p]["file"], dtype=np.uint16, mode="r")
                            .reshape(npos, nt) for p in src_names]

    vxs, vys = source_nets(arch)

    def cell_matrix(pi: int, target_srb: int) -> np.ndarray:
        px, py = positions[pi]
        x, y = int(insts[target_srb, 0]), int(insts[target_srb, 1])
        fx = np.clip(px + vxs, 0, 119)
        fy = np.clip(py + vys, 0, 549)
        gs = (np.abs(vp[fx] - vp[px]) + np.abs(vp[x] - vp[fx]) +
              np.abs(hp[fy] - hp[py]) + np.abs(hp[y] - hp[fy]))
        mat = np.empty((ns, nt), dtype=np.float32)
        for s, mm in enumerate(fields[(px, py)]):
            mat[s] = mm[target_srb]
        mat -= gs.astype(np.float32)[:, None]
        mat[mat > 60000] = np.nan          # unreachable sentinel from the dump
        return mat

    def cell_target(ix: int, iy: int):
        pi = int(cover_pos[ix, iy])
        px, py = positions[pi]
        return pi, pos_of[(int(dxs[ix]) + px) * 10000 + (int(dys[iy]) + py)]

    rng = np.random.default_rng(2027)
    cov_idx = np.nonzero(covered.ravel())[0]
    sample = rng.choice(cov_idx, size=min(args.sample_cells, len(cov_idx)), replace=False)
    sdx = np.array([divmod(int(c), NY)[0] - 119 for c in sample], dtype=np.float32)
    sdy = np.array([divmod(int(c), NY)[1] - 549 for c in sample], dtype=np.float32)
    X = np.empty((pairs, len(sample)), dtype=np.float32)
    for j, c in enumerate(sample):
        ix, iy = divmod(int(c), NY)
        pi, tpos = cell_target(ix, iy)
        X[:, j] = cell_matrix(pi, tpos).ravel()
    valid = ~np.isnan(X)
    with np.errstate(invalid="ignore"):
        pair_mean = np.where(valid, X, 0).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
    global_mean = float(np.where(valid, X, 0).sum() / max(valid.sum(), 1))
    pair_mean = np.where(valid.sum(axis=1) > 0, pair_mean, global_mean).astype(np.float32)
    print(f"holes in sample: {int((~valid).sum())} of {X.size}", flush=True)
    X[~valid] = np.broadcast_to(pair_mean[:, None], X.shape)[~valid]
    D = np.stack([np.ones(len(sample)), sdx, sdy], axis=1).astype(np.float64)
    lin = X.astype(np.float64) @ np.linalg.pinv(D.T)
    A = lin[:, 0].astype(np.float32)
    Bx = lin[:, 1].astype(np.float32)
    By = lin[:, 2].astype(np.float32)
    R = X - (lin @ D.T).astype(np.float32)
    Rc = R.astype(np.float64)
    gram = Rc.T @ Rc
    eigval, eigvec = np.linalg.eigh(gram)
    order = np.argsort(-eigval)
    eigval, eigvec = eigval[order], eigvec[:, order]
    r = min(args.rank, X.shape[1])
    U = (Rc @ eigvec[:, :r]) / np.sqrt(np.maximum(eigval[:r], 1e-12))[None, :]
    print(f"basis {U.shape} energy {float(eigval[:r].sum() / eigval.sum()):.6f}", flush=True)
    del X, R, Rc, gram

    qmax_u = 127 if args.u_dtype == "int8" else 32767
    qmax_c = 127 if args.c_dtype == "int8" else 32767
    su = float(np.abs(U).max()) / qmax_u if np.abs(U).max() > 0 else 1.0
    U8 = np.rint(U / su).clip(-qmax_u, qmax_u).astype(np.int8 if qmax_u == 127 else np.int16)
    Uf = U8.astype(np.float32) * su

    C = np.zeros((r, CELLS), dtype=np.int8 if qmax_c == 127 else np.int16)
    scale = np.ones(CELLS, dtype=np.float32)
    t0 = time.time()
    for start in range(0, len(cov_idx), args.chunk):
        block = cov_idx[start:start + args.chunk]
        mats = np.empty((pairs, len(block)), dtype=np.float32)
        for j, c in enumerate(block):
            ix, iy = divmod(int(c), NY)
            pi, tpos = cell_target(ix, iy)
            col = cell_matrix(pi, tpos).ravel()
            bad = np.isnan(col)
            if bad.any():
                col[bad] = pair_mean[bad]
            mats[:, j] = col
        dxb = np.array([divmod(int(c), NY)[0] - 119 for c in block], dtype=np.float32)
        dyb = np.array([divmod(int(c), NY)[1] - 549 for c in block], dtype=np.float32)
        mats -= (A[:, None] + Bx[:, None] * dxb[None, :] + By[:, None] * dyb[None, :])
        coef = Uf.T @ mats
        sc = np.abs(coef).max(axis=0) / qmax_c
        sc[sc == 0] = 1.0
        C[:, block] = np.rint(coef / sc[None, :]).clip(-qmax_c, qmax_c).astype(C.dtype)
        scale[block] = sc
        if (start // args.chunk) % 20 == 0:
            print(f"  cells {start + len(block)}/{len(cov_idx)} {time.time() - t0:.1f}s", flush=True)
    print(f"coefficients in {time.time() - t0:.1f}s", flush=True)

    if not covered.all():
        from scipy import ndimage
        inv = ~covered
        residue = (dxs[:, None] % 10) * 12 + (dys[None, :] % 12)
        idxmap = ndimage.distance_transform_edt(inv, return_distances=False, return_indices=True)
        flat_src = (idxmap[0] * NY + idxmap[1]).ravel()
        for cls in range(120):
            cls_mask = residue == cls
            src_cells = np.nonzero(cls_mask & covered)[0]
            dst_cells = np.nonzero(cls_mask.ravel() & inv.ravel())[0]
            if len(dst_cells) == 0:
                continue
            if len(src_cells) > 0:
                coef_mean = C[:, src_cells].astype(np.float64).mean(axis=1)
                sc = float(np.abs(coef_mean).max()) / qmax_c
                sc = sc if sc > 0 else 1.0
                C[:, dst_cells] = np.rint(coef_mean / sc).clip(-qmax_c, qmax_c).astype(C.dtype)[:, None]
                scale[dst_cells] = sc
            else:
                C[:, dst_cells] = C[:, flat_src[dst_cells]]
                scale[dst_cells] = scale[flat_src[dst_cells]]
        print(f"filled {int(inv.sum())} uncovered cells", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, U8=U8, su=np.float32(su), A=A, Bx=Bx, By=By,
                        C=C, scale=scale, rank=np.int32(r),
                        meta=np.array([NX, NY, ns, nt], dtype=np.int32))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)", flush=True)

    if args.public_rows:
        data = np.load(args.out)
        evaluate_public(args, arch, {k: data[k] for k in data.files}, vp, hp)


if __name__ == "__main__":
    main()
