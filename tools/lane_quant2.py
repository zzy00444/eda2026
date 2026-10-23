"""B3: mean-removed int8 quantization of the lane model.

Raw int8 quantization of G fails (89% acc) because the dominant SVD components
carry the large per-(s,k) level.  If that level is stored separately as an int16
table (33,280 entries = 66 KB), the *residual* is much smaller and may quantize to
int8 without losing accuracy.  That would halve the per-query table traffic
(2 lanes x 256 B instead of 1 KB) and bring the runtime back into budget.

Model:
    G(s,k,cell) ~= mean[s,k] + sum_r U[(s,k),r] * C[cell,r]
with `mean` int16, U/C int8 when --dtype int8.

Usage
    python tools/lane_quant2.py --rank 256 --dtype int8 --e2e-cells 96
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ROOT / "build" / "fields"
FIELDS_ALL = ROOT / "build" / "fields_all"


def gap_for(arch, sx, sy, x, y) -> int:
    total = 0
    for line in arch.vertical_lines:
        lo, hi = (sx, x) if sx < x else (x, sx)
        if lo <= line.site < hi:
            total += line.delay
    for line in arch.horizontal_lines:
        lo, hi = (sy, y) if sy < y else (y, sy)
        if lo <= line.site < hi:
            total += line.delay
    return total


def accuracy(truth: np.ndarray, pred: np.ndarray) -> float:
    pred = np.maximum(0, np.rint(pred))
    nonzero = truth != 0
    points = np.zeros(truth.shape, dtype=np.float64)
    points[nonzero] = 100.0 * (1.0 - np.tanh(4.0 * np.abs(pred[nonzero] - truth[nonzero]) / truth[nonzero]))
    points[~nonzero] = 100.0 * (pred[~nonzero] == 0)
    return float(points.mean())


def quant_rows(m: np.ndarray, qmax: int):
    s = np.abs(m).max(axis=1, keepdims=True) / qmax
    s[s == 0] = 1.0
    return np.rint(m / s).clip(-qmax, qmax).astype(np.int16), s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--dtype", choices=["int8", "int16"], default="int8")
    parser.add_argument("--grid-step", default="4,10")
    parser.add_argument("--e2e-cells", type=int, default=96)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    from arch import load_architecture

    arch = load_architecture()
    insts = np.array(arch.insts, dtype=np.int32)
    src_names = [arch.ports[i] for i in arch.query_sources]
    ns = len(src_names)
    i_ports = [i for i in range(len(arch.ports)) if arch.ports[i].startswith("I")]
    nk = len(i_ports)
    nt = len(arch.targets)

    index = json.loads((FIELDS_ALL / "index.json").read_text(encoding="utf-8"))["fields"]
    by_port = {f["port"]: f for f in index if (f["x"], f["y"]) == (40, 275)}
    order = [by_port[p] for p in src_names]

    step_x, step_y = (int(v) for v in args.grid_step.split(","))
    cells = np.array([i for i, (x, y) in enumerate(arch.insts) if x % step_x == 0 and y % step_y == 0])
    rng = np.random.default_rng(31)
    e2e = np.sort(rng.choice(len(cells), size=min(args.e2e_cells, len(cells)), replace=False))
    gaps = np.array([gap_for(arch, 40, 275, int(insts[c, 0]), int(insts[c, 1])) for c in cells], dtype=np.float64)

    graph = coo_matrix(([a.delay for a in arch.arcs], ([a.src for a in arch.arcs], [a.dst for a in arch.arcs])),
                       shape=(len(arch.ports), len(arch.ports))).tocsr()
    internal = dijkstra(graph, directed=True)
    B = internal[np.ix_(i_ports, arch.targets)].astype(np.float64)
    B[~np.isfinite(B)] = 1e9

    G = np.empty((ns, nk, len(cells)), dtype=np.float32)
    M = np.zeros((ns, nt, len(e2e)), dtype=np.float32)
    for s, entry in enumerate(order):
        data = np.memmap(FIELDS_ALL / entry["file"], dtype=np.uint16, mode="r").reshape(len(arch.insts), len(arch.ports))
        G[s] = data[cells][:, i_ports].T.astype(np.float32)
        M[s] = np.memmap(FIELDS / entry["file"], dtype=np.uint16, mode="r").reshape(len(arch.insts), nt)[cells[e2e]].T.astype(np.float32)
    G -= gaps.astype(np.float32)[None, None, :]
    M -= gaps[e2e].astype(np.float32)[None, None, :]

    Gf = G.reshape(ns * nk, len(cells)).astype(np.float64)
    mean = Gf.mean(axis=1, keepdims=True)
    R = Gf - mean
    gram = R.T @ R
    eigval, eigvec = np.linalg.eigh(gram)
    o = np.argsort(-eigval)
    eigval, eigvec = eigval[o], eigvec[:, o]
    r = args.rank
    basis = R @ eigvec[:, :r] / np.sqrt(np.maximum(eigval[:r], 1e-12))[None, :]
    coef = basis.T @ R

    qmax = 127 if args.dtype == "int8" else 32767
    bq, bs = quant_rows(basis, qmax)
    cq, cs = quant_rows(coef.T, qmax)
    approx = (mean + (bq.astype(np.float64) * bs) @ (cq.T.astype(np.float64) * cs.T)).reshape(ns, nk, len(cells))
    gerr = float(np.abs(approx - G).mean())

    pred = np.empty((ns, nt, len(e2e)))
    for s in range(ns):
        blk = approx[s][:, e2e]
        pred[s] = (blk[:, :, None] + B[:, None, :]).min(axis=0).T
    accs = []
    for s in range(ns):
        for j in range(len(e2e)):
            accs.append(accuracy(M[s, :, j] + gaps[e2e[j]], pred[s, :, j] + gaps[e2e[j]]))
    nbytes = {
        "mean_int16": int(ns * nk * 2),
        "basis": int(ns * nk * r * (1 if args.dtype == "int8" else 2)),
        "coef": int(60200 * r * (1 if args.dtype == "int8" else 2)),
    }
    summary = {"rank": r, "dtype": args.dtype, "g_mae_ps": gerr,
               "accuracy": float(np.mean(accs)),
               "bytes": nbytes, "bytes_total": int(sum(nbytes.values()))}
    print(json.dumps(summary, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
