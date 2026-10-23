"""S3-M2 part 6: int8 quantization of the lane model.

The lane model stores `basis[(source,lane), rank]` and `coef[cell, rank]`.  For a
package budget it must be quantized.  This probe fits the rank-r global basis on G,
quantizes both factors to int8 with per-row / per-column scales, and re-measures
the end-to-end accuracy of `min_k (G_quant + B)` against the exact delay.

Usage
    python tools/lane_quant.py --rank 256 --grid-step 4,10 --e2e-cells 96
    python tools/lane_quant.py --rank 384 --dtype int16
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


def quantize_rows(matrix: np.ndarray, qmax: int):
    scale = np.abs(matrix).max(axis=1, keepdims=True) / qmax
    scale[scale == 0] = 1.0
    q = np.rint(matrix / scale).clip(-qmax, qmax).astype(np.int16)
    return q, scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--grid-step", default="4,10")
    parser.add_argument("--e2e-cells", type=int, default=96)
    parser.add_argument("--dtype", choices=["int8", "int16", "float"], default="int8")
    parser.add_argument("--scheme", choices=["row", "component"], default="component")
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
    rng = np.random.default_rng(21)
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
    gram = Gf.T @ Gf
    eigval, eigvec = np.linalg.eigh(gram)
    order_idx = np.argsort(-eigval)
    eigval, eigvec = eigval[order_idx], eigvec[:, order_idx]
    r = args.rank
    basis = Gf @ eigvec[:, :r] / np.sqrt(np.maximum(eigval[:r], 1e-12))[None, :]
    coef = basis.T @ Gf

    if args.dtype == "float":
        approx = (basis @ coef).reshape(ns, nk, len(cells))
        bytes_total = (ns * nk + 60200) * r * 4
    else:
        qmax = 127 if args.dtype == "int8" else 32767
        if args.scheme == "component":
            # per rank-component scale: matches the very different magnitudes of the
            # singular components, unlike per-row scaling.
            bs = (np.abs(basis).max(axis=0, keepdims=True) / qmax)
            bs[bs == 0] = 1.0
            bq = np.rint(basis / bs).clip(-qmax, qmax).astype(np.int16)
            cs = (np.abs(coef).max(axis=1, keepdims=True) / qmax)
            cs[cs == 0] = 1.0
            coef_q = np.rint(coef / cs).clip(-qmax, qmax).astype(np.int16)
        else:
            bq, bs = quantize_rows(basis, qmax)
            cq, cs2 = quantize_rows(coef.T, qmax)
            coef_q = cq.T
            cs = cs2.T
        # dequantized product
        if args.scheme == "component":
            approx = ((bq.astype(np.float64) * bs) @ (coef_q.astype(np.float64) * cs)).reshape(ns, nk, len(cells))
        else:
            approx = ((bq.astype(np.float64) * bs) @ (coef_q.astype(np.float64) * cs)).reshape(ns, nk, len(cells))
        bytes_total = (ns * nk + 60200) * r * (1 if args.dtype == "int8" else 2)

    gerr = float(np.abs(approx - G).mean())
    pred = np.empty((ns, nt, len(e2e)))
    for s in range(ns):
        blk = approx[s][:, e2e]
        pred[s] = (blk[:, :, None] + B[:, None, :]).min(axis=0).T
    accs = []
    for s in range(ns):
        for j in range(len(e2e)):
            accs.append(accuracy(M[s, :, j] + gaps[e2e[j]], pred[s, :, j] + gaps[e2e[j]]))
    summary = {
        "rank": r, "dtype": args.dtype, "scheme": args.scheme,
        "g_mae_ps": gerr,
        "accuracy": float(np.mean(accs)),
        "bytes_total": int(bytes_total),
    }
    print(json.dumps(summary, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
