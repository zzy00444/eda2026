"""S3-M2 part 4: lane-level decomposition (the "G" object).

Split the problem at the arrival lanes:

    M(s,t,cell) = min_k [ G(s,k,cell) + B(k,t) ]

  * `G(s,k,cell)` = exact distance from source pin `s` to node (cell, I_k), available
    from the all-ports fields (`fields_all`);
  * `B(k,t)`      = internal shortest path I_k -> target t, *exact*, 160x288 int16
    (92 KB, cache resident).

Only `G` needs a model.  G is `208 x 160` per cell (1.8x smaller than M) and it
removes the target-side last-hop variation, a prime suspect for the rotation that
defeats global-basis compression of M.

The probe measures the global-basis rank curve of G and the *end-to-end* accuracy of
`min_k (G_rank + B)` against the exact delay (computed from the target fields).

Usage
    python tools/lane_probe.py --grid-step 4,10 --ranks 8,16,32,64,128 --json-out build/lane_probe.json
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-step", default="4,10")
    parser.add_argument("--ranks", default="8,16,32,64,128")
    parser.add_argument("--e2e-cells", type=int, default=256, help="cells used for end-to-end accuracy")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tag", default="lane")
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
    print(f"sources {ns}, I lanes {nk}, targets {nt}", flush=True)

    index = json.loads((FIELDS_ALL / "index.json").read_text(encoding="utf-8"))["fields"]
    by_port = {f["port"]: f for f in index if (f["x"], f["y"]) == (40, 275)}
    missing = [p for p in src_names if p not in by_port]
    if missing:
        raise SystemExit(f"missing all-ports fields for {len(missing)} ports, e.g. {missing[:3]}")
    order = [by_port[p] for p in src_names]

    step_x, step_y = (int(v) for v in args.grid_step.split(","))
    cells = np.array([i for i, (x, y) in enumerate(arch.insts) if x % step_x == 0 and y % step_y == 0])
    print(f"cells {len(cells)}", flush=True)
    rng = np.random.default_rng(args.seed)
    e2e_cells = np.sort(rng.choice(len(cells), size=min(args.e2e_cells, len(cells)), replace=False))
    print(f"end-to-end cells {len(e2e_cells)}", flush=True)

    pid = {name: i for i, name in enumerate(arch.ports)}
    np_ = len(arch.ports)
    rows = [a.src for a in arch.arcs]
    cols = [a.dst for a in arch.arcs]
    vals = [a.delay for a in arch.arcs]
    graph = coo_matrix((vals, (rows, cols)), shape=(np_, np_)).tocsr()
    internal = dijkstra(graph, directed=True)
    B = internal[np.ix_(i_ports, arch.targets)].astype(np.float64)
    B[~np.isfinite(B)] = 1e9

    gaps = np.array([gap_for(arch, 40, 275, int(insts[c, 0]), int(insts[c, 1])) for c in cells], dtype=np.float64)
    G = np.empty((ns, nk, len(cells)), dtype=np.float32)
    M = np.zeros((ns, nt, len(e2e_cells)), dtype=np.float32)
    for s, entry in enumerate(order):
        path = FIELDS_ALL / entry["file"]
        data = np.memmap(path, dtype=np.uint16, mode="r").reshape(len(arch.insts), np_)
        G[s] = data[cells][:, i_ports].T.astype(np.float32)
        target_path = FIELDS / entry["file"]
        if target_path.exists():
            tdata = np.memmap(target_path, dtype=np.uint16, mode="r").reshape(len(arch.insts), nt)
            M[s] = tdata[cells[e2e_cells]].T.astype(np.float32)
    G -= gaps.astype(np.float32)[None, None, :]
    M -= gaps[e2e_cells].astype(np.float32)[None, None, :]
    print("G", G.shape, "M", M.shape, flush=True)

    Gf = G.reshape(ns * nk, len(cells)).astype(np.float64)
    gram = Gf.T @ Gf
    print("gram", gram.shape, flush=True)
    eigval, eigvec = np.linalg.eigh(gram)
    order_idx = np.argsort(-eigval)
    eigval, eigvec = eigval[order_idx], eigvec[:, order_idx]
    total = eigval.sum()
    print("explained energy at 8/16/32/64/128/192:",
          [float(eigval[:r].sum() / total) for r in (8, 16, 32, 64, 128, 192)], flush=True)

    results = []
    for r in [int(v) for v in args.ranks.split(",")]:
        basis = Gf @ eigvec[:, :r] / np.sqrt(np.maximum(eigval[:r], 1e-12))[None, :]
        coef = basis.T @ Gf
        approx = (basis @ coef).reshape(ns, nk, len(cells))
        gerr = float(np.abs(approx - G).mean())
        errs = []
        accs = []
        for si, s in enumerate(range(ns)):
            blk = approx[s][:, e2e_cells]                     # (nk, n_e2e)
            pred = (blk[:, :, None] + B[:, None, :]).min(axis=0)   # (n_e2e, nt)
            pred_delay = pred + gaps[e2e_cells][:, None]
            truth_delay = M[s].T + gaps[e2e_cells][:, None]
            errs.append(np.abs(pred_delay - truth_delay).ravel())
            for j in range(len(e2e_cells)):
                accs.append(accuracy(truth_delay[j], pred_delay[j]))
        errs = np.concatenate(errs)
        row = {
            "rank": r,
            "g_mae_ps": gerr,
            "e2e_mae_ps": float(errs.mean()),
            "e2e_p90_ps": float(np.percentile(errs, 90)),
            "accuracy": float(np.mean(accs)),
            "bytes": int(r * ns * nk + 60200 * r),
        }
        results.append(row)
        print(row, flush=True)

    summary = {"tag": args.tag, "sources": ns, "lanes": nk, "cells": len(cells),
               "e2e_cells": len(e2e_cells), "rank_table": results}
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
