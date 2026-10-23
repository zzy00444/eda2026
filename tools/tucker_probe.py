"""S3: Tucker decomposition of (source port, target port, displacement cell).

The shared-basis probe (`basis_probe.py`) saturates around 97.9% accuracy because
the port-pair pattern *rotates* from cell to cell.  A Tucker model keeps shared
port factors but a per-cell mixing matrix:

    M(cell, s, t) ~= sum_{a,b} A[s,a] * G[cell,a,b] * B[t,b]

so the per-cell structure is a small (a x b) core instead of a scalar multiple of
a global basis.  Storage is `208*a + 288*b + cells*a*b` int8 values, and a query
is an a*b dot product (plus the separable gap-line sum).

Usage
    python tools/tucker_probe.py --cells 400 --tag tucker
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ROOT / "build" / "fields"


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
    parser.add_argument("--cells", type=int, default=400)
    parser.add_argument("--tag", default="tucker")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    from arch import load_architecture

    arch = load_architecture()
    index = json.loads((FIELDS / "index.json").read_text(encoding="utf-8"))["fields"]
    index = sorted([f for f in index if (f["x"], f["y"]) == (40, 275)], key=lambda f: f["port"])
    ns, nt = len(index), len(arch.targets)
    maps = [np.memmap(FIELDS / f["file"], dtype=np.uint16, mode="r").reshape(len(arch.insts), nt)
            for f in index]
    insts = np.array(arch.insts, dtype=np.int32)
    rng = np.random.default_rng(4242)

    wall = np.array([(76 <= x <= 89) for x, _ in arch.insts])
    edge = np.array([x <= 12 or x >= 107 or y <= 12 or y >= 537 for x, y in arch.insts])
    interior = ~wall & ~edge
    n = args.cells
    picks = np.concatenate([
        rng.choice(np.nonzero(interior)[0], size=n * 6 // 10, replace=False),
        rng.choice(np.nonzero(wall)[0], size=n * 2 // 10, replace=False),
        rng.choice(np.nonzero(edge)[0], size=n * 2 // 10, replace=False),
    ])

    tensor = np.empty((ns, nt, len(picks)), dtype=np.float32)
    gaps = np.empty(len(picks), dtype=np.int64)
    for j, p in enumerate(picks):
        x, y = int(insts[p, 0]), int(insts[p, 1])
        for i, mm in enumerate(maps):
            tensor[i, :, j] = np.asarray(mm[p], dtype=np.float32)
        g = gap_for(arch, 40, 275, x, y)
        gaps[j] = g
        tensor[:, :, j] -= g

    train = np.arange(len(picks)) % 2 == 0
    test = ~train
    X = tensor[:, :, train]
    # HOSVD factors from the training cells.
    a_unfold = X.transpose(0, 2, 1).reshape(ns, -1)
    b_unfold = X.transpose(1, 2, 0).reshape(nt, -1)
    ua = np.linalg.svd(a_unfold.astype(np.float64), full_matrices=False)[0]
    ub = np.linalg.svd(b_unfold.astype(np.float64), full_matrices=False)[0]

    rows = []
    for ra, rb in [(4, 4), (8, 8), (8, 16), (16, 16), (16, 24), (24, 24), (32, 32), (32, 64), (64, 64)]:
        A = ua[:, :ra]
        B = ub[:, :rb]
        core = np.einsum("sa,stc,tb->abc", A, X.astype(np.float64), B)
        approx = np.einsum("sa,abc,tb->stc", A, core, B)
        err_train = np.abs(approx - X).mean()
        # evaluate on held-out cells: project each cell onto the fixed factors
        Xt = tensor[:, :, test]
        core_t = np.einsum("sa,stc,tb->abc", A, Xt.astype(np.float64), B)
        approx_t = np.einsum("sa,abc,tb->stc", A, core_t, B)
        err = np.abs(approx_t - Xt)
        accs = []
        test_idx = np.nonzero(test)[0]
        for j in range(len(test_idx)):
            pred = approx_t[:, :, j] + gaps[test_idx[j]]
            truth = Xt[:, :, j] + gaps[test_idx[j]]
            accs.append(accuracy(truth, pred))
        rows.append({"ra": ra, "rb": rb, "mae_test_ps": float(err.mean()),
                     "p90_test_ps": float(np.percentile(err, 90)),
                     "mae_train_ps": float(err_train),
                     "accuracy": float(np.mean(accs))})
        print(rows[-1])

    summary = {
        "tag": args.tag,
        "cells_sampled": len(picks),
        "pairs": [ns, nt],
        "rank_table": rows,
        "core_bytes_all_cells": [
            {"ra": r["ra"], "rb": r["rb"], "bytes": 60200 * r["ra"] * r["rb"]} for r in rows
        ],
    }
    print(json.dumps(summary, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
