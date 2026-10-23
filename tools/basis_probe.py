"""S3: accuracy vs a *shared* port basis (the "rank x dot-product" model shape).

The old route stores a rank-r dot product per displacement cell.  Here we measure
the best-case version of that shape with exact labels: one shared pair basis
`A[s,t,r]` (208x288) plus per-cell coefficients `c[cell,r]`,

    M(cell, s, t) ~= sum_r A[s,t,r] * c[cell,r]

which is a rank-r factorization of the (208*288) x cells matrix.  We fit the basis
on half of the sampled cells and evaluate on the other half (coefficients are
always obtained from the exact oracle offline, so they are free to store).

Usage
    python tools/basis_probe.py --cells 400 --tag basis
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
    parser.add_argument("--tag", default="basis")
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

    metrics = np.empty((ns * nt, len(picks)), dtype=np.float32)
    gaps = np.empty(len(picks), dtype=np.int64)
    for j, p in enumerate(picks):
        x, y = int(insts[p, 0]), int(insts[p, 1])
        block = np.empty((ns, nt), dtype=np.float32)
        for i, mm in enumerate(maps):
            block[i] = np.asarray(mm[p], dtype=np.float32)
        g = gap_for(arch, 40, 275, x, y)
        gaps[j] = g
        metrics[:, j] = (block - g).ravel()

    train = np.arange(len(picks)) % 2 == 0
    test = ~train
    u, s, vt = np.linalg.svd(metrics[:, train].astype(np.float64), full_matrices=False)
    ranks = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    rows = []
    truth_flat = metrics[:, test]
    n_test = int(test.sum())
    for r in ranks:
        basis = u[:, :r]                       # orthonormal pair basis (pairs x r)
        coef = basis.T @ truth_flat            # per-cell coefficients (r x cells)
        approx = basis @ coef                  # exact projection onto the basis
        err = np.abs(approx - truth_flat)
        accs = []
        for j in range(n_test):
            pred = approx[:, j].reshape(ns, nt) + gaps[test][j]
            truth = truth_flat[:, j].reshape(ns, nt) + gaps[test][j]
            accs.append(accuracy(truth, pred))
        rows.append({"rank": r, "mae_ps": float(err.mean()), "p90_ps": float(np.percentile(err, 90)),
                     "accuracy": float(np.mean(accs))})
        print(rows[-1])

    summary = {
        "tag": args.tag,
        "cells_sampled": len(picks),
        "train_cells": int(train.sum()),
        "pairs": ns * nt,
        "rank_table": rows,
        "model_bytes_at_all_cells": [
            {"rank": r, "bytes": 208 * 288 * r + 60200 * r} for r in ranks
        ],
    }
    print(json.dumps(summary, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
