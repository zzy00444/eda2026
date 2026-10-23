"""S3-M2 viability: does a *cell-adaptive* basis fit the budget?

The S3 probes showed the pair pattern rotates with the displacement: one global
basis saturates at ~97.96% accuracy (rank 256) while per-cell rank-32 reaches
99.43% at an impossible 956 MB.  The deciding question is whether the rotation can
be captured by a small number of *pattern clusters*:

    M(cell, s, t) ~= sum_a A_cluster(cell)[s, a] * coef[cell, a]

i.e. `k` shared rank-r bases (one per pattern cluster) plus one r-vector per cell.
Size = k*r*59904 + 60200*r bytes (int8).

Cells are clustered by their pattern (normalised coefficients in a global basis),
each cluster's basis is fitted on half of its cells and evaluated on the other
half.  Cells are sampled uniformly (60% interior / 20% wall / 20% edge).

Usage
    python tools/adaptive_probe.py --cells 4096 --json-out build/adaptive_probe.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans2

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
    parser.add_argument("--cells", type=int, default=4096)
    parser.add_argument("--ks", default="8,16,32,64")
    parser.add_argument("--ranks", default="8,16,24")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tag", default="adaptive")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    from arch import load_architecture

    arch = load_architecture()
    index = json.loads((FIELDS / "index.json").read_text(encoding="utf-8"))["fields"]
    index = sorted([f for f in index if (f["x"], f["y"]) == (40, 275)], key=lambda f: f["port"])
    ns, nt = len(index), len(arch.targets)
    insts = np.array(arch.insts, dtype=np.int32)
    rng = np.random.default_rng(args.seed)

    print(f"loading {len(index)} fields ...", flush=True)
    fields = np.empty((len(arch.insts), nt, ns), dtype=np.uint16)
    for i, entry in enumerate(index):
        fields[:, :, i] = np.memmap(FIELDS / entry["file"], dtype=np.uint16, mode="r").reshape(len(arch.insts), nt)
    print("loaded", fields.nbytes / (1 << 30), "GiB", flush=True)

    wall = np.array([(76 <= x <= 89) for x, _ in arch.insts])
    edge = np.array([x <= 12 or x >= 107 or y <= 12 or y >= 537 for x, y in arch.insts])
    interior = ~wall & ~edge
    n = args.cells
    picks = np.concatenate([
        rng.choice(np.nonzero(interior)[0], size=n * 6 // 10, replace=False),
        rng.choice(np.nonzero(wall)[0], size=n * 2 // 10, replace=False),
        rng.choice(np.nonzero(edge)[0], size=n * 2 // 10, replace=False),
    ])
    gaps = np.array([gap_for(arch, 40, 275, int(insts[p, 0]), int(insts[p, 1])) for p in picks])
    M = fields[picks].transpose(1, 2, 0).reshape(ns * nt, len(picks)).astype(np.float32)
    M -= gaps.astype(np.float32)[None, :]
    print(f"M: {M.shape} {M.nbytes / (1 << 30):.2f} GiB", flush=True)

    # pattern space: top right singular vectors of M (cells x r), energy weighted
    gram = (M.astype(np.float64).T @ M.astype(np.float64))
    eigval, eigvec = np.linalg.eigh(gram)
    order = np.argsort(-eigval)
    eigval, eigvec = eigval[order][:32], eigvec[:, order][:, :32]
    keep = eigval > 1e-6
    pattern = eigvec[:, keep] * np.sqrt(np.maximum(eigval[keep], 0.0))[None, :]
    pattern /= np.maximum(np.linalg.norm(pattern, axis=1, keepdims=True), 1e-12)
    print(f"pattern space: {pattern.shape}", flush=True)

    results = []
    ks = [int(v) for v in args.ks.split(",")]
    ranks = [int(v) for v in args.ranks.split(",")]
    for k in ks:
        cent, label = kmeans2(pattern, k, minit="++", seed=args.seed, missing="raise")
        for r in ranks:
            errors = []
            accs = []
            for c in range(k):
                idx = np.nonzero(label == c)[0]
                if len(idx) < 4:
                    continue
                perm = rng.permutation(len(idx))
                half = len(idx) // 2
                tr, te = idx[perm[:half]], idx[perm[half:]]
                X = M[:, tr].astype(np.float64)
                r_eff = min(r, X.shape[1])
                u, s, _ = np.linalg.svd(X, full_matrices=False)
                basis = u[:, :r_eff]
                Xt = M[:, te].astype(np.float64)
                approx = basis @ (basis.T @ Xt)
                errors.append(np.abs(approx - Xt).ravel())
                for j in range(Xt.shape[1]):
                    accs.append(accuracy(Xt[:, j].reshape(ns, nt), approx[:, j].reshape(ns, nt)))
            errors = np.concatenate(errors)
            row = {
                "k": k, "rank": r,
                "mae_ps": float(errors.mean()),
                "p90_ps": float(np.percentile(errors, 90)),
                "accuracy": float(np.mean(accs)),
                "bytes": k * r * ns * nt + 60200 * r,
            }
            results.append(row)
            print(row, flush=True)

    summary = {"tag": args.tag, "cells": len(picks), "pairs": [ns, nt], "rank_table": results}
    print(json.dumps(summary, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
