"""S3: per-cluster low-rank basis (cell-adaptive), the model shape that can beat
the shared-basis wall.

`basis_probe.py` shows one global pair basis saturates at ~97.95% accuracy even at
rank 256: the pair pattern rotates with the displacement.  This probe asks whether
the rotation is captured by a *displacement class*: cells are clustered by
`(dx mod 10, dy mod 12)` and each cluster gets its own rank-r basis, fitted on half
of the cluster's cells and evaluated on the other half.

Size of the shipped model would be `clusters * r * pairs * 1 byte + cells * r`.

Usage
    python tools/cluster_probe.py --cells 2400 --tag cluster
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
    parser.add_argument("--cells", type=int, default=2400)
    parser.add_argument("--ranks", default="4,8,16")
    parser.add_argument("--tag", default="cluster")
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
    rng = np.random.default_rng(99)

    picks = rng.choice(len(arch.insts), size=args.cells, replace=False)
    dx = insts[picks, 0] - 40
    dy = insts[picks, 1] - 275
    cluster = ((dx % 10) * 12 + (dy % 12)).astype(np.int64)

    matrices = np.empty((ns * nt, len(picks)), dtype=np.float32)
    gaps = np.empty(len(picks), dtype=np.int64)
    for j, p in enumerate(picks):
        x, y = int(insts[p, 0]), int(insts[p, 1])
        for i, mm in enumerate(maps):
            matrices[i * nt:(i + 1) * nt, j] = np.asarray(mm[p], dtype=np.float32)
        gaps[j] = gap_for(arch, 40, 275, x, y)
        matrices[:, j] -= gaps[j]

    ranks = [int(v) for v in args.ranks.split(",")]
    records = []
    for r in ranks:
        errors = np.empty(0)
        accs = []
        for c in range(120):
            idx = np.nonzero(cluster == c)[0]
            if len(idx) < 4:
                continue
            perm = rng.permutation(len(idx))
            half = len(idx) // 2
            tr, te = idx[perm[:half]], idx[perm[half:]]
            X = matrices[:, tr].astype(np.float64)
            u, s, _ = np.linalg.svd(X, full_matrices=False)
            r_eff = min(r, len(tr))
            basis = u[:, :r_eff]
            Xt = matrices[:, te].astype(np.float64)
            approx = basis @ (basis.T @ Xt)
            err = np.abs(approx - Xt)
            errors = np.concatenate([errors, err.ravel()])
            for j in range(Xt.shape[1]):
                accs.append(accuracy(Xt[:, j].reshape(ns, nt), approx[:, j].reshape(ns, nt)))
        records.append({
            "rank": r,
            "mae_ps": float(errors.mean()),
            "p90_ps": float(np.percentile(errors, 90)),
            "accuracy": float(np.mean(accs)),
            "bytes": 120 * r * ns * nt + 60200 * r,
        })
        print(records[-1])

    summary = {"tag": args.tag, "cells": len(picks), "clusters": 120, "rank_table": records}
    print(json.dumps(summary, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
