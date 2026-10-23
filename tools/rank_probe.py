"""S3: rank/accuracy trade-off of the (source, target) port dimension.

For a sampled target position `p` the 208 exact source fields give the full
`208 x 288` delay matrix at displacement `dx = p.x - x0, dy = p.y - y0`.
Removing the gap-line sum (separable, exact) leaves the metric matrix.  We then
measure, for a range of ranks, the reconstruction error and the task accuracy of

    pred = rank_r(metric) + gap_sum

which is the "fine displacement cell x low-rank ports" representation used by the
old route.  This tells us how the byte budget splits between displacement cells
and port rank, and whether the old route is rank-bound.

Usage
    python tools/rank_probe.py --positions 120 --tag rank
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
    parser.add_argument("--positions", type=int, default=120)
    parser.add_argument("--tag", default="rank")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    from arch import load_architecture

    arch = load_architecture()
    index = json.loads((FIELDS / "index.json").read_text(encoding="utf-8"))["fields"]
    index = [f for f in index if (f["x"], f["y"]) == (40, 275)]
    index.sort(key=lambda f: f["port"])
    assert len(index) == 208, len(index)
    maps = [np.memmap(FIELDS / f["file"], dtype=np.uint16, mode="r").reshape(len(arch.insts), len(arch.targets))
            for f in index]
    insts = np.array(arch.insts, dtype=np.int32)
    rng = np.random.default_rng(2026)

    wall = np.array([(76 <= x <= 89) for x, _ in arch.insts])
    edge = np.array([x <= 12 or x >= 107 or y <= 12 or y >= 537 for x, y in arch.insts])
    interior = ~wall & ~edge
    picks = np.concatenate([
        rng.choice(np.nonzero(interior)[0], size=args.positions * 6 // 10, replace=False),
        rng.choice(np.nonzero(wall)[0], size=args.positions * 2 // 10, replace=False),
        rng.choice(np.nonzero(edge)[0], size=args.positions * 2 // 10, replace=False),
    ])

    ranks = [4, 8, 16, 32, 64, 96, 128, 192, 256]
    errors = {r: [] for r in ranks}
    accs = {r: [] for r in ranks}
    for p in picks:
        x, y = int(insts[p, 0]), int(insts[p, 1])
        matrix = np.empty((len(index), len(arch.targets)), dtype=np.float64)
        for i, mm in enumerate(maps):
            row = np.asarray(mm[p], dtype=np.float64)
            matrix[i] = row
        g = gap_for(arch, 40, 275, x, y)
        metric = matrix - g
        truth = matrix
        # subtract the global mean before SVD; it is stored for free as a scalar
        mean = metric.mean()
        centered = metric - mean
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        for r in ranks:
            approx = (u[:, :r] * s[:r]) @ vt[:r] + mean
            err = np.abs(approx - metric)
            errors[r].append(float(err.mean()))
            accs[r].append(accuracy(truth, approx + g))

    summary = {
        "tag": args.tag,
        "positions": len(picks),
        "sources": len(index),
        "targets": len(arch.targets),
        "rank_table": [
            {"rank": r,
             "mae_ps": float(np.mean(errors[r])),
             "p90_ps": float(np.percentile(errors[r], 90)),
             "accuracy": float(np.mean(accs[r]))}
            for r in ranks
        ],
    }
    print(json.dumps(summary, indent=2))
    print("\nrank | mae ps | accuracy | bytes/value at 100MB for 266305 cells")
    for row in summary["rank_table"]:
        per_cell = row["rank"] * (len(index) + len(arch.targets))
        print(f"{row['rank']:>4} | {row['mae_ps']:>6.2f} | {row['accuracy']:>8.3f} | {per_cell:>8} ")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
