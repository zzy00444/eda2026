"""S3-M2 viability, part 2: displacement *tile* shared bases.

Pattern clustering on the coefficient space failed (`adaptive_probe.py`): the pair
pattern varies continuously with the displacement, so discrete clusters are worse
than one global basis.  A tile partition of the displacement plane instead uses a
basis per *spatial neighbourhood* of displacements, which is the natural way to
exploit smooth rotation.

For each tile (tw x th cells) a rank-r basis is fitted on half of the tile's cells
and evaluated on the other half.  Size = tiles*r*59904 + 60200*r bytes.

Cells are sampled on a regular grid of the displacement plane (uniform coverage),
so each tile has enough cells even for fine partitions.

Usage
    python tools/tile_probe.py --grid-step 2,5 --json-out build/tile_probe.json
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
    parser.add_argument("--grid-step", default="2,5", help="sampling step over x and y")
    parser.add_argument("--tiles", default="60x275,30x138,15x69,12x55,8x37")
    parser.add_argument("--ranks", default="8,16,24")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--tag", default="tile")
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

    step_x, step_y = (int(v) for v in args.grid_step.split(","))
    cells = [i for i, (x, y) in enumerate(arch.insts) if x % step_x == 0 and y % step_y == 0]
    picks = np.array(cells, dtype=np.int64)
    print(f"sampled {len(picks)} cells on a {step_x}x{step_y} grid", flush=True)
    gaps = np.array([gap_for(arch, 40, 275, int(insts[p, 0]), int(insts[p, 1])) for p in picks])
    M = fields[picks].transpose(1, 2, 0).reshape(ns * nt, len(picks)).astype(np.float32)
    M -= gaps.astype(np.float32)[None, :]
    print(f"M: {M.shape} {M.nbytes / (1 << 30):.2f} GiB", flush=True)

    x = insts[picks, 0]
    y = insts[picks, 1]
    results = []
    for spec in args.tiles.split(","):
        tw, th = (int(v) for v in spec.split("x"))
        tile = (x // tw).astype(np.int64) * 1000 + (y // th)
        for r in [int(v) for v in args.ranks.split(",")]:
            errors = []
            accs = []
            for t in np.unique(tile):
                idx = np.nonzero(tile == t)[0]
                if len(idx) < 4:
                    continue
                perm = rng.permutation(len(idx))
                half = len(idx) // 2
                tr, te = idx[perm[:half]], idx[perm[half:]]
                X = M[:, tr].astype(np.float64)
                r_eff = min(r, X.shape[1])
                u, _, _ = np.linalg.svd(X, full_matrices=False)
                basis = u[:, :r_eff]
                Xt = M[:, te].astype(np.float64)
                approx = basis @ (basis.T @ Xt)
                errors.append(np.abs(approx - Xt).ravel())
                for j in range(Xt.shape[1]):
                    accs.append(accuracy(Xt[:, j].reshape(ns, nt), approx[:, j].reshape(ns, nt)))
            errors = np.concatenate(errors)
            n_tiles = len(np.unique(tile))
            row = {
                "tile": spec, "tiles": int(n_tiles), "rank": r,
                "cells_per_tile": float(len(picks) / n_tiles),
                "mae_ps": float(errors.mean()),
                "p90_ps": float(np.percentile(errors, 90)),
                "accuracy": float(np.mean(accs)),
                "bytes": int(n_tiles) * r * ns * nt + 60200 * r,
            }
            results.append(row)
            print(row, flush=True)

    summary = {"tag": args.tag, "cells": len(picks), "pairs": [ns, nt], "rank_table": results}
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
