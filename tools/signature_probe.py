"""S3-M2 part 3: cluster displacement cells by their *combinatorial hop signature*.

Pattern clustering and spatial tiles both failed, so the rotation is neither
discrete in pattern space nor smooth in displacement space.  The physical driver is
the optimal hop multiset: how many span-1/2/4/10 (x) and 1/2/4/12 (y) hops the
cheapest route uses.  That multiset is computable per displacement with a tiny
coin DP, so it is a *free* per-cell label at query time.

This probe groups cells by signature and measures the held-out reconstruction
error of a per-signature rank-r basis, plus the dictionary dimension needed to
store the signature bases compactly.

Usage
    python tools/signature_probe.py --json-out build/signature_probe.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ROOT / "build" / "fields"

# cheapest hop cost per move span (from the exact lane cost matrix, rounded down)
X_COINS = {1: 48, 2: 52, 4: 73, 10: 103}
Y_COINS = {1: 48, 2: 52, 4: 73, 12: 104}


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


def coin_signature(max_d: int, coins: dict[int, int]):
    """Best multiset of spans for every distance 0..max_d (min cost, prefer long spans)."""
    inf = 1 << 30
    best = np.full(max_d + 1, inf, dtype=np.int64)
    prev = np.full(max_d + 1, -1, dtype=np.int64)
    choice = np.full(max_d + 1, -1, dtype=np.int64)
    best[0] = 0
    spans = sorted(coins, reverse=True)
    for d in range(1, max_d + 1):
        for s in spans:
            if s <= d and best[d - s] + coins[s] <= best[d]:
                best[d] = best[d - s] + coins[s]
                prev[d] = d - s
                choice[d] = s
    counts = np.zeros((max_d + 1, len(spans)), dtype=np.int8)
    for d in range(1, max_d + 1):
        counts[d] = counts[prev[d]]
        counts[d, spans.index(choice[d])] += 1
    return counts, spans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-step", default="2,5")
    parser.add_argument("--ranks", default="8,16")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--tag", default="signature")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    from arch import load_architecture

    arch = load_architecture()
    index = json.loads((FIELDS / "index.json").read_text(encoding="utf-8"))["fields"]
    index = sorted([f for f in index if (f["x"], f["y"]) == (40, 275)], key=lambda f: f["port"])
    ns, nt = len(index), len(arch.targets)
    insts = np.array(arch.insts, dtype=np.int32)
    rng = np.random.default_rng(args.seed)

    cx, sx_spans = coin_signature(119, X_COINS)
    cy, sy_spans = coin_signature(549, Y_COINS)
    print("x coin multiset: distance 20 ->", dict(zip(sx_spans, cx[20].tolist())),
          "| distance 17 ->", dict(zip(sx_spans, cx[17].tolist())))
    print("y coin multiset: distance 100 ->", dict(zip(sy_spans, cy[100].tolist())))

    print(f"loading {len(index)} fields ...", flush=True)
    fields = np.empty((len(arch.insts), nt, ns), dtype=np.uint16)
    for i, entry in enumerate(index):
        fields[:, :, i] = np.memmap(FIELDS / entry["file"], dtype=np.uint16, mode="r").reshape(len(arch.insts), nt)
    print("loaded", fields.nbytes / (1 << 30), "GiB", flush=True)

    step_x, step_y = (int(v) for v in args.grid_step.split(","))
    cells = np.array([i for i, (x, y) in enumerate(arch.insts) if x % step_x == 0 and y % step_y == 0])
    x = insts[cells, 0]
    y = insts[cells, 1]
    ddx = np.abs(x - 40)
    ddy = np.abs(y - 275)
    # signature: sign bit + multiset indices packed
    sig = np.zeros(len(cells), dtype=np.int64)
    for k in range(len(sx_spans)):
        sig = sig * 16 + cx[ddx, k]
    for k in range(len(sy_spans)):
        sig = sig * 64 + cy[ddy, k]
    sig = sig * 4 + (x >= 40).astype(np.int64) * 2 + (y >= 275).astype(np.int64)
    print(f"cells {len(cells)}, distinct signatures {len(np.unique(sig))}", flush=True)

    gaps = np.array([gap_for(arch, 40, 275, int(xx), int(yy)) for xx, yy in zip(x, y)])
    M = fields[cells].transpose(1, 2, 0).reshape(ns * nt, len(cells)).astype(np.float32)
    M -= gaps.astype(np.float32)[None, :]

    results = []
    for r in [int(v) for v in args.ranks.split(",")]:
        errors = []
        accs = []
        bases = []
        for s in np.unique(sig):
            idx = np.nonzero(sig == s)[0]
            if len(idx) < 4:
                continue
            perm = rng.permutation(len(idx))
            half = len(idx) // 2
            tr, te = idx[perm[:half]], idx[perm[half:]]
            X = M[:, tr].astype(np.float64)
            r_eff = min(r, X.shape[1])
            u, _, _ = np.linalg.svd(X, full_matrices=False)
            basis = u[:, :r_eff]
            bases.append(basis)
            Xt = M[:, te].astype(np.float64)
            approx = basis @ (basis.T @ Xt)
            errors.append(np.abs(approx - Xt).ravel())
            for j in range(Xt.shape[1]):
                accs.append(accuracy(Xt[:, j].reshape(ns, nt), approx[:, j].reshape(ns, nt)))
        errors = np.concatenate(errors)
        n_sig = len(bases)
        # dictionary test: how many atoms explain the signature bases themselves?
        stacked = np.concatenate(bases, axis=1) if bases else np.zeros((ns * nt, 1))
        sv = np.linalg.svd(stacked.T @ stacked, compute_uv=False)
        energy = np.cumsum(sv) / sv.sum()
        n_atoms = int(np.searchsorted(energy, 0.999)) + 1
        row = {
            "rank": r, "signatures_used": n_sig,
            "mae_ps": float(errors.mean()),
            "p90_ps": float(np.percentile(errors, 90)),
            "accuracy": float(np.mean(accs)),
            "bytes_direct": n_sig * r * ns * nt + 60200 * r,
            "dictionary_atoms_for_999": n_atoms,
            "bytes_with_dictionary": n_atoms * ns * nt + n_sig * n_atoms + 60200 * r,
        }
        results.append(row)
        print(row, flush=True)

    summary = {"tag": args.tag, "cells": len(cells), "signatures": int(len(np.unique(sig))),
               "rank_table": results}
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
