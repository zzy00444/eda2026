"""B1: can the displacement dependence of the lane-level metric be parameterised by
a handful of scalars per (source, lane) plus one shared periodic table?

    G(s,k,cell) ~= a(s,k) + lam_x*dx + lam_y*dy + R[dx mod 10][dy mod 12]

with a *per-(s,k)* intercept and slopes, and one *shared* residue table R.  If the
residual is <=2 ps this is the structural breakthrough Plan B needs: the whole
model becomes ~33k int16 intercepts + 33k int16 slopes + one 120-entry table
(~200 KB), and a query evaluates 20 lanes with 20 lookups and a few adds.

The fit is done on half the cells and evaluated on the other half (per (s,k) pair).

Usage
    python tools/spectrum_probe.py --pairs 200 --grid-step 4,10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--grid-step", default="4,10")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    from arch import load_architecture

    arch = load_architecture()
    insts = np.array(arch.insts, dtype=np.int32)
    src_names = [arch.ports[i] for i in arch.query_sources]
    ns = len(src_names)
    i_ports = [i for i in range(len(arch.ports)) if arch.ports[i].startswith("I")]
    nk = len(i_ports)

    index = json.loads((FIELDS_ALL / "index.json").read_text(encoding="utf-8"))["fields"]
    by_port = {f["port"]: f for f in index if (f["x"], f["y"]) == (40, 275)}
    order = [by_port[p] for p in src_names]

    step_x, step_y = (int(v) for v in args.grid_step.split(","))
    cells = np.array([i for i, (x, y) in enumerate(arch.insts) if x % step_x == 0 and y % step_y == 0])
    dx_all = (insts[cells, 0] - 40).astype(np.float64)
    dy_all = (insts[cells, 1] - 275).astype(np.float64)
    gaps = np.array([gap_for(arch, 40, 275, int(insts[c, 0]), int(insts[c, 1])) for c in cells], dtype=np.float64)
    resid_key = ((insts[cells, 0] - 40) % 10) * 12 + ((insts[cells, 1] - 275) % 12)
    quad = ((dx_all < 0).astype(np.int64) * 2 + (dy_all < 0).astype(np.int64))
    print(f"cells {len(cells)}", flush=True)

    rng = np.random.default_rng(args.seed)
    pair_list = [(s, k) for s in range(ns) for k in range(nk)]
    picked = [pair_list[i] for i in rng.choice(len(pair_list), size=min(args.pairs, len(pair_list)), replace=False)]

    G_by_src = {}
    for s, entry in enumerate(order):
        path = FIELDS_ALL / entry["file"]
        data = np.memmap(path, dtype=np.uint16, mode="r").reshape(len(arch.insts), len(arch.ports))
        G_by_src[s] = data[cells][:, i_ports].astype(np.float32) - gaps.astype(np.float32)[:, None]
    print("loaded", flush=True)

    train = np.arange(len(cells)) % 2 == 0
    test = ~train
    rows = []
    for s, k in picked:
        raw = G_by_src[s][:, k].astype(np.float64)
        reach = raw < 60000.0
        y = raw
        if reach.sum() < 200:
            continue
        # design: 4 quadrant slopes + intercept + 120 residue dummies (shareable)
        n = len(y)
        X = np.zeros((n, 4 * 3 + 120))
        for q in range(4):
            m = quad == q
            X[m, 3 * q + 0] = dx_all[m]
            X[m, 3 * q + 1] = dy_all[m]
            X[m, 3 * q + 2] = 1.0
        X[np.arange(n), 12 + resid_key] = 1.0
        X[:, 12] = 0.0
        tr = train & reach
        te = test & reach
        coef, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        r = y - X @ coef
        rows.append({
            "s": int(s), "k": int(k),
            "rows": int(reach.sum()),
            "rmse_train": float(np.sqrt((r[tr] ** 2).mean())),
            "rmse_test": float(np.sqrt((r[te] ** 2).mean())),
            "p95_test": float(np.percentile(np.abs(r[te]), 95)),
            "max_test": float(np.abs(r[te]).max()),
        })
    if not rows:
        raise SystemExit("no reachable pairs")
    table = np.array([r["rmse_test"] for r in rows])
    summary = {
        "pairs": len(rows),
        "cells": len(cells),
        "rmse_test_mean": float(table.mean()),
        "rmse_test_median": float(np.median(table)),
        "rmse_test_p90": float(np.percentile(table, 90)),
        "rmse_test_max": float(table.max()),
        "params": {"intercepts_slopes": 12 * ns * nk, "residue_table": 120},
        "verdict": "PASS" if table.mean() <= 2.0 else ("MARGINAL" if table.mean() <= 5.0 else "FAIL"),
    }
    print(json.dumps(summary, indent=2))
    worst = sorted(rows, key=lambda r: -r["rmse_test"])[:8]
    for r in worst:
        print(f"  src={r['s']:>3} lane={r['k']:>3} rmse_test={r['rmse_test']:7.2f} p95={r['p95_test']:7.2f}")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({"summary": summary, "pairs": rows}, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
