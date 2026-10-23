"""Evaluate a lean model (fit_lean output) on the public labels with coverage split.

    python tools/eval_lean.py --model build/lean_s5k.npz --rows 1000000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from arch import load_architecture  # noqa: E402
from fit_lean import gap_arrays, source_nets, telescope_gap, PUBLIC  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=0)
    args = parser.parse_args()

    arch = load_architecture()
    vp, hp = gap_arrays(arch)
    insts = np.array(arch.insts)
    occ = np.zeros((550, 120), dtype=bool)
    for x, y in arch.insts:
        occ[y, x] = True
    d = np.load(args.model)
    U8, su, C, scale = d["U8"], float(d["su"]), d["C"], d["scale"]
    A, Bx, By = d["A"], d["Bx"], d["By"]
    nx, ny, ns, nt = [int(v) for v in d["meta"]]
    rank = int(d["rank"])

    src = [arch.ports[i] for i in arch.query_sources]
    sm = {p: i for i, p in enumerate(src)}
    tm = {arch.ports[t]: i for i, t in enumerate(arch.targets)}
    df = pd.read_csv(PUBLIC, nrows=args.rows if args.rows else None)
    sp = df["From"].str.extract(r"^SRB_(\d+)_(\d+)/(.+)$")
    tp = df["To"].str.extract(r"^SRB_(\d+)_(\d+)/(.+)$")
    si = sp[2].map(sm).to_numpy(np.int64); ti = tp[2].map(tm).to_numpy(np.int64)
    sx = sp[0].to_numpy(np.int32); sy = sp[1].to_numpy(np.int32)
    tx = tp[0].to_numpy(np.int32); ty = tp[1].to_numpy(np.int32)
    cell = (tx - sx + 119) * ny + (ty - sy + 549)
    pair = si * nt + ti
    pred = np.zeros(len(df))
    for st in range(0, len(df), 200_000):
        sl = slice(st, min(st + 200_000, len(df)))
        c = C[:, cell[sl]].astype(np.int32).T
        u = U8[pair[sl]].astype(np.int32)
        pred[sl] = (c * u).sum(axis=1) * (scale[cell[sl]] * su)
    vxs, vys = source_nets(arch)
    g = telescope_gap(vp, hp, vxs[si], vys[si], sx, sy, tx, ty)
    p = np.rint(pred + g + A[pair] + Bx[pair] * (tx - sx).astype(np.float32) + By[pair] * (ty - sy).astype(np.float32))
    truth = df["delay"].to_numpy(dtype=np.int64)

    # coverage of this model's fit positions is not stored; approximate "covered"
    # by the displacement being realisable from the first dump position (40,275)
    # only when the model was fitted with a single position.  Report per-displacement
    # rows instead when multiple positions were used.
    xs = tx - sx + 40; ys = ty - sy + 275
    inside = (xs >= 0) & (xs < 120) & (ys >= 0) & (ys < 550)
    covered = inside & occ[np.clip(ys, 0, 549), np.clip(xs, 0, 119)]

    out = {"model": str(args.model), "rank": rank, "rows": int(len(df))}
    def report(mask, name):
        q = 100.0 * (1.0 - np.tanh(4.0 * np.abs(p[mask] - truth[mask]) / np.maximum(truth[mask], 1)))
        e = np.abs(p[mask] - truth[mask])
        out[name] = {"n": int(mask.sum()), "accuracy": float(q.mean()), "mae": float(e.mean()),
                     "p50_p90_p99": [float(np.percentile(e, k)) for k in (50, 90, 99)]}
    allm = np.ones(len(df), dtype=bool)
    report(allm, "all")
    report(covered, "covered_40_275")
    report(~covered, "uncovered_40_275")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
