"""S3: measure how well a "quadrant slope + residue table" (M1) represents the
exact metric, using the exact field dumps as ground truth.

For each source field and a sample of target ports we take the exact delay over
all 60,200 SRB positions, remove the monotone gap-line sum (which is separable
and exact), and fit

    M(dx, dy) ~= a_q*dx + b_q*dy + c_q + R[dx mod 10][dy mod 12]

per sign quadrant q, on half of the positions, then evaluate on the other half.
That residual is the *representation error floor* of the old route's table shape
(no rank truncation involved), so it separates "table shape" from "compression".

Usage
    python tools/fit_metric.py --pins 4 --targets 12 --tag m1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ROOT / "build" / "fields"


def fit_pair(dx: np.ndarray, dy: np.ndarray, metric: np.ndarray, train: np.ndarray):
    """Solve the least-squares design for one (source, target) pair."""
    quad = (dx < 0).astype(np.int64) * 2 + (dy < 0).astype(np.int64)
    resid = ((dx % 10) * 12 + (dy % 12)).astype(np.int64)
    n = len(dx)
    X = np.zeros((n, 12 + 120))
    for q in range(4):
        m = quad == q
        X[m, 3 * q + 0] = dx[m]
        X[m, 3 * q + 1] = dy[m]
        X[m, 3 * q + 2] = 1.0
    X[np.arange(n), 12 + resid] = 1.0
    X[:, 12] = 0.0  # drop one residue class to avoid collinearity
    coef, *_ = np.linalg.lstsq(X[train], metric[train], rcond=None)
    pred = X @ coef
    return pred, np.abs(pred - metric)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pins", type=int, default=4, help="number of source fields")
    parser.add_argument("--stride", type=int, default=23, help="field stride across the index")
    parser.add_argument("--targets", type=int, default=12, help="target ports sampled per field")
    parser.add_argument("--tag", default="m1")
    parser.add_argument("--far", action="store_true", help="fit/evaluate only |dx|>20 or |dy|>24")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    index = json.loads((FIELDS / "index.json").read_text(encoding="utf-8"))["fields"]
    fields = index[:: args.stride][: args.pins]

    from arch import load_architecture  # local import keeps CLI startup light

    arch = load_architecture()
    insts = np.array(arch.insts, dtype=np.int32)
    sx_all = insts[:, 0].astype(np.int64)
    sy_all = insts[:, 1].astype(np.int64)
    rng = np.random.default_rng(11)
    records = []
    for entry in fields:
        field = np.fromfile(FIELDS / entry["file"], dtype=np.uint16).reshape(len(arch.insts), len(arch.targets))
        sx, sy = entry["x"], entry["y"]
        tx = sx_all
        ty = sy_all
        dx = tx - sx
        dy = ty - sy
        gap = np.zeros(len(insts), dtype=np.int64)
        for line in arch.vertical_lines:
            lo, hi = np.minimum(sx, tx), np.maximum(sx, tx)
            gap += line.delay * ((lo <= line.site) & (line.site < hi))
        for line in arch.horizontal_lines:
            lo, hi = np.minimum(sy, ty), np.maximum(sy, ty)
            gap += line.delay * ((lo <= line.site) & (line.site < hi))
        train = np.arange(len(insts)) % 2 == 0
        far = (np.abs(dx) > 20) | (np.abs(dy) > 24)
        if args.far:
            train = train & far
        for k in rng.choice(len(arch.targets), size=args.targets, replace=False):
            exact = field[:, k].astype(np.int64)
            if (exact == 65535).any():
                continue
            metric = exact - gap
            pred, err = fit_pair(dx, dy, metric, train)
            test = far if args.far else ~train
            records.append({
                "src": f"{sx},{sy},{entry['port']}",
                "tgt": arch.ports[arch.targets[k]],
                "tgt_class": arch.ports[arch.targets[k]][0],
                "mae_train": float(err[train].mean()),
                "mae_test": float(err[test].mean()),
                "p90_test": float(np.percentile(err[test], 90)),
                "p99_test": float(np.percentile(err[test], 99)),
                "signed_test": float((pred - metric)[test].mean()),
            })
    table = pd.DataFrame(records)
    summary = {
        "tag": args.tag,
        "pairs": len(table),
        "fields": [f"{e['x']},{e['y']},{e['port']}" for e in fields],
        "mae_test_mean": float(table.mae_test.mean()),
        "mae_test_median": float(table.mae_test.median()),
        "p90_test_mean": float(table.p90_test.mean()),
        "p99_test_mean": float(table.p99_test.mean()),
        "by_tgt_class": table.groupby("tgt_class")[["mae_test", "p90_test"]].mean().round(3).to_dict(),
    }
    print(json.dumps(summary, indent=2))
    print(table.sort_values("mae_test", ascending=False).head(12).to_string(index=False))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({"summary": summary, "pairs": records}, indent=2) + "\n",
                                 encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
