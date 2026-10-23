"""S2 error decomposition for a candidate's predictions against exact labels.

Joins predictions with the exact-label feature table produced by tools/labels.py
and reports, per group, the accuracy, mae and the share of the total accuracy
loss.  Also reports the "opportunity" of each group: the overall accuracy if that
group were predicted exactly, which ranks where modelling effort pays off.

Usage
    python tools/decompose.py --features build/s2_features.parquet \
        --predictions build/s2_old_result.csv --tag s2_old [--json-out build/s2_old.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def points(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    pred = np.maximum(0, np.rint(pred))
    out = np.zeros(len(truth), dtype=np.float64)
    nz = truth != 0
    out[nz] = 100.0 * (1.0 - np.tanh(4.0 * np.abs(pred[nz] - truth[nz]) / truth[nz]))
    out[~nz] = 100.0 * (pred[~nz] == 0)
    return out


def group_report(data: pd.DataFrame, points_: np.ndarray, by: str, edges=None) -> list[dict]:
    if edges is None:
        groups = data.groupby(by, dropna=False)
    else:
        groups = data.groupby(pd.cut(data[by], edges), dropna=False, observed=True)
    base = points_.mean()
    rows = []
    for key, frame in groups:
        idx = frame.index
        pts = points_[idx]
        loss = (100.0 - pts).mean() * len(idx) / len(points_)
        exact = (pts >= 100.0 - 1e-9)
        hypothetical = (base * len(points_) - pts.sum() + 100.0 * len(idx)) / len(points_)
        rows.append({
            "group": str(key),
            "rows": int(len(idx)),
            "share": len(idx) / len(points_),
            "acc": float(pts.mean()),
            "mae": float(np.abs(frame.pred - frame.delay).mean()),
            "exact_frac": float(exact.mean()),
            "loss_share": float(loss / max(1e-9, (100.0 - base))),
            "loss_points": float(loss),
            "acc_if_exact": float(hypothetical),
        })
    rows.sort(key=lambda r: -r["loss_points"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--tag", default="decompose")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    data = pd.read_parquet(args.features)
    pred = pd.read_csv(args.predictions)
    if len(pred) != len(data):
        raise SystemExit(f"row mismatch: features {len(data)} vs predictions {len(pred)}")
    if not (pred["From"].to_numpy() == data["From"].to_numpy()).all():
        raise SystemExit("prediction rows are not aligned with the feature table")
    data["pred"] = pred["delay"].to_numpy()
    truth = data["delay"].to_numpy(dtype=np.float64)
    pts = points(truth, data["pred"].to_numpy())
    base = pts.mean()

    summary = {
        "tag": args.tag,
        "rows": len(data),
        "accuracy": float(base),
        "mae_ps": float(np.abs(data.pred - data.delay).mean()),
        "signed_ps": float((data.pred - data.delay).mean()),
        "abs_p50_p90_p99": [float(np.percentile(np.abs(data.pred - data.delay), q)) for q in (50, 90, 99)],
        "loss_points": float(100.0 - base),
    }
    print(json.dumps(summary, indent=2))

    groups = {
        "delay": (None, [0, 300, 600, 1000, 1500, 2000, 3000, 4000, 6000, 10**9]),
        "dist": (None, [0, 12, 30, 60, 120, 300, 600, 10**9]),
        "cross_wall": (None, None),
        "corridor": (None, None),
        "near_edge": (None, None),
        "src_port": (None, None),
        "tgt_class": (None, None),
    }
    data["tgt_class"] = data["tgt_port"].str[0]
    results = {}
    for name, (_, edges) in groups.items():
        rows = group_report(data, pts, name, edges)
        results[name] = rows
        print(f"\n== by {name} ==")
        print(f"{'group':<24}{'rows':>8}{'acc':>9}{'mae':>8}{'exact':>7}{'loss':>8}{'acc_if_exact':>13}")
        for row in rows[:12]:
            print(f"{row['group']:<24}{row['rows']:>8}{row['acc']:>9.3f}{row['mae']:>8.2f}"
                  f"{row['exact_frac']:>7.3f}{row['loss_points']:>8.3f}{row['acc_if_exact']:>13.3f}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({"summary": summary, "groups": results}, indent=2) + "\n",
                                 encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
