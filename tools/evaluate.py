"""Evaluate a result CSV against golden labels.

Usage
    python tools/evaluate.py --result build/s0_result.csv --golden build/s0_golden.csv
    python tools/evaluate.py --result ... --golden ... --require-exact   # S0 gate

Reports accuracy (task formula), mae/rmse, exact-match count, error percentiles,
and a delay-bucket breakdown.  Exit code 1 when an assertion fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    lower = {c.lower(): c for c in frame.columns}
    if "from" not in lower or "to" not in lower or "delay" not in lower:
        raise SystemExit(f"{path} must have From,To,delay columns, got {list(frame.columns)}")
    frame = frame.rename(columns={lower["from"]: "From", lower["to"]: "To", lower["delay"]: "delay"})
    return frame[["From", "To", "delay"]]


def score(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, np.ndarray]:
    predicted = np.maximum(0, np.rint(predicted))
    points = np.zeros(len(actual), dtype=np.float64)
    nonzero = actual != 0
    points[nonzero] = 100.0 * (1.0 - np.tanh(4.0 * np.abs(predicted[nonzero] - actual[nonzero]) / actual[nonzero]))
    points[~nonzero] = 100.0 * (predicted[~nonzero] == 0)
    return float(points.mean()), points


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--require-exact", action="store_true", help="fail unless every row matches exactly")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    result = load(args.result)
    golden = load(args.golden)
    if len(result) != len(golden):
        raise SystemExit(f"row count mismatch: result {len(result)} vs golden {len(golden)}")
    if not (result["From"].to_numpy() == golden["From"].to_numpy()).all() or \
       not (result["To"].to_numpy() == golden["To"].to_numpy()).all():
        raise SystemExit("result rows are not in the same order/keys as golden")

    truth = golden["delay"].to_numpy(dtype=np.int64)
    predicted = result["delay"].to_numpy(dtype=np.int64)
    error = predicted - truth
    accuracy, points = score(truth, predicted)
    exact = int((error == 0).sum())

    report = {
        "label": args.label,
        "rows": len(truth),
        "accuracy": accuracy,
        "exact_rows": exact,
        "exact_fraction": exact / len(truth),
        "mae_ps": float(np.abs(error).mean()),
        "rmse_ps": float(np.sqrt((error.astype(np.float64) ** 2).mean())),
        "signed_error_ps": float(error.mean()),
        "abs_error_p50_p90_p99": [float(np.percentile(np.abs(error), q)) for q in (50, 90, 99)],
        "max_abs_error_ps": int(np.abs(error).max()),
    }
    print(json.dumps(report, indent=2))

    print("\nby truth delay bucket:")
    edges = [0, 300, 600, 1000, 1500, 2000, 3000, 4000, 6000, 10**9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (truth >= lo) & (truth < hi)
        if not mask.any():
            continue
        print(f"  [{lo:>5},{hi if hi < 10**9 else 'inf':>5}) n={int(mask.sum()):>6} "
              f"acc={points[mask].mean():8.4f} mae={np.abs(error[mask]).mean():7.2f} "
              f"exact={int((error[mask] == 0).sum()):>6}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.require_exact and exact != len(truth):
        print(f"\nFAIL: {len(truth) - exact} rows differ from golden", file=sys.stderr)
        worst = np.argsort(-np.abs(error))[:10]
        for i in worst:
            if error[i] != 0:
                print(f"  {golden['From'].iat[i]} -> {golden['To'].iat[i]} golden={truth[i]} pred={predicted[i]}",
                      file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
