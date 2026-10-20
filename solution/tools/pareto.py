"""Build a candidate Pareto table: accuracy, measured CPU, projected 1B time, total.

Every candidate written by tools/build_candidate_wsl.sh leaves a
<name>_result.csv plus a <name>_benchmark.json (1M, two runs) in
solution/build/local_runs; tools/validate_stage5_wsl.sh adds a
<name>_benchmark_10m.json (10M, five runs) for the same name.  This script
pairs them, scores the predictions against the public labels, and reports the
quantity the competition actually ranks:

    total = 0.8 * accuracy + 0.15 * max(0, 1 - cpu_seconds_1b / 1800) * 100 + 5

The final 5 assumes five identical runs, so use the 10M report when it exists.
Accuracy uses the fixed rng7 split (validation [300000:350000], test
[350000:400000]) and is a local regression metric, not an official score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import DATA

DEFAULT_RUNS = Path(__file__).resolve().parents[1] / "build" / "local_runs"


def competition_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    predicted = np.maximum(0, np.rint(predicted))
    points = np.zeros(len(actual), dtype=np.float64)
    nonzero = actual != 0
    points[nonzero] = 100 * (1 - np.tanh(4 * np.abs(predicted[nonzero] - actual[nonzero]) / actual[nonzero]))
    points[~nonzero] = 100 * (predicted[~nonzero] == 0)
    return float(points.mean())


def split_indices(rows: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.random.default_rng(7).permutation(rows)
    return order[300000:350000], order[350000:400000]


def load_answers(rows: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(DATA / "data" / "delay_estimate_ans.csv")
    if rows is not None and rows != len(frame):
        raise SystemExit(f"result has {rows} rows but the public set has {len(frame)}")
    return frame


def read_report(path: Path) -> tuple[float, float, int] | None:
    """Return (mean cpu seconds, rows, runs) from a benchmark report."""
    if not path.exists():
        return None
    runs = json.loads(path.read_text())
    if not runs:
        return None
    rows = runs[0].get("rows")
    if rows is None:  # 1M reports do not record the row count
        rows = 1_000_000
    return (float(np.mean([r["cpu_seconds"] for r in runs])), int(rows), len(runs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=str(DEFAULT_RUNS), help="directory holding *_result.csv")
    parser.add_argument("--out", default="", help="optional JSON output path")
    parser.add_argument("--markdown", default="", help="optional markdown output path")
    args = parser.parse_args()

    directory = Path(args.runs)
    results = sorted(directory.glob("*_result.csv"))
    if not results:
        raise SystemExit(f"no *_result.csv under {directory}")
    rows_out = []
    answers = None
    for result_path in results:
        name = result_path.name[: -len("_result.csv")]
        predicted = pd.read_csv(result_path)
        if answers is None:
            answers = load_answers(len(predicted))
        if len(predicted) != len(answers):
            print(f"skip {name}: {len(predicted)} rows", flush=True)
            continue
        if not (predicted.From.values == answers.From.values).all() or not (
            predicted.To.values == answers.To.values).all():
            print(f"skip {name}: endpoint order differs from the public set", flush=True)
            continue
        truth = answers.delay.to_numpy()
        guess = predicted.delay.to_numpy()
        valid, test = split_indices(len(truth))
        timing = read_report(directory / f"{name}_benchmark_10m.json")
        timing_source = "10m"
        if timing is None:
            timing = read_report(directory / f"{name}_benchmark.json")
            timing_source = "1m"
        entry = {
            "candidate": name,
            "rows": int(len(truth)),
            "full_accuracy": competition_score(truth, guess),
            "valid_accuracy": competition_score(truth[valid], guess[valid]),
            "test_accuracy": competition_score(truth[test], guess[test]),
        }
        if timing is not None:
            cpu_seconds, timed_rows, runs = timing
            cpu_1b = cpu_seconds * 1_000_000_000 / timed_rows
            time_score = max(0.0, 1.0 - cpu_1b / 1800.0) * 100
            entry.update({
                "timing_source": timing_source,
                "benchmark_runs": runs,
                "cpu_seconds_measured": cpu_seconds,
                "timed_rows": timed_rows,
                "projected_1b_cpu_minutes": cpu_1b / 60,
                "time_score": time_score,
                "projected_total": 0.8 * entry["test_accuracy"] + 0.15 * time_score + 5,
            })
        rows_out.append(entry)

    rows_out.sort(key=lambda r: r.get("projected_total", -1), reverse=True)
    header = f"{'candidate':28s} {'test_acc':>9s} {'cpu_1m_s':>9s} {'cpu_1b_min':>10s} {'time':>7s} {'total':>7s} {'src':>4s}"
    lines = [header, "-" * len(header)]
    for row in rows_out:
        if "projected_total" not in row:
            lines.append(f"{row['candidate']:28s} {row['test_accuracy']:9.4f} {'-':>9s} {'-':>10s} {'-':>7s} {'-':>7s} {'-':>4s}")
            continue
        lines.append(
            f"{row['candidate']:28s} {row['test_accuracy']:9.4f} "
            f"{row['cpu_seconds_measured']:9.4f} {row['projected_1b_cpu_minutes']:10.3f} "
            f"{row['time_score']:7.2f} {row['projected_total']:7.3f} {row['timing_source']:>4s}"
        )
    text = "\n".join(lines)
    print(text, flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(rows_out, indent=2), encoding="utf-8")
    if args.markdown:
        Path(args.markdown).write_text("```text\n" + text + "\n```\n", encoding="utf-8")


if __name__ == "__main__":
    main()
