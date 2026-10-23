"""Estimate competition scores with the frozen official/local CPU ratio."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from project_paths import REPO_ROOT


def estimate(accuracy: float, local_cpu_seconds: float, ratio: float) -> dict:
    official_cpu = local_cpu_seconds * ratio
    time_score = 100 * max(0.0, 1.0 - official_cpu / 1800.0)
    return {
        "accuracy": accuracy,
        "local_projected_1b_cpu_seconds": local_cpu_seconds,
        "calibrated_1b_cpu_seconds": official_cpu,
        "calibrated_time_score": time_score,
        "estimated_total": 0.8 * accuracy + 0.15 * time_score + 5.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path, help="displacement_run.py benchmark JSON")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    alignment_path = REPO_ROOT / "docs/stage11_official_benchmark_alignment.json"
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    ratio = alignment["alignment"]["runtime_ratio"]
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    report = {
        "benchmark": str(args.benchmark),
        "alignment": "docs/stage11_official_benchmark_alignment.json",
        "runtime_ratio": ratio,
        "accuracy_source": "historical public test split",
        "official_score": False,
        "candidates": {},
    }
    for name, item in benchmark["candidates"].items():
        analysis_path = REPO_ROOT / "docs" / f"{name}_analysis.json"
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        report["candidates"][name] = estimate(
            analysis["test"]["accuracy"], item["projected_1b_cpu_seconds"], ratio
        )
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
