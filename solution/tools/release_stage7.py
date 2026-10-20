"""Verify stage-7 evidence, package the folded candidate, and write its manifest.

Stage 7 folds the residual model into the architecture table, so the ELF no
longer contains trees.  Accuracy therefore differs from stage 6 and is taken
from stage7_fold_metrics.json (recomputed from the public 1M output), while the
runtime evidence is the five-run 10M benchmark.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
import zipfile

from project_paths import BUILD as B, REPO_ROOT

S = REPO_ROOT / "solution"
GENERATED = "generated_s7_fold"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name: str):
    return json.loads((B / name).read_text(encoding="utf-8"))


def package(binary: Path, target: Path) -> dict:
    data = binary.read_bytes()
    assert data[:4] == b"\x7fELF", "expected a Linux ELF"
    info = zipfile.ZipInfo("bin/estimate", date_time=(2026, 10, 20, 0, 0, 0))
    info.create_system = 3
    info.external_attr = 0o100755 << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    assert target.stat().st_size < 100_000_000
    with zipfile.ZipFile(target) as archive:
        assert archive.namelist() == ["bin/estimate"]
        assert archive.read("bin/estimate") == data
        assert archive.getinfo("bin/estimate").external_attr >> 16 == 0o100755
    return {"binary_bytes": len(data), "zip_bytes": target.stat().st_size,
            "binary_sha256": sha(binary), "zip_sha256": sha(target),
            "contents": ["bin/estimate"], "unix_mode": "100755"}


def main() -> None:
    candidate = "stage7_fold"
    benchmark = read(f"{candidate}_benchmark_10m.json")
    integration = read(f"{candidate}_integration.json")
    metrics = read("stage7_fold_metrics.json")
    assert len(benchmark) == 5, "five 10M runs required"
    digests = {run["output_sha256"] for run in benchmark}
    assert len(digests) == 1, "10M runs disagree"
    for run in benchmark:
        assert run["rows"] == 10_000_000 and run["repeated_output_verified"]
        assert run["max_observed_threads"] == 1
        assert run["container_peak_bytes"] < 2_000_000_000
    assert integration["interface_checks"]
    assert metrics["folded"] is True
    export = read(f"{GENERATED}/export.json")
    assert export["folded"] is True
    assert export["net_coefficients"] == metrics["net_coefficients"]

    binary = S / "bin" / "estimate_stage7_fold"
    assert binary.exists(), "build stage7 first"
    cpu_seconds = sum(run["cpu_seconds"] for run in benchmark) / len(benchmark)
    cpu_1b = cpu_seconds * 100
    time_score = 15 * max(0.0, 1 - cpu_1b / 1800)
    valid, test = metrics["valid"]["score"], metrics["test"]["score"]
    previous = read("stage6_manifest.json")

    target = S / "stage7_fold_submission.zip"
    package_info = package(binary, target)
    shutil.copy2(binary, S / "bin/estimate")
    shutil.copy2(target, S / "submission.zip")
    (B / "generated").mkdir(exist_ok=True)
    for item in (B / GENERATED).iterdir():
        if item.is_file():
            shutil.copy2(item, B / "generated" / item.name)

    manifest = {
        "stage": 7,
        "date": "2026-10-20",
        "default": "fold",
        "candidate": "stage7_fold",
        "generated_dir": GENERATED,
        "binary": "bin/estimate_stage7_fold",
        "design": "residual model folded into the architecture table; runtime is one lookup",
        "model": export["model"],
        "model_trees": export["trees"],
        "predictor": "table[key] where key = (direction, source, dx%10, dy%12, target)",
        "net_correction": "five fitted structural coefficients, unchanged in count",
        "net_coefficients": export["net_coefficients"],
        "validation_accuracy": valid,
        "test_accuracy": test,
        "full_public_accuracy": metrics["full_score"],
        "mae": metrics["mae"],
        "distance_groups": metrics["groups"],
        "average_10m_cpu_seconds": cpu_seconds,
        "average_projected_1b_cpu_seconds": cpu_1b,
        "average_projected_1b_cpu_minutes": cpu_1b / 60,
        "cpu_10m_range": [min(r["cpu_seconds"] for r in benchmark),
                          max(r["cpu_seconds"] for r in benchmark)],
        "time_score": time_score,
        "projected_total_from_validation": 0.8 * valid + time_score + 5,
        "projected_total_from_test": 0.8 * test + time_score + 5,
        "stage6_projected_total_from_test": previous["projected_total_from_test"],
        "stage6_average_projected_1b_cpu_seconds": previous["average_projected_1b_cpu_seconds"],
        "cpu_reduction_vs_stage6": 1 - cpu_1b / previous["average_projected_1b_cpu_seconds"],
        "max_process_rss_kib": max(r["process_peak_rss_kib"] for r in benchmark),
        "max_container_bytes": max(r["container_peak_bytes"] for r in benchmark),
        "benchmark_runs": len(benchmark),
        "max_threads": 1,
        "output_10m_sha256": digests.pop(),
        "generated_header_sha256": sha(B / GENERATED / "model.hpp"),
        "folded_table_bytes": (B / GENERATED / "architecture.bin").stat().st_size,
        "package": package_info,
        "submission": target.name,
        "compiler": "GCC 11.4.0",
        "docker_image": "fpga-ziguang:latest",
        "compiler_flags": "-O3 -march=x86-64 -ffp-contract=off -fprofile-use=profile -fprofile-correction -DNDEBUG -std=c++17",
        "full_1b_run_completed": False,
        "score_caveat": "Local held-out accuracy plus 10M timing extrapolation, not an official hidden-set score.",
        "source_hashes": {str(p.relative_to(S)): sha(p) for p in (
            S / "src/main.cpp", S / "tools/train_foldable.py", S / "tools/fold_features.py",
            S / "tools/fold_model.py", S / "tools/fit_netcorr_folded.py",
            S / "tools/export_cpp.py", S / "tools/pareto.py")},
    }
    assert manifest["projected_total_from_test"] > previous["projected_total_from_test"], \
        "stage 7 must beat stage 6 on the projected total"
    (B / "stage7_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in (
        "validation_accuracy", "test_accuracy", "average_10m_cpu_seconds",
        "average_projected_1b_cpu_minutes", "time_score", "projected_total_from_test",
        "cpu_reduction_vs_stage6", "package")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
