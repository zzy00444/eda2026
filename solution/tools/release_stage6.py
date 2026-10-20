"""Verify stage-6 evidence, package the accepted candidate, and write its manifest.

Stage 6 keeps the stage-5 model, net-correction coefficients and architecture
table; only the runtime and the port perfect hash changed, so the predictions
are required to be byte-identical to the stage-5 artefact.  The script asserts
that instead of re-deriving accuracy, and it refuses to publish if any of the
five 10M runs disagree, use more than one thread, or break the memory cap.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
import zipfile

from project_paths import BUILD as B, REPO_ROOT

S = REPO_ROOT / "solution"
GENERATED = "generated_s6_score"


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
    candidate = "stage6_v7"
    benchmark = read(f"{candidate}_benchmark_10m.json")
    integration = read(f"{candidate}_integration.json")
    assert len(benchmark) == 5, "five 10M runs required"
    digests = {run["output_sha256"] for run in benchmark}
    assert len(digests) == 1, "10M runs disagree"
    for run in benchmark:
        assert run["rows"] == 10_000_000 and run["repeated_output_verified"]
        assert run["max_observed_threads"] == 1
        assert run["container_peak_bytes"] < 2_000_000_000
    assert integration["interface_checks"]
    stage5 = read("stage5_score_metrics.json")
    export = read(f"{GENERATED}/export.json")
    assert export["model"] == stage5["model"]
    assert export["trees"] == stage5["trees"]
    assert export["net_coefficients"] == stage5["net_coefficients"]
    assert export["packed_ports"] and export["fast_categories"]
    # Byte-identical predictions: the recorded stage-5 output digest must match.
    stage5_runs = read("stage5_generated_s5_score_benchmark_10m.json")
    assert {r["output_sha256"] for r in stage5_runs} == digests, "predictions changed"

    binary = S / "bin" / "estimate_stage6_score"
    assert binary.exists(), "build stage6_score first"
    cpu_seconds = sum(run["cpu_seconds"] for run in benchmark) / len(benchmark)
    cpu_1b = cpu_seconds * 100
    time_score = 15 * max(0.0, 1 - cpu_1b / 1800)
    valid, test = stage5["valid"]["score"], stage5["test"]["score"]
    previous = read("stage5_manifest.json")
    previous_cpu = previous["variants"]["score"]["average_projected_1b_cpu_seconds"]

    target = S / "stage6_score_submission.zip"
    package_info = package(binary, target)
    shutil.copy2(binary, S / "bin/estimate")
    shutil.copy2(target, S / "submission.zip")
    (B / "generated").mkdir(exist_ok=True)
    for item in (B / GENERATED).iterdir():
        if item.is_file():
            shutil.copy2(item, B / "generated" / item.name)

    manifest = {
        "stage": 6,
        "date": "2026-10-20",
        "default": "score",
        "candidate": "stage6_score",
        "generated_dir": GENERATED,
        "binary": "bin/estimate_stage6_score",
        "model": export["model"],
        "trees": export["trees"],
        "net_correction": export["net_correction"],
        "net_coefficients": export["net_coefficients"],
        "port_hash": "open-addressed 2048-slot table with exact 64-bit key check",
        "validation_accuracy": valid,
        "test_accuracy": test,
        "accuracy_note": "identical to stage 5: the 1M and 10M outputs are byte-identical",
        "average_10m_cpu_seconds": cpu_seconds,
        "average_projected_1b_cpu_seconds": cpu_1b,
        "average_projected_1b_cpu_minutes": cpu_1b / 60,
        "cpu_10m_range": [min(r["cpu_seconds"] for r in benchmark),
                          max(r["cpu_seconds"] for r in benchmark)],
        "time_score": time_score,
        "projected_total_from_validation": 0.8 * valid + time_score + 5,
        "projected_total_from_test": 0.8 * test + time_score + 5,
        "stage5_projected_1b_cpu_seconds": previous_cpu,
        "cpu_reduction_fraction": 1 - cpu_1b / previous_cpu,
        "max_process_rss_kib": max(r["process_peak_rss_kib"] for r in benchmark),
        "max_container_bytes": max(r["container_peak_bytes"] for r in benchmark),
        "benchmark_runs": len(benchmark),
        "max_threads": 1,
        "output_10m_sha256": digests.pop(),
        "primary_model_sha256": sha(B / (export["model"] + ".txt")),
        "generated_header_sha256": sha(B / GENERATED / "model.hpp"),
        "package": package_info,
        "submission": target.name,
        "compiler": "GCC 11.4.0",
        "docker_image": "fpga-ziguang:latest",
        "compiler_flags": "-O3 -march=x86-64 -ffp-contract=off -fprofile-use=profile -fprofile-correction -DNDEBUG -std=c++17",
        "full_1b_run_completed": False,
        "score_caveat": "Local held-out accuracy plus 10M timing extrapolation, not an official hidden-set score.",
        "source_hashes": {str(p.relative_to(S)): sha(p) for p in (
            S / "src/main.cpp", S / "tools/export_cpp.py", S / "tools/pareto.py",
            S / "tools/build_candidate_wsl.sh", S / "tools/validate_stage5_wsl.sh",
            S / "tools/ensure_request_header.sh")},
    }
    assert manifest["projected_total_from_test"] > previous["variants"]["score"]["projected_total_from_test"]
    ab_path = B / "stage6_ab10m.json"
    if ab_path.exists():
        ab = json.loads(ab_path.read_text(encoding="utf-8"))
        assert ab["output_sha256"] == manifest["output_10m_sha256"]
        manifest["same_session_interleaved"] = ab
    (B / "stage6_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
