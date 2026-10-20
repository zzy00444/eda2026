"""Archive the accepted stage-3 source, generated parameters and evidence."""
from __future__ import annotations

import hashlib
import json
import platform
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy

from project_paths import BUILD, DATA, REPO_ROOT

ROOT = REPO_ROOT
SOLUTION = ROOT / "solution"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(name):
    return json.loads((BUILD / name).read_text(encoding="utf-8"))


def main():
    metrics = read("stage3_final_metrics.json")
    benchmark = read("stage3_benchmark_10m.json")
    package = read("stage3_package.json")
    standalone = read("stage3_standalone.json")
    export = read("generated/export.json")
    assert metrics["valid"]["score"] >= 96 and metrics["test"]["score"] >= 96
    assert metrics["full_parity"]["python_cpp_mismatch_rows"] == 0
    assert len(benchmark) == 3
    assert all(b["projected_1b_cpu_minutes"] < 30 and b["max_observed_threads"] == 1
               and b["container_peak_bytes"] < 2 * 1024 ** 3
               and b["repeated_output_verified"] for b in benchmark)
    assert len({b["output_sha256"] for b in benchmark}) == 1
    assert standalone["standalone_zip_test"] == "passed"
    assert sha(SOLUTION / "bin/estimate") == package["binary_sha256"] == standalone["binary_sha256"]
    assert sha(SOLUTION / "submission.zip") == package["zip_sha256"]
    assert sha(SOLUTION / "stage2_submission.zip") == "aca93d91b5a739378d37a983ec0b13d82503b4ac89374055c2160cbd139b0541"

    order = np.random.default_rng(7).permutation(1_000_000)
    valid, test, train = order[300000:350000], order[350000:400000], np.r_[order[:300000], order[400000:]]
    assert not np.intersect1d(train, valid).size and not np.intersect1d(train, test).size
    np.savez_compressed(BUILD / "stage3_fixed_split.npz", train=train, valid=valid, test=test)
    manifest = {
        "stage": 3, "date": "2026-10-19", "model": export["model"],
        "train_size": len(train), "valid_size": len(valid), "test_size": len(test), "seed": 7,
        "valid_score": metrics["valid"]["score"], "test_score": metrics["test"]["score"],
        "trees": export["trees"], "fast_trees": export["fast_trees"],
        "refine_distance": export["refine_distance"], "float32": export["float"],
        "fast_categories": export["fast_categories"], "pgo": True,
        "measured_rows_per_run": 10_000_000, "full_1b_run_completed": False,
        "worst_projected_1b_cpu_minutes": max(b["projected_1b_cpu_minutes"] for b in benchmark),
        "docker_image": "fpga-ziguang:latest",
        "docker_image_id": "sha256:075ea21ff06bffd3ac1e938e01b7bbd1bce4fcbee7fe7c3fa274851f02c6b8ec",
        "compiler": "g++ 11.4.0", "cpu": "Intel Core i9-14900HX",
        "compiler_flags": "-O3 -march=native -ffp-contract=off -fprofile-use=profile -fprofile-correction -DNDEBUG -std=c++17",
        "python": platform.python_version(),
        "packages": {"numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "lightgbm": lgb.__version__},
        "data_sha256": sha(DATA / "data/delay_estimate_ans.csv"),
        "architecture_sha256": {p.name: sha(p) for p in sorted((DATA / "arch").glob("*.json"))},
        "training_command": "python tools/topology_experiment.py --leaves 511 --rounds 25 --learning-rate 0.32 --prefix s3_topology_lr32",
        "export_command": "python tools/export_cpp.py --model s3_topology_lr32_511l_25r --trees 25 --fast-trees 18 --refine-distance 100 --fast-categories --output-dir generated_s3_final_pgo",
        "build_command": "wsl -d Ubuntu-20.04 -- bash /mnt/f/EDA/solution/tools/build_candidate_wsl.sh generated_s3_final_pgo pgo",
        "package": package,
    }
    (BUILD / "stage3_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paths = [SOLUTION / "README.md", SOLUTION / "requirements.txt",
             SOLUTION / "FPGA延时估算第三阶段结果.md", SOLUTION / "src/main.cpp"]
    paths += sorted(p for p in (SOLUTION / "tools").iterdir() if p.suffix in (".py", ".sh", ".cpp"))
    paths += [BUILD / "s3_topology_lr32_511l_25r.txt", BUILD / "s3_topology_lr32_511l_25r_metrics.json",
              BUILD / "architecture_meta.json", BUILD / "stage3_fixed_split.npz", BUILD / "stage3_manifest.json"]
    paths += [BUILD / n for n in ("stage3_final_metrics.json", "stage3_full_parity.json",
              "stage3_benchmark_10m.json", "generated_s3_final_pgo_benchmark_1m.json",
              "stage3_integration.json", "stage3_standalone.json", "stage3_package.json")]
    paths += sorted((BUILD / "generated").iterdir())
    records = [{"path": p.relative_to(SOLUTION).as_posix(), "bytes": p.stat().st_size,
                "sha256": sha(p)} for p in paths]
    archive = SOLUTION / "stage3_source_archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as out:
        for p, record in zip(paths, records):
            entry = zipfile.ZipInfo(record["path"], date_time=(2026, 10, 19, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = (0o100755 if p.suffix == ".sh" else 0o100644) << 16
            out.writestr(entry, p.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        out.writestr("ARCHIVE_MANIFEST.json", json.dumps({**manifest, "files": records}, indent=2))
    report = {"archive": archive.name, "bytes": archive.stat().st_size, "sha256": sha(archive),
              "files": len(records), "contents": records}
    (BUILD / "stage3_archive_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "contents"}, indent=2))


if __name__ == "__main__":
    main()
