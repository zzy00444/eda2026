"""Create a deterministic source and evidence archive for stage 5."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from project_paths import BUILD as B, REPO_ROOT

ROOT = REPO_ROOT
S = ROOT / "solution"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    names = [
        "README.md", "requirements.txt", "FPGA延时估算第五阶段结果.md", "src/main.cpp",
        "tools/architecture.py", "tools/archive_stage5.py", "tools/build_candidate_wsl.sh",
        "tools/check_stage5_packages_wsl.sh", "tools/export_cpp.py", "tools/net_correction_experiment.py",
        "tools/project_paths.py", "tools/release_stage5.py", "tools/validate_stage5_wsl.sh", "tools/verify_cpp.py",
    ]
    paths = [S / name for name in names]
    paths += [B / name for name in (
        "s3_topology_lr32_511l_25r.txt", "s3_topology_lr32_511l_25r_metrics.json",
        "stage5_manifest.json", "stage5_score_metrics.json", "stage5_accuracy_metrics.json",
        "stage5_score_standalone.json", "stage5_accuracy_standalone.json",
        "stage5_generated_s5_score_benchmark_10m.json", "stage5_generated_s5_accuracy_benchmark_10m.json",
        "stage5_generated_s5_score_integration.json", "stage5_generated_s5_accuracy_integration.json",
    )]
    for variant in ("score", "accuracy"):
        generated = B / f"generated_s5_{variant}"
        paths += [generated / name for name in ("architecture.bin", "architecture.S", "export.json", "model.hpp")]
        paths.append(B / f"net_correction_f{'1' if variant == 'score' else '18'}_c0/metrics.json")
    records = [{"path": p.relative_to(S).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)} for p in paths]
    archive = S / "stage5_source_archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as out:
        for path, record in zip(paths, records):
            info = zipfile.ZipInfo(record["path"], date_time=(2026, 10, 20, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (0o100755 if path.suffix == ".sh" else 0o100644) << 16
            out.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        out.writestr("ARCHIVE_MANIFEST.json", json.dumps({"stage": 5, "files": records}, indent=2))
    with zipfile.ZipFile(archive) as check:
        assert len(check.namelist()) == len(records) + 1
        assert all(check.read(record["path"]) == path.read_bytes() for path, record in zip(paths, records))
    manifest = {"archive": archive.name, "bytes": archive.stat().st_size, "sha256": sha(archive), "files": len(records)}
    (B / "stage5_archive_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
