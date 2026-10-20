"""Create a deterministic source/reproduction archive for stage 2."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[2]
SOLUTION = ROOT / "solution"
ARCHIVE = SOLUTION / "stage2_source_archive.zip"
STAMP = (2026, 10, 17, 0, 0, 0)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def archive_name(path: Path) -> str:
    try:
        return path.relative_to(SOLUTION).as_posix()
    except ValueError:
        return f"docs/{path.name}"


paths = [
    SOLUTION / "requirements.txt",
    SOLUTION / "src/main.cpp",
    *sorted((SOLUTION / "tools").glob("*.py")),
    *sorted((SOLUTION / "tools").glob("*.cpp")),
    *sorted((SOLUTION / "tools").glob("*.sh")),
    SOLUTION / "build/arch_fast25.txt",
    SOLUTION / "build/arch_fast25_features.json",
    SOLUTION / "build/arch_fast25_metrics.json",
    SOLUTION / "build/architecture_meta.json",
    SOLUTION / "build/fixed_split.npz",
    SOLUTION / "build/generated/model.hpp",
    SOLUTION / "build/generated/architecture.S",
    SOLUTION / "build/generated/architecture.bin",
    SOLUTION / "build/generated/export.json",
    SOLUTION / "build/final_metrics.json",
    SOLUTION / "build/benchmark_final.json",
    SOLUTION / "build/benchmark_10m.json",
    SOLUTION / "build/benchmark_fast25_10m.json",
    SOLUTION / "build/integration.json",
    SOLUTION / "build/manifest.json",
    SOLUTION / "build/package.json",
    ROOT / "FPGA延时估算开发计划.md",
    ROOT / "FPGA延时估算第一阶段结果.md",
]
paths = list(dict.fromkeys(paths))
missing = [str(path) for path in paths if not path.is_file()]
if missing:
    raise FileNotFoundError("Missing archive inputs:\n" + "\n".join(missing))

records = [
    {"path": archive_name(path), "bytes": path.stat().st_size, "sha256": digest(path)}
    for path in paths
]
manifest = {
    "stage": 2,
    "model": "arch_fast25",
    "trees": 25,
    "float32": True,
    "archive_contents": records,
}
manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"

with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as output:
    for path, record in zip(paths, records):
        entry = zipfile.ZipInfo(record["path"], date_time=STAMP)
        entry.create_system = 3
        entry.external_attr = (0o100755 if path.suffix == ".sh" else 0o100644) << 16
        entry.compress_type = zipfile.ZIP_DEFLATED
        output.writestr(entry, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    entry = zipfile.ZipInfo("ARCHIVE_MANIFEST.json", date_time=STAMP)
    entry.create_system = 3
    entry.external_attr = 0o100644 << 16
    entry.compress_type = zipfile.ZIP_DEFLATED
    output.writestr(entry, manifest_text, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)

archive_report = {
    "archive": str(ARCHIVE.relative_to(ROOT)),
    "bytes": ARCHIVE.stat().st_size,
    "sha256": digest(ARCHIVE),
    "files": len(records),
    "manifest": "ARCHIVE_MANIFEST.json",
}
(SOLUTION / "build/stage2_archive_manifest.json").write_text(
    json.dumps({**manifest, **archive_report}, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(archive_report, ensure_ascii=False, indent=2), flush=True)
