"""Append the lean model blob to the compiled executable and build submission.zip.

    python tools/pack_lean.py --elf build/estimate --model models/lean.bin \
        --out submission/bin/estimate --zip submission.zip

Layout: [elf bytes][model bytes]["LEANTAIL"][u64 offset][u64 size]
The runtime reads the last 24 bytes to find the blob.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elf", type=Path, default=ROOT / "build" / "estimate")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "lean.bin")
    parser.add_argument("--out", type=Path, default=ROOT / "submission" / "bin" / "estimate")
    parser.add_argument("--zip", type=Path, default=ROOT / "submission.zip")
    args = parser.parse_args()

    elf = args.elf.read_bytes()
    model = args.model.read_bytes()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        f.write(elf)
        f.write(model)
        f.write(b"LEANTAIL")
        f.write(struct.pack("<QQ", len(elf), len(model)))
    args.out.chmod(0o755)

    args.zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.write(args.out, "bin/estimate")
    print(json.dumps({
        "elf_bytes": len(elf), "model_bytes": len(model),
        "packed_bytes": args.out.stat().st_size,
        "zip_bytes": args.zip.stat().st_size,
        "zip_ok": args.zip.stat().st_size <= 100 * 1024 * 1024,
    }, indent=2))


if __name__ == "__main__":
    main()
