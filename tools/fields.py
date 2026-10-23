"""Exact field factory: dump full-chip distance fields from a set of source pins.

Two kinds of dump:

* `field`    -> distances to the 288 architecturally valid target ports,
                `uint16[60200][288]` (34.7 MB), for query-level labels;
* `fieldall` -> distances to all 496 ports, `uint16[60200][496]` (59.8 MB), which
                exposes the lane-level object `G(s, lane, cell)` used by the
                lane-decomposition model.

Each dump costs ~3 s of CPU, so the whole preset matrix is cheap.  Because the
interior metric is translation invariant in the *displacement* (the per-column
gap-line sums are separable and can be removed analytically), one dump per source
port at a single interior position already gives the complete defect-free metric
for every displacement.  Additional source positions cover the defect classes:
vertical-line columns, the wall corridor and its doors, block0, and the edges.

Usage
    python tools/fields.py dump --preset core                 # 208 sources at (40, 275)
    python tools/fields.py dump --preset core --allports      # lane-level fields
    python tools/fields.py dump --preset defect
    python tools/fields.py index                              # (re)build index.json

Outputs live in build/fields/ and build/fields_all/ (gitignored).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from arch import load_architecture  # noqa: E402

WSL_REPO = "/mnt/f/EDA/solution/eda2026"
FIELDS = ROOT / "build" / "fields"
FIELDS_ALL = ROOT / "build" / "fields_all"

INTERIOR = (40, 275)
PHASE_POSITIONS = [(41, 277), (43, 269), (44, 283), (46, 291)]
DEFECT_PORTS = ["ZSEA[0]", "ZLN[0]", "ZSSB[3]", "S_HY", "A_AQ0"]
VLINE_X = [9, 19, 33, 39, 45, 51, 57, 63, 69, 95, 99, 105, 111, 117]
WALL_X = [75, 76, 79, 83, 86, 89, 90, 91]
DOOR_Y = [49, 50, 51, 99, 100, 101, 149, 150, 199, 200, 449, 450, 499, 500, 549]
EDGE_X = [0, 1, 12, 118, 119]
BLOCK0_EDGE = [(16, 0), (16, 50), (17, 99), (20, 20), (24, 30), (30, 60)]


def field_name(x: int, y: int, port: str) -> str:
    return f"{x}_{y}__{port}.bin"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def preset_pins(arch, preset: str) -> list[tuple[int, int, str]]:
    sources = [arch.ports[i] for i in arch.query_sources]
    if preset == "core":
        return [(INTERIOR[0], INTERIOR[1], p) for p in sources]
    if preset == "phase":
        return [(x, y, p) for (x, y) in PHASE_POSITIONS for p in sources[:12]]
    if preset == "defect":
        out = []
        for x in VLINE_X:
            out.append((x, 275, "ZSEA[0]"))
        for x in WALL_X:
            for y in (25, 75, 125, 175):
                out.append((x, y, "ZLN[0]"))
        for y in DOOR_Y:
            out.append((80, y, "ZSSB[3]"))
        for x in EDGE_X:
            for y in (0, 12, 275, 537, 549):
                out.append((x, y, "S_HY"))
        for (x, y) in BLOCK0_EDGE:
            out.append((x, y, "A_AQ0"))
        return out
    if preset == "all":
        return preset_pins(arch, "core") + preset_pins(arch, "phase") + preset_pins(arch, "defect")
    raise SystemExit(f"unknown preset {preset}")


def dump(arch, preset: str, limit: int = 0, image: str = "fpga-ziguang:latest",
         allports: bool = False) -> None:
    outdir = FIELDS_ALL if allports else FIELDS
    subdir = "fields_all" if allports else "fields"
    mode = "fieldall" if allports else "field"
    columns = len(arch.ports) if allports else len(arch.targets)
    pins = preset_pins(arch, preset)
    if limit:
        pins = pins[:limit]
    outdir.mkdir(parents=True, exist_ok=True)
    todo = [(x, y, p) for (x, y, p) in pins if not (outdir / field_name(x, y, p)).exists()]
    print(f"preset {preset}: {len(pins)} pins, {len(pins) - len(todo)} present, {len(todo)} to dump")
    if not todo:
        return
    script = outdir / "_dump.sh"
    lines = ["#!/bin/bash", "set -e", "cd /ws", "export TIMEFORMAT='%R'"]
    for x, y, p in todo:
        out = f"build/{subdir}/{field_name(x, y, p)}"
        lines.append(f'./build/oracle {mode} build/arch.txt "SRB_{x}_{y}/{p}" {out}')
    script.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"running {len(todo)} dumps in one container ...")
    proc = subprocess.run(
        ["wsl", "-d", "Ubuntu-20.04", "--", "bash", "-lc",
         f"docker run --rm --user $(id -u):$(id -g) -v {WSL_REPO}:/ws -w /ws {image} "
         f"bash /ws/build/{subdir}/_dump.sh"],
        capture_output=True, text=True)
    tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
    if proc.returncode != 0:
        raise SystemExit(f"dump failed ({proc.returncode})\n{tail}")
    print(tail)
    index(outdir, columns)
    for x, y, p in todo:
        path = outdir / field_name(x, y, p)
        if path.stat().st_size != 60200 * columns * 2:
            raise SystemExit(f"unexpected field size for {path.name}: {path.stat().st_size}")


def index(outdir: Path = FIELDS, columns: int = 288) -> None:
    arch = load_architecture()
    entries = []
    for path in sorted(outdir.glob("*.bin")):
        stem = path.name[:-4]
        coord, port = stem.split("__", 1)
        x, y = (int(v) for v in coord.split("_"))
        entries.append({
            "file": path.name,
            "x": x,
            "y": y,
            "port": port,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "shape": [len(arch.insts), columns],
        })
    out = outdir / "index.json"
    out.write_text(json.dumps({"fields": entries, "count": len(entries)}, indent=2) + "\n", encoding="utf-8")
    print(f"index: {len(entries)} fields, {sum(e['bytes'] for e in entries) / (1 << 30):.2f} GiB -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["dump", "index"])
    parser.add_argument("--preset", default="core", choices=["core", "phase", "defect", "all"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--image", default="fpga-ziguang:latest")
    parser.add_argument("--allports", action="store_true", help="dump distances to all 496 ports")
    args = parser.parse_args()
    if args.action == "index":
        index(FIELDS_ALL if args.allports else FIELDS, 496 if args.allports else 288)
        return
    arch = load_architecture()
    dump(arch, args.preset, args.limit, args.image, args.allports)


if __name__ == "__main__":
    main()
