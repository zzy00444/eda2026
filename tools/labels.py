"""Build labelled query sets from exact field dumps.

A field dumped from pin (x0, y0, port) gives the exact delay for every query
`SRB_x0_y0/port -> SRB_x_y/tport` for all 60,200 SRB positions and 288 target
ports.  This tool samples such pairs (uniform over target position and target
port), writes a request CSV / golden CSV for downstream runs, and records the
features needed for the S2 error decomposition.

Usage
    python tools/labels.py --pins 8 --rows-per-pin 25000 --tag s2_interior
    python tools/labels.py --pin-filter core --rows-per-pin 50000 --tag s2_core
    python tools/labels.py --list
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from arch import load_architecture  # noqa: E402

FIELDS = ROOT / "build" / "fields"


def load_index() -> list[dict]:
    index = json.loads((FIELDS / "index.json").read_text(encoding="utf-8"))
    return index["fields"]


def feature_frame(arch, source: dict, srb: np.ndarray, tgt: np.ndarray) -> pd.DataFrame:
    sx, sy = source["x"], source["y"]
    insts = np.array(arch.insts, dtype=np.int32)
    tx = insts[srb, 0]
    ty = insts[srb, 1]
    dx = tx - sx
    dy = ty - sy
    gap = np.zeros(len(srb), dtype=np.int32)
    for line in arch.vertical_lines:
        lo, hi = np.minimum(sx, tx), np.maximum(sx, tx)
        gap += line.delay * ((lo <= line.site) & (line.site < hi))
    for line in arch.horizontal_lines:
        lo, hi = np.minimum(sy, ty), np.maximum(sy, ty)
        gap += line.delay * ((lo <= line.site) & (line.site < hi))
    frame = pd.DataFrame({
        "sx": sx, "sy": sy, "tx": tx, "ty": ty, "dx": dx, "dy": dy,
        "gap": gap,
        "src_port": source["port"],
        "tgt_port": [arch.ports[arch.targets[k]] for k in tgt],
    })
    frame["cross_wall"] = ((frame.sx < 76) & (frame.tx > 89)) | ((frame.tx < 76) & (frame.sx > 89))
    frame["corridor"] = ((frame.sx >= 76) & (frame.sx <= 89)) | ((frame.tx >= 76) & (frame.tx <= 89))
    frame["near_edge"] = (frame.tx <= 12) | (frame.tx >= 107) | (frame.ty <= 12) | (frame.ty >= 537)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pins", type=int, default=8, help="number of source fields to use")
    parser.add_argument("--stride", type=int, default=1, help="take every k-th field instead of the first block")
    parser.add_argument("--pin-filter", default="", help="substring filter on port name")
    parser.add_argument("--rows-per-pin", type=int, default=25000)
    parser.add_argument("--tag", default="labels")
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    fields = load_index()
    if args.list:
        for entry in fields:
            print(f"{entry['x']:>3} {entry['y']:>3} {entry['port']:<12} {entry['bytes']/1e6:8.1f} MB")
        print(f"{len(fields)} fields")
        return
    if args.pin_filter:
        fields = [f for f in fields if args.pin_filter in f["port"]]
    if args.stride > 1:
        fields = fields[:: args.stride]
    fields = fields[: args.pins]
    if not fields:
        raise SystemExit("no fields selected")

    arch = load_architecture()
    insts = np.array(arch.insts, dtype=np.int32)
    rng = np.random.default_rng(args.seed)
    frames = []
    for entry in fields:
        field = np.fromfile(FIELDS / entry["file"], dtype=np.uint16).reshape(len(arch.insts), len(arch.targets))
        srb = rng.integers(0, len(arch.insts), size=args.rows_per_pin)
        tgt = rng.integers(0, len(arch.targets), size=args.rows_per_pin)
        frame = feature_frame(arch, entry, srb, tgt)
        frame["delay"] = field[srb, tgt].astype(np.int64)
        frame["From"] = [f"SRB_{entry['x']}_{entry['y']}/{entry['port']}"] * len(srb)
        frame["To"] = [f"SRB_{x}_{y}/{p}" for (x, y), p in zip(insts[srb], frame["tgt_port"])]
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["dist"] = data.dx.abs() + data.dy.abs()

    (ROOT / "build").mkdir(exist_ok=True)
    data[["From", "To"]].to_csv(ROOT / "build" / f"{args.tag}_request.csv", index=False, lineterminator="\n")
    data[["From", "To", "delay"]].to_csv(ROOT / "build" / f"{args.tag}_golden.csv", index=False, lineterminator="\n")
    data.to_parquet(ROOT / "build" / f"{args.tag}_features.parquet", index=False)

    print(f"{len(data)} rows from {len(fields)} fields -> build/{args.tag}_request.csv / _golden.csv / _features.parquet")
    print(f"delay: mean={data.delay.mean():.0f} p50={data.delay.median():.0f} max={data.delay.max()}")
    print(f"cross_wall={data.cross_wall.mean():.3f} corridor={data.corridor.mean():.3f} "
          f"near_edge={data.near_edge.mean():.3f}")
    print(f"src ports: {sorted(data.src_port.unique())[:6]} ...")


if __name__ == "__main__":
    main()
