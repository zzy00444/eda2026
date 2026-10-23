"""Export a lean model (.npz from fit_lean) into the runtime binary read by src/estimate.cpp.

Layout (little endian):

    header    : char magic[8]="LEAN0001", u32 version, u32 rank, u32 nx, ny, ns, nt,
                u32 nports, f32 su, u32 flags (bit0 U int8, bit1 C int8)      -> 44 B
    sections  : A, Bx, By (f32 ns*nt each), U (int8/int16 ns*nt*rank),
                C.T (int8/int16 nx*ny*rank, cell-major), scale (f32 nx*ny),
                vp (i32 121), hp (i32 551), vx (i8 ns), vy (i8 ns),
                ports (nports x {char name[24]; i16 src; i16 tgt})
    footer    : char magic[8]="LEANIDX1", u32 count, count x (u64 offset, u64 size)

The runtime reads the footer at (blob_end - FOOTER_SIZE) so no offset arithmetic
is duplicated between writer and reader.

Usage: python tools/export_lean.py --model build/lean_7pos.npz --out models/lean.bin
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from arch import load_architecture  # noqa: E402
from fit_lean import gap_arrays, source_nets  # noqa: E402

SECTIONS = ("A", "Bx", "By", "U", "C", "scale", "vp", "hp", "vx", "vy", "ports")
FOOTER_SIZE = 8 + 4 + 16 * len(SECTIONS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "lean.bin")
    args = parser.parse_args()

    arch = load_architecture()
    vp, hp = gap_arrays(arch)
    vxs, vys = source_nets(arch)
    d = np.load(args.model)
    U8, su, A, Bx, By = d["U8"], float(d["su"]), d["A"], d["Bx"], d["By"]
    C, scale, rank = d["C"], d["scale"], int(d["rank"])
    nx, ny, ns, nt = [int(v) for v in d["meta"]]
    nports = len(arch.ports)
    src_of = {p: i for i, p in enumerate(arch.query_sources)}
    tgt_of = {p: i for i, p in enumerate(arch.targets)}
    flags = (1 if U8.dtype == np.int8 else 0) | (2 if C.dtype == np.int8 else 0)

    chunks = {
        "A": A.astype("<f4").tobytes(),
        "Bx": Bx.astype("<f4").tobytes(),
        "By": By.astype("<f4").tobytes(),
        "U": U8.astype("<i1" if U8.dtype == np.int8 else "<i2").tobytes(),
        "C": C.T.astype("<i1" if C.dtype == np.int8 else "<i2").tobytes(),  # cell-major
        "scale": scale.astype("<f4").tobytes(),
        "vp": vp.astype("<i4").tobytes(),
        "hp": hp.astype("<i4").tobytes(),
        "vx": vxs.astype(np.int8).tobytes(),
        "vy": vys.astype(np.int8).tobytes(),
        "ports": b"".join(
            str(arch.ports[i]).encode("ascii")[:23].ljust(24, b"\x00")
            + struct.pack("<hh", src_of.get(i, -1), tgt_of.get(i, -1))
            for i in range(nports)
        ),
    }

    blob = bytearray()
    blob += b"LEAN0001"
    blob += struct.pack("<I", 1)
    blob += struct.pack("<I", rank)
    blob += struct.pack("<4I", nx, ny, ns, nt)
    blob += struct.pack("<I", nports)
    blob += struct.pack("<f", su)
    blob += struct.pack("<I", flags)
    offsets = []
    for name in SECTIONS:
        data = chunks[name]
        offsets.append((name, len(blob), len(data)))
        blob += data
    blob += b"LEANIDX1"
    blob += struct.pack("<I", len(offsets))
    for _, off, sz in offsets:
        blob += struct.pack("<QQ", off, sz)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(bytes(blob))
    print(json.dumps({
        "out": str(args.out), "bytes": len(blob), "rank": rank,
        "u_dtype": str(U8.dtype), "c_dtype": str(C.dtype), "footer": FOOTER_SIZE,
        "sections": {n: [o, s] for n, o, s in offsets},
    }, indent=2))


if __name__ == "__main__":
    main()
