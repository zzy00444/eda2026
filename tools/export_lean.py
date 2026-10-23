"""Export a lean model (.npz from fit_lean) into the runtime binary read by src/estimate.cpp.

Layout (little endian):

    char   magic[8]      = "LEAN0001"
    u32    version       = 1
    u32    rank
    u32    nx, ny        displacement grid (239 x 1099)
    u32    ns, nt        source/target port counts (208, 288)
    u32    nports        (496)
    f32    su            basis scale
    u32    flags         bit0: U int8 (else int16), bit1: C int8 (else int16)
    f32    A[ns*nt]
    f32    Bx[ns*nt]
    f32    By[ns*nt]
    <U>    ns*nt*rank    int8/int16
    <C>    nx*ny*rank    int8/int16
    f32    scale[nx*ny]
    i32    vp[120]
    i32    hp[550]
    i8     src_vx[ns], src_vy[ns]
    port table: nports x { char name[24]; i16 src; i16 tgt; }

Usage: python tools/export_lean.py --model build/lean.tel.npz --out models/lean.bin
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
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        f.write(b"LEAN0001")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<I", rank))
        f.write(struct.pack("<4I", nx, ny, ns, nt))
        f.write(struct.pack("<I", nports))
        f.write(struct.pack("<f", su))
        f.write(struct.pack("<I", flags))
        for arr in (A, Bx, By):
            f.write(np.asarray(arr, dtype=np.float32).tobytes())
        f.write(np.ascontiguousarray(U8).tobytes())
        f.write(np.ascontiguousarray(C).tobytes())
        f.write(np.asarray(scale, dtype=np.float32).tobytes())
        f.write(np.asarray(vp, dtype=np.int32).tobytes())
        f.write(np.asarray(hp, dtype=np.int32).tobytes())
        f.write(np.asarray(vxs, dtype=np.int8).tobytes())
        f.write(np.asarray(vys, dtype=np.int8).tobytes())
        table = np.zeros(nports, dtype=[("name", "S24"), ("src", "<i2"), ("tgt", "<i2")])
        for i, name in enumerate(arch.ports):
            table["name"][i] = name.encode("ascii")[:23]
            table["src"][i] = src_of.get(i, -1)
            table["tgt"][i] = tgt_of.get(i, -1)
        f.write(table.tobytes())
    size = args.out.stat().st_size
    print(json.dumps({
        "out": str(args.out), "bytes": size, "rank": rank,
        "u_dtype": str(U8.dtype), "c_dtype": str(C.dtype),
        "expected": (8 + 4 + 4 + 16 + 4 + 4 + 4) + 12 * ns * nt + U8.nbytes + C.nbytes + 4 * nx * ny + 4 * (120 + 550) + 2 * ns + nports * 28,
    }, indent=2))


if __name__ == "__main__":
    main()
