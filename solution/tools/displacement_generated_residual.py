"""Fit an additive low-rank residual to the read-only generated distance tables.

The model uses architecture-generated values and caller training geometry only;
public delay labels are not read.  It writes separate candidate assets and never
changes the imported baseline or the external source tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

for _key in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_key] = "6"

import numpy as np


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_table(path: Path, *, dtype: str, offset: int, shape: tuple[int, ...]) -> np.ndarray:
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape)


def round_away(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, np.floor(x + .5), np.ceil(x - .5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="stage13_global64")
    ap.add_argument("--raw", type=Path, default=Path("F:/EDA/pango_delay_handover_v3/build"))
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--sample-columns", type=int, default=8192)
    ap.add_argument("--lx", type=int, default=32)
    ap.add_argument("--ly", type=int, default=160)
    ap.add_argument("--seed", type=int, default=20261023)
    args = ap.parse_args()
    if not 1 <= args.rank <= 128 or args.sample_columns < args.rank + 32:
        raise ValueError("rank must be 1..128 and sample columns must exceed rank+32")

    started = time.perf_counter()
    root = Path(__file__).resolve().parents[2]
    out = root / "solution/build/displacement"
    oracle = out / "oracle"

    with (oracle / "oracle_vt.bin").open("rb") as f:
        vh = np.fromfile(f, "<i4", 6)
        vscale = np.fromfile(f, "<f4", int(vh[0]))
    rank0, nc, ncx, ncy, cx0, cy0 = map(int, vh)
    v0 = read_table(oracle / "oracle_vt.bin", dtype="<i2", offset=24 + 4 * rank0,
                    shape=(nc, rank0))

    with (out / "U_refit_q16.bin").open("rb") as f:
        uh = np.fromfile(f, "<i4", 6)
        uscale = np.fromfile(f, "<f4", int(uh[2]))
        uq = np.fromfile(f, "<i2").reshape(-1, int(uh[2]))
    assert tuple(map(int, uh[:3])) == (208, 288, rank0)
    udecoded = uq.astype(np.float32) * uscale
    uruntime = udecoded * vscale
    m = np.fromfile(out / "M_refit.bin", "<i2", offset=12).astype(np.float32)
    assert len(m) == len(udecoded) == 208 * 288
    ids = np.loadtxt(oracle / "oracle_ids.txt", dtype=np.int32)[:208]
    assert sha(args.raw / "arch.txt") == sha(oracle / "oracle_arch.txt")

    # Public training geometry supplies only a displacement sampling prior.
    geo = np.load(out / "geometry.npz")
    n = len(geo["sx"])
    order = np.random.default_rng(7).permutation(n)
    train = np.r_[order[:300000], order[400000:]]
    sx, sy, tx, ty = (geo[k].astype(np.int32) for k in ("sx", "sy", "tx", "ty"))
    dx, dy = tx[train] - sx[train], ty[train] - sy[train]
    cross = ((sx[train] <= 75) & (tx[train] >= 90)) | ((tx[train] <= 75) & (sx[train] >= 90))
    row0 = ((dx - np.where(cross, np.where(dx > 0, 14, -14), 0) + 120) * 1105 + dy + 552)
    row1 = (dx + 120) * 1105 + dy + 552
    cols = np.r_[row0[(row0 >= 0) & (row0 < nc)], row1[(row1 >= 0) & (row1 < nc)]]
    rng = np.random.default_rng(args.seed)
    n_geom = args.sample_columns * 3 // 4
    xx, yy = np.meshgrid(np.arange(-args.lx, args.lx + 1),
                         np.arange(-args.ly, args.ly + 1), indexing="ij")
    locations = ((xx + 120) * 1105 + yy + 552).astype(np.int32).ravel()
    locations = locations[(locations >= 0) & (locations < nc)]
    cols = cols[np.isin(cols, locations)]
    if len(cols) == 0:
        raise ValueError("training geometry has no columns in the selected region")
    picked = rng.choice(cols, n_geom, replace=True)
    picked = np.r_[picked, rng.choice(locations, args.sample_columns - n_geom, replace=True)]
    rng.shuffle(picked)

    # Residual matrix for randomized range finding; columns are deliberately
    # sampled with replacement so frequently queried displacements get weight.
    residual = np.empty((208 * 288, len(picked)), dtype=np.float32)
    before = []
    base_v = v0[picked].astype(np.float32)
    for si, pid in enumerate(ids):
        path = args.raw / f"all_C_{pid}.bin"
        st = path.stat()
        if st.st_size != 288 * nc * 2:
            raise ValueError(f"unexpected table size: {path}")
        before.append((path, st.st_size, st.st_mtime_ns))
        raw = np.memmap(path, "<i2", "r", shape=(288, nc))[:, picked].astype(np.float32)
        sl = slice(si * 288, (si + 1) * 288)
        residual[sl] = raw - (uruntime[sl] @ base_v.T + m[sl, None])
        if (si + 1) % 32 == 0:
            print("sample", si + 1, "elapsed", round(time.perf_counter() - started, 1), flush=True)

    width = min(args.rank + 32, residual.shape[1])
    omega = rng.standard_normal((len(picked), width), dtype=np.float32)
    y = residual @ omega
    q, _ = np.linalg.qr(y, mode="reduced")
    small = q.T @ residual
    left, singular, _ = np.linalg.svd(small, full_matrices=False)
    q = (q @ left[:, :args.rank]).astype(np.float32)
    del residual, y, omega, small, left
    print("range basis", q.shape, "singular", singular[:8].tolist(),
          "elapsed", round(time.perf_counter() - started, 1), flush=True)

    # Project the complete architecture-generated matrix once, in source-row
    # blocks.  Algebraically subtract the old model after projection to avoid
    # forming another 59,904 x 266,305 dense matrix.
    projected = np.zeros((args.rank, len(locations)), dtype=np.float32)
    for si, pid in enumerate(ids):
        path = args.raw / f"all_C_{pid}.bin"
        mm = np.memmap(path, "<i2", "r", shape=(288, nc))
        pair0 = si * 288
        for lo in range(0, 288, 64):
            hi = min(288, lo + 64)
            raw = mm[lo:hi, locations].astype(np.float32)
            projected += q[pair0 + lo:pair0 + hi].T @ raw
        if (si + 1) % 32 == 0:
            print("project", si + 1, "elapsed", round(time.perf_counter() - started, 1), flush=True)
    qu = q.T @ uruntime
    qm = q.T @ m
    projected -= qu @ v0[locations].astype(np.float32).T
    projected -= qm[:, None]
    del qu, qm
    for path, size, mtime in before:
        st = path.stat()
        assert (st.st_size, st.st_mtime_ns) == (size, mtime), f"source modified: {path}"

    # Store a local-runtime-compatible additive factorization.
    residual_scale = np.maximum(np.max(np.abs(projected), axis=1) / 32767.0,
                                np.finfo(np.float32).tiny).astype("<f4")
    vq = np.rint(projected / residual_scale[:, None]).clip(-32767, 32767).astype("<i2")
    factors_path = out / f"{args.name}_factors.npz"
    np.savez_compressed(factors_path, coefficients=q, v=vq, scales=residual_scale,
                        locations=locations)

    report = {
        "candidate": args.name,
        "rank_base": rank0,
        "rank_residual": args.rank,
        "rank_total_with_base": rank0 + args.rank,
        "displacement_window": {"dx": [-args.lx, args.lx], "dy": [-args.ly, args.ly]},
        "columns": len(locations),
        "sample_columns": len(picked),
        "sampled_public_training_geometry_only": True,
        "public_delay_labels_used": False,
        "external_tables_modified": False,
        "external_architecture_sha256": sha(args.raw / "arch.txt"),
        "sources": len(ids),
        "source_table_bytes_each": 288 * nc * 2,
        "source_tables": [
            {"id": int(pid), "bytes": size, "mtime_ns": mtime}
            for pid, (_, size, mtime) in zip(ids, before)
        ],
        "factors_file": str(factors_path),
        "factors_sha256": sha(factors_path),
        "factors_bytes": factors_path.stat().st_size,
        "residual_singular_values": singular[:min(64, len(singular))].tolist(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (root / "docs" / f"{args.name}_fit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (root / "docs" / f"{args.name}_training.json").write_text(json.dumps({
        "name": args.name, "rank": args.rank, "rows": 208 * 288,
        "columns": len(locations), "direct": False, "refit_base": True,
        "public_delay_labels_used": False, "geometry_training_rows_only": True,
        "relative_weight": "sampled displacement residual from architecture-generated all_C tables",
        "factors_sha256": report["factors_sha256"],
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
