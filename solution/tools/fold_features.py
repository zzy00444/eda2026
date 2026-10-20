"""Shared foldable feature construction for the stage-7 folding experiments."""
import json

import numpy as np

from project_paths import BUILD

FOLDABLE = [
    "source_port", "source_family", "source_bit",
    "target_port", "target_family", "target_bit",
    "first_dx", "first_dy",
    "dx_mod", "dy_mod", "ax_mod", "ay_mod",
    "remaining_x_mod", "remaining_y_mod",
    "port_pair", "family_pair", "bit_pair", "direction",
    "table_scaled", "envelope_proxy",
]


def foldable_matrix(x, base):
    """Return (frame with FOLDABLE columns in canonical order, names, enriched x)."""
    meta = json.loads((BUILD / "architecture_meta.json").read_text())
    slopes = np.array(meta["slopes_scaled12"])
    dx = x.dx.to_numpy(np.int64)
    dy = x.dy.to_numpy(np.int64)
    direction = (2 * (dx < 0) + (dy < 0)).astype(np.int64)
    linear12 = slopes[direction, 0] * dx + slopes[direction, 1] * dy
    gaps = x.gap_x.to_numpy(np.int64) + x.gap_y.to_numpy(np.int64)
    table = np.load(BUILD / "architecture_table.npy", mmap_mode="r")
    src_map = {p: i for i, p in enumerate(meta["sources"])}
    tgt_map = {p: i for i, p in enumerate(meta["targets"])}
    sidx = np.array([src_map[p] for p in x.source_port.astype(np.int32).to_numpy()], dtype=np.int64)
    tidx = np.array([tgt_map[p] for p in x.target_port.astype(np.int32).to_numpy()], dtype=np.int64)
    r = (dx % 10) * 12 + (dy % 12)
    bounds = np.stack([table[d, sidx, r, tidx] for d in range(4)])
    bounds = np.where(bounds == 32767, -32768, bounds)
    enriched = x.copy()
    enriched["table_scaled"] = np.rint((base - gaps) * 12 - linear12)
    enriched["envelope_proxy"] = bounds.max(axis=0)
    return enriched[FOLDABLE], list(FOLDABLE), enriched
