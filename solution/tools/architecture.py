"""Compile a translation-invariant shortest-path lower bound from the architecture.

Positions are reduced modulo (10, 12). Subtract a feasible linear spatial
potential from each edge, then use Johnson reweighting and exact Dijkstra on
the finite quotient graph. No public delay labels enter this calculation.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from project_paths import BUILD, DATA

ARCH = DATA / "arch"


def compile_tables() -> None:
    ports = [p["Name"] for p in json.loads((ARCH / "SRB_Port.json").read_text())["Port"]]
    ids = {p: i for i, p in enumerate(ports)}
    arcs = json.loads((ARCH / "SRB_Arc.json").read_text())["Arcs"]
    nets = json.loads((ARCH / "SRB_Net.json").read_text())["Nets"]
    edges = [(ids[a["from"]], ids[a["to"]], a["delay"], 0, 0) for a in arcs]
    edges += [(ids[n["from"]], ids[n["to"]], 0, n["delta x"], n["delta y"]) for n in nets]
    u, v, delay, dx, dy = np.asarray(edges, dtype=np.int32).T
    sources = sorted(set(u.tolist()))
    # I* are transit vertices rather than allowed endpoints.
    sources = [i for i in sources if not ports[i].startswith("I")]
    targets = sorted(set(ids[a["to"]] for a in arcs))
    nports = len(ports)
    residues = np.arange(120)
    expanded_u = (residues[:, None] * nports + u).ravel()
    dest_r = (((residues[:, None] // 12 + dx) % 10) * 12
              + (residues[:, None] % 12 + dy) % 12)
    expanded_v = (dest_r * nports + v).ravel()
    all_tables = []
    slopes = []
    for sign_x, sign_y in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        started = time.perf_counter()
        ax, by = 150 * sign_x, (127 if sign_y > 0 else 125) * sign_y
        for attempt in range(10):
            reduced = delay * 12 - ax * dx - by * dy
            potential = np.zeros(nports, dtype=np.int64)
            for iteration in range(nports):
                updated = potential.copy()
                np.minimum.at(updated, v, potential[u] + reduced)
                if np.array_equal(updated, potential):
                    break
                potential = updated
            else:
                ax *= .99
                by *= .99
                continue
            break
        else:
            raise RuntimeError("Unable to find a feasible spatial potential")
        if not (float(ax).is_integer() and float(by).is_integer()):
            raise RuntimeError(f"Fractional fallback slope needs exact storage: {ax}, {by}")
        weights = reduced + potential[u] - potential[v]
        assert weights.min() >= 0
        graph = coo_matrix((np.tile(weights.astype(float), 120),
                            (expanded_u, expanded_v)), shape=(120*nports, 120*nports)).tocsr()
        distance = dijkstra(graph, directed=True, indices=sources)
        distance = distance.reshape(len(sources), 120, nports)[:, :, targets]
        distance = distance - potential[np.array(sources), None, None] + potential[np.array(targets)]
        finite = np.isfinite(distance)
        assert np.all(distance[finite] >= -32768) and np.all(distance[finite] < 32767)
        table = np.where(finite, distance, 32767).astype(np.int16)
        all_tables.append(table)
        slopes.append((int(ax), int(by)))
        print("direction", sign_x, sign_y, "slopes/12", ax, by,
              "range", table[finite].min(), table[finite].max(),
              "seconds", time.perf_counter()-started, flush=True)
    BUILD.mkdir(exist_ok=True, parents=True)
    np.save(BUILD / "architecture_table.npy", np.stack(all_tables))
    (BUILD / "architecture_meta.json").write_text(json.dumps({
        "ports": ports, "sources": sources, "targets": targets,
        "slopes_scaled12": slopes, "scale": 12, "period_x": 10, "period_y": 12,
    }, indent=2), encoding="utf-8")


def predict(frame):
    from experiment import pin_parts
    meta = json.loads((BUILD / "architecture_meta.json").read_text())
    table = np.load(BUILD / "architecture_table.npy", mmap_mode="r")
    sx, sy, sp = pin_parts(frame["From"])
    tx, ty, tp = pin_parts(frame["To"])
    ports = meta["ports"]
    source_map = {ports[p]: i for i, p in enumerate(meta["sources"])}
    target_map = {ports[p]: i for i, p in enumerate(meta["targets"])}
    sidx = sp.map(source_map).fillna(0).to_numpy(dtype=np.int32)
    tidx = tp.map(target_map).fillna(0).to_numpy(dtype=np.int32)
    dx, dy = tx - sx, ty - sy
    direction = (2*(dx < 0) + (dy < 0)).astype(np.int8)
    r = (dx % 10)*12 + dy % 12
    slopes = np.array(meta["slopes_scaled12"])
    result = (slopes[direction, 0]*dx + slopes[direction, 1]*dy + table[direction, sidx, r, tidx])/12
    # All monotone routes pay the same Line gap sum. Detours may pay more.
    gaps = json.loads((ARCH / "SRB_Gap.json").read_text())["Gap"]
    for line in gaps["Line"]:
        lo, hi = (np.minimum(sx, tx), np.maximum(sx, tx)) if line["direction"] == "vertical" else (np.minimum(sy, ty), np.maximum(sy, ty))
        result += line["delay"] * ((lo <= line["site"]) & (line["site"] < hi))
    result[(sx == tx) & (sy == ty) & (sp.to_numpy() == tp.to_numpy())] = 0
    return result


if __name__ == "__main__":
    compile_tables()
