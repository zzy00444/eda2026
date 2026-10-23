"""Canonical architecture model for the Pango SRB delay-estimation task.

This module is the *single source of truth* for the five JSON description files.
Every other tool (oracle, field factory, evaluators, exporters) must consume the
graph and the rules through here, so that a rule change is made in exactly one
place.

Delays are modelled as (Q&A A1/A2, task PDF section 5):

    delay = sum(SRB internal arc delays)
          + sum(gap-line delays crossed by nets)
          + sum(crossable gap-block crossing delays)

  * a net itself has zero base delay;
  * a gap line at `site` is crossed by a net whose coordinate moves across it,
    i.e. when `min(a,b) <= site < max(a,b)`;
  * a gap-block cell has no SRB; a net landing on a block is either broken
    (not crossable in the travel direction) or shifted to the far side of the
    block, counting only non-block cells in the travel direction;
  * crossing a crossable block adds its crossing delay.

CLI
    python arch.py --out build/arch.txt            # export for the C++ oracle
    python arch.py --report                        # print structural statistics
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT.parents[1] / "EDA-PANGO-SRB-Arch-Delay" / "EDA-PANGO-SRB-Arch-Delay" / "arch"
if not ARCHIVE.is_dir():  # fall back to an explicit local copy
    ARCHIVE = ROOT / "arch"

HORIZONTAL = 0  # block travel axis helpers
VERTICAL = 1


@dataclass(frozen=True)
class Arc:
    src: int
    dst: int
    delay: int


@dataclass(frozen=True)
class Net:
    src: int  # Z port index
    dst: int  # I port index
    dx: int
    dy: int

    @property
    def horizontal(self) -> bool:
        return self.dx != 0


@dataclass(frozen=True)
class Block:
    id: int
    lower: int
    upper: int
    left: int
    right: int
    vertical_crossable: bool
    vertical_delay: int
    horizontal_crossable: bool
    horizontal_delay: int

    def contains(self, x: int, y: int) -> bool:
        return self.left <= x <= self.right and self.lower <= y <= self.upper

    def width(self) -> int:
        return self.right - self.left + 1


@dataclass(frozen=True)
class Line:
    id: int
    vertical: bool
    site: int
    delay: int


@dataclass
class Architecture:
    width: int
    height: int
    ports: list[str]
    direction: list[str]
    arcs: list[Arc]
    nets: list[Net]
    insts: list[tuple[int, int]]
    lines: list[Line]
    blocks: list[Block]

    # -- derived tables -------------------------------------------------
    def __post_init__(self) -> None:
        pid = {name: i for i, name in enumerate(self.ports)}
        self.port_id = pid
        self.srb_index = {xy: i for i, xy in enumerate(self.insts)}
        self.arcs_from: list[list[Arc]] = [[] for _ in self.ports]
        for a in self.arcs:
            self.arcs_from[a.src].append(a)
        self.nets_from: list[list[Net]] = [[] for _ in self.ports]
        for n in self.nets:
            self.nets_from[n.src].append(n)
        self.sources = [i for i in range(len(self.ports)) if self.arcs_from[i] or self.nets_from[i]]
        # Query endpoints are all non-I ports (Q&A: "I* 端口以外均可能作为请求端点").
        # A logic load port has no outgoing edge, so the reachable From universe is
        # exactly the non-I ports with at least one outgoing arc or net (208 here).
        self.query_sources = [i for i in self.sources if not self.ports[i].startswith("I")]
        self.targets = sorted({a.dst for a in self.arcs} | {n.src for n in self.nets})
        self.vertical_lines = [l for l in self.lines if l.vertical]
        self.horizontal_lines = [l for l in self.lines if not l.vertical]

    # -- rules ----------------------------------------------------------
    def srb_at(self, x: int, y: int) -> int:
        return self.srb_index.get((x, y), -1)

    def block_at(self, x: int, y: int) -> Block | None:
        for b in self.blocks:
            if b.contains(x, y):
                return b
        return None

    def net_target(self, net: Net, x: int, y: int) -> tuple[int, int] | None:
        """Return (target SRB index, net cost) or None if the net is broken."""
        nx, ny = x + net.dx, y + net.dy
        if not (0 <= nx < self.width and 0 <= ny < self.height):
            return None
        target = self.srb_at(nx, ny)
        if target < 0:
            block = self.block_at(nx, ny)
            if block is None:
                return None
            crossable = block.horizontal_crossable if net.horizontal else block.vertical_crossable
            if not crossable:
                return None
            step_x = 0 if net.dx == 0 else (1 if net.dx > 0 else -1)
            step_y = 0 if net.dy == 0 else (1 if net.dy > 0 else -1)
            need = abs(net.dx) if net.horizontal else abs(net.dy)
            got = 0
            cx, cy = x, y
            while got < need:
                cx += step_x
                cy += step_y
                if not (0 <= cx < self.width and 0 <= cy < self.height):
                    return None
                if self.srb_at(cx, cy) >= 0:
                    got += 1
            nx, ny = cx, cy
            target = self.srb_at(nx, ny)
            if target < 0:  # should not happen: we counted non-block cells
                return None
        cost = 0
        if x != nx:
            lo, hi = (x, nx) if x < nx else (nx, x)
            for line in self.vertical_lines:
                if lo <= line.site < hi:
                    cost += line.delay
        if y != ny:
            lo, hi = (y, ny) if y < ny else (ny, y)
            for line in self.horizontal_lines:
                if lo <= line.site < hi:
                    cost += line.delay
        for b in self.blocks:
            if net.horizontal:
                if not b.horizontal_crossable:
                    continue
                if b.lower <= y <= b.upper:
                    lo, hi = (x, nx) if x < nx else (nx, x)
                    if not (hi < b.left or lo > b.right):
                        cost += b.horizontal_delay
            else:
                if not b.vertical_crossable:
                    continue
                if b.left <= x <= b.right:
                    lo, hi = (y, ny) if y < ny else (ny, y)
                    if not (hi < b.lower or lo > b.upper):
                        cost += b.vertical_delay
        return target, cost

    def gap_cost(self, sx: int, sy: int, tx: int, ty: int) -> int:
        """Monotone gap-line crossing cost between two SRB coordinates."""
        cost = 0
        for line in self.vertical_lines:
            lo, hi = (sx, tx) if sx < tx else (tx, sx)
            if lo <= line.site < hi:
                cost += line.delay
        for line in self.horizontal_lines:
            lo, hi = (sy, ty) if sy < ty else (ty, sy)
            if lo <= line.site < hi:
                cost += line.delay
        return cost


def load_architecture(root: Path | None = None) -> Architecture:
    root = Path(root) if root is not None else ARCHIVE

    def read(name: str):
        return json.loads((root / name).read_text(encoding="utf-8"))

    port_doc = read("SRB_Port.json")["Port"]
    ports = [p["Name"] for p in port_doc]
    direction = [p["Direction"] for p in port_doc]
    pid = {name: i for i, name in enumerate(ports)}
    arcs = [Arc(pid[a["from"]], pid[a["to"]], int(a["delay"])) for a in read("SRB_Arc.json")["Arcs"]]
    nets = [Net(pid[n["from"]], pid[n["to"]], int(n["delta x"]), int(n["delta y"])) for n in read("SRB_Net.json")["Nets"]]
    insts = [(int(i["x"]), int(i["y"])) for i in read("SRB_Inst.json")["Inst"]]
    gap = read("SRB_Gap.json")["Gap"]
    lines = [
        Line(int(l["id"]), l["direction"] == "vertical", int(l["site"]), int(l["delay"]))
        for l in gap["Line"]
    ]
    blocks = [
        Block(
            int(b["id"]), int(b["lower"]), int(b["upper"]), int(b["left"]), int(b["right"]),
            bool(b["vertical crossable"]), int(b["vertical cross delay"]),
            bool(b["horizontal crossable"]), int(b["horizontal cross delay"]),
        )
        for b in gap["Block"]
    ]
    width = max(x for x, _ in insts) + 1
    height = max(y for _, y in insts) + 1
    return Architecture(width, height, ports, direction, arcs, nets, insts, lines, blocks)


def render_arch_text(arch: Architecture) -> str:
    """Text format consumed by tools/oracle.cpp (all references are port indices)."""
    out: list[str] = []
    out.append(f"GRID {arch.width} {arch.height}")
    out.append(f"PORTS {len(arch.ports)}")
    for i, name in enumerate(arch.ports):
        out.append(f"{i} {name}")
    out.append(f"ARCS {len(arch.arcs)}")
    for a in arch.arcs:
        out.append(f"{a.src} {a.dst} {a.delay}")
    out.append(f"NETS {len(arch.nets)}")
    for n in arch.nets:
        out.append(f"{n.src} {n.dst} {n.dx} {n.dy}")
    out.append(f"SRBS {len(arch.insts)}")
    for x, y in arch.insts:
        out.append(f"{x} {y}")
    out.append(f"LINES {len(arch.lines)}")
    for l in arch.lines:
        out.append(f"{1 if l.vertical else 0} {l.site} {l.delay}")
    out.append(f"BLOCKS {len(arch.blocks)}")
    for b in arch.blocks:
        out.append(
            f"{b.lower} {b.upper} {b.left} {b.right} "
            f"{1 if b.vertical_crossable else 0} {b.vertical_delay} "
            f"{1 if b.horizontal_crossable else 0} {b.horizontal_delay}"
        )
    out.append(f"SOURCES {len(arch.sources)} " + " ".join(str(i) for i in arch.sources))
    out.append(f"TARGETS {len(arch.targets)} " + " ".join(str(i) for i in arch.targets))
    out.append("END")
    return "\n".join(out) + "\n"


def report(arch: Architecture) -> dict:
    from collections import Counter

    z = sum(1 for p in arch.ports if p.startswith("Z"))
    i = sum(1 for p in arch.ports if p.startswith("I"))
    logic = len(arch.ports) - z - i
    return {
        "grid": [arch.width, arch.height],
        "insts": len(arch.insts),
        "ports": len(arch.ports),
        "ports_z": z,
        "ports_i": i,
        "ports_logic": logic,
        "directions": dict(Counter(arch.direction)),
        "arcs": len(arch.arcs),
        "arc_from_classes": dict(Counter(arch.ports[a.src][0] for a in arch.arcs)),
        "nets": len(arch.nets),
        "net_vectors": dict(sorted((f"{dx},{dy}", c) for (dx, dy), c in Counter((n.dx, n.dy) for n in arch.nets).items())),
        "sources": len(arch.sources),
        "sources_query": len(arch.query_sources),
        "targets": len(arch.targets),
        "lines": len(arch.lines),
        "blocks": len(arch.blocks),
        "block_absent_cells": sum(b.width() * (b.upper - b.lower + 1) for b in arch.blocks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="write arch.txt for tools/oracle.cpp")
    parser.add_argument("--report", action="store_true", help="print structural statistics as JSON")
    parser.add_argument("--arch", type=Path, default=None, help="override architecture directory")
    args = parser.parse_args()

    arch = load_architecture(args.arch)
    if args.report or args.out is None:
        print(json.dumps(report(arch), indent=2, ensure_ascii=False, default=str))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(render_arch_text(arch), encoding="utf-8", newline="\n")
        print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
