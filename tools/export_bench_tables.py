"""Export the small tables the C++ lane micro-benchmark needs.

* build/bench_ports.txt : 496 lines "<name> <source_index|-1> <target_index|-1>"
* build/bench_gap.txt   : vertical-line prefix sums over x (120) and horizontal over
                          y (550), so the benchmark computes the exact gap in O(1)
                          exactly like the real runtime will.

The large model tables (coef, basis, shortlist, B) are synthesised inside the C++
benchmark with a deterministic PRNG; only their *shapes and access pattern* matter
for the cost measurement.

Usage: python tools/export_bench_tables.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from arch import load_architecture  # noqa: E402


def main() -> None:
    arch = load_architecture()
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    src_index = {p: i for i, p in enumerate(arch.query_sources)}
    tgt_index = {p: i for i, p in enumerate(arch.targets)}
    lines = []
    for pid, name in enumerate(arch.ports):
        lines.append(f"{name} {src_index.get(pid, -1)} {tgt_index.get(pid, -1)}")
    (build / "bench_ports.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {build/'bench_ports.txt'} ({len(lines)} ports)")

    vp = [0] * (arch.width + 1)
    for line in arch.vertical_lines:
        for x in range(line.site + 1, arch.width + 1):
            vp[x] += line.delay
    hp = [0] * (arch.height + 1)
    for line in arch.horizontal_lines:
        for y in range(line.site + 1, arch.height + 1):
            hp[y] += line.delay
    out = [f"{arch.width} {arch.height}"]
    out.append(" ".join(str(v) for v in vp))
    out.append(" ".join(str(v) for v in hp))
    (build / "bench_gap.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {build/'bench_gap.txt'} (vp {len(vp)}, hp {len(hp)})")


if __name__ == "__main__":
    main()
