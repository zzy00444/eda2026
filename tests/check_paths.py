"""Validate the architecture rules in tools/arch.py against the official check paths.

Each check row gives `From,To,Min Delay,Path` where Path lists every port the
minimum-delay route visits.  We replay the path:
  * consecutive ports in the same SRB  -> one arc (must exist, cost = arc delay)
  * consecutive ports in different SRBs -> one net (delta must match, cost from
    gap lines + crossable blocks, landing must match net_target)
and require the sum to equal `Min Delay` exactly.

Run:  python tests/check_paths.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from arch import load_architecture  # noqa: E402

DATA = ROOT.parents[1] / "EDA-PANGO-SRB-Arch-Delay" / "EDA-PANGO-SRB-Arch-Delay" / "data"


def parse_pin(text: str) -> tuple[int, int, str]:
    srb, port = text.split("/")
    _, x, y = srb.split("_")
    return int(x), int(y), port


def main() -> int:
    arch = load_architecture()
    arc_cost = {(a.src, a.dst): a.delay for a in arch.arcs}
    net_by_ports = {(n.src, n.dst): n for n in arch.nets}
    pid = arch.port_id
    rows = list(csv.DictReader((DATA / "delay_estimate_check.csv").open(encoding="utf-8-sig", newline="")))
    failures = 0
    for row in rows:
        path = [p.strip() for p in row["Path"].split("|")]
        total = 0
        ok = True
        for a, b in zip(path, path[1:]):
            ax, ay, ap = parse_pin(a)
            bx, by, bp = parse_pin(b)
            if (ax, ay) == (bx, by):
                w = arc_cost.get((pid[ap], pid[bp]))
                if w is None:
                    print(f"  missing arc {a} -> {b}")
                    ok = False
                    break
                total += w
            else:
                net = net_by_ports.get((pid[ap], pid[bp]))
                if net is None:
                    print(f"  missing net {a} -> {b}")
                    ok = False
                    break
                if (net.dx, net.dy) != (bx - ax, by - ay):
                    print(f"  net delta mismatch {a} -> {b}: {(net.dx, net.dy)} vs {(bx - ax, by - ay)}")
                    ok = False
                    break
                landed = arch.net_target(net, ax, ay)
                srbs = arch.srb_index
                if landed is None or srbs.get((bx, by)) != landed[0]:
                    print(f"  landing mismatch {a} -> {b}: {landed}")
                    ok = False
                    break
                total += landed[1]
        golden = int(row["Min Delay"])
        status = "OK " if ok and total == golden else "FAIL"
        if status == "FAIL":
            failures += 1
        print(f"[{status}] {row['From']} -> {row['To']}  golden={golden} replay={total} legs={len(path) - 1}")
    print(f"\n{len(rows) - failures}/{len(rows)} check paths reproduced exactly")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
