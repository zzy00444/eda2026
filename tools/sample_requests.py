"""Build request files (and golden sidecars) for checks and benchmarks.

Usage
    python tools/sample_requests.py check                     # 8 official check paths
    python tools/sample_requests.py public --limit 200000     # public 1M subset
    python tools/sample_requests.py s0                        # stratified S0 suite (300 rows)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parents[1] / "EDA-PANGO-SRB-Arch-Delay" / "EDA-PANGO-SRB-Arch-Delay" / "data"
BUILD = ROOT / "build"


def write(rows: pd.DataFrame, request: Path, golden: Path | None = None) -> None:
    request.parent.mkdir(parents=True, exist_ok=True)
    request.write_text("From,To\n" + "\n".join(f"{a},{b}" for a, b in zip(rows["From"], rows["To"])) + "\n",
                       encoding="utf-8", newline="\n")
    if golden is not None:
        golden.write_text("From,To,delay\n"
                          + "\n".join(f"{a},{b},{int(d)}" for a, b, d in zip(rows["From"], rows["To"], rows["delay"]))
                          + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {request} ({len(rows)} rows)" + (f" and {golden}" if golden else ""))


def public_frame() -> pd.DataFrame:
    frame = pd.read_csv(DATA / "delay_estimate_ans.csv")
    parts = frame["From"].str.extract(r"^SRB_(\d+)_(\d+)/")
    tparts = frame["To"].str.extract(r"^SRB_(\d+)_(\d+)/")
    frame["sx"] = parts[0].astype(int)
    frame["sy"] = parts[1].astype(int)
    frame["tx"] = tparts[0].astype(int)
    frame["ty"] = tparts[1].astype(int)
    frame["dx"] = frame["tx"] - frame["sx"]
    frame["dy"] = frame["ty"] - frame["sy"]
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["check", "public", "s0"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    BUILD.mkdir(parents=True, exist_ok=True)

    if args.kind == "check":
        with (DATA / "delay_estimate_check.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = pd.DataFrame(list(csv.DictReader(stream)))
        rows = rows.rename(columns={"Min Delay": "delay"})[["From", "To", "delay"]]
        rows["delay"] = rows["delay"].astype(int)
        write(rows, BUILD / "check_request.csv", BUILD / "check_golden.csv")
        return

    frame = public_frame()
    rng = np.random.default_rng(args.seed)
    if args.kind == "public":
        n = args.limit if args.limit > 0 else len(frame)
        pick = frame.iloc[rng.choice(len(frame), size=min(n, len(frame)), replace=False)]
        write(pick, BUILD / "public_request.csv", BUILD / "public_golden.csv")
        return

    # S0 stratified suite: deliberately over-sample the defect classes.
    wall = (frame["sx"] < 76) & (frame["tx"] > 89) | (frame["tx"] < 76) & (frame["sx"] > 89)
    near_horizontal_edge = (frame["sy"] <= 12) | (frame["sy"] >= 537) | (frame["ty"] <= 12) | (frame["ty"] >= 537)
    near_vertical_edge = (frame["sx"] <= 12) | (frame["sx"] >= 107) | (frame["tx"] <= 12) | (frame["tx"] >= 107)
    long_haul = (frame["dx"].abs() + frame["dy"].abs()) > 600
    short_delay = frame["delay"] < 600
    wall_corridor = ((frame["sx"] >= 74) & (frame["sx"] <= 91)) | ((frame["tx"] >= 74) & (frame["tx"] <= 91))
    near_gap_line = frame["sx"].isin([9, 19, 33, 39, 45, 51, 57, 63, 69, 95, 99, 105, 111, 117]) \
        | frame["tx"].isin([9, 19, 33, 39, 45, 51, 57, 63, 69, 95, 99, 105, 111, 117])

    classes = {
        "wall": wall,
        "edge_y": near_horizontal_edge,
        "edge_x": near_vertical_edge,
        "long": long_haul,
        "short": short_delay,
        "corridor": wall_corridor,
        "gap_line": near_gap_line,
    }
    picks = []
    used = set()
    for name, mask in classes.items():
        idx = frame.index[mask].to_numpy()
        idx = np.array([i for i in idx if i not in used])
        take = rng.choice(idx, size=min(40, len(idx)), replace=False) if len(idx) else idx
        used.update(int(i) for i in take)
        picks.append(frame.loc[take])
        print(f"{name}: {len(take)} rows (pool {int(mask.sum())})")
    rest = np.array([i for i in frame.index if i not in used])
    picks.append(frame.loc[rng.choice(rest, size=20, replace=False)])
    suite = pd.concat(picks).drop_duplicates(subset=["From", "To"])
    write(suite, BUILD / "s0_request.csv", BUILD / "s0_golden.csv")


if __name__ == "__main__":
    main()
