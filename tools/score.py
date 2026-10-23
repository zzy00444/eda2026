"""Score arithmetic for the Pango SRB delay task.

The single official formula is

    total = 0.8*acc + 0.15*time + 0.05*consistency
    time  = 100 * max(0, 1 - cpu_seconds/1800)

with `consistency` assumed 100 (five identical runs) unless stated.

Examples
    python tools/score.py --table
    python tools/score.py --acc 99.3 --official-cpu 130
    python tools/score.py --acc 99.2 --local-100m 8.5 --ratio 1.1686
    python tools/score.py --required-acc 98.5 --official-cpu 110
"""

from __future__ import annotations

import argparse


def time_score(cpu_seconds: float) -> float:
    return 100.0 * max(0.0, 1.0 - cpu_seconds / 1800.0)


def total(acc: float, cpu_seconds: float, consistency: float = 100.0) -> float:
    return 0.8 * acc + 0.15 * time_score(cpu_seconds) + 0.05 * consistency


def required_acc(target: float, cpu_seconds: float, consistency: float = 100.0) -> float:
    return (target - 0.15 * time_score(cpu_seconds) - 0.05 * consistency) / 0.8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acc", type=float, default=None)
    parser.add_argument("--official-cpu", type=float, default=None, help="official 1B CPU seconds (5-run mean)")
    parser.add_argument("--local-100m", type=float, default=None, help="local 100M CPU seconds (5-run mean)")
    parser.add_argument("--ratio", type=float, default=None, help="official/local ratio used to calibrate")
    parser.add_argument("--required-acc", type=float, default=None, help="solve acc needed for this total")
    parser.add_argument("--target", type=float, default=98.0)
    parser.add_argument("--consistency", type=float, default=100.0)
    parser.add_argument("--table", action="store_true")
    args = parser.parse_args()

    if args.table:
        print("| cpu_s | time | total@acc98.2 | total@acc98.5 | total@acc99.0 | total@acc99.3 | total@acc99.5 |")
        print("|---:|---:|---:|---:|---:|---:|---:|")
        for cpu in (60, 80, 90, 100, 110, 120, 130, 150, 180, 220, 281):
            row = [f"{cpu}", f"{time_score(cpu):.2f}"]
            row += [f"{total(a, cpu, args.consistency):.3f}" for a in (98.2, 98.5, 99.0, 99.3, 99.5)]
            print("| " + " | ".join(row) + " |")
        print("\nrequired acc for a target total:")
        for target in (98.0, 98.5, 99.0):
            print(f"  total={target}: " + "  ".join(f"cpu={c}:{required_acc(target, c, args.consistency):.3f}"
                                                    for c in (80, 100, 120, 140, 160)))
        return

    cpu = args.official_cpu
    if cpu is None and args.local_100m is not None:
        if args.ratio is None:
            raise SystemExit("--local-100m needs --ratio (official/local)")
        cpu = args.local_100m * 10.0 * args.ratio
        print(f"calibrated official 1B CPU = {args.local_100m} * 10 * {args.ratio} = {cpu:.3f} s")

    if args.required_acc is not None:
        if cpu is None:
            raise SystemExit("--required-acc needs --official-cpu or --local-100m+--ratio")
        print(f"required acc for total {args.required_acc}: {required_acc(args.required_acc, cpu, args.consistency):.4f}")
        return

    if args.acc is None or cpu is None:
        raise SystemExit("need --acc and a CPU value (or use --table / --required-acc)")
    ts = time_score(cpu)
    print(f"time_score = 100*(1-{cpu}/{1800}) = {ts:.4f}")
    print(f"total      = 0.8*{args.acc} + 0.15*{ts:.4f} + {0.05 * args.consistency} = "
          f"{total(args.acc, cpu, args.consistency):.4f}")
    print(f"required acc for total {args.target} at this CPU: {required_acc(args.target, cpu, args.consistency):.4f}")


if __name__ == "__main__":
    main()
