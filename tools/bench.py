"""Run a candidate ELF inside the official image and report CPU time.

This is the harness used by the S5 speed gate.  It runs the binary `--runs`
times (alternating order is the caller's job for multi-candidate batches),
records `user + sys` CPU time, wall time, and the SHA-256 of every output.

The 100M protocol (public 1M requests repeated 100 times) requires the repeated
request file to exist beforehand; `--make-repeat` can build it from a base file
if the target does not exist (writes into the same directory, ~3.5 GB).

Examples
    python tools/bench.py --binary build/estimate --request build/req_1m.csv
    python tools/bench.py --binary build/estimate --request build/req_1m.csv --runs 5
    python tools/bench.py --binary build/estimate --request build/local_runs/request_1m.csv \
        --make-repeat out100m.csv --repeat 100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WSL_REPO = "/mnt/f/EDA/solution/eda2026"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def wsl(command: str) -> str:
    proc = subprocess.run(["wsl", "-d", "Ubuntu-20.04", "--", "bash", "-lc", command],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or proc.stdout.strip())
    return proc.stdout


def make_repeat(base: Path, target: Path, repeat: int) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as out:
        header, _, body = base.read_bytes().partition(b"\n")
        out.write(header + b"\n")
        for _ in range(repeat):
            out.write(body)
    print(f"wrote {target} ({target.stat().st_size} bytes)")


def run_once(binary: Path, request: Path, result: Path, repo: Path) -> dict:
    relative_bin = binary.relative_to(repo).as_posix() if binary.is_relative_to(repo) else None
    relative_req = request.relative_to(repo).as_posix() if request.is_relative_to(repo) else None
    relative_out = result.relative_to(repo).as_posix() if result.is_relative_to(repo) else None
    if not all((relative_bin, relative_req, relative_out)):
        raise SystemExit("binary, request and result must live inside the repository")
    script = (
        "export TIMEFORMAT='wall=%R user=%U sys=%S'; "
        f"{{ time ./{relative_bin} -in ./{relative_req} -out ./{relative_out}; }} 2>&1; "
        "cat /sys/fs/cgroup/memory.peak"
    )
    start = time.perf_counter()
    out = wsl(
        "docker run --rm --cpus=1 --memory=2000000000 --memory-swap=2000000000 "
        f"-v {WSL_REPO}:/ws -w /ws fpga-ziguang:latest bash -lc \"{script}\""
    )
    wall = time.perf_counter() - start
    metrics = {"wall_s": round(wall, 3)}
    for token in out.split():
        if "=" in token:
            key, value = token.split("=", 1)
            try:
                metrics[f"{key}_s"] = float(value)
            except ValueError:
                pass
        elif token.isdigit():
            metrics["memory_peak_bytes"] = int(token)
    metrics["cpu_s"] = round(metrics.get("user_s", 0.0) + metrics.get("sys_s", 0.0), 4)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--tag", default="bench")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--make-repeat", type=Path, default=None)
    args = parser.parse_args()

    repo = args.repo.resolve()
    binary = (repo / args.binary).resolve() if not args.binary.is_absolute() else args.binary
    request = (repo / args.request).resolve() if not args.request.is_absolute() else args.request
    if args.repeat and args.make_repeat:
        target = (repo / args.make_repeat).resolve() if not args.make_repeat.is_absolute() else args.make_repeat
        make_repeat(request, target, args.repeat)
        request = target

    results = []
    for i in range(args.runs):
        result = repo / "build" / f"{args.tag}_run{i}.csv"
        metrics = run_once(binary, request, result, repo)
        metrics["sha256"] = sha256(result)
        metrics["rows"] = sum(1 for _ in result.open("rb")) - 1
        metrics["run"] = i
        print(json.dumps(metrics))
        results.append(metrics)

    hashes = {r["sha256"] for r in results}
    cpus = [r["cpu_s"] for r in results]
    summary = {
        "tag": args.tag,
        "runs": args.runs,
        "rows": results[0]["rows"],
        "cpu_mean": round(sum(cpus) / len(cpus), 4),
        "cpu_min": min(cpus),
        "cpu_max": max(cpus),
        "distinct_output_hashes": len(hashes),
        "consistent": len(hashes) == 1,
        "memory_peak_bytes": max(r.get("memory_peak_bytes", 0) for r in results),
    }
    print(json.dumps(summary, indent=2))
    out = repo / "build" / f"{args.tag}_bench.json"
    out.write_text(json.dumps({"summary": summary, "runs": results}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
