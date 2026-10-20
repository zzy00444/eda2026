"""Measure estimator CPU, wall, process RSS, container peak and output SHA256."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="request.csv")
parser.add_argument("--output", default="result.csv")
parser.add_argument("--report", default="benchmark.json")
parser.add_argument("--runs", type=int, default=1)
parser.add_argument("--executable", default="./bin/estimate")
args = parser.parse_args()
reports = []
for run in range(args.runs):
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    process = subprocess.Popen([args.executable, "-in", args.input, "-out", args.output])
    max_threads = 0
    # Sampling is performed by this helper process; estimator creates no threads.
    while process.poll() is None:
        try:
            max_threads = max(max_threads, len(list(Path(f"/proc/{process.pid}/task").iterdir())))
        except FileNotFoundError:
            pass
        time.sleep(.02)
    wall = time.perf_counter()-start
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    if process.returncode:
        raise RuntimeError(f"Estimator exit status {process.returncode}")
    report = {"run": run+1, "wall_seconds": wall,
              "user_seconds": after.ru_utime-before.ru_utime,
              "sys_seconds": after.ru_stime-before.ru_stime,
              "cpu_seconds": (after.ru_utime-before.ru_utime)+(after.ru_stime-before.ru_stime),
              "process_peak_rss_kib": after.ru_maxrss, "max_observed_threads": max_threads}
    peak = Path("/sys/fs/cgroup/memory.peak")
    report["container_peak_bytes"] = int(peak.read_text()) if peak.exists() else None
    digest = hashlib.sha256()
    with open(args.output, "rb") as stream:
        for chunk in iter(lambda: stream.read(1<<20), b""):
            digest.update(chunk)
    report["output_sha256"] = digest.hexdigest()
    reports.append(report)
    print(json.dumps(report), flush=True)
Path(args.report).write_text(json.dumps(reports,indent=2)+"\n")
