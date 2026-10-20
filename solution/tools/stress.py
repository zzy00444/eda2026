"""Expand requests for throughput measurement and verify the entire repeated output."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--copies", type=int, default=10)
parser.add_argument("--input", default="request.csv")
parser.add_argument("--expanded", default="request_10m.csv")
parser.add_argument("--output", default="result_10m.csv")
parser.add_argument("--report", default="benchmark_10m.json")
parser.add_argument("--reference", default="result.csv")
parser.add_argument("--executable", default="./bin/estimate")
parser.add_argument("--runs", type=int, default=1)
args = parser.parse_args()
assert args.copies > 0 and args.runs > 0
copies = args.copies
request = Path(args.input)
expanded = Path(args.expanded)
assert request.resolve() != expanded.resolve()
with request.open("rb") as src:
    src.readline()
    rows = sum(chunk.count(b"\n") for chunk in iter(lambda: src.read(1 << 20), b""))
with request.open("rb") as src, expanded.open("wb") as dst:
    header = src.readline()
    body_start = src.tell()
    dst.write(header)
    for _ in range(copies):
        src.seek(body_start)
        shutil.copyfileobj(src,dst,length=1<<20)
subprocess.run(["python3","tools/benchmark.py","--input",str(expanded),
                "--output",args.output,"--report",args.report,
                "--executable",args.executable,"--runs",str(args.runs)],check=True)
expected = hashlib.sha256()
with Path(args.reference).open("rb") as stream:
    header = stream.readline()
    start = stream.tell()
    expected.update(header)
    for _ in range(copies):
        stream.seek(start)
        for chunk in iter(lambda:stream.read(1<<20),b""):
            expected.update(chunk)
report=json.loads(Path(args.report).read_text())
for run in report:
    assert expected.hexdigest()==run["output_sha256"], "Repeated output mismatch"
    run["rows"]=rows*copies
    run["input_copies"]=copies
    run["repeated_output_verified"]=True
    run["projected_1b_cpu_minutes"]=run["cpu_seconds"]*1_000_000_000/(rows*copies)/60
Path(args.report).write_text(json.dumps(report,indent=2))
print(f"All {rows*copies:,} output rows verified by SHA256 ({args.runs} runs)",flush=True)
