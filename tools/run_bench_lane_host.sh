#!/bin/bash
# C++ micro-benchmark for the lane-level runtime shape, run directly on the WSL host
# (no Docker).  Use tools/run_bench_lane.sh for the official image when Docker is up.
# Usage: bash tools/run_bench_lane_host.sh [request.csv] [rank] [lanes]
set -euo pipefail
REPO=/mnt/f/EDA/solution/eda2026
REQ="${1:-$REPO/build/req_public_1m.csv}"
RANK="${2:-256}"
LANES="${3:-13}"
cd "$REPO"
g++ -O3 -mavx2 -std=c++17 -o build/bench_lane tools/bench_lane.cpp
cp "$REQ" /tmp/bench_req.csv
cp build/bench_ports.txt build/bench_gap.txt /tmp/
export TIMEFORMAT='wall=%R user=%U sys=%S'
for mode in copy parse full; do
  echo "== $mode (rank=$RANK lanes=$LANES) =="
  time ./build/bench_lane /tmp/bench_ports.txt /tmp/bench_gap.txt /tmp/bench_req.csv "/tmp/bench_out_$mode.csv" "$mode" "$RANK" "$LANES" 2
  ls -la "/tmp/bench_out_$mode.csv"
done
echo "== full rank=192 lanes=13 =="
time ./build/bench_lane /tmp/bench_ports.txt /tmp/bench_gap.txt /tmp/bench_req.csv /tmp/o.csv full 192 13 2
echo "== full rank=256 lanes=4 =="
time ./build/bench_lane /tmp/bench_ports.txt /tmp/bench_gap.txt /tmp/bench_req.csv /tmp/o.csv full 256 4 2
echo "== full rank=256 lanes=1 =="
time ./build/bench_lane /tmp/bench_ports.txt /tmp/bench_gap.txt /tmp/bench_req.csv /tmp/o.csv full 256 1 2
