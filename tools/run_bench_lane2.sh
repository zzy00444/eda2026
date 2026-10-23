#!/bin/bash
# Lane runtime prototype benchmark inside the official image.
# Usage: bash tools/run_bench_lane2.sh [request.csv] [rank] [lanes] [select]
set -euo pipefail
REQ="${1:-build/req_public_1m.csv}"
RANK="${2:-256}"
LANES="${3:-13}"
SELECT="${4:-2}"
cd /ws
g++ -O3 -mavx2 -std=c++17 -o build/bench_lane2 tools/bench_lane2.cpp
cp "$REQ" /tmp/bench_req.csv
cp build/bench_ports.txt build/bench_gap.txt /tmp/
export TIMEFORMAT='wall=%R user=%U sys=%S'
for mode in copy readonly parse full fullnoprefetch; do
  echo "== $mode (rank=$RANK lanes=$LANES select=$SELECT) =="
  time ./build/bench_lane2 /tmp/bench_ports.txt /tmp/bench_gap.txt /tmp/bench_req.csv "/tmp/out_$mode.csv" "$mode" "$RANK" 8 "$LANES" "$SELECT"
  ls -la "/tmp/out_$mode.csv"
done
echo "== full rank=128 =="
time ./build/bench_lane2 /tmp/bench_ports.txt /tmp/bench_gap.txt /tmp/bench_req.csv /tmp/o.csv full 128 8 "$LANES" "$SELECT"
echo "== full rank=192 =="
time ./build/bench_lane2 /tmp/bench_ports.txt /tmp/bench_gap.txt /tmp/bench_req.csv /tmp/o.csv full 192 8 "$LANES" "$SELECT"
echo "== full select=1 =="
time ./build/bench_lane2 /tmp/bench_ports.txt /tmp/bench_gap.txt /tmp/bench_req.csv /tmp/o.csv full "$RANK" 8 "$LANES" 1
echo "== full select=3 =="
time ./build/bench_lane2 /tmp/bench_ports.txt /tmp/bench_gap.txt /tmp/bench_req.csv /tmp/o.csv full "$RANK" 8 "$LANES" 3
