#!/bin/bash
# Lane runtime prototype v3 benchmark inside the official image (double-buffered pipeline).
set -euo pipefail
cd /ws
g++ -O3 -mavx2 -std=c++17 -o build/bench_lane3 tools/bench_lane3.cpp
cp build/req_public_1m.csv /tmp/bench_req.csv
cp build/bench_ports.txt build/bench_gap.txt /tmp/
export TIMEFORMAT='wall=%R user=%U sys=%S'
for mode in copy readonly parse full noprefetch; do
  echo "== $mode =="
  time ./build/bench_lane3 /tmp/bench_ports.txt /tmp/bench_gap.txt /tmp/bench_req.csv "/tmp/o3_$mode.csv" "$mode" 256 8 13 2
done
echo "== full rank=192 =="
time ./build/bench_lane3 /tmp/bench_ports.txt /tmp/bench_gap.txt /tmp/bench_req.csv /tmp/o3.csv full 192 8 13 2
echo "== full select=1 =="
time ./build/bench_lane3 /tmp/bench_ports.txt /tmp/bench_gap.txt /tmp/bench_req.csv /tmp/o3.csv full 256 8 13 1
echo "== full select=3 =="
time ./build/bench_lane3 /tmp/bench_ports.txt /tmp/bench_gap.txt /tmp/bench_req.csv /tmp/o3.csv full 256 8 13 3
ls -la /tmp/o3_full.csv /tmp/o3_copy.csv
