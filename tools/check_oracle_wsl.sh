#!/bin/bash
# S0 gate: run the exact oracle on the 8 official check rows and on the
# stratified S0 suite.  Results are compared with golden by tools/evaluate.py.
# Usage: bash tools/check_oracle_wsl.sh [repo-root]
set -euo pipefail
ROOT="${1:-/mnt/f/EDA/solution/eda2026}"
IMAGE="${IMAGE:-fpga-ziguang:latest}"
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$ROOT":/ws -w /ws "$IMAGE" bash -c '
set -e
export TIMEFORMAT="wall=%R cpu_user=%U cpu_sys=%S"
echo "== check paths =="
time ./build/oracle queries build/arch.txt build/check_request.csv build/check_result.csv
echo "== S0 stratified suite =="
time ./build/oracle queries build/arch.txt build/s0_request.csv build/s0_result.csv
'
