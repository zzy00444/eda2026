#!/usr/bin/env bash
set -euo pipefail
project=/home/zzy/fpga-delay-solution
cp /mnt/f/EDA/solution/tools/integration.py "$project/tools/"
cp /mnt/f/EDA/EDA-PANGO-SRB-Arch-Delay/EDA-PANGO-SRB-Arch-Delay/data/delay_estimate_check.csv "$project/check.csv"
docker run --rm --user 1000:1000 --cpus=1 --memory=2g --memory-swap=2g \
  --mount type=bind,source="$project",target=/workspace -w /workspace fpga-ziguang:latest \
  python3 tools/integration.py
docker run --rm --user 1000:1000 --cpus=1 --memory=2g --memory-swap=2g \
  --mount type=bind,source="$project",target=/workspace -w /workspace fpga-ziguang:latest \
  python3 tools/benchmark.py --runs 5 --report benchmark_final.json
cp "$project/benchmark_final.json" "$project/integration.json" /mnt/f/EDA/solution/build/
cp "$project/result.csv" /mnt/f/EDA/solution/build/final_result.csv
