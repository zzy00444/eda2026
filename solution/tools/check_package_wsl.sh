#!/usr/bin/env bash
set -euo pipefail
project=/home/zzy/fpga-delay-solution
cp /mnt/f/EDA/solution/submission.zip "$project/submission.zip"
cp /mnt/f/EDA/solution/tools/standalone.py "$project/tools/"
docker run --rm --user 1000:1000 --cpus=1 --memory=2g --memory-swap=2g \
  --mount type=bind,source="$project",target=/workspace -w /workspace fpga-ziguang:latest \
  python3 tools/standalone.py
cp "$project/standalone.json" /mnt/f/EDA/solution/build/
