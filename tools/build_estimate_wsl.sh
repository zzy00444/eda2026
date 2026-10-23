#!/bin/bash
# Build the Plan A runtime inside the official image and pack the model blob.
# Usage: bash tools/build_estimate_wsl.sh [repo-root]
set -euo pipefail
ROOT="${1:-/mnt/f/EDA/solution/eda2026}"
IMAGE="${IMAGE:-fpga-ziguang:latest}"
docker run --rm --user "$(id -u):$(id -g)" -v "$ROOT":/ws -w /ws "$IMAGE" \
  g++ -O3 -mavx2 -std=c++17 -o build/estimate src/estimate.cpp
ls -la "$ROOT/build/estimate"
