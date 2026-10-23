#!/bin/bash
# Compile the exact oracle inside the official evaluation image.
# Usage: bash tools/build_oracle_wsl.sh [repo-root]
set -euo pipefail
ROOT="${1:-/mnt/f/EDA/solution/eda2026}"
IMAGE="${IMAGE:-fpga-ziguang:latest}"
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$ROOT":/ws -w /ws "$IMAGE" \
  g++ -O3 -std=c++17 -march=x86-64 -o build/oracle tools/oracle.cpp
ls -la "$ROOT/build/oracle"
