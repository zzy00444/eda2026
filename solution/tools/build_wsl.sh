#!/usr/bin/env bash
set -euo pipefail
project=/home/zzy/fpga-delay-solution
mkdir -p "$project/src" "$project/generated" "$project/bin" "$project/tools"
cp /mnt/f/EDA/solution/src/main.cpp "$project/src/"
cp /mnt/f/EDA/solution/build/generated/model.hpp /mnt/f/EDA/solution/build/generated/architecture.S /mnt/f/EDA/solution/build/generated/architecture.bin "$project/generated/"
cp /mnt/f/EDA/solution/tools/benchmark.py "$project/tools/"
docker run --rm --user 1000:1000 --memory=2g --mount type=bind,source="$project",target=/workspace -w /workspace fpga-ziguang:latest \
  g++ -O3 -march=native -ffp-contract=off -flto -DNDEBUG -std=c++17 -Wall -Wextra -Igenerated src/main.cpp generated/architecture.S -Wa,-Igenerated -o bin/estimate
mkdir -p /mnt/f/EDA/solution/bin
cp "$project/bin/estimate" /mnt/f/EDA/solution/bin/estimate
