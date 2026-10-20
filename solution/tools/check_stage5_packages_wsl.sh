#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
work=${EDA_WSL_WORKDIR:-"$HOME/eda2026-work"}
mkdir -p "$work/tools" "$work/candidates" "$repo/solution/build/local_runs"
cp "$repo/solution/tools/standalone.py" "$work/tools/"
for variant in score accuracy; do
  package="$repo/solution/stage5_${variant}_submission.zip"
  candidate="$repo/solution/bin/estimate_stage5_${variant}"
  [[ -f "$package" && -x "$candidate" ]] || {
    echo "Missing stage5_${variant} package or executable; package/build it first" >&2
    exit 2
  }
  cp "$package" "$work/stage5_${variant}_submission.zip"
  cp "$candidate" "$work/candidates/estimate_stage5_${variant}"
  docker run --rm --user "$(id -u):$(id -g)" --cpus=1 --memory=2g --memory-swap=2g \
    --mount "type=bind,source=$work,target=/workspace" -w /workspace fpga-ziguang:latest \
    python3 tools/standalone.py --package "stage5_${variant}_submission.zip" \
      --reference-executable "candidates/estimate_stage5_${variant}" \
      --report "stage5_${variant}_standalone.json"
  cp "$work/stage5_${variant}_standalone.json" \
    "$repo/solution/build/local_runs/stage5_${variant}_standalone.json"
done
