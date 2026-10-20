#!/usr/bin/env bash
# Interface checks plus a 5-run 10M stress test for each named candidate.
# Defaults to the two stage5 artefacts; pass other candidate names (the string
# that build_candidate_wsl.sh received as its artifact argument) to validate
# experimental builds the same way.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
source "$(dirname "$0")/ensure_request_header.sh"
request=${EDA_REQUEST_CSV_WSL:-}
if [[ -z "$request" || ! -f "$request" ]]; then
  echo "Set EDA_REQUEST_CSV_WSL to a WSL-visible From,To CSV" >&2
  exit 2
fi
work=${EDA_WSL_WORKDIR:-"$HOME/eda2026-work"}
mkdir -p "$work/tools" "$work/candidates" "$repo/solution/build/local_runs"
cp "$repo/solution/tools"/{benchmark.py,stress.py,integration.py} "$work/tools/"
cp "$request" "$work/request.csv"
ensure_request_header "$work/request.csv"
candidates=("$@")
if [[ ${#candidates[@]} -eq 0 ]]; then
  candidates=(stage5_score stage5_accuracy)
fi
for candidate in "${candidates[@]}"; do
  binary="$repo/solution/bin/estimate_$candidate"
  [[ -x "$binary" ]] || { echo "Build $candidate first with build_candidate_wsl.sh" >&2; exit 2; }
  cp "$binary" "$work/candidates/$candidate"
  chmod +x "$work/candidates/$candidate"
  docker run --rm --user "$(id -u):$(id -g)" --cpus=1 --memory=2g --memory-swap=2g \
    --mount "type=bind,source=$work,target=/workspace" -w /workspace fpga-ziguang:latest \
    python3 tools/integration.py --executable "candidates/$candidate" --report "$candidate-integration.json"
  docker run --rm --user "$(id -u):$(id -g)" --cpus=1 --memory=2g --memory-swap=2g \
    --mount "type=bind,source=$work,target=/workspace" -w /workspace fpga-ziguang:latest \
    "./candidates/$candidate" -in request.csv -out "$candidate-result-1m.csv"
  docker run --rm --user "$(id -u):$(id -g)" --cpus=1 --memory=2g --memory-swap=2g \
    --mount "type=bind,source=$work,target=/workspace" -w /workspace fpga-ziguang:latest \
    python3 tools/stress.py --copies 10 --input request.csv --expanded "$candidate-request-10m.csv" \
      --executable "./candidates/$candidate" --output "$candidate-result-10m.csv" \
      --reference "$candidate-result-1m.csv" --report "$candidate-benchmark-10m.json" --runs 5
  cp "$work/$candidate-integration.json" "$repo/solution/build/local_runs/${candidate}_integration.json"
  cp "$work/$candidate-benchmark-10m.json" "$repo/solution/build/local_runs/${candidate}_benchmark_10m.json"
done
