#!/usr/bin/env bash
# Build and benchmark an isolated candidate in the configured competition image.
set -euo pipefail
candidate=${1:?candidate directory/name required}
profile_mode=${2:-none}
cpu_target=${3:-x86-64}
artifact=${4:-$candidate}
[[ "$candidate" =~ ^[a-zA-Z0-9_]+$ ]] || exit 2
[[ "$cpu_target" == native || "$cpu_target" == x86-64 ]] || exit 2
[[ "$artifact" =~ ^[a-zA-Z0-9_]+$ ]] || exit 2

repo=$(cd "$(dirname "$0")/../.." && pwd)
request=${EDA_REQUEST_CSV_WSL:-}
if [[ -z "$request" || ! -f "$request" ]]; then
  echo "Set EDA_REQUEST_CSV_WSL to a WSL-visible From,To CSV" >&2
  exit 2
fi
candidate_dir="$repo/solution/build/$candidate"
for file in model.hpp architecture.S architecture.bin export.json; do
  [[ -f "$candidate_dir/$file" ]] || { echo "Missing $candidate_dir/$file" >&2; exit 2; }
done

work=${EDA_WSL_WORKDIR:-"$HOME/eda2026-work"}
mkdir -p "$work/src" "$work/generated" "$work/tools" "$work/profile"
cp "$repo/solution/src/main.cpp" "$work/src/"
cp "$candidate_dir"/{model.hpp,architecture.S,architecture.bin,export.json} "$work/generated/"
cp "$repo/solution/tools/benchmark.py" "$work/tools/"
cp "$request" "$work/request.csv"
cd "$work"

docker_args=(--rm --user "$(id -u):$(id -g)" --memory=2g --memory-swap=2g \
  --mount "type=bind,source=$work,target=/workspace" -w /workspace)
if [[ "$profile_mode" == pgo ]]; then
  rm -f profile/*.gcda
  docker run "${docker_args[@]}" fpga-ziguang:latest \
    g++ -O3 -march="$cpu_target" -ffp-contract=off -fprofile-generate=profile \
    -DNDEBUG -std=c++17 -Igenerated src/main.cpp generated/architecture.S -Wa,-Igenerated -o estimate
  docker run "${docker_args[@]}" --cpus=1 fpga-ziguang:latest \
    ./estimate -in /workspace/request.csv -out /dev/null
  docker run "${docker_args[@]}" fpga-ziguang:latest \
    g++ -O3 -march="$cpu_target" -ffp-contract=off -fprofile-use=profile \
    -fprofile-correction -DNDEBUG -std=c++17 -Wall -Wextra \
    -Igenerated src/main.cpp generated/architecture.S -Wa,-Igenerated -o estimate
else
  docker run "${docker_args[@]}" fpga-ziguang:latest \
    g++ -O3 -march="$cpu_target" -ffp-contract=off -flto -DNDEBUG -std=c++17 -Wall -Wextra \
    -Igenerated src/main.cpp generated/architecture.S -Wa,-Igenerated -o estimate
fi
docker run "${docker_args[@]}" --cpus=1 fpga-ziguang:latest \
  python3 tools/benchmark.py --executable ./estimate --input request.csv \
  --output result.csv --report benchmark.json --runs 2

mkdir -p "$repo/solution/bin" "$repo/solution/build/local_runs"
cp estimate "$repo/solution/bin/estimate_$artifact"
cp benchmark.json "$repo/solution/build/local_runs/${candidate}_benchmark.json"
cp result.csv "$repo/solution/build/local_runs/${candidate}_result.csv"
printf 'ELF: %s\nBenchmark: %s\n' \
  "$repo/solution/bin/estimate_$artifact" \
  "$repo/solution/build/local_runs/${candidate}_benchmark.json"
