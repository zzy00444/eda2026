$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $root
foreach ($trees in @(1, 3, 6, 10)) {
    & .\solution\.venv\Scripts\python.exe -u solution/tools/export_cpp.py `
        --model s3_topology_lr32_511l_25r --trees 25 --fast-trees $trees `
        --refine-distance 100 --fast-categories --output-dir "generated_s5_prefix$trees"
    if ($LASTEXITCODE) { throw "Export prefix $trees failed" }
    wsl -d Ubuntu-20.04 -- bash /mnt/f/EDA/solution/tools/build_candidate_wsl.sh "generated_s5_prefix$trees" pgo
    if ($LASTEXITCODE) { throw "Build prefix $trees failed" }
}
