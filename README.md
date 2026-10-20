# EDA 2026 FPGA 延时估算

本仓库管理“面向超大规模 FPGA 的布局布线延时估算算法”赛题实现、实验报告和本机复测流程。当前推荐候选是第五阶段总分优先版本，C++ 源码位于 [`solution/src`](solution/src)，训练和验证脚本位于 [`solution/tools`](solution/tools)，阶段报告位于 [`solution`](solution)。

## 仓库内容

- `solution/src`：可提交的 C++ 估算程序。
- `solution/tools`：模型训练、特征导出、验证、打包、镜像编译和归档脚本。
- `solution/build`：纳入版本管理的第五阶段模型、C++ 模型头文件、导出参数和关键验证证据。架构表可从官方架构 JSON 重新生成；缓存、生成架构表、临时结果、ELF 和提交 ZIP 不纳入 Git。
- `solution/FPGA延时估算*阶段结果.md`：阶段实验记录和当前方案说明。
- `solution/实现对比与适配评估.md`：与 Pango 实现的算法比较及镜像适配评估。
- [`HANDOFF.md`](HANDOFF.md)：交接入口，列出当前基线、复测步骤、未完成验证和后续优化方向。

官方架构 JSON 和公开数据集由竞赛资料单独提供，不复制进此仓库。默认期望其位于 `F:\EDA\EDA-PANGO-SRB-Arch-Delay\EDA-PANGO-SRB-Arch-Delay`，即从仓库根目录的 `..\..\EDA-PANGO-SRB-Arch-Delay\EDA-PANGO-SRB-Arch-Delay`；也可用 `EDA_DATA_DIR` 指定该目录。需要端到端压测时，还需准备只有 `From,To` 两列的请求 CSV，并在 WSL 构建命令中通过 `EDA_REQUEST_CSV_WSL` 指定其 `/mnt/...` 路径。这些外部数据和本地路径不应提交到 Git。

## 当前第五阶段结果

README 与阶段报告记录了本地固定公开集指标、10M 单线程 CPU 实测及外推。默认版本本地测试准确度为 94.9711，1B CPU 外推约 5.05 分钟，预计总分约 93.45；高准确度候选分别约为 96.2257、21.98 分钟和 85.99。完整 1B 运行和官方隐藏集验证尚未完成，这些数值不是官方成绩。

## 本机镜像

本机已验证的环境记录见仓库外的 `F:\EDA\评测环境记录.md`：Windows、WSL2 `Ubuntu-20.04`、Docker Desktop WSL 集成、Linux/amd64 镜像 `fpga-ziguang:latest`，容器内 Ubuntu 22.04.5、GCC 11.4.0、Python 3.12.13。官方约束是单线程、内存不超过 2 GB；本地 Docker 使用 `--cpus=1 --memory=2g --memory-swap=2g`，其中 `2g` 实际为 2 GiB（2,147,483,648 字节）。本机镜像与发行版状态可能变化，接手时先按以下步骤检查。

### 启动 WSL 与 Docker

1. 启动 Docker Desktop，等待 Engine 显示运行中。在 Docker Desktop 的 **Settings → Resources → WSL Integration** 中启用 `Ubuntu-20.04`。
2. 在 PowerShell 检查 WSL2 发行版和 Docker Desktop 集成：

```powershell
wsl --status
wsl --list --verbose
wsl -d Ubuntu-20.04 -- docker version
wsl -d Ubuntu-20.04 -- docker image inspect fpga-ziguang:latest
```

若 `docker image inspect` 提示镜像不存在，且本机有归档文件，再导入镜像：

```powershell
wsl -d Ubuntu-20.04 -- docker load -i /mnt/f/EDA/fpga-ziguang-latest.tar
wsl -d Ubuntu-20.04 -- docker image inspect fpga-ziguang:latest
```

可启动受限交互容器检查镜像工具链。进入容器 shell 后可执行 `g++ --version`、`python3 --version`、`cat /etc/os-release`；退出使用 `exit`：

```powershell
wsl -d Ubuntu-20.04 -- docker run --rm -it --cpus=1 --memory=2g --memory-swap=2g --entrypoint /bin/bash fpga-ziguang:latest
```

本仓库位于 Windows `F:\EDA\solution\eda2026`，在 WSL 中对应 `/mnt/f/EDA/solution/eda2026`。构建脚本会从脚本自身位置定位仓库，并在 WSL 的 `$HOME/eda2026-work` 建立临时工作目录；不要从 Windows Python 直接调用 Linux ELF。构建与压测通过 WSL shell 调起 Docker Desktop 容器。

### 准备数据与 Python

从 PowerShell 进入仓库根目录。外部赛题资料默认位于 `F:\EDA\EDA-PANGO-SRB-Arch-Delay\EDA-PANGO-SRB-Arch-Delay`，必须包含 `arch\SRB_*.json` 和 `data\delay_estimate_ans.csv`。端到端构建/压测还需要 `From,To` 两列的请求文件；当前本机样例为 `F:\EDA\pango_delay_handover_v3\local_build\request.csv`。换机器时设置为实际路径，且确保它在 WSL 中可通过 `/mnt/...` 访问。

```powershell
Set-Location F:\EDA\solution\eda2026
Test-Path ..\..\EDA-PANGO-SRB-Arch-Delay\EDA-PANGO-SRB-Arch-Delay\arch\SRB_Arc.json
Test-Path ..\..\EDA-PANGO-SRB-Arch-Delay\EDA-PANGO-SRB-Arch-Delay\data\delay_estimate_ans.csv
Test-Path F:\EDA\pango_delay_handover_v3\local_build\request.csv
$env:EDA_DATA_DIR = (Resolve-Path ..\..\EDA-PANGO-SRB-Arch-Delay\EDA-PANGO-SRB-Arch-Delay).Path
$env:PYTHONPATH = (Resolve-Path solution\tools).Path
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r solution\requirements.txt
```

镜像构建脚本需要 Linux 可见的请求文件路径。当前请求样例在 WSL 中为 `/mnt/f/EDA/pango_delay_handover_v3/local_build/request.csv`；构建命令通过 `EDA_REQUEST_CSV_WSL` 显式传给脚本。

### 重建当前候选

完成上面的数据和 Python 准备后，从仓库根目录运行训练、导出及镜像构建：

```powershell
.\.venv\Scripts\python.exe -u solution\tools\architecture.py

.\.venv\Scripts\python.exe -u solution\tools\topology_experiment.py `
  --leaves 511 --rounds 25 --learning-rate 0.32 --prefix s3_topology_lr32
.\.venv\Scripts\python.exe -u solution\tools\net_correction_experiment.py `
  --short-trees 1 --fast-trees 1 --cascade-trees 0 --l1-only `
  --output-dir net_correction_f1_c0
.\.venv\Scripts\python.exe -u solution\tools\export_cpp.py `
  --model s3_topology_lr32_511l_25r --trees 1 --fast-categories --packed-ports `
  --net-correction net_correction_f1_c0/metrics.json --output-dir generated_s5_score
wsl -d Ubuntu-20.04 -- bash -lc 'EDA_REQUEST_CSV_WSL=/mnt/f/EDA/pango_delay_handover_v3/local_build/request.csv bash /mnt/f/EDA/solution/eda2026/solution/tools/build_candidate_wsl.sh generated_s5_score pgo x86-64 stage5_score'
```

训练输出和导出目录均位于 `solution/build`。镜像构建脚本从仓库内读取 C++ 和生成文件，使用 `EDA_REQUEST_CSV_WSL` 提供 PGO 训练及基准请求；该变量须是 WSL 可见的 `/mnt/...` 路径。脚本只构建到隔离候选目录，不覆盖提交包。上面的请求文件路径是这台工作机目前已有的本地请求样例，换机器时请改为对应文件。

全公开集 C++/Python 一致性验证：

```powershell
.\.venv\Scripts\python.exe -u solution\tools\verify_cpp.py `
  --result solution\build\local_runs\stage5_score_result.csv `
  --generated-dir generated_s5_score --name stage5_score --all-rows
```

`verify_cpp.py` 需要先用公开请求运行候选 ELF，生成预测 CSV。两个候选构建完成后，可依次打包、执行镜像内接口与 10M 五次压测，再验证 ZIP 解压后的独立运行：

```powershell
.\.venv\Scripts\python.exe solution\tools\package.py --binary solution\bin\estimate_stage5_score --output solution\stage5_score_submission.zip
.\.venv\Scripts\python.exe solution\tools\package.py --binary solution\bin\estimate_stage5_accuracy --output solution\stage5_accuracy_submission.zip
$env:EDA_REQUEST_CSV_WSL = "/mnt/f/EDA/pango_delay_handover_v3/local_build/request.csv"
wsl -d Ubuntu-20.04 -- bash -lc 'EDA_REQUEST_CSV_WSL=/mnt/f/EDA/pango_delay_handover_v3/local_build/request.csv bash /mnt/f/EDA/solution/eda2026/solution/tools/validate_stage5_wsl.sh'
wsl -d Ubuntu-20.04 -- bash /mnt/f/EDA/solution/eda2026/solution/tools/check_stage5_packages_wsl.sh
```

压测报告保存在 `solution/build/local_runs`。上述流程验证接口、资源和重复输出；它不执行完整 1B 官方隐藏集评测。

## 赛题资料

请先阅读 [`solution/README.md`](solution/README.md) 了解接口、评分和约束，再查看第五阶段报告。提交包只需包含 `bin/estimate`，上限为 100 MB；不能把本地测试成绩或 1B 时间外推写成官方成绩。
