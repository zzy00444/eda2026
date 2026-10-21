# EDA 2026 FPGA 延时估算

本仓库管理“面向超大规模 FPGA 的布局布线延时估算算法”赛题实现、实验报告和本机复测流程。当前推荐候选是第七阶段默认版本（残差模型折叠进架构表，运行时只有一次查表），C++ 源码位于 [`solution/src`](solution/src)，训练和验证脚本位于 [`solution/tools`](solution/tools)，阶段报告位于 [`solution`](solution)。

## 仓库内容

- `solution/src`：可提交的 C++ 估算程序。
- `solution/tools`：模型训练、特征导出、验证、打包、镜像编译和归档脚本。
- `solution/build`：纳入版本管理的第五至第七阶段模型、C++ 模型头文件、导出参数和关键验证证据。架构表可从官方架构 JSON 重新生成；缓存、生成架构表、临时结果、ELF 和提交 ZIP 不纳入 Git。
- `solution/FPGA延时估算*阶段结果.md`：阶段实验记录和当前方案说明。
- `solution/实现对比与适配评估.md`：与 Pango 实现的算法比较及镜像适配评估。
- [`HANDOFF.md`](HANDOFF.md)：交接入口，列出当前基线、复测步骤、未完成验证和后续优化方向。

官方架构 JSON 和公开数据集由竞赛资料单独提供，不复制进此仓库。默认期望其位于 `F:\EDA\EDA-PANGO-SRB-Arch-Delay\EDA-PANGO-SRB-Arch-Delay`，即从仓库根目录的 `..\..\EDA-PANGO-SRB-Arch-Delay\EDA-PANGO-SRB-Arch-Delay`；也可用 `EDA_DATA_DIR` 指定该目录。需要端到端压测时，还需准备只有 `From,To` 两列的请求 CSV，并在 WSL 构建命令中通过 `EDA_REQUEST_CSV_WSL` 指定其 `/mnt/...` 路径。这些外部数据和本地路径不应提交到 Git。

## 当前第七阶段结果

第七阶段把残差模型**离线折叠进架构表**：每个 `(方向, 源端口, dx%10, dy%12, 目标端口)` 表项 = 周期下界 + 训练修正。运行时不再有决策树和特征计算，只剩一次查表、线性项、Gap 前缀和 5 项 Net 结构校正。准确度小幅下降，CPU 相对第六阶段再降 39%（相对第五阶段 −57%）。

| 候选 | 固定验证 | 固定测试 | 10M CPU 均值 | 1B 外推 | 预计总分 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 第七阶段默认（折叠） | 94.6212 | 94.6074 | 1.3498 秒 | 2.25 分钟 | **94.56** |
| 第六阶段默认 | 94.9601 | 94.9711 | 2.2244 秒 | 3.71 分钟 | 94.12 |
| 第五阶段默认（同机复测） | 94.9601 | 94.9711 | 3.1562 秒 | 5.26 分钟 | 93.35 |

折叠的代价是修正只能是键的函数：绝对位置、连续位移、`gap_x/gap_y`、`remaining_ax_mod` 和真包络都被排除；换来的是模型容量免费（300 棵树不再占运行时间）。给键增加“网络位移是否超过位移”的比特已经测过：未加 Net 校正的测试分只从 94.1816 提高到 94.3784（+0.197），并非早期估计的 +0.6，最终未采用。详见 [`FPGA延时估算第七阶段结果.md`](solution/FPGA延时估算第七阶段结果.md) 与 `solution/build/stage7_manifest.json`。

94.61 是当前特征、训练和表表示下的实测结果，不能视为赛题的准确度上限。2026-10-21 同机对照已完成：第七阶段与 Pango 未校准版测试准确度为 94.6074/97.5955，10M CPU 为 1.7310/2.3186 秒，预计总分为 94.2434/96.1442。Pango 准确度及预计总分更优，第七阶段更快；本轮与历史测时分开保留。完整位移低秩表示、资产核查与后续实验顺序见 [`实现对比与适配评估.md`](solution/实现对比与适配评估.md)，原始记录见 [`docs/pango_comparison_20261021.json`](docs/pango_comparison_20261021.json)。

## 第六阶段历史结果

第六阶段只改运行时与端口解析，模型、Net 结构校正和架构表不变，1M 与 10M 输出与第五阶段逐字节相同，因此准确度不变而 CPU 明显下降。同机交错 10M 对照：第五阶段产物 3.1562 秒、第六阶段 2.1990 秒，**CPU 降低 30.3%**；第六阶段单独 10M 五次均值 2.2244 秒，1B 外推约 3.71 分钟，时间分 87.6，预计总分 **94.12**（第五阶段同机复测约 93.35，记录值 93.45）。

第六阶段同时确认了两个方向性结论：继续加树不划算（2 棵树需买回 0.32 分准确度才抵平，实测第 2 棵树约 0.30 且递减），而低成本的确定性结构校正性价比约为加树的 10 倍（5 项 Net 校正花 11 ns 换回 1.19 分准确度）。详见 [`FPGA延时估算第六阶段结果.md`](solution/FPGA延时估算第六阶段结果.md) 与 `build/stage6_manifest.json`。

## 第五阶段历史结果

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

本机已有虚拟环境为 `F:\EDA\solution\.venv`；可直接将下文 `.\.venv\Scripts\python.exe` 替换为 `F:\EDA\solution\.venv\Scripts\python.exe`，无需再创建一份。

镜像构建脚本需要 Linux 可见的请求文件路径。当前请求样例在 WSL 中为 `/mnt/f/EDA/pango_delay_handover_v3/local_build/request.csv`；构建命令通过 `EDA_REQUEST_CSV_WSL` 显式传给脚本。该样例没有 `From,To` 表头（从答案 CSV 裁列时被去掉），`build_candidate_wsl.sh` 和 `validate_stage5_wsl.sh` 会用 `solution/tools/ensure_request_header.sh` 自动补齐；带表头的文件不会被改动。也可以改用 `solution/build/local_runs/request_1m.csv`，该文件由 `delay_estimate_ans.csv` 重新生成并保留表头。

新克隆没有请求缓存时，在设置 `EDA_DATA_DIR` 后生成下文使用的带表头文件：

```powershell
.\.venv\Scripts\python.exe -c "import os; from pathlib import Path; import pandas as pd; p=Path('solution/build/local_runs/request_1m.csv'); p.parent.mkdir(parents=True, exist_ok=True); pd.read_csv(Path(os.environ['EDA_DATA_DIR'])/'data'/'delay_estimate_ans.csv', usecols=['From','To']).to_csv(p, index=False)"
```

### 重建当前候选

当前默认是 `stage7_fold`。已跟踪的 `foldable_lr08_300r.txt` 和 `stage7_net_correction_folded.json` 足以重建折叠表，不必先重训第五、六阶段模型。完成数据和 Python 准备后执行：

```powershell
.\.venv\Scripts\python.exe -u solution\tools\architecture.py
.\.venv\Scripts\python.exe -u solution\tools\fold_model.py `
  --model foldable_lr08_300r --trees 300 --out generated_s7_fold `
  --net-coefficients solution\build\stage7_net_correction_folded.json
wsl -d Ubuntu-20.04 -- env EDA_REQUEST_CSV_WSL=/mnt/f/EDA/solution/eda2026/solution/build/local_runs/request_1m.csv bash /mnt/f/EDA/solution/eda2026/solution/tools/build_candidate_wsl.sh generated_s7_fold pgo x86-64 stage7_fold
```

训练输出和导出目录均位于 `solution/build`。镜像构建脚本从仓库内读取 C++ 和生成文件，使用 `EDA_REQUEST_CSV_WSL` 提供 PGO 训练及基准请求；该变量须是 WSL 可见的 `/mnt/...` 路径。脚本只构建到隔离候选目录，不覆盖提交包。上面的请求文件路径是这台工作机目前已有的本地请求样例，换机器时请改为对应文件。

`verify_cpp.py` 对应第五、六阶段的树模型，不能直接校验第七阶段的折叠表。第七阶段需要按折叠键和表量化规则重算 Python 参考值，再与新 ELF 的百万行输出核对；已归档的指标不能替代新产物的验证。构建完成后，接口与 10M 五次压测按候选名执行：

```powershell
wsl -d Ubuntu-20.04 -- env EDA_REQUEST_CSV_WSL=/mnt/f/EDA/solution/eda2026/solution/build/local_runs/request_1m.csv bash /mnt/f/EDA/solution/eda2026/solution/tools/validate_stage5_wsl.sh stage7_fold
```

`release_stage7.py` 读取 `build/stage7_fold_metrics.json`、`stage7_fold_benchmark_10m.json` 和 `stage7_fold_integration.json`；验证脚本的新报告先写入 `build/local_runs`，发布前必须核对新结果并归档到上述文件。发布脚本检查五次一致性、资源与预计总分，但不会自动重算准确度，也不会替代提交 ZIP 的独立运行检查：

```powershell
.\.venv\Scripts\python.exe -u solution\tools\release_stage7.py
```

完整重训流程见第七阶段报告。第五、六阶段的复现命令保留在各阶段报告中，重建历史候选时使用独立目录，避免覆盖默认提交。

压测报告保存在 `solution/build/local_runs`。上述流程验证接口、资源和重复输出；它不执行完整 1B 官方隐藏集评测。

## 赛题资料

请先阅读 [`solution/README.md`](solution/README.md) 了解接口、评分和约束，再查看第七阶段报告。提交包只需包含 `bin/estimate`，上限为 100 MB；不能把本地测试成绩或 1B 时间外推写成官方成绩。

2026-10-21 已清理旧版 ELF、重复架构表、预测 CSV 和训练特征缓存。保留当前第七阶段产物、基础架构表、模型和报告；第五、六阶段的旧二进制及预测 CSV 可能需要重建。原始赛题数据、Python 环境和 Docker 镜像均保留。
