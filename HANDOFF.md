# EDA 2026 赛题交接

## 目标与入口

这是“面向超大规模 FPGA 的布局布线延时估算算法”实现仓库。接手时先阅读仓库根目录 [`README.md`](README.md) 的环境准备，再看 [`solution/README.md`](solution/README.md) 的接口、评分和算法结构，最后查阅 [`solution/FPGA延时估算第五阶段结果.md`](solution/FPGA延时估算第五阶段结果.md) 的实验依据。

- 提交程序：`solution/src/main.cpp`
- 特征、训练和导出：`solution/tools/`
- 共享路径配置：`solution/tools/project_paths.py`
- 总分优先模型和导出文件：`solution/build/s3_topology_lr32_511l_25r.txt`、`solution/build/generated_s5_score/`
- 高准确度导出文件：`solution/build/generated_s5_accuracy/`
- 当前算法和环境验证证据：`solution/build/stage5_manifest.json`、`solution/build/stage5_*`、`solution/build/stage5_generated_s5_*`
- 镜像构建、压测和 ZIP 独立运行：`solution/tools/build_candidate_wsl.sh`、`solution/tools/validate_stage5_wsl.sh`、`solution/tools/check_stage5_packages_wsl.sh`

Git 有意不保存公开赛题数据、架构表缓存、临时 CSV、ELF、提交 ZIP 和训练缓存。候选模型、C++ 头文件、导出参数及压缩后的验证证据已纳入仓库。

## 当前基线

第五阶段保留两个候选，优化目标是官方总分而非单独追求准确度。公开集固定切分、指标解释和验证细节以第五阶段报告及 `solution/build/stage5_manifest.json` 为准。

| 候选 | 固定验证准确度 | 固定测试准确度 | 10M CPU 均值 | 1B CPU 外推 | 预计总分 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stage5_score`，默认 | 94.9601 | 94.9711 | 3.0286 秒 | 5.0477 分钟 | 93.4531 |
| `stage5_accuracy` | 96.2345 | 96.2257 | 13.1892 秒 | 21.9820 分钟 | 85.9896 |

默认候选使用主 LightGBM 模型 1 棵树；高准确度候选按距离门控使用最多 25 棵。二者都叠加由训练数据拟合的 5 项源端 Net 结构校正，并使用架构下界和打包端口解析。模型推理不依赖运行时 Python、JSON 或外部模型文件。更多方法细节和哈希见第五阶段报告。

上述分数来自本地固定公开集及本机镜像压测的外推，不是官方成绩。已经有迁移前的公开 1M C++/Python 全量一致性、接口测试、10M 五次确定性压测和 ZIP 独立运行记录；完整 1B 官方规模终测和隐藏集评测仍未完成。固定测试切分在开发过程中已多次查看，应作为回归集，不应宣称为未触碰的泛化集。

## 本机前置条件

当前登记的复测主机为 Windows + WSL2 `Ubuntu-20.04` + Docker Desktop。Docker Desktop 的 WSL Integration 要启用该发行版。提供的 `fpga-ziguang:latest` 镜像记录为 Linux/amd64、Ubuntu 22.04.5、GCC 11.4.0、Python 3.12.13；这些版本以 [`solution/README.md`](solution/README.md) 和本机 `F:\EDA\评测环境记录.md` 为参照，接手时需要实际检查。

本仓库位置是 `F:\EDA\solution\eda2026`，WSL 路径为 `/mnt/f/EDA/solution/eda2026`。公开数据默认位于 `F:\EDA\EDA-PANGO-SRB-Arch-Delay\EDA-PANGO-SRB-Arch-Delay`，其下需要 `arch/` 和 `data/delay_estimate_ans.csv`。当前请求样例登记为 `F:\EDA\pango_delay_handover_v3\local_build\request.csv`，需要 `From,To` 两列。请求样例只用于构建期 PGO 和本机接口/压力测试，不是隐藏测试集。

容器执行候选时设置 `--cpus=1 --memory=2g --memory-swap=2g`。Docker 的 `2g` 是 2 GiB；赛题规则的 2 GB 应单独理解。正式规则、评分公式和接口约束以赛题 PDF/Q&A 的整理版 [`solution/README.md`](solution/README.md) 为准。

## 从基线复测

在 PowerShell 进入仓库根目录，先完成 README 中的 WSL/Docker 检查和镜像 shell 冒烟测试，再准备 Python 环境与数据路径：

```powershell
Set-Location F:\EDA\solution\eda2026
$env:EDA_DATA_DIR = (Resolve-Path ..\..\EDA-PANGO-SRB-Arch-Delay\EDA-PANGO-SRB-Arch-Delay).Path
$env:PYTHONPATH = (Resolve-Path solution\tools).Path
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r solution\requirements.txt
```

若已有虚拟环境，无需重复创建；换机器时按实际位置设置 `EDA_DATA_DIR` 和请求文件路径。迁移时 `F:\EDA\solution\.venv` 已存在且版本与本仓库 `solution/requirements.txt` 一致，可直接用 `F:\EDA\solution\.venv\Scripts\python.exe` 代替 `.\\.venv\\Scripts\\python.exe`。

Git 不保存 `solution/build/architecture_meta.json`、`architecture_table.npy`、`architecture_prediction.npy` 和 `stage3_features.pkl`，因此克隆后第一次重建必须先生成周期架构表；跳过这一步会让 `topology_experiment.py`、`export_cpp.py`、`verify_cpp.py` 因缺少架构表而报错。默认总分优先候选的最小重建序列：

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
wsl -d Ubuntu-20.04 -- bash -lc 'EDA_REQUEST_CSV_WSL=/mnt/f/EDA/solution/eda2026/solution/build/local_runs/request_1m.csv bash /mnt/f/EDA/solution/eda2026/solution/tools/build_candidate_wsl.sh generated_s5_score pgo x86-64 stage5_score'
```

请求 CSV 首行不是 `From,To` 时，`build_candidate_wsl.sh` 与 `validate_stage5_wsl.sh` 会通过 `tools/ensure_request_header.sh` 自动补表头；官方输入本身带表头，本地从 `delay_estimate_ans.csv` 裁出的 `From,To` 文件通常不带。`F:\EDA\pango_delay_handover_v3\local_build\request.csv` 就是这种无表头文件，不要直接把它当作已可用样本。

候选 ELF 写入 `solution/bin/estimate_stage5_score`，构建期基准输出写入 `solution/build/local_runs/`。如果重建了模型或 C++，重新运行公开集一致性检查、接口/10M 压测和 ZIP 独立运行检查，不可将旧证据或旧哈希当作新二进制的验证结果。相应命令见根目录 README。

高准确度候选的结构校正和导出参数见第五阶段报告“复现高准确度候选”小节；构建脚本命令将候选目录改为 `generated_s5_accuracy`，产物标识改为 `stage5_accuracy`。

## 已知边界与优化顺序

1. 先在接手机器确认 WSL 发行版、Docker Desktop 集成和镜像标签可用；检查数据 JSON/CSV 与请求样例在 Windows 和 WSL 两侧都可读。未准备这些外部文件时，仓库本身仍可供代码阅读，但训练和端到端验证无法启动。
2. 先按当前参数重建 `stage5_score`，比较新旧导出文件和公开集指标，验证 README 命令与迁移后的目录结构可复现。当前迁移后已做 Python `compileall`、三个 shell 脚本 `bash -n` 和 `git diff --check`；WSL 启动异常导致迁移后的 Docker 编译/压测没有重跑。
3. 以官方总分公式建立候选 Pareto 表，分别记录准确度、单线程 CPU 时间、峰值内存、可执行大小、确定性和验证集/固定测试集指标。默认总分优先候选是性能基准，高准确度候选是精度对照。
4. 优先做可解释的小步实验：主模型树数/距离门控、Net 校正特征与系数、C++ 热路径与端口解析。每次只改变一类因素，系数只用训练集拟合，验证集用于选型；固定测试集用于回归比较并注明已有反复查看。
5. 性能改进达到稳定收益后，再在受限镜像中做多轮大规模压力验证。最终需要官方 1B 规模及隐藏集评测结果才能确认真实排名；10M 线性外推只能用于筛选候选。

避免把本地预测的成绩写成官方成绩，也不要在没有新验证记录的情况下覆盖当前基线证据。新实验应使用独立输出目录，并记录参数、源码/模型哈希和运行环境。
