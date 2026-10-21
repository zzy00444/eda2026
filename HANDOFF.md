# EDA 2026 赛题交接

## 目标与入口

这是“面向超大规模 FPGA 的布局布线延时估算算法”实现仓库。接手时先阅读仓库根目录 [`README.md`](README.md) 的环境准备，再看 [`solution/README.md`](solution/README.md) 的接口、评分和算法结构，最后查阅[第七阶段报告](solution/FPGA延时估算第七阶段结果.md)及[实现比较](solution/实现对比与适配评估.md)。本交接说明更新于 2026-10-21。

- 提交程序：`solution/src/main.cpp`
- 特征、训练和导出：`solution/tools/`
- 共享路径配置：`solution/tools/project_paths.py`
- 默认模型和导出文件：`solution/build/foldable_lr08_300r.txt`、`solution/build/generated_s7_fold/`
- 默认 Net 系数：`solution/build/stage7_net_correction_folded.json`
- 第五、六阶段历史主模型：`solution/build/s3_topology_lr32_511l_25r.txt`
- 高准确度导出文件：`solution/build/generated_s5_accuracy/`
- 当前算法和环境验证证据：`solution/build/stage7_manifest.json`、`solution/build/stage7_fold_*`
- 镜像构建和压测：`solution/tools/build_candidate_wsl.sh`、`solution/tools/validate_stage5_wsl.sh stage7_fold`
- 独立运行检查工具：`solution/tools/standalone.py`；`check_stage5_packages_wsl.sh` 是第五阶段专用包装脚本

Git 有意不保存公开赛题数据、架构表缓存、临时 CSV、ELF、提交 ZIP 和训练缓存。候选模型、C++ 头文件、导出参数及压缩后的验证证据已纳入仓库。

## 当前基线

第七阶段的默认候选把残差模型折叠进架构表，运行时只有一次查表；相对第五阶段同机复测 CPU 降低约 57%，预计总分提高约 1.21。指标解释与验证细节以[第七阶段报告](solution/FPGA延时估算第七阶段结果.md)、[第六阶段报告](solution/FPGA延时估算第六阶段结果.md)和 `solution/build/stage7_manifest.json`、`stage6_manifest.json` 为准。

| 候选 | 固定验证准确度 | 固定测试准确度 | 10M CPU 均值 | 1B CPU 外推 | 预计总分 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stage7_fold`，默认 | 94.6212 | 94.6074 | 1.3498 秒 | 2.25 分钟 | **94.5611** |
| `stage6_score` | 94.9601 | 94.9711 | 2.2244 秒 | 3.71 分钟 | 94.1233 |
| `stage5_score`（同机交错复测） | 94.9601 | 94.9711 | 3.1562 秒 | 5.26 分钟 | 93.3471 |
| `stage5_accuracy` | 96.2345 | 96.2257 | 13.1892 秒 | 21.9820 分钟 | 85.9896 |

默认候选的运行时是一张 28,753,920 项的 int16 表（键 = 方向 × 源端口 × 位移余数 × 目标端口），表项已含训练修正；随后叠加线性空间项、Gap 前缀和 5 项 Net 结构校正。第六阶段的"树 + 特征"实现保留在同一份 `main.cpp` 的 `#else` 分支里，由 `model.hpp` 的 `FOLDED_MODEL` 宏切换。

三条决策依据：一棵 511 叶树约 31 ns（1B 约 0.26 分），需买回 0.32 分准确度才划算，实测第 2 棵树只有约 0.30 且递减，堆树不划算；5 项 Net 结构校正只花 11 ns 换回 1.19 分；把模型折进表里之后容量免费，代价只剩"修正必须是键的函数"，实测该限制损失 1.9 分准确度（相对 25 棵树），再用结构校正补回 0.43 分。

上述分数来自本地固定公开集及本机镜像压测的外推，不是官方成绩。已经有公开 1M C++/Python 全量一致性、接口测试、10M 五次确定性压测和 ZIP 独立运行记录；完整 1B 官方规模终测和隐藏集评测仍未完成。固定测试切分在开发过程中已多次查看，应作为回归集，不应宣称为未触碰的泛化集。

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

Git 不保存 `solution/build/architecture_meta.json`、`architecture_table.npy`、`architecture_prediction.npy` 和 `stage3_features.pkl`。克隆后先生成周期架构表；从已跟踪模型重建当前默认候选的最小序列如下。请求文件需提前准备为 `From,To` 两列，路径可替换为本机实际样例：

```powershell
.\.venv\Scripts\python.exe -u solution\tools\architecture.py
.\.venv\Scripts\python.exe -u solution\tools\fold_model.py `
  --model foldable_lr08_300r --trees 300 --out generated_s7_fold `
  --net-coefficients solution\build\stage7_net_correction_folded.json
wsl -d Ubuntu-20.04 -- env EDA_REQUEST_CSV_WSL=/mnt/f/EDA/solution/eda2026/solution/build/local_runs/request_1m.csv bash /mnt/f/EDA/solution/eda2026/solution/tools/build_candidate_wsl.sh generated_s7_fold pgo x86-64 stage7_fold
wsl -d Ubuntu-20.04 -- env EDA_REQUEST_CSV_WSL=/mnt/f/EDA/solution/eda2026/solution/build/local_runs/request_1m.csv bash /mnt/f/EDA/solution/eda2026/solution/tools/validate_stage5_wsl.sh stage7_fold
```

请求 CSV 首行不是 `From,To` 时，`build_candidate_wsl.sh` 与 `validate_stage5_wsl.sh` 会通过 `tools/ensure_request_header.sh` 自动补表头；官方输入本身带表头，本地从 `delay_estimate_ans.csv` 裁出的 `From,To` 文件通常不带。`F:\EDA\pango_delay_handover_v3\local_build\request.csv` 就是这种无表头文件，脚本会补表头，但更推荐直接用带表头的 `solution/build/local_runs/request_1m.csv`。

候选 ELF 写入 `solution/bin/estimate_stage7_fold`，构建期结果及验证报告写入 `solution/build/local_runs/`。如果重建了模型或 C++，应重新核对折叠表量化后的 Python 参考预测、准确度、接口和五次 10M 输出。树模型专用 `verify_cpp.py` 不能直接验证折叠模型。

核验新结果后，将指标与压测报告归档到 `build/stage7_fold_metrics.json`、`stage7_fold_benchmark_10m.json` 和 `stage7_fold_integration.json`，再运行 `release_stage7.py`。该脚本读取这些归档证据，检查一致性、资源和预计总分并打包；它不会自动计算新准确度或执行 ZIP 独立运行，不能直接沿用旧证据。

需要完整重训时，按[第七阶段报告](solution/FPGA延时估算第七阶段结果.md)执行 `train_foldable.py` → `fit_netcorr_folded.py` → `fold_model.py`。仅重建现有候选时可以跳过重训；`architecture.bin` 不入库。

高准确度候选的结构校正和导出参数见第五阶段报告“复现高准确度候选”小节；构建脚本命令将候选目录改为 `generated_s5_accuracy`，产物标识改为 `stage5_accuracy`。

## 已知边界与优化顺序

1. 先在接手机器确认 WSL 发行版、Docker Desktop 集成和镜像标签可用；检查数据 JSON/CSV 与请求样例在 Windows 和 WSL 两侧都可读。未准备这些外部文件时，仓库本身仍可供代码阅读，但训练和端到端验证无法启动。
2. 先核对当前 `stage7_fold` 的产物与清单。2026-10-21 清理保留了当前 ELF、ZIP、折叠表、基础架构表和全部受 Git 跟踪的文件；旧 ELF、重复表及预测 CSV 已删除，`stage3_features.pkl` 等缓存需按需再生。
3. Pango 完整目录为 `F:\EDA\pango_delay_handover_v3`，已包含模型、提交件和本机重编产物；旧 lite 目录缺表的结论不能套用。同机比较已完成，见[原始记录](docs/pango_comparison_20261021.json)：第七阶段/Pango 未校准版测试 94.6074/97.5955，10M CPU 1.7310/2.3186 秒，预计总分 94.2434/96.1442。下一阶段采用 Pango 为准确度/总分参照、第七阶段为速度参照；历史清单不覆盖，官方成绩与本地外推分列。Pango 校准版使用全部公开标签，随机切片不构成干净留出。
4. 下一阶段优先研究完整位移与端口对的紧凑表示、短路径结构和位置相关误差。当前折叠键上的加树、1023 叶、超调比特扩表等已测实验不支持继续堆容量，但这些负结果不构成新表示方法的上限。
5. 性能改进达到稳定收益后，再在受限镜像中做多轮大规模压力验证。最终需要官方 1B 规模及隐藏集评测结果才能确认真实排名；10M 线性外推只能用于筛选候选。

30 分钟是时间分归零点；任一次超过 1 小时或超过 2 GB 才触发对应硬限制。避免把本地预测写成官方成绩。新实验使用独立目录，记录参数、源码/模型哈希和环境；系数用训练集拟合，选型使用验证集，并保留新的位置分组留出检查。
