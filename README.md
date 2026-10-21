# EDA 2026 FPGA 延时估算

2026-10-21 已完成[第九阶段计划](docs/第九阶段98分优化计划.md)：固定验证及测试回归准确度均超过 98。当前默认为 `stage9_fast16`；按用户要求，额外保留本轮已验收的最高精度版 `stage9_accurate64`，并保留第八阶段速度方案。

## 产物与结果

下表为同机、相同输入、正反交错五轮 10M 的 CPU 均值。准确度是赛题相对误差得分；固定 rng7 切片已在历史开发中查看。全量 1B 和官方隐藏集尚未评测，1B 用时及总分均为外推。

| 方案 | 验证准确度 | 测试回归准确度 | 10M CPU | 1B CPU 外推 | 预计总分 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 第八阶段速度方案 | 97.638770 | 97.650829 | 1.886051 s | 3.143 分钟 | **96.548954** |
| **第九阶段默认 16 树** | 98.021374 | 98.031248 | 7.630726 s | 12.718 分钟 | 92.066060 |
| **额外保留的高精度 64 树** | **98.057516** | **98.069069** | 16.872524 s | **28.121 分钟** | 84.394818 |

准确度提升带来明显 CPU 成本，预计综合分仍以第八阶段更高。默认 16 树满足本轮超过 98 的目标；高精度包额外保留，本机五轮每次外推均小于 30 分钟，范围为 27.760～28.599 分钟。远程平台存在用时波动，需另行实测。

| 用途 | 提交包 | ZIP 字节 | 发布证据 |
| --- | --- | ---: | --- |
| 默认，超过 98 | [submission.zip](solution/submission.zip)，同 [stage9_fast16_submission.zip](solution/stage9_fast16_submission.zip) | 98,376,571 | [默认清单](docs/stage9_manifest.json) |
| 本轮已验收最高精度，额外保留 | [stage9_accurate64_submission.zip](solution/stage9_accurate64_submission.zip) | 98,540,421 | [高精度清单](docs/stage9_accuracy_manifest.json) |
| 速度与综合分参照 | [displacement_cal16_submission.zip](solution/displacement_cal16_submission.zip) | 80,797,119 | [第八阶段历史清单](docs/stage8_manifest.json) |

两个新包均只包含 `bin/estimate`，通过百万行 Python/C++ 零差异、五次 10M 输出一致与重复内容核对、11 项接口、11 项额外错误/边界及 ZIP 空目录独立运行。容器峰值分别为 649,625,600 / 649,138,176 B，观察线程数 1；容器内存限额精确为 2,000,000,000 B。见[最终基准](docs/stage9_final_benchmark.json)、[16 树验收](docs/stage9_fast16_validation.json)和[64 树验收](docs/stage9_accurate64_validation.json)。

新增残差的源坐标留出准确度为 98.046356 / 98.087727；继承的第八阶段参数曾使用这些位置，所以该检查仅隔离本轮新增参数。48 条保留位置精确查询中，第八阶段为 88.050655，16 树为 87.936545，64 树为 88.578985，仍有结构误差。这些结果不能证明隐藏集达到 98，详见[泛化检查](docs/stage9_generalization.json)。

## 环境与数据

本机为 Windows、WSL2 `Ubuntu-20.04`、Docker Desktop，镜像 `fpga-ziguang:latest`，登记工具链 GCC 11.4.0、Linux/amd64。Docker Desktop 须启用该发行版的 WSL Integration；镜像含新运行时需要的 zlib。

```powershell
wsl --list --verbose
wsl -d Ubuntu-20.04 -- docker version
wsl -d Ubuntu-20.04 -- docker image inspect fpga-ziguang:latest
```

若镜像缺失且本机有归档，可执行 `wsl -d Ubuntu-20.04 -- docker load -i /mnt/f/EDA/fpga-ziguang-latest.tar`。环境历史在仓库外 `F:\EDA\评测环境记录.md`。

官方数据不入 Git，默认 `F:\EDA\EDA-PANGO-SRB-Arch-Delay\EDA-PANGO-SRB-Arch-Delay`，应含 `arch/SRB_*.json` 和 `data/delay_estimate_ans.csv`；也可设置 `EDA_DATA_DIR`。本机 Windows Python 为 `F:\EDA\solution\.venv\Scripts\python.exe`，依赖见 `solution/requirements.txt`。新环境可执行 `python -m venv .venv`，再用其中的 Python 安装依赖。

新克隆没有请求缓存时，从仓库根目录生成：

```powershell
$env:EDA_DATA_DIR = 'F:\EDA\EDA-PANGO-SRB-Arch-Delay\EDA-PANGO-SRB-Arch-Delay'
& F:\EDA\solution\.venv\Scripts\python.exe -c "import os; from pathlib import Path; import pandas as pd; p=Path('solution/build/local_runs/request_1m.csv'); p.parent.mkdir(parents=True, exist_ok=True); pd.read_csv(Path(os.environ['EDA_DATA_DIR'])/'data/delay_estimate_ans.csv', usecols=['From','To']).to_csv(p,index=False)"
```

## 重建与验收

源码为 [displacement.cpp](solution/src/displacement.cpp) 和 [残差运行时](solution/src/displacement_residual_runtime.hpp)。第八阶段冻结资产来自用户提供的 Pango 工程，见[导入指纹](docs/displacement_import.json)和[上一轮参数重建](docs/完整位移模型优化记录.md)。外部工程仍有人修改，本轮只使用仓库已冻结资产，不改动外部工程或赛题附件。

Git 不包含 ELF、ZIP、二进制模型、公开数据和大缓存，不能仅凭克隆直接构建。已有基础模型、几何及预测缓存时，按[第九阶段记录](docs/第九阶段优化记录.md)训练、无损压缩包络并导出；缺少周期架构表时先运行 `solution/tools/architecture.py`。完整复现和两候选验收命令都在该记录中。快速重建默认：

```powershell
wsl -d Ubuntu-20.04 -- python3 /mnt/f/EDA/solution/eda2026/solution/tools/displacement_run.py build stage9_fast16 --u /mnt/f/EDA/solution/eda2026/solution/build/displacement/U_refit_q16.bin --m /mnt/f/EDA/solution/eda2026/solution/build/displacement/M_refit.bin --calibration /mnt/f/EDA/solution/eda2026/solution/build/displacement/calibration_train.bin --residual-header /mnt/f/EDA/solution/eda2026/solution/build/displacement/stage9_fast16.hpp --bound-file /mnt/f/EDA/solution/eda2026/solution/build/displacement/stage9_bound.z --define EST_RESIDUAL_BATCH
wsl -d Ubuntu-20.04 -- python3 /mnt/f/EDA/solution/eda2026/solution/tools/displacement_run.py run stage9_fast16
& F:\EDA\solution\.venv\Scripts\python.exe solution\tools\displacement_analyze.py stage9_fast16
& F:\EDA\solution\.venv\Scripts\python.exe solution\tools\displacement_package.py stage9_fast16
wsl -d Ubuntu-20.04 -- python3 /mnt/f/EDA/solution/eda2026/solution/tools/displacement_run.py validate stage9_fast16
```

候选构建共用 WSL 头文件，必须顺序执行，工作目录为 `/home/zzy/eda2026-displacement`。Linux ELF 不从 Windows 直接运行。重新发布前，两候选需完成精确参考查询、逐行参考核对及同批五轮基准：

```powershell
& F:\EDA\solution\.venv\Scripts\python.exe solution\tools\displacement_residual_verify.py stage9_fast16 stage9_accurate64
wsl -d Ubuntu-20.04 -- python3 /mnt/f/EDA/solution/eda2026/solution/tools/displacement_run.py benchmark displacement_cal16 stage9_fast16 stage9_accurate64 --runs 5 --tag stage9_final
& F:\EDA\solution\.venv\Scripts\python.exe solution\tools\release_stage9.py stage9_accurate64 --retain
& F:\EDA\solution\.venv\Scripts\python.exe solution\tools\release_stage9.py stage9_fast16
```

发布器核对实际源码、生成头、模型、六段基础资产、包络、ELF/ZIP、准确度、五轮、资源与独立运行证据，并要求每轮 1B 外推小于 1800 秒。`--retain` 只登记高精度备选，不覆盖默认。源码或产物变化后必须重新验收，旧报告不能证明新产物。测时期间避免训练、压缩和其他估算器任务，10M 临时输入输出会自动删除。

## 交接与历史

- [HANDOFF](HANDOFF.md)：当前文件、保留策略及下一步。
- [solution/README](solution/README.md)：赛题、评分、接口和算法。
- [第九阶段记录](docs/第九阶段优化记录.md)：本轮方法、负结果、推理提速及完整复现。
- [第八阶段计划](docs/优化推进计划.md)、[第八阶段记录](docs/完整位移模型优化记录.md)：完整位移接入、压缩和训练。
- [实现对比](solution/实现对比与适配评估.md)、[第七阶段报告](solution/FPGA延时估算第七阶段结果.md)：历史比较，测时不与本轮混算。
- [开发计划](solution/FPGA延时估算开发计划.md)：初始计划与当前进展入口。

第八/七阶段 ELF、ZIP、模型及原清单保留。淘汰的运行实现与压缩试验大文件可按脚本再生，历史路径不保证仍存在；紧凑报告保留。下一轮优先降低超过 98 模型的 CPU 成本，并补充短距离、墙门、边界与稀少端口的精确参考。
