# EDA 2026 FPGA 延时估算

2026-10-21 已完成[第十阶段联合优化计划](docs/第十阶段精度与综合分联合优化计划.md)：默认改为 `stage10_local96w64`，验证/测试准确度均超过 98，预计综合分高于同批第八阶段。另保留本轮已验收最高精度 `stage10_local160w160`；第九阶段两个包及第八阶段速度包全部保留。方法、负结果和复现见[第十阶段记录](docs/第十阶段优化记录.md)。本轮仅做本地 Docker 测试，未提交赛事平台。

## 产物与结果

下表为同机、相同输入、正反交错五轮 10M 的 CPU 均值。准确度是赛题相对误差得分；固定 rng7 切片已在历史开发中查看。全量 1B 和官方隐藏集尚未评测，1B 用时及总分均为外推。

| 方案 | 验证准确度 | 测试回归准确度 | 10M CPU | 1B CPU 外推 | 预计总分 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 第八阶段速度参照 | 97.638770 | 97.650829 | 2.420913 s | 4.035 分钟 | 96.103236 |
| **第十阶段默认，96 维+稀疏墙内 64 树** | 98.015370 | 98.021603 | 2.691408 s | 4.486 分钟 | **96.174442** |
| **额外保留高精度，160 维+稀疏墙内 160 树** | **98.079389** | **98.074532** | 2.973247 s | **4.955 分钟** | 95.981920 |

默认版比同批第八阶段预计总分高 **0.071206**，优势较小；高精度版未过综合分门槛，仅按用户要求额外保留。高精度版五轮每次外推 4.488～5.388 分钟，均低于 30 分钟。历史第八阶段曾测得 96.548954，本轮默认比该历史数值低 0.374512；CPU 波动明显，按预定口径以同批比较验收，不能跨批拼接时间声称绝对超越历史。远程用时尚未测量。

| 用途 | 提交包 | ZIP 字节 | 发布证据 |
| --- | --- | ---: | --- |
| 默认，联合验收通过 | [submission.zip](solution/submission.zip)，同 [stage10_local96w64_submission.zip](solution/stage10_local96w64_submission.zip) | 87,769,854 | [默认清单](docs/stage10_manifest.json) |
| 本轮已验收最高精度，额外保留 | [stage10_local160w160_submission.zip](solution/stage10_local160w160_submission.zip) | 92,486,699 | [高精度清单](docs/stage10_accuracy_manifest.json) |
| 速度与综合分参照 | [displacement_cal16_submission.zip](solution/displacement_cal16_submission.zip) | 80,797,119 | [第八阶段历史清单](docs/stage8_manifest.json) |
| 第九阶段旧默认 | [stage9_fast16_submission.zip](solution/stage9_fast16_submission.zip) | 98,376,571 | [第九阶段清单](docs/stage9_manifest.json) |
| 第九阶段旧高精度 | [stage9_accurate64_submission.zip](solution/stage9_accurate64_submission.zip) | 98,540,421 | [第九阶段高精度清单](docs/stage9_accuracy_manifest.json) |

两个新包均只包含 `bin/estimate`，通过百万行独立 Python/C++ 零差异、五次 10M 输出一致与重复内容核对、11 项接口、11 项额外错误/边界及 ZIP 空目录独立运行。容器峰值分别为 616,259,584 / 635,576,320 B，观察线程数 1；容器限额精确为 2,000,000,000 B。见[最终基准](docs/stage10_final_benchmark.json)、[默认验收](docs/stage10_local96w64_validation.json)和[高精度验收](docs/stage10_local160w160_validation.json)。

新增参数的源坐标留出准确度为 98.041809 / 98.095581；继承的第八阶段参数曾使用这些位置，所以它不是全系统位置隔离。48 条保留位置精确查询中，第八阶段为 88.050655，新默认为 93.903862，高精度为 94.506240，仍有结构误差。这些结果不能证明隐藏集达到 98，详见[默认参考](docs/stage10_local96w64_reference.json)和[高精度参考](docs/stage10_local160w160_reference.json)。

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

源码为 [displacement.cpp](solution/src/displacement.cpp)、[局部残差运行时](solution/src/displacement_local_runtime.hpp)和[稀疏墙内运行时](solution/src/displacement_wall_runtime.hpp)。基础冻结资产来自用户提供的 Pango 工程，见[导入指纹](docs/displacement_import.json)和[基础重建](docs/完整位移模型优化记录.md)。本轮只读外部已完成生成距离表并记录来源，不改动外部工程或赛题附件，也未运行外部训练脚本。

Git 不包含 ELF、ZIP、二进制模型、公开数据和大缓存，不能仅凭克隆直接构建。按[第十阶段记录](docs/第十阶段优化记录.md)依次生成局部因子、导出量化资产、运行局部基线、训练并导出稀疏墙内树。当前两候选不使用第九阶段包络资产。已有冻结资产时，快速重建默认：

```powershell
wsl -d Ubuntu-20.04 -- python3 /mnt/f/EDA/solution/eda2026/solution/tools/displacement_run.py build stage10_local96w64 --u /mnt/f/EDA/solution/eda2026/solution/build/displacement/U_refit_q16.bin --m /mnt/f/EDA/solution/eda2026/solution/build/displacement/M_refit.bin --calibration /mnt/f/EDA/solution/eda2026/solution/build/displacement/calibration_train.bin --local-file /mnt/f/EDA/solution/eda2026/solution/build/displacement/stage10_local96.local --wall-header /mnt/f/EDA/solution/eda2026/solution/build/displacement/stage10_local96w64.wall.hpp --define EST_FAST_ROUND
wsl -d Ubuntu-20.04 -- python3 /mnt/f/EDA/solution/eda2026/solution/tools/displacement_run.py run stage10_local96w64
& F:\EDA\solution\.venv\Scripts\python.exe solution\tools\displacement_analyze.py stage10_local96w64
& F:\EDA\solution\.venv\Scripts\python.exe solution\tools\displacement_package.py stage10_local96w64
wsl -d Ubuntu-20.04 -- python3 /mnt/f/EDA/solution/eda2026/solution/tools/displacement_run.py validate stage10_local96w64
```

候选构建共用 WSL 头文件，必须顺序执行，工作目录为 `/home/zzy/eda2026-displacement`。Linux ELF 不从 Windows 直接运行。重新发布前，两候选需完成精确参考查询、逐行参考核对及同批五轮基准：

```powershell
wsl -d Ubuntu-20.04 -- python3 /mnt/f/EDA/solution/eda2026/solution/tools/displacement_run.py run stage10_local96w64 stage10_local160w160 --request /mnt/f/EDA/solution/eda2026/solution/build/displacement/oracle_request.csv --suffix oracle
& F:\EDA\solution\.venv\Scripts\python.exe solution\tools\displacement_local_verify.py stage10_local96w64 --local-file stage10_local96.local --wall-model stage10_local96_wall --trees 64
& F:\EDA\solution\.venv\Scripts\python.exe solution\tools\displacement_local_verify.py stage10_local160w160 --local-file stage10_local160.local --wall-model stage10_local160_wall --trees 160
wsl -d Ubuntu-20.04 -- python3 /mnt/f/EDA/solution/eda2026/solution/tools/displacement_run.py benchmark displacement_cal16 stage10_local96w64 stage10_local160w160 --runs 5 --tag stage10_final
& F:\EDA\solution\.venv\Scripts\python.exe solution\tools\release_stage10.py stage10_local160w160 --retain
& F:\EDA\solution\.venv\Scripts\python.exe solution\tools\release_stage10.py stage10_local96w64
```

发布器核对源码、生成头、模型、六段基础资产及局部段、ELF/ZIP、准确度、五轮和独立运行证据，要求每轮 1B 外推小于 1800 秒；切换默认还必须通过同批第八阶段综合分门槛。`--retain` 只登记精度更高的备选，不覆盖默认。源码或产物变化须重新验收，旧报告不能证明新产物。测时期间避免训练、压缩和其他估算器，10M 临时输入输出会自动删除。

## 交接与历史

- [HANDOFF](HANDOFF.md)：当前文件、保留策略及下一步。
- [solution/README](solution/README.md)：赛题、评分、接口和算法。
- [第十阶段记录](docs/第十阶段优化记录.md)：本轮方法、负结果、联合验收及完整复现。
- [第九阶段记录](docs/第九阶段优化记录.md)：历史全局包络/残差模型。
- [第八阶段计划](docs/优化推进计划.md)、[第八阶段记录](docs/完整位移模型优化记录.md)：完整位移接入、压缩和训练。
- [实现对比](solution/实现对比与适配评估.md)、[第七阶段报告](solution/FPGA延时估算第七阶段结果.md)：历史比较，测时不与本轮混算。
- [开发计划](solution/FPGA延时估算开发计划.md)：初始计划与当前进展入口。

第九/八/七阶段 ELF、ZIP、模型及历史清单保留。历史清单中的默认路径已被新版本接替，旧产物用具名备份查找；旧发布版应使用各自发布提交重建。下一轮优先扩大联合分数余量，再研究局部因子压缩与墙门/边界/稀少端口覆盖。
