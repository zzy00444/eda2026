# EDA 2026 FPGA 延时估算

2026-10-21 完成[第十一阶段](docs/第十一阶段优化记录.md)：当前默认为 `stage11_balancedw64`，另保留合法包中精度最高的 `stage11_compactw160`。第十阶段两版及第七至九阶段旧发布版全部保留。本轮仅用本地 Docker，未提交赛事远程平台，未推送 Git。

## 当前结果与产物

下表来自同批五轮 **100M** 的 CPU 均值，时间乘 10 外推 1B。准确度是赛题相对误差得分；历史公开切片不是盲测，外推总分不是官方成绩。

| 方案 | 验证准确度 | 测试准确度 | 100M CPU | 1B CPU 外推 | 预计总分 |
| --- | ---: | ---: | ---: | ---: | ---: |
| displacement_cal16 | 97.638770 | 97.650829 | 20.785931 s | 207.859 s | 96.388502 |
| stage10_local96w64 | 98.015370 | 98.021603 | 21.246623 s | 212.466 s | 96.646730 |
| stage11_fastw64 | 98.077092 | 98.073604 | 19.887726 s | 198.877 s | 96.801573 |
| **stage11_balancedw64** | 98.202309 | 98.192013 | 20.045812 s | 200.458 s | 96.883126 |
| stage11_compactw160 | 98.258566 | 98.247955 | 24.471854 s | 244.719 s | 96.559043 |

默认比同批第八阶段的预计综合分高 **0.494624**，比同批第十阶段默认高 **0.236396**。精度版验证/测试为 **98.258566 / 98.247955**，测试比 98.25 低约 0.002045，严格目标尚未完全达到；其综合分 高于同批第八阶段 0.170541，但低于同批第十阶段默认 0.087687，按用户要求额外保留，不能把两套配置的精度和速度拼成一个成绩。

| 用途 | 提交包 | ZIP 字节 | 证据 |
| --- | --- | ---: | --- |
| 默认 | [submission.zip](solution/submission.zip)，同 [stage11_balancedw64_submission.zip](solution/stage11_balancedw64_submission.zip) | 98,094,824 | [默认清单](docs/stage11_manifest.json) |
| 额外最高精度 | [stage11_compactw160_submission.zip](solution/stage11_compactw160_submission.zip) | 99,807,086 | [高精度清单](docs/stage11_accuracy_manifest.json) |
| 已验收速度候选 | [stage11_fastw64_submission.zip](solution/stage11_fastw64_submission.zip) | 94,987,753 | [独立参考](docs/stage11_fastw64_reference.json) |
| 第十阶段旧默认 / 旧精度版 | [96维](solution/stage10_local96w64_submission.zip) / [160维](solution/stage10_local160w160_submission.zip) | 87,769,854 / 92,486,699 | [历史默认](docs/stage10_manifest.json) / [历史精度](docs/stage10_accuracy_manifest.json) |
| 第八阶段参照 | [displacement_cal16_submission.zip](solution/displacement_cal16_submission.zip) | 80,797,119 | [历史清单](docs/stage8_manifest.json) |

精度版每轮 1B 外推 4.008～4.186 分钟，均小于 30 分钟。三个新候选均通过百万行独立 Python/C++ 零差异、56 条精确查询、五轮重复内容核对、11 项接口、11 项边界及 ZIP 空目录独立运行。100M 的 cgroup 峰值含文件页缓存，最大 1,999,998,976 B；进程峰值 RSS 见原始报告。

## 环境与数据

Windows、WSL2 `Ubuntu-20.04`、Docker Desktop，镜像 `fpga-ziguang:latest`，GCC 11.4、Linux/amd64。Docker 须开启该发行版的 WSL Integration。精度版使用镜像已有的 `liblzma.so.5` 解压内嵌局部矩阵，不需要外部模型文件。

官方数据默认 `F:\EDA\EDA-PANGO-SRB-Arch-Delay\EDA-PANGO-SRB-Arch-Delay`，应含 `arch/SRB_*.json` 和 `data/delay_estimate_ans.csv`；可设置 `EDA_DATA_DIR`。Windows Python 为 `F:\EDA\solution\.venv\Scripts\python.exe`，依赖见 `solution/requirements.txt`。外部 `F:\EDA\pango_delay_handover_v3` 只读，仅读取已完成生成距离表，未运行其脚本。

新克隆生成请求缓存：

```powershell
$env:EDA_DATA_DIR = 'F:\EDA\EDA-PANGO-SRB-Arch-Delay\EDA-PANGO-SRB-Arch-Delay'
& F:\EDA\solution\.venv\Scripts\python.exe -c "import os; from pathlib import Path; import pandas as pd; p=Path('solution/build/local_runs/request_1m.csv'); p.parent.mkdir(parents=True, exist_ok=True); pd.read_csv(Path(os.environ['EDA_DATA_DIR'])/'data/delay_estimate_ans.csv', usecols=['From','To']).to_csv(p,index=False)"
```

## 重建与验收

源码为 [displacement.cpp](solution/src/displacement.cpp)、[局部压缩运行时](solution/src/displacement_local_runtime.hpp)、[稀疏墙内运行时](solution/src/displacement_wall_runtime.hpp)。基础为完整位移 rank128；局部使用互斥位移区域、生成距离相对误差权重、稀疏端口映射以及独立秩。匹配消融发现旧 M/校准调整只有很小收益，因此保留旧校准，新局部残差直接匹配 refit 基线。

Git 不含 ELF、ZIP、二进制模型、公开数据及大缓存。先按[第八阶段记录](docs/完整位移模型优化记录.md)重建基础冻结资产，再按[第十一阶段记录](docs/第十一阶段优化记录.md)再生分区因子、从冻结 profile 导出资产、训练墙内模型。已有资产时的构建、独立参考、打包、验收命令也在该记录中。

构建共用 `/home/zzy/eda2026-displacement` 的生成头，必须顺序执行；测时不运行训练、压缩或其他估算器。最终基准：

```powershell
wsl -d Ubuntu-20.04 -- python3 /mnt/f/EDA/solution/eda2026/solution/tools/displacement_run.py benchmark displacement_cal16 stage10_local96w64 stage11_fastw64 stage11_balancedw64 stage11_compactw160 --runs 5 --repeat 100 --tag stage11_final
& F:\EDA\solution\.venv\Scripts\python.exe solution/tools/release_stage11.py stage11_compactw160 --retain
& F:\EDA\solution\.venv\Scripts\python.exe solution/tools/release_stage11.py stage11_balancedw64
```

发布器核对源码、模型、内嵌资产、ELF/ZIP、独立参考、资源及五轮实际行数。默认还须不低于同批第八阶段和第十阶段默认综合分；`--retain` 只登记精度备选，不覆盖默认。旧报告不能证明新产物。历史第八阶段 96.548954、旧 10M 与本轮 100M 的时间口径不混用。

## 交接与历史

- [HANDOFF](HANDOFF.md)：当前资产、保留策略与下一步。
- [solution/README](solution/README.md)：赛题规则、接口、评分及算法。
- [第十一阶段计划](docs/第十一阶段优化计划.md) / [记录](docs/第十一阶段优化记录.md)：消融、负结果、冻结配置和复现。
- [第十阶段记录](docs/第十阶段优化记录.md)、[第九阶段记录](docs/第九阶段优化记录.md)、[第八阶段记录](docs/完整位移模型优化记录.md)：历史方案与证据。
- [开发计划](solution/FPGA延时估算开发计划.md)：初始计划及进展入口。

历史 canonical 路径由新默认接替，旧版本用具名备份；旧发布版应在各自发布提交重建。超限诊断包 `stage11_accurate_oversize_submission.zip` 及实验 ELF 保留供分析，不属于合格提交包。
