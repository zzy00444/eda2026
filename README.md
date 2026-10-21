# EDA 2026 FPGA 延时估算

本仓库管理“面向超大规模 FPGA 的布局布线延时估算算法”的实现、实验报告与复测流程。2026-10-21 已按[优化推进计划](docs/优化推进计划.md)完成 P0～P5；当前默认为第八阶段 `displacement_cal16`：完整位移 rank128 模型、逐分量 U-int16、训练折内 M 修正与校准。提交文件为 `solution/submission.zip`，内含 `bin/estimate`。

实验与参数重建见[完整位移模型优化记录](docs/完整位移模型优化记录.md)，产物及证据指纹见[第八阶段清单](docs/stage8_manifest.json)。源码在 [solution/src/displacement.cpp](solution/src/displacement.cpp)，交接入口为 [HANDOFF.md](HANDOFF.md)。

## 当前结果

下表采用本轮同机、相同输入、正反交错五轮 10M CPU 测量。准确度是赛题相对误差得分，固定 rng7 测试切片已在历史开发中查看；完整 1B 和官方隐藏集尚未评测。

| 候选 | 固定测试准确度 | 10M CPU 均值 | 1B CPU 外推 | 预计总分 |
| --- | ---: | ---: | ---: | ---: |
| 第七阶段 `stage7_fold` | 94.607402 | 1.343385 s | 134.339 s | 94.566433 |
| Pango 未校准基线 | 97.595517 | 1.963007 s | 196.301 s | 96.440575 |
| **第八阶段 `displacement_cal16`** | **97.650829** | **2.022604 s** | **202.260 s** | **96.435160** |

新默认相对第七阶段测试准确度 +3.043427、预计总分 +1.868726；相对 Pango 未校准基线准确度 +0.055312。后者的预计总分差为 −0.005415，结合测时波动只能视为接近，不能声称总分和速度全面胜出。选择新默认依据是准确度、更小包体和完整的输入/错误处理；64 MB 分块输出相同，本轮 CPU 更高且占用更多内存，未采用。

新默认 ELF 为 84,423,084 B，ZIP 为 **80,797,119 B**，容器峰值 **582,066,176 B**，观察线程数 1。百万行格式与评分核对、五次 10M 输出一致及重复内容检查、11 项接口、11 项额外错误/边界检查、ZIP 空目录独立运行全部通过。见[最终压测](docs/displacement_final_benchmark.json)和[验收汇总](docs/displacement_cal16_validation.json)。本轮容器使用精确的 `--memory=2000000000 --memory-swap=2000000000`。

结构误差仍未解决：48 条新位置精确查询的相对误差得分为 88.248625→88.050655，虽 RMSE 下降，但并非所有位置改善。99 分仍需研究位置相关的合法路径、墙门与边界机制，不能靠公开随机切片校准证明已达到。详见[泛化检查](docs/displacement_generalization.json)。

## 本机准备

本机为 Windows、WSL2 `Ubuntu-20.04`、Docker Desktop，镜像 `fpga-ziguang:latest`。登记工具链为 GCC 11.4.0、Linux/amd64；接手时实际检查。Docker Desktop 的 WSL Integration 须启用该发行版。

```powershell
wsl --list --verbose
wsl -d Ubuntu-20.04 -- docker version
wsl -d Ubuntu-20.04 -- docker image inspect fpga-ziguang:latest
```

若镜像缺失且本机有归档，再执行 `wsl -d Ubuntu-20.04 -- docker load -i /mnt/f/EDA/fpga-ziguang-latest.tar`。环境历史记录位于仓库外 `F:\EDA\评测环境记录.md`。

官方数据不入 Git。默认目录 `F:\EDA\EDA-PANGO-SRB-Arch-Delay\EDA-PANGO-SRB-Arch-Delay` 应包含 `arch/SRB_*.json` 和 `data/delay_estimate_ans.csv`；也可设置 `EDA_DATA_DIR`。Windows 训练/评分需要 `solution/requirements.txt` 的依赖；WSL 驱动只使用 Python 标准库，编译与执行在 Docker 中完成。

```powershell
Set-Location F:\EDA\solution\eda2026
$env:EDA_DATA_DIR = (Resolve-Path ..\..\EDA-PANGO-SRB-Arch-Delay\EDA-PANGO-SRB-Arch-Delay).Path
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r solution\requirements.txt
```

本机已有环境 `F:\EDA\solution\.venv`，可直接用其中的 `Scripts\python.exe` 替代上述 Python，不必再创建一份。以下命令使用该已有环境。

请求文件应有 `From,To` 表头。本机保留 `solution/build/local_runs/request_1m.csv`；新克隆没有缓存时生成：

```powershell
& F:\EDA\solution\.venv\Scripts\python.exe -c "import os; from pathlib import Path; import pandas as pd; p=Path('solution/build/local_runs/request_1m.csv'); p.parent.mkdir(parents=True, exist_ok=True); pd.read_csv(Path(os.environ['EDA_DATA_DIR'])/'data'/'delay_estimate_ans.csv', usecols=['From','To']).to_csv(p, index=False)"
```

## 重建与验收

模型原料来自用户提供的 `F:\EDA\pango_delay_handover_v3`。本机已冻结在 `solution/build/displacement/oracle`，外部源码后续变化不影响本仓库构建。新克隆需先按[执行记录](docs/完整位移模型优化记录.md)的 P1～P3 导入、生成诊断记录、拟合 M 与校准表；核对导入资产与 `docs/displacement_import.json` 指纹。Git 不包含这些二进制模型，不能仅靠克隆直接构建，也不需要为本轮重新生成约 32 GB 稠密原料。

已有冻结模型和训练产物时，从仓库根目录执行：

```powershell
wsl -d Ubuntu-20.04 -- python3 /mnt/f/EDA/solution/eda2026/solution/tools/displacement_run.py build displacement_cal16 --u /mnt/f/EDA/solution/eda2026/solution/build/displacement/U_refit_q16.bin --m /mnt/f/EDA/solution/eda2026/solution/build/displacement/M_refit.bin --calibration /mnt/f/EDA/solution/eda2026/solution/build/displacement/calibration_train.bin
wsl -d Ubuntu-20.04 -- python3 /mnt/f/EDA/solution/eda2026/solution/tools/displacement_run.py run displacement_cal16
& F:\EDA\solution\.venv\Scripts\python.exe solution\tools\displacement_analyze.py displacement_cal16
& F:\EDA\solution\.venv\Scripts\python.exe solution\tools\displacement_package.py displacement_cal16
wsl -d Ubuntu-20.04 -- python3 /mnt/f/EDA/solution/eda2026/solution/tools/displacement_run.py validate displacement_cal16
```

驱动在 WSL `/home/zzy/eda2026-displacement` 中运行，ELF 写回 `solution/bin/estimate_<候选名>`，参数/预测在 `solution/build/displacement`，紧凑报告在 `docs`。不要从 Windows 直接执行 Linux ELF。候选构建不会覆盖默认提交。

重新发布前，需要准备第七阶段与原始完整位移基线的 ELF，并为相同最终模型构建 `displacement_cal64`（增加 `--define EST_BLOCK_MB=64`）。已清理的候选按执行记录重建；旧源版本对应的二进制及性能以原清单为准。运行同批基准后，发布脚本检查实际源码、嵌入模型、ELF/ZIP、评分输出、五轮及独立运行证据，全部通过才更新默认文件：

```powershell
wsl -d Ubuntu-20.04 -- python3 /mnt/f/EDA/solution/eda2026/solution/tools/displacement_run.py benchmark stage7_fold displacement_baseline displacement_cal16 displacement_cal64 --runs 5 --tag displacement_final
& F:\EDA\solution\.venv\Scripts\python.exe solution\tools\release_stage8.py displacement_cal16
```

源码、模型或二进制改变后须重新评分和验收，不能沿用旧报告证明新产物。10M 输入/输出在基准完成后自动删除；压测时避免同时训练或压缩。

## 仓库与历史记录

- [solution/README.md](solution/README.md)：接口、评分、当前算法及约束。
- [优化推进计划](docs/优化推进计划.md)：本轮计划与执行台账。
- [完整位移模型优化记录](docs/完整位移模型优化记录.md)：基线、量化、训练、负结果与复现。
- [实现对比与适配评估](solution/实现对比与适配评估.md)：首轮 Pango 核查和误差贡献比较。
- [第七阶段报告](solution/FPGA延时估算第七阶段结果.md)、第六/五阶段报告：历史折叠表与树模型，其历史测时不与本轮混算。
- [开发计划](solution/FPGA延时估算开发计划.md)：初始计划及当前进度入口。

第七阶段 ELF、ZIP、模型和原清单保留作为速度参照及回退。旧版/淘汰候选的 ELF、重复提交包和临时预测 CSV 已按需清理，历史报告不保证其中所有产物路径仍存在。代码、参数生成脚本与紧凑报告入 Git；公开数据、模型二进制、ELF、ZIP 和大缓存不入 Git。
