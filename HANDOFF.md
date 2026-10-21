# EDA 2026 赛题交接

第十一阶段进行中，见[计划](docs/第十一阶段优化计划.md)及[实时记录](docs/第十一阶段优化记录.md)；第十阶段发布产物继续保留。

更新于 2026-10-21。[第十阶段联合优化计划](docs/第十阶段精度与综合分联合优化计划.md)已完成：默认 `stage10_local96w64` 的验证/测试准确度均超过 98，预计总分高于同批第八阶段 0.071206。额外保留已验收最高精度 `stage10_local160w160`，五轮每次外推均小于 30 分钟。第九阶段两个包和第八/七阶段速度方案继续保留。本轮只用本地 Docker，未向赛事平台提交，也未推送 Git。

## 当前产物

| 用途 | ELF | ZIP | 证据 |
| --- | --- | --- | --- |
| 默认，联合门槛通过 | `solution/bin/estimate`，同 `estimate_stage10_local96w64` | `solution/submission.zip`，同 `stage10_local96w64_submission.zip` | [默认清单](docs/stage10_manifest.json) |
| 额外保留最高精度 | `solution/bin/estimate_stage10_local160w160` | `solution/stage10_local160w160_submission.zip` | [高精度清单](docs/stage10_accuracy_manifest.json) |
| 第九阶段旧默认 | `solution/bin/estimate_stage9_fast16` | `solution/stage9_fast16_submission.zip` | [历史清单](docs/stage9_manifest.json) |
| 第九阶段旧高精度 | `solution/bin/estimate_stage9_accurate64` | `solution/stage9_accurate64_submission.zip` | [历史高精度清单](docs/stage9_accuracy_manifest.json) |
| 速度与综合分参照 | `solution/bin/estimate_displacement_cal16` | `solution/displacement_cal16_submission.zip` | [第八阶段历史清单](docs/stage8_manifest.json) |
| 历史速度回退 | `solution/bin/estimate_stage7_fold` | `solution/stage7_fold_submission.zip` | `solution/build/stage7_manifest.json` |

| 方案 | 测试回归准确度 | 同批 10M CPU 均值 | 1B CPU 外推 | 预计总分 |
| --- | ---: | ---: | ---: | ---: |
| 第八阶段 | 97.650829 | 2.420913 s | 4.035 分钟 | 96.103236 |
| 第十阶段默认 96 维+稀疏墙内 64 树 | 98.021603 | 2.691408 s | 4.486 分钟 | 96.174442 |
| 第十阶段额外高精度 160 维+稀疏墙内 160 树 | 98.074532 | 2.973247 s | 4.955 分钟 | 95.981920 |

上表来自本轮同批五次 10M 交替测速。默认联合验收通过但分数余量较小；额外高精度版综合分未过第八阶段，仅作保留备选。高精度五轮外推 4.488～5.388 分钟，远低于 30 分钟。历史第八阶段 96.548954 与新默认之差为 -0.374512，不能跨批混用时间。尚未执行完整 1B 或远程验证，公开切片均非官方隐藏集。

默认/高精度 ZIP 为 87,769,854 / 92,486,699 B，容器峰值 616,259,584 / 635,576,320 B，线程数 1。两者通过百万行独立 Python/C++ 零差异、五轮 10M 一致与重复内容核对、11+11 项接口/边界和 ZIP 独立运行。参见[最终基准](docs/stage10_final_benchmark.json)和两份清单。

## 实现与再生

- [根 README](README.md)：环境、数据、构建、发布与产物链接。
- [solution/README](solution/README.md)：规则、评分、接口及当前算法。
- [第十阶段记录](docs/第十阶段优化记录.md)：生成残差、稀疏墙内模型、负结果及完整复现。
- [第九阶段记录](docs/第九阶段优化记录.md)：历史全局包络和残差模型。
- [第八阶段记录](docs/完整位移模型优化记录.md)：冻结基础资产和第八阶段参数的生成。

源码为 `solution/src/displacement.cpp`、`displacement_local_runtime.hpp`、`displacement_wall_runtime.hpp`。基础资产为 `oracle/`、`U_refit_q16.bin`、`M_refit.bin`、`calibration_train.bin`。新增资产位于 `solution/build/displacement/`：`stage10_local_weighted_factors.npz`、`stage10_local96.local` / `stage10_local160.local`、`stage10_local96_wall.txt` / `stage10_local160_wall.txt`、两个 `*.wall.hpp` 及冻结配置/独立参考。第九阶段包络和模型仍保留；本轮候选不加载它们。

局部模型拟合架构生成距离残差，不用公开延迟标签；端口频率权重仅用 719,583 行训练几何。墙内残差在其中 4,233 行训练，使用相对误差加权 L1。180,375 行位置评估为 98.041809 / 98.095581，继承基础参数曾用主训练集，因此不是全系统盲测。48 条精确保留位置为 93.903862 / 94.506240；第八阶段为 88.050655。参考报告分别为 `docs/stage10_local96w64_reference.json` / `stage10_local160w160_reference.json`。

本机 Windows Python 为 `F:\EDA\solution\.venv\Scripts\python.exe`；WSL `Ubuntu-20.04`，仓库映射 `/mnt/f/EDA/solution/eda2026`，工作目录 `/home/zzy/eda2026-displacement`，镜像 `fpga-ziguang:latest`。容器精确限制为单 CPU、2,000,000,000 B，镜像已有 zlib。二进制、包和缓存不入 Git，新克隆须先重建本地资产。

构建使用 `--local-file ... --wall-header ... --define EST_FAST_ROUND`。先训练/导出，再 `build` → `run` → `analyze` → `package` → 精确查询 → `local_verify` → `validate` → 同批五轮 `benchmark`。候选共用编译头，构建必须顺序；测时避免其他估算器、训练或压缩。完整命令见第十阶段记录；参考验证器显式重建双分支和 float32 FMA，不能用旧单分支筛选充当一致性证据。

发布入口为 `release_stage10.py stage10_local160w160 --retain` 和 `release_stage10.py stage10_local96w64`。前者登记高精度备选，后者还须通过同批第八阶段综合分门槛才覆盖默认；二者都核对实际代码、模型、ELF/ZIP、参考、五轮、资源及独立运行，每轮外推低于 1800 秒。源码或模型变化须重新验收。

`stage8_manifest.json`、第九阶段两个 manifest 是历史快照，canonical 路径已被本轮接替；用具名备份找到旧产物。第八阶段源码位于 `0232704`，第九阶段发布提交为 `91f253e`，旧发布版应从相应提交重建。

外部 `F:\EDA\pango_delay_handover_v3` 仍有人修改。本轮仅读取 `build/all_C_<source>.bin`，架构 SHA 与仓库冻结资产相同，各源尺寸/mtime 记录在 `stage10_local_weighted_training.json`；未修改、清理或运行该工程脚本，也未导入其新增模型。

## 后续优先级

1. 本轮已经完成联合验收；下一轮优先扩大 0.0712 分的余量，研究局部位移分区/非均匀秩、更适合短距离相对误差的生成残差权重，先筛选再测实际双分支成本。
2. 精度最高备选仅比旧高精度测试回归高约 0.0055，不把它解释为隐藏集显著提升；继续额外保留最快和最高精度的已验收产物。
3. 扩充墙门、边界和稀少端口的精确参考，解释保留位置仍未达到 98 的误差。仅当收到新的远程测试安排时才向赛事平台提交，目前继续以本地镜像为准。
4. 默认/高精度 ZIP 余量约 12.23/7.51 MB。不要重建约 32 GB 原料；可再生缓存的清理不能删除正式速度、高精度备份、生成配置和证据。

