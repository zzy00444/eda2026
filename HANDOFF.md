# EDA 2026 赛题交接

更新于 2026-10-21。[第九阶段计划](docs/第九阶段98分优化计划.md)已完成：C++ 固定验证和测试回归均超过 98。默认选择 16 树版；按用户后续要求，额外保留本轮已验收的最高精度且本机外推低于 30 分钟的 64 树版，同时保留第八/七阶段速度方案。

## 当前产物

| 用途 | ELF | ZIP | 证据 |
| --- | --- | --- | --- |
| 默认，超过 98 | `solution/bin/estimate`，同 `estimate_stage9_fast16` | `solution/submission.zip`，同 `stage9_fast16_submission.zip` | [默认清单](docs/stage9_manifest.json) |
| 额外保留高精度 | `solution/bin/estimate_stage9_accurate64` | `solution/stage9_accurate64_submission.zip` | [高精度清单](docs/stage9_accuracy_manifest.json) |
| 速度与综合分参照 | `solution/bin/estimate_displacement_cal16` | `solution/displacement_cal16_submission.zip` | [第八阶段历史清单](docs/stage8_manifest.json) |
| 历史速度回退 | `solution/bin/estimate_stage7_fold` | `solution/stage7_fold_submission.zip` | `solution/build/stage7_manifest.json` |

| 方案 | 测试回归准确度 | 同批 10M CPU 均值 | 1B CPU 外推 | 预计总分 |
| --- | ---: | ---: | ---: | ---: |
| 第八阶段 | 97.650829 | 1.886051 s | 3.143 分钟 | 96.548954 |
| 第九阶段 16 树默认 | 98.031248 | 7.630726 s | 12.718 分钟 | 92.066060 |
| 第九阶段 64 树额外保留 | 98.069069 | 16.872524 s | 28.121 分钟 | 84.394818 |

准确度目标已在本地回归达到，综合分因 CPU 成本下降，不能声称全面超过旧方案。64 树五次外推范围为 27.760～28.599 分钟，均小于用户指定的 30 分钟预算；远程平台有波动，尚未执行完整 1B 或远程验证。所有公开切片均非官方隐藏集。

16/64 树 ZIP 为 98,376,571 / 98,540,421 B，容器峰值 649,625,600 / 649,138,176 B，观察线程数 1。两者通过百万行 Python/C++ 零差异、五轮 10M 逐字节一致及重复内容核对、11+11 项接口/边界和 ZIP 独立运行。参见[最终基准](docs/stage9_final_benchmark.json)和两份发布清单。

## 实现与再生

- [根 README](README.md)：环境、数据、构建、发布与产物链接。
- [solution/README](solution/README.md)：规则、评分、接口及当前算法。
- [第九阶段记录](docs/第九阶段优化记录.md)：训练、导出、压缩、性能修正、负结果和完整复现。
- [第八阶段记录](docs/完整位移模型优化记录.md)：冻结基础资产和第八阶段参数的生成。

源码为 `solution/src/displacement.cpp`、`displacement_residual_runtime.hpp`，生成头为本地 `solution/build/displacement/<候选>.hpp`。基础资产为 `oracle/`、`U_refit_q16.bin`、`M_refit.bin`、`calibration_train.bin`。新增资产为 `stage9_bound.z`、两个冻结模型原文 `residual_bound_255_fast.txt` / `residual_bound_127.txt`、导出头、`*_frozen.json` 和参考数组。保留完整 128 轮训练模型；正式候选分别使用前 16/64 棵树。

新残差使用相对误差加权 L1，在 719,583 行拟合；180,375 行按源坐标留出，只隔离新增参数。基础第八阶段 M/校准曾用主训练集，不称其为整个系统的新盲测。验证选择，配置冻结后测试回归；历史切片已被开发反复查看。48 条保留位置精确查询仍有结构误差，16 树 87.936545、64 树 88.578985；见[本轮泛化报告](docs/stage9_generalization.json)。

本机 Windows Python 为 `F:\EDA\solution\.venv\Scripts\python.exe`；WSL `Ubuntu-20.04`，仓库映射 `/mnt/f/EDA/solution/eda2026`，工作目录 `/home/zzy/eda2026-displacement`，镜像 `fpga-ziguang:latest`。容器精确限制为单 CPU、2,000,000,000 B，镜像已有 zlib。二进制、包和缓存不入 Git，新克隆须先重建本地资产。

构建使用 `--residual-header ... --bound-file ... --define EST_RESIDUAL_BATCH`；导出使用 `--direct --integer`。先训练/导出，再 `build` → `run` → `analyze` → `package` → 精确查询 → `validate`，最后两个候选一起 `residual_verify` 和五轮 `benchmark`。候选共用编译头，构建必须顺序；测时避免其他估算器、训练或压缩。

发布先运行 `release_stage9.py stage9_accurate64 --retain`，再运行 `release_stage9.py stage9_fast16`。前者只登记备选，后者覆盖默认；两者都核对实际代码、模型、ELF/ZIP、准确度、五轮、内存、单线程及独立运行，要求每轮 1B 外推低于 1800 秒。源码或模型改变须重算，禁止沿用旧报告证明新产物。

`stage8_manifest.json` 是历史发布快照，其中 canonical 路径已被第九阶段接替，应通过具名备份找到旧产物。第八阶段源码位于 Git `0232704`；旧发布器不能用当前新增残差源码去通过其源码指纹检查。

外部 `F:\EDA\pango_delay_handover_v3` 仍有人修改。本轮未改动或使用其新增模型；重新导入须核对 [displacement_import.json](docs/displacement_import.json) 指纹，也不改动原始附件。

## 后续优先级

1. 在远程平台分别测试默认、高精度和第八阶段包，保存平台准确度与五次 CPU，用实际环境决定使用哪个包。高精度版的本地外推距 30 分钟约 1.88 分钟，不保证远程仍在预算内。
2. 继续压缩树推理成本，在保持已达到的准确度时争取恢复时间分；通用树表、逐请求直接分支、浮点小批次的负结果及最终整数批处理均已有报告。每轮继续额外保留已验收且满足时间预算的最高精度版本。
3. 扩充短距离、墙门、边界和稀少输出端口的精确参考，先验证合法路径及候选覆盖；不把逐样本 oracle 最优当作在线收益。
4. 研究更紧凑的架构包络/残差表示。当前 ZIP 余量约 1.5 MB，避免重建约 32 GB 稠密原料；清理可再生试验大文件时保留正式高精度、速度备份、生成参数和证据。
# 第十阶段进行中

按[联合优化计划](docs/第十阶段精度与综合分联合优化计划.md)仅在本地 Docker 继续优化，未提交赛事远程平台。精度达到 98 的混合候选仍在降耗，尚未通过综合分门槛，默认产物暂不切换。进度及失败试验见[第十阶段记录](docs/第十阶段优化记录.md)。第八阶段速度包与第九阶段额外高精度包继续保留；重建旧发布版须使用其发布提交，因为当前源码正在优化。

