# EDA 2026 赛题交接

2026-10-23 第十三阶段已冻结：停止继续优化和远程提交。`stage12_b128_submission.zip` 已完成独立打包和验收，同批五轮 100M 校准估分 96.284189，未达到 97；默认包与最高精度备选不变。后续若恢复工作，估分统一使用官方对齐的 1.401789 CPU 比例。详见[第十三阶段记录](docs/第十三阶段优化记录.md)。下方第十一阶段表格保留历史未校准口径。

更新于 2026-10-23。第十一阶段完成，默认 `stage11_balancedw64`，额外保留合法最高精度 `stage11_compactw160`。已收到默认包一次官方平台成绩，运行时间校准记录见下文及 `docs/stage11_official_benchmark_alignment.json`。第十阶段两版及第七至九阶段旧发布版继续保留；未执行新的远程提交，未推送 Git。

## 当前结果

| 方案 | 验证准确度 | 测试准确度 | 100M CPU | 1B CPU 外推 | 预计总分 |
| --- | ---: | ---: | ---: | ---: | ---: |
| displacement_cal16 | 97.638770 | 97.650829 | 20.785931 s | 207.859 s | 96.388502 |
| stage10_local96w64 | 98.015370 | 98.021603 | 21.246623 s | 212.466 s | 96.646730 |
| stage11_fastw64 | 98.077092 | 98.073604 | 19.887726 s | 198.877 s | 96.801573 |
| **stage11_balancedw64** | 98.202309 | 98.192013 | 20.045812 s | 200.458 s | 96.883126 |
| stage11_compactw160 | 98.258566 | 98.247955 | 24.471854 s | 244.719 s | 96.559043 |

上表保留未校准的同批五轮 100M 本机 Docker 数据，乘 10 外推 1B，详见 `docs/stage11_final_benchmark.json`。默认原始本机预计综合分比同批第八阶段高 0.494624，比同批第十阶段默认高 0.236396。精度版验证 98.258566、测试 98.247955，测试比 98.25 低 0.002045；不能四舍五入后声称严格达标。精度版综合分高于同批第八阶段 0.170541，但低于同批第十阶段默认 0.087687，只作额外精度保留。历史切片反复使用，不是盲测或官方隐藏集。

官方平台两任务 CPU 为 287.120 / 274.880 s，平均 281.000 s；相对本机 1B CPU 外推 200.458 s 的比例为 1.401789。校准后本机时间分为 84.3889，按本机测试准确度计算的估计总分为 96.211944。官方实际准确度/总分为 98.07 / 96.12，单独记录；这是一次官方测评，不代表稳定运行比例或隐藏集泛化结论。

## 当前资产与保留策略

- 默认：`solution/bin/estimate` / `solution/submission.zip`，与具名 `stage11_balancedw64` 相同；证据 `docs/stage11_manifest.json`。
- 合法最高精度：`solution/bin/estimate_stage11_compactw160` / `solution/stage11_compactw160_submission.zip`，ZIP 99,807,086 B；证据 `docs/stage11_accuracy_manifest.json`。每轮 1B 外推 4.008～4.186 分钟。
- 本轮速度候选 `stage11_fastw64`、平衡候选 `stage11_balancedw64` 均具名保留，均已完整验收。
- 第十阶段 `stage10_local96w64` / `stage10_local160w160`、第九阶段两版、第八阶段 `displacement_cal16`、第七阶段 `stage7_fold` 全部保留。旧清单是历史快照，canonical 路径已由新默认接替。
- `stage11_accurate_oversize_submission.zip` / `estimate_stage11_accurate` 是超限实验诊断，不用于提交；模型和配置仍保留。

基础资产仍是 `oracle/`、`U_refit_q16.bin`、`M_refit.bin`、`calibration_train.bin`。新增资产位于 `solution/build/displacement/`：24 个 `stage11_tile_*_factors.npz`、`stage11_outer_rel_factors.npz`、三套 `.local`、`stage11_compact.local.xz`、各自墙内模型和生成头。三个 `*_profile.json` 明确冻结的每区表示、秩和精度，不能从最新探索结果推测发布配置。

## 方法、训练与验收

先验证真实双分支的 M/校准匹配，改动只带来约 0.003～0.005，位置留出不支持更改 M。随后将 `dx=-32..32,dy=-160..160` 分为 12 个互斥区域，按生成距离逆平方的行/列边际和训练端口频率压缩；每区独立选择直接距离或基础残差及秩。训练/存储 23,321 个训练几何出现的端口组合，未覆盖组合回退旧基线。精度版再加外区残差。

局部因子不直接拟合公开 delay 标签；继承的基础 M/校准使用过公开训练集。新增端口统计和墙内训练排除既定源坐标留出，墙内训练 4,233 行、位置评估 180,375 行，因此不声称全系统位置隔离。最终配置先冻结，再读取 test，未根据 test 的 0.002 差距继续调参。

精度版局部 V 用 lag-12 差分和双字节平面，再 XZ 压缩。封装工具核对压缩前后字节一致，运行时从 ELF 调用镜像已有 `liblzma.so.5` 解压。独立参考重建全部基础/局部 FMA、真实两候选和墙内树，三候选百万行及 56 条精确查询都零差异。11+11 项接口/边界、ZIP 空目录运行、五轮重复内容 SHA 均通过。

## 重建与发布

完整命令在[第十一阶段记录](docs/第十一阶段优化记录.md)。Windows Python `F:\EDA\solution\.venv\Scripts\python.exe`；WSL `Ubuntu-20.04`，仓库 `/mnt/f/EDA/solution/eda2026`，工作目录 `/home/zzy/eda2026-displacement`，镜像 `fpga-ziguang:latest`。构建共用生成头，必须顺序。

`displacement_tile_train.py` → 冻结 profile 导出 → 无墙内基线 `build/run` → `displacement_wall_train.py` / `wall_export.py` → 最终 `build/run/analyze` → 保留查询 → `local_verify.py` → `package/validate` → 五轮 `benchmark --repeat 100` → `release_stage11.py`。

精度版 build 额外传 `--local-xz .../stage11_compact.local.xz`，基础 `.local` 仍传入以核对逐字节回环。发布器 `--retain` 不覆盖默认；默认还验证不低于同批第八阶段及第十阶段。源码或资产改动必须重新验收，不能用历史报告证明新二进制。

外部 `F:\EDA\pango_delay_handover_v3` 有其他人在工作，继续只读；不得修改、清理或运行其脚本。Git 不含大型模型/缓存/二进制，新的克隆需先重建。

## 冻结后的状态

当前不安排新的实验。保留默认提交、最高准确度备选和 `stage12_b128` 速度候选；官方实际分数与本机估分继续分开记录。
