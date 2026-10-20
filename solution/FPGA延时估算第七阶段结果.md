# FPGA 延时估算第七阶段结果

记录日期：2026-10-20。本阶段把残差模型**离线折叠进架构表**：每个 `(方向, 源端口, dx%10, dy%12, 目标端口)` 键的条目不再是纯周期下界，而是"下界 + 训练得到的修正"。运行时因此只剩一次查表、一次线性项和 Gap 前缀，不再执行决策树、不再计算 73 个特征。准确度略有下降，但 CPU 再降 32%，综合分继续上升。

## 结论

| 候选 | 固定验证 | 固定测试 | 10M CPU 均值 | 1B 外推 | 时间分 | 预计总分 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 第五阶段默认（本会话同机复测） | 94.9601 | 94.9711 | 3.1562 s | 5.26 min | 82.5 | 93.35 |
| 第六阶段默认 | 94.9601 | 94.9711 | 2.2244 s | 3.71 min | 87.6 | 94.12 |
| **第七阶段默认** | 94.6212 | 94.6074 | **1.5105 s** | **2.52 min** | 91.6 | **94.43** |

第七阶段相对第六阶段：准确度 −0.36，CPU −32%，综合分 +0.31；相对本会话开头的第五阶段产物：CPU −52%，综合分 +1.08。所有数字仍是本地固定集与 10M 外推，不是官方成绩。

## 为什么可以折叠，以及折叠换来了什么

第六阶段的成本结构里，一棵 511 叶树要 31 ns、特征填充 32 ns、Net 校正 11 ns，而一次架构表查询是 40 ns 左右。既然查表无法避免，把模型**折进被查的那张表**就等于让模型容量完全免费——训练得再大，运行时也只多算一次离线生成。

折叠的代价是：修正只能是 `(方向, 源端口, dx%10, dy%12, 目标端口)` 的函数。以下特征因此被排除：

| 被排除的特征 | 原因 |
| --- | --- |
| `sx/sy/tx/ty`、Block 与行带特征 | 依赖绝对位置 |
| `dx/dy/ax/ay/manhattan`、距离分档 | 连续位移，不是键 |
| `gap_x/gap_y`、`architecture_delay` | 含"跨了哪几条 Gap Line"，取决于绝对位置 |
| `remaining_ax_mod` / `remaining_ay_mod` | `|dx - 网络位移|` 需要差的符号，`dx mod 10` 不够 |
| `architecture_envelope` | 取四方向包络需要 `slopes·(dx,dy)`，依赖 dx/dy 本身；改用"四个方向表项取最大"的可折叠代理 `envelope_proxy` |

实测代价（固定测试集，不含 Net 校正）：

| 特征集 | 1 棵树 | 3 棵 | 25 棵 | 300 棵 |
| --- | ---: | ---: | ---: | ---: |
| 全特征（第六阶段可用） | 93.7799 | 94.7330 | 96.1239 | – |
| 只含不变特征（含 `remaining_ax_mod`、真包络） | 93.7229 | 94.4938 | 95.5489 | 94.7978 |
| 可折叠特征集（本阶段采用） | – | – | – | 94.1816 |

也就是说：折叠损失约 1.9 分准确度（相对 25 棵树），但换来 300 棵树免费 + 树与特征计算从运行时消失。可折叠模型在 100 棵树后基本饱和，继续加树没有收益，这一点与第六阶段的结论一致——瓶颈不在容量，而在被排除的那 5 个特征。

折叠后准确度 94.1816 再叠加**重新拟合**的 5 项 Net 结构校正，得到验证 94.6212 / 测试 94.6074。系数必须重拟合：第五阶段那组是针对 1 棵树弱基线的残差拟合的，换基线后增益从 +1.19 降到 +0.43。

## 实现

1. `tools/train_foldable.py` 用可折叠特征集训练 L1 残差模型（511 叶、300 轮、学习率 0.08、训练集权重 `1/max(y,200)`），只保存模型。
2. `tools/fit_netcorr_folded.py` 用同一套结构特征、以加权 L1 线性规划重拟合 5 个系数；系数只用训练集拟合，验证集用于确认。
3. `tools/fold_model.py` 枚举全部 28,753,920 个键（4 × 208 源 × 120 余数 × 288 目标），构造可折叠特征、预测修正、以 `table_scaled + round(12 × 修正)` 写回 int16 表，并生成不含任何树代码的 `model.hpp`。`32767` 的"不可达"哨兵保留。
4. `src/main.cpp` 增加 `FOLDED_MODEL` 分支：同引脚 0 → 两个 Gap 前缀 → 一次查表 → 线性项 + 校正 → Net 结构校正 → 舍入。第六阶段的实现完整保留在 `#else` 分支里，两个模型共用同一份源码。

表规模与第五阶段相同（57,507,840 B），ELF 57,599,944 B，因为树代码被去掉、表大小不变；提交 ZIP 46,580,810 B，比第六阶段的 41.7 MB 大，折叠后的表项更难压缩。

## 已验证

- 10M 五次 CPU 1.4277～1.5995 s，五次输出 SHA256 全部相同（`87f0875a…`）。
- 接口 11 项通过（LF、CRLF、BOM、无末尾换行、空行、顺序、同引脚零延时、只有表头、非法端口、非法字段数、非法表头）。
- 单线程、进程峰值 RSS 50,432 KiB、容器峰值 806,817,792 B。
- 1M 预测与离线 Python 计算逐行一致（valid 94.6212 / test 94.6074 两侧相同），说明折叠与训练用的是同一套键语义。
- 提交 ZIP 在空容器内独立运行通过，二进制哈希与打包前一致。
- 分组分数显示损失集中在短距离：0–10 从 61.3 降到 56.4、10–30 从 78.8 降到 75.8、30–100 从 91.6 降到 89.9，而 300–1000 从 97.04 升到 97.28。

## 折叠加树的前沿（已测，结论是"不加树"）

折叠表看不见连续位移和绝对位置，这些信息仍然有值。用折叠预测当基线再训一棵能看到全特征的树（`s8_onfold_511l_25r`，511 叶、25 轮、学习率 0.32，训练目标为 `真实值 − 折叠预测`）：

| 树数 | 固定验证 | 固定测试 | 相对折叠基线 |
| ---: | ---: | ---: | ---: |
| 0（折叠 + Net 校正） | 94.6212 | 94.6074 | – |
| 1 | 94.6898 | 94.6457 | +0.46（未加校正） |
| 2 | 95.0142 | 94.9671 | +0.79 |
| 3 | 95.2223 | 95.1903 | +1.01 |
| 5 | 95.4655 | 95.4383 | +1.26 |
| 10 | 95.7545 | 95.7252 | +1.54 |
| 25 | 96.1218 | 96.0961 | +1.91 |

单看准确度，25 棵树能把折叠基线推回 96.1（与第五阶段 25 棵树持平）。但每棵树运行时要 31 ns、特征填充 32 ns，合计 63 ns ≈ 0.52 分；1 棵树只买回 0.46 分准确度（0.37 分），3 棵树买回 1.01 分（0.81 分）但要再花 62 ns（0.52 分）。**所有树数下都是净亏**，与第六阶段的结论一致：瓶颈是"每棵树 31 ns"而不是"基线不够好"。

## 三条零成本猜想（已否证）

1. **小型辅助表**：在折叠预测后叠加一张按 `(方向, 剩余位移 mod 10/12, bit 对/距离档)` 索引的修正表，按训练集均值拟合。480 项、9600 项、6480 项、38880 项四种键全部让验证集变差（−0.02 到 −0.32），说明折叠模型的残差在这些低阶特征上没有可利用结构。
2. **重拟合线性空间项**：折叠预测里的线性项沿用架构推导的斜率（150/127/125），不一定最优。按方向用加权 L1 拟合两个系数后得到 **恰好 0**，即现有斜率已经是最优，残差里没有线性分量残留。
3. **目标端 Net 落点校正**：探测显示所有 Net 都落在目标 SRB 本身（Net 终点是 I 端口，与目标同址），目标端不存在"越界落点"，因此源端的 5 项结构校正没有镜像版本。

## 与文献的关系

折叠这一步是 Map Lookahead 的自然延伸：VTR 的 `router_lookahead_map` 也是把"某一类源端 + 相对位移"的代价预计算成表，让布线器在扩展节点时不再做完整搜索（见 [VTR 8](https://dl.acm.org/doi/pdf/10.1145/3728467)、[`router_lookahead_map.cpp`](https://github.com/verilog-to-routing/vtr-verilog-to-routing/blob/66f35d9f26255de8e01d68f0395b84ed006ca28f/vpr/src/route/router_lookahead_map.cpp)）。它们的表只放**结构代价**，重建性模型另行处理；本赛题没有重建负担，于是可以把学习到的修正也写进同一张表，这是本阶段与文献做法的主要差别。AIR 的"lazy"思想（只计算会影响结果的部分）在这里体现为"只保留键的函数"——代价是必须放弃连续位移。

## 下一步（按收益排序）

1. **给键加一个"网络位移是否超过位移"的比特**。`remaining_ax_mod/ay_mod` 只有在 `dx < 网络位移` 时不可折叠；把这一位加进键即可恢复它们，代价是表翻倍到 115 MB。当前 ZIP 46.6 MB，翻倍后估计 85～90 MB，仍在 100 MB 限制内但需要实测。这是最直接拿回 0.6 分准确度的办法。
2. **短距离用完整模型、长距离用折叠表**。短距离（< 100）只占 20.5% 的行，用第六阶段的树补上可获得约 +0.33 准确度，代价约 13 ns/行（0.1 分），净收益约 +0.16。
3. 可折叠特征集里再找结构项（例如把 Net 校正的 5 项也做成可折叠代理），当前 5 项校正仍依赖绝对位置。

## 复现

```powershell
Set-Location F:\EDA\solution\eda2026
$env:PYTHONPATH = (Resolve-Path solution\tools).Path
$env:EDA_DATA_DIR = (Resolve-Path F:\EDA\EDA-PANGO-SRB-Arch-Delay\EDA-PANGO-SRB-Arch-Delay).Path

# 前置：架构表与特征缓存
.\.venv\Scripts\python.exe -u solution\tools\architecture.py
.\.venv\Scripts\python.exe -u solution\tools\topology_experiment.py `
  --leaves 511 --rounds 25 --learning-rate 0.32 --prefix s3_topology_lr32

# 可折叠模型 → 重拟合 Net 校正 → 折叠成表
.\.venv\Scripts\python.exe -u solution\tools\train_foldable.py 300 0.08 foldable_lr08_300r
.\.venv\Scripts\python.exe -u solution\tools\fit_netcorr_folded.py foldable_lr08_300r 300 net_correction_folded.json
.\.venv\Scripts\python.exe -u solution\tools\fold_model.py --model foldable_lr08_300r --trees 300 `
  --out generated_s7_fold --net-coefficients solution\build\net_correction_folded.json

# 镜像内 PGO 构建与验证
wsl -d Ubuntu-20.04 -- bash -lc 'EDA_REQUEST_CSV_WSL=/mnt/f/EDA/solution/eda2026/solution/build/local_runs/request_1m.csv bash /mnt/f/EDA/solution/eda2026/solution/tools/build_candidate_wsl.sh generated_s7_fold pgo x86-64 stage7_fold'
wsl -d Ubuntu-20.04 -- bash -lc 'EDA_REQUEST_CSV_WSL=/mnt/f/EDA/solution/eda2026/solution/build/local_runs/request_1m.csv bash /mnt/f/EDA/solution/eda2026/solution/tools/validate_stage5_wsl.sh stage7_fold'
.\.venv\Scripts\python.exe -u solution\tools\release_stage7.py
```

`tools/release_stage7.py` 会校验五次 10M 输出一致、单线程、内存低于 2 GB、折叠标记与 Net 系数一致，并要求投影总分严格高于第六阶段，否则拒绝打包。
