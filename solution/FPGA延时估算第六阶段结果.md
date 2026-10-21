# FPGA 延时估算第六阶段结果

记录日期：2026-10-20。本阶段只改运行时与端口解析，模型、Net 结构校正系数、架构表全部保持不变，因此预测逐字节不变、准确度不变；综合分的提升全部来自 CPU 时间。所有数字都是本地固定集与 10M 压测外推，不是官方成绩。

本文为第六阶段历史快照；现行产物见 [README](README.md)。2026-10-21 已清理第六阶段旧二进制、重复架构表和预测 CSV，保留了模型、生成源码、参数和验证报告，重跑需先重建产物。

## 结论

| 候选 | 固定验证 | 固定测试 | 10M CPU 均值 | 1B 外推 | 时间分 | 预计总分 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 第五阶段默认（本会话复测，同机交错） | 94.9601 | 94.9711 | 3.1562 s | 5.26 min | 82.5 | 93.35 |
| **第六阶段默认** | 94.9601 | 94.9711 | **2.2244 s** | **3.71 min** | 87.6 | **94.12** |
| 记录中的第五阶段默认 | 94.9601 | 94.9711 | 3.0286 s | 5.05 min | 83.2 | 93.45 |

同一台机器、同一镜像、同样 `--cpus=1 --memory=2g` 条件下交错测量：第五阶段产物 3.1562 s/10M，第六阶段产物 2.1990 s/10M，**CPU 降低 30.3%**，预计总分提高约 0.83。第五阶段记录值 3.0286 s 是在更早、机器更凉的状态下测的，跨时段直接比较会被 3～5% 的漂移污染，所以本阶段主要引用交错对照的比例。

1M 与 10M 输出的 SHA256 与第五阶段完全一致（10M：`2cb05566105cc53e99d9fd4132bcd03d111458eaeb5bd8835208fcca7f879dc0`），所以准确度逐位不变不是推断而是测出来的。

## 为什么这一阶段先做运行时

官方总分 `0.8×准确度 + 0.15×时间分 + 0.05×一致性分`，在 CPU < 30 分钟、五次一致时等价于：

```text
总分 = 0.8 × 准确度 + 20 - CPU秒数 / 120
```

即 1 秒 CPU 值 0.00833 分，1 分准确度值 0.8 分。11 条消融实验给出当前单行成本（1M 请求、非 PGO 编译、逐项关闭）：

| 变体 | CPU 秒/1M | 与 base 之差 | 含义 |
| --- | ---: | ---: | --- |
| base | 0.2369 | – | 完整体 |
| `noarch`（架构表载入换成常数） | 0.1547 | −0.0822 | 57 MB 表的随机访问延迟 82 ns |
| `notree`（跳过残差树） | 0.2056 | −0.0313 | 一棵 511 叶树 31 ns |
| `nocorr`（跳过 Net 校正） | 0.2257 | −0.0112 | 5 项结构校正 11 ns |
| `nopref`（去掉预取） | 0.2497 | +0.0128 | 原有的行内预取已省 13 ns |
| `noest`（只解析+读写） | 0.0812 | −0.1557 | 解析与 I/O 共 81 ns |

结论：固定成本（解析、I/O、架构表）占 60% 以上，一棵树只占 13%。在这种结构下继续加树并不划算——见下面 Pareto 表——所以先把固定成本压下来，再谈模型。

## 四项运行时改动

1. **显式 4 MiB 输入输出缓冲**。原来每行调用 `getline` 加两次 `fwrite`，单线程下也各带一次锁操作；现在自己维护缓冲，用 `memchr` 找行尾、`memcpy` 拼输出。
2. **解析去掉异常与 `std::from_chars`**。热路径改成数字循环并返回错误字符串，只有真正出错时才构造异常，避免每行两次可能抛出的调用；坐标上限做了溢出保护，接受/拒绝的分类与原来一致。
3. **架构表预取 + 8 行一批的软件流水**。行被分两批：一批解析并"准备"（算出表索引、发出预取），另一批求值，于是每次查表前面有整整一批工作在排队，而不是十几条指令。
4. **端口完美哈希改成 L1 可驻留的开放寻址表**。原来的置换表是 65536 个 `short`（128 KB），会被 57 MB 的架构表从 cache 里挤掉，每个引脚都要再吃一次冷缺失；现在是 2048 槽（键 16 KB + 编号 4 KB），仍然做完整 64 位键比较，行为不变。

第 3 项有一个反直觉的教训值得记下来：**只做流水线不定 noinline 会慢 19%**。导出的单棵树是一个约 22 KB 的表达式，编译器为每个缓冲行复制了一份，文本段因此增长 123 KB，指令缓存被冲垮。把求值函数标记 `noinline`、只保留一份拷贝之后才变成净收益。变体 ELF 体积可以直接看出这个问题：57735200 B（noinline）对 57854000 B（复制）。

## 候选 Pareto 表

`tools/pareto.py` 会把 `build/local_runs` 里每个 `<name>_result.csv` 与对应压测报告配对，按固定 rng7 划分重算准确度并给出投影总分。当前点：

| 候选 | 树数 | Net 校正 | 固定测试准确度 | 10M/1M CPU | 1B 外推 | 预计总分 |
| --- | ---: | :---: | ---: | ---: | ---: | ---: |
| `stage6_t0` 纯架构下界 | 0 | 否 | 91.3959 | 0.1647 s/1M | 2.75 min | 91.74 |
| **`stage6_score` 默认** | 1 | 是 | **94.9711** | 2.2244 s/10M | **3.71 min** | **94.12** |
| `stage6_t3` | 3 | 否 | 94.7330 | 0.3317 s/1M | 5.53 min | 93.02 |
| `stage6_t6` | 6 | 否 | 95.3925 | 0.4357 s/1M | 7.26 min | 92.68 |
| 第五阶段高准确度 | 25 | 是 | 96.2257 | 13.1892 s/10M | 21.98 min | 85.99 |

两个可直接用于决策的边际结论：

- **第 2、3 棵树**：从默认候选换成 `stage6_t3`（多 2 棵树、去掉校正）准确度反而低 0.24，CPU 多 109 秒。要划算，一棵树必须买回 0.32 分准确度，而实测第 2 棵树只有约 0.30，之后递减到 0.06～0.15。
- **结构校正的性价比约为加树的 10 倍**：5 项 Net 校正只花 11 ns（约 11 秒/1B，0.09 分），却买回 1.19 分准确度（0.95 分）。这是本阶段最重要的方向性结论——**继续在低成本的确定性结构项上做文章，比堆模型划算**。

## 已验证

- 1M 输出 SHA256 `477f44d6…` 与第五阶段逐字节相同；10M 五次输出 SHA256 全部相同且等于 `2cb05566…`。
- 接口检查 11 项通过：LF、CRLF、UTF-8 BOM、无末尾换行、空行、顺序与端点、同引脚零延时、只有表头、非法端口、非法字段数、非法表头。
- 单线程（采样最大线程数 1）、进程峰值 RSS 50,944 KiB、容器峰值 806,952,960 B，均远低于 2 GB。
- 提交包仅含 `bin/estimate`，权限 100755，41,692,644 B（上限 100 MB）。
- 端到端可复现：`tools/release_stage6.py` 会重新校验五次压测一致性、输出哈希与第五阶段的等价性，然后才允许打包和写 `build/stage6_manifest.json`。

## 试过但没采用

记录在这里是为了避免重复投入。

1. **绝对位置残差特征**（`sx%10`、`sy%12` 等）。怀疑架构表的周期结构没有利用源端绝对相位；实测短距离残差与这些相位几乎无关：按 `(sx%10,sy%12)` 分组的组均值散布只有 4.5 ps，而组内标准差是 62.6 ps。加这类特征无望。
2. **`(方向, 源, 目标)` 偏差表**。训练集拟合每组平均残差、收缩后加到测试集：240k 项换 0.185 准确度，加上距离分档（2.4M 项）换 0.32 准确度。折算只有 0.15～0.26 分，却要引入一次新的随机访存（约 0.17～0.34 分），净收益在噪声以内，未做进 C++。
3. **更深的流水线**（8 行一批 + 无 noinline、以及 1 行深度的两段式）。前者慢 19%（见上），后者慢 9%；只有在 noinline 修复后才转正。
4. **`--generic` 节点表求值**。未采用：展开式虽然分支多，但每层的特征下标是编译期常量、载入可以并行发射；节点表版本每层多一次依赖载入，实测并不更快。

## 复现

```powershell
Set-Location F:\EDA\solution\eda2026
$env:PYTHONPATH = (Resolve-Path solution\tools).Path
$env:EDA_DATA_DIR = (Resolve-Path F:\EDA\EDA-PANGO-SRB-Arch-Delay\EDA-PANGO-SRB-Arch-Delay).Path

# 架构表与模型（Git 不保存，克隆后必须先跑第一步）
.\.venv\Scripts\python.exe -u solution\tools\architecture.py
.\.venv\Scripts\python.exe -u solution\tools\topology_experiment.py `
  --leaves 511 --rounds 25 --learning-rate 0.32 --prefix s3_topology_lr32
.\.venv\Scripts\python.exe -u solution\tools\net_correction_experiment.py `
  --short-trees 1 --fast-trees 1 --cascade-trees 0 --l1-only --output-dir net_correction_f1_c0

# 导出（生成的 model.hpp/architecture.S/export.json 已纳入仓库，可跳过上面三步直接导出）
.\.venv\Scripts\python.exe -u solution\tools\export_cpp.py `
  --model s3_topology_lr32_511l_25r --trees 1 --fast-categories --packed-ports `
  --net-correction net_correction_f1_c0/metrics.json --output-dir generated_s6_score

# 镜像内 PGO 构建 + 接口与 10M 五次压测 + 打包校验
wsl -d Ubuntu-20.04 -- bash -lc 'EDA_REQUEST_CSV_WSL=/mnt/f/EDA/solution/eda2026/solution/build/local_runs/request_1m.csv bash /mnt/f/EDA/solution/eda2026/solution/tools/build_candidate_wsl.sh generated_s6_score pgo x86-64 stage6_score'
wsl -d Ubuntu-20.04 -- bash -lc 'EDA_REQUEST_CSV_WSL=/mnt/f/EDA/solution/eda2026/solution/build/local_runs/request_1m.csv bash /mnt/f/EDA/solution/eda2026/solution/tools/validate_stage5_wsl.sh stage6_score'
.\.venv\Scripts\python.exe -u solution\tools\release_stage6.py
```

请求 CSV 首行不是 `From,To` 时，构建/验证脚本会通过 `tools/ensure_request_header.sh` 自动补齐；`solution/build/local_runs/request_1m.csv` 是从官方答案 CSV 重新生成并保留表头的 1M 样本。
