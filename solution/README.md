# FPGA 布局布线延时估算器

当前默认提交按官方总分选择，采用“架构表 + 1 棵残差树 + 5 项源端 Net 结构校正”。提交文件为 [`submission.zip`](submission.zip)，仅含 `bin/estimate`。另保留 [`stage5_accuracy_submission.zip`](stage5_accuracy_submission.zip)，用于需要更高准确度的场景。旧第四阶段级联包已备份为 `stage4_submission.zip`。

## 当前候选结果

固定划分使用 `numpy.random.default_rng(7)` 的 100 万条公开样本随机排列：验证集为 `[300000:350000]`，测试集为 `[350000:400000]`，其余 900,000 条训练。结构校正系数只在训练集拟合，验证集用于选型；测试集已在多轮开发中查看，是本地回归指标，不能代替官方隐藏集成绩。

| 指标 | 默认：总分优先 | 高准确度候选 |
| --- | ---: | ---: |
| 固定验证集准确度 | 94.9601032 | 96.2344961 |
| 固定测试集准确度 | 94.9711467 | 96.2257253 |
| 全公开集 C++/Python 不一致行数 | 0 | 0 |
| 同引脚零延时样本 | 1/1 正确 | 1/1 正确 |
| 10M 单线程 CPU 秒（各五次） | 2.968～3.072 | 12.935～13.434 |
| 外推 1B 平均 CPU 分钟 | 5.0477 | 21.9820 |
| 本地测试准确度 + CPU 外推的预计总分 | 93.4531 | 85.9896 |
| 容器峰值内存 | 844,095,488 B | 843,706,368 B |
| ELF 大小 | 57,726,920 B | 58,437,424 B |
| 提交 ZIP 大小 | 41,688,436 B | 42,016,719 B |

10M 压测把公开 100 万条请求重复 10 次，各候选五次输出哈希完全相同，并逐字节核对重复内容。未执行完整 1B 终测；表中预计总分不是官方成绩。相同会话重新测得旧第四阶段的预计总分为 83.7738。默认方案降低了准确度、显著缩短 CPU 时间；高准确度候选则同时改善了旧方案的准确度和耗时。见 [`FPGA延时估算第五阶段结果.md`](FPGA延时估算第五阶段结果.md) 和 `build/stage5_manifest.json`。

## 赛题接口和评分

赛题提供一个 120×550 的 SRB 阵列，以及固定的 `SRB_Inst.json`、`SRB_Port.json`、`SRB_Arc.json`、`SRB_Net.json` 和 `SRB_Gap.json` 五个架构文件。正式架构和延时数据固定；公开请求只覆盖部分端口和样本。按官方 Q&A，隐藏请求会覆盖架构描述中的各类 output 端口；作为请求端点时，除 `I*` 端口以外的其他端口类型都可能出现。请求顺序可能打乱。正式隐藏集与公开集不同且原则上不重复；打榜阶段的 1B 页面则会把公开 100 万条随机复制扩充到 10 亿条。

程序调用方式：

```bash
bin/estimate -in request.csv -out result.csv
```

输入表头必须为 `From,To`，输出表头为 `From,To,delay`。输出必须保留输入端点和行顺序，`delay` 为非负整数；同一引脚到自身的延时为 0。当前实现另外支持 LF、CRLF、UTF-8 BOM、文件尾无换行和空行，并在表头、字段数或端口非法时返回非零状态；这些属于实现兼容行为，不是赛题另行规定的评分项。

准确度按每条查询计算：当真实延时大于 0 时，单条得分为 `100 × (1 - tanh(4 × |estimate - truth| / truth))`；真实延时为 0 时，只有估计值为 0 才得满分。准确度是所有查询单条得分的平均值。总分按赛题规则计算：

```text
总分 = 0.80 × 准确度 + 0.15 × 时间分 + 0.05 × 一致性分
时间分 = max(0, 1 - 平均 CPU 秒数 / 1800) × 100
```

正式评测使用约 10 亿条查询并运行 5 次。单线程、禁止 GPU；`avg_time` 为 5 次 CPU time 的平均值。本地按 `user + sys` 测量，包含解析和读写消耗的 CPU，不含等待 I/O 的墙钟时间。任一次运行超过 1 小时或内存超过官方规定的 2 GB，作品无效并判 0 分；5 次输出完全一致时一致性分为 100，否则为 0，且准确度按 5 次结果的平均准确度计算。`avg_time` 超过 30 分钟时仅时间分为 0，准确度和一致性仍按规则计算。提交压缩包上限为 100 MB；本机 Docker 的 `--memory=2g` 按 2 GiB（2^31 字节）配置，属于本地验证设置。

## 算法结构

`tools/architecture.py` 根据 Arc、Net 和 Gap JSON 生成周期架构表。`tools/net_correction_experiment.py` 根据源 Net 远端落点计算额外 Gap 穿越和 Block 返回/跨越特征，再用训练集拟合 5 个线性系数。运行时用前缀表和稀疏分支计算校正，无查询答案缓存。

默认仅使用 `s3_topology_lr32_511l_25r` 的第 1 棵树；高准确度版本距离小于 100 时用 25 棵，其他情况用前 18 棵。两者均追加各自拟合的 5 项结构校正，并使用验证键值的 64 位端口完美哈希。模型和约 57.5 MB 架构表嵌入 ELF，运行时不依赖 Python、JSON 或外部模型。最终构建使用 `-march=x86-64`，不要求本机专有指令。

## 构建和验证

在仓库根目录 `F:\EDA\solution\eda2026` 的 PowerShell 中，先确保 WSL 的 Ubuntu-20.04 已启用 Docker Desktop 集成，并设置赛题数据及请求文件路径，然后执行以下命令。训练脚本的输出目录相对于 `solution\build`，导出目录参数也会写入该 build 目录：

```powershell
$env:PYTHONPATH = (Resolve-Path solution\tools).Path
$env:EDA_DATA_DIR = (Resolve-Path ..\..\EDA-PANGO-SRB-Arch-Delay\EDA-PANGO-SRB-Arch-Delay).Path
.\.venv\Scripts\python.exe -u solution\tools\topology_experiment.py `
  --leaves 511 --rounds 25 --learning-rate 0.32 --prefix s3_topology_lr32
.\.venv\Scripts\python.exe -u solution\tools\net_correction_experiment.py `
  --short-trees 1 --fast-trees 1 --cascade-trees 0 --l1-only `
  --output-dir net_correction_f1_c0
.\.venv\Scripts\python.exe -u solution\tools\export_cpp.py `
  --model s3_topology_lr32_511l_25r --trees 1 --fast-categories --packed-ports `
  --net-correction net_correction_f1_c0/metrics.json --output-dir generated_s5_score
wsl -d Ubuntu-20.04 -- bash -lc 'EDA_REQUEST_CSV_WSL=/mnt/f/EDA/pango_delay_handover_v3/local_build/request.csv bash /mnt/f/EDA/solution/eda2026/solution/tools/build_candidate_wsl.sh generated_s5_score pgo x86-64 stage5_score'
.\.venv\Scripts\python.exe -u solution\tools\verify_cpp.py `
  --result solution/build/generated_s5_score_result.csv `
  --generated-dir generated_s5_score --name stage5_score --all-rows
```

`tools/validate_stage5_wsl.sh` 对两个候选执行接口和 10M 压测；`tools/check_stage5_packages_wsl.sh` 验证 ZIP 解压后无外部模型依赖。构建工作区默认位于 WSL `$HOME/eda2026-work`，ELF 和报告分别回写到 `solution/bin` 与 `solution/build/local_runs`；构建脚本不会自动覆盖默认提交。应在 `fpga-ziguang:latest` 镜像内编译，避免宿主机 C++ 运行库不兼容。

训练依赖和完整实验记录保存在 `build/`；临时实验脚本不属于提交包。
