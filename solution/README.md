# FPGA 布局布线延时估算器

更新日期：2026-10-21。当前默认提交为第七阶段 `stage7_fold`：残差模型离线折叠进架构表，在线执行“一次查表 + 空间线性项 + Gap 前缀和 + 5 项源端 Net 结构校正”。[`submission.zip`](submission.zip) 与 `stage7_fold_submission.zip` 均只含 `bin/estimate`。

第五阶段高准确度候选仍有记录和导出参数；本机备选包与 ELF 保留在仓库外的 `F:\EDA\solution\stage5_accuracy_submission.zip` 和 `F:\EDA\solution\bin\estimate_stage5_accuracy`。其他旧提交包、旧 ELF 和临时预测大文件已在 2026-10-21 清理，不能假定旧报告中的产物路径仍存在。

## 当前候选结果

固定划分使用 `numpy.random.default_rng(7)` 的 100 万条公开样本随机排列：验证集为 `[300000:350000]`，测试集为 `[350000:400000]`，其余 900,000 条训练。结构校正系数只在训练集拟合，验证集用于选型；测试集已在多轮开发中查看，是本地回归指标，不能代替官方隐藏集成绩。

| 指标 | 当前：第七阶段 | 历史：第六阶段 | 第五阶段高准确度备选 |
| --- | ---: | ---: | ---: |
| 固定验证集准确度 | 94.6212479 | 94.9601032 | 96.2344961 |
| 固定测试集准确度 | 94.6074016 | 94.9711467 | 96.2257253 |
| 10M 单线程 CPU 均值 | 1.3498318 秒 | 2.2244016 秒 | 13.189198 秒 |
| 外推 1B 平均 CPU 分钟 | 2.2497 | 3.7073 | 21.9820 |
| 按本地测试与 CPU 外推的预计总分 | 94.5611 | 94.1233 | 85.9896 |
| 容器峰值内存 | 806,977,536 B | 见第六阶段清单 | 843,706,368 B |
| ELF 大小 | 57,595,848 B | 57,624,656 B | 58,437,424 B |
| 提交 ZIP 大小 | 46,580,381 B | 41,692,644 B | 42,016,719 B |

10M 压测把公开 100 万条请求重复 10 次，五次输出哈希完全相同，并核对了重复内容。当前第七阶段还记录了百万行 C++/Python 预测一致、11 项接口检查和 ZIP 独立运行通过。未执行完整 1B 终测；表中预计总分不是官方成绩。准确度是赛题的相对误差得分，不是精确命中比例。详情见[第七阶段报告](FPGA延时估算第七阶段结果.md)和 `build/stage7_manifest.json`。

2026-10-21 新增 Pango 同机对照：第七阶段/Pango 未校准版测试准确度 94.6074/97.5955，10M CPU 1.7310/2.3186 秒，预计总分 94.2434/96.1442。测时有批次差异，以上历史指标不改写；完整比较及后续优先级见[实现对比与适配评估](实现对比与适配评估.md)。

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

`tools/architecture.py` 从 Arc/Net 构造 10×12 周期图，通过空间势函数、Johnson 重加权和 Dijkstra 生成四方向架构下界表；Gap Line 在预测时单独相加。`tools/train_foldable.py` 只用表键可决定的 20 个特征训练 300 棵残差树，`fit_netcorr_folded.py` 在训练集重拟合 5 个结构校正系数，`fold_model.py` 把模型修正写入 28,753,920 项 int16 表。

当前运行时没有树推理，表键为 `(方向, 源端口, dx%10, dy%12, 目标端口)`，随后叠加空间线性项、Gap 前缀和及 Net 超调/跨墙校正。解析使用校验完整名称的 64 位开放寻址哈希，4 MiB 输入输出缓冲与 8 行批处理隐藏访存延迟。表和参数嵌入 ELF，运行时不依赖 Python、JSON 或外部模型；构建使用 PGO 与 `-march=x86-64`。

第五、六阶段的树实现仍保留在同一份 `src/main.cpp` 中，通过生成头文件的 `FOLDED_MODEL` 宏选择。第五阶段高准确度模型距离小于 100 时用 25 棵树，其他情况用前 18 棵。完整 Pango 工程采用另一种完整位移低秩表示，最新资产核查和比较见[实现对比与适配评估](实现对比与适配评估.md)。

## 构建和验证

在仓库根目录的 PowerShell 中，先按[根目录 README](../README.md)准备 Python、官方数据、请求 CSV 和 WSL/Docker。利用已跟踪模型重建当前默认候选：

```powershell
$env:PYTHONPATH = (Resolve-Path solution\tools).Path
$env:EDA_DATA_DIR = (Resolve-Path ..\..\EDA-PANGO-SRB-Arch-Delay\EDA-PANGO-SRB-Arch-Delay).Path
.\.venv\Scripts\python.exe -u solution\tools\architecture.py
.\.venv\Scripts\python.exe -u solution\tools\fold_model.py `
  --model foldable_lr08_300r --trees 300 --out generated_s7_fold `
  --net-coefficients solution\build\stage7_net_correction_folded.json
wsl -d Ubuntu-20.04 -- env EDA_REQUEST_CSV_WSL=/mnt/f/EDA/solution/eda2026/solution/build/local_runs/request_1m.csv bash /mnt/f/EDA/solution/eda2026/solution/tools/build_candidate_wsl.sh generated_s7_fold pgo x86-64 stage7_fold
wsl -d Ubuntu-20.04 -- env EDA_REQUEST_CSV_WSL=/mnt/f/EDA/solution/eda2026/solution/build/local_runs/request_1m.csv bash /mnt/f/EDA/solution/eda2026/solution/tools/validate_stage5_wsl.sh stage7_fold
```

请求 CSV 路径须存在，首行无表头时构建/验证脚本会补齐。本机已保留 `solution/build/local_runs/request_1m.csv`。构建工作区默认位于 WSL `$HOME/eda2026-work`，ELF 和结果回写到 `solution/bin` 与 `solution/build/local_runs`；构建脚本不会覆盖默认提交。

第七阶段需按折叠表的量化规则重新核对 Python/C++ 输出；`verify_cpp.py` 目前只适用于树模型。新报告核验并归档到 `build/stage7_fold_*.json` 后，才运行 `release_stage7.py` 打包。发布脚本不会自动重算准确度或执行 ZIP 独立运行检查；不能用旧指标证明新二进制已验证。完整重训和历史候选流程见各阶段报告。
