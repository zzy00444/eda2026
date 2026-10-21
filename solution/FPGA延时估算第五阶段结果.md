# FPGA 延时估算第五阶段结果

记录日期：2026-10-20。本阶段纠正了此前只看准确度、忽略时间分线性扣分的选型方式，并完成源端 Net 结构校正和端口解析优化。

本文为第五阶段历史快照，“默认”指本阶段当时的选择；现行产物见 [README](README.md)。2026-10-21 已清理旧提交包、旧 ELF 和运行缓存；高准确度备选包保留在仓库外的 `F:\EDA\solution\stage5_accuracy_submission.zip`，其余旧产物路径不保证仍存在。

## 最终结果和默认选择

所有结果均为本地固定集准确度和本机镜像压测；没有执行完整 1B 终测，也没有获得新的官方隐藏集成绩。

| 候选 | 固定验证准确度 | 固定测试准确度 | 外推 1B 平均 CPU 分钟 | 预计总分 |
| --- | ---: | ---: | ---: | ---: |
| 旧第四阶段级联（本会话复测对照） | 96.0936254 | 96.0784893 | 26.1780 | 83.7738 |
| 第五阶段总分优先，当时默认 | 94.9601032 | 94.9711467 | 5.0477 | **93.4531** |
| 第五阶段高准确度 | **96.2344961** | **96.2257253** | 21.9820 | 85.9896 |

预计总分使用表中的固定测试准确度、10M CPU 均值线性外推以及一致性分 100 计算。默认候选相对旧第四阶段预计总分提高约 9.6793；代价是准确度降低约 1.1073。高准确度候选的测试准确度提高约 0.1472，外推 CPU 缩短约 4.20 分钟，预计总分提高约 2.2158。默认按验证集准确度和测量时间选择，表中测试指标用于本地回归比较。

## 为什么改变选型目标

当 CPU 时间低于 1800 秒、输出一致性满分时，官方公式等价于：

```text
total = 0.8 × accuracy + 20 - CPU_seconds / 120
```

每多用 1 秒 CPU，总分降低约 0.008333；提高 1 分准确度最多抵消 96 秒 CPU。第三至第四阶段准确度只提高约 0.0256，允许的额外十亿条 CPU 预算仅约 2.46 秒，而历史 10M 压测外推增加了约 116.6 秒。此前第四阶段文档中的“净收益”只能理解为准确度收益，不能理解为官方总分提高。

本阶段先在验证集比较树前缀，编译 1/3/6/10 棵普通查询前缀以及纯架构基线，测量其时间。最终保留两个用途明确的候选：总分优先版本仅执行主模型第 1 棵树；高准确度版本继续使用距离小于 100 时 25 棵、其余 18 棵的门控，去掉中距离 10 棵级联。

## 结构校正和训练隔离

两个候选都从 `s3_topology_lr32_511l_25r` 取得残差树，再追加 5 项线性校正。源 Net 位移由 `SRB_Net.json` 给出，Gap 和 Block 结构来自 `SRB_Gap.json`：

1. 源 Net 名义落点进入允许水平跨越的 Block 时，按 Block 宽度延伸落点；不可跨越的 Block 0 不参与。
2. 计算 `gap(source, landing) + gap(landing, target) - gap(source, target)`。落点在端点包围盒内时增量为零；否则只需通过前缀表计算多穿越两次的 Gap。
3. 根据落点到目标是否需要返回跨墙、两端行区间内是否有门，以及目标是否位于墙列范围，形成四项互斥的墙特征。其数值使用 JSON 中的跨越延时 292，额外幅度由训练集拟合。

5 个系数通过训练集上的加权 L1 线性规划拟合，权重为 `1 / max(truth, 200)`。验证集比较不校正和校正方案，系数不使用验证或测试标签拟合。

| 系数对应特征 | 总分优先 | 高准确度 |
| --- | ---: | ---: |
| Gap 落点额外穿越 | 0.9405376344 | 0.3002822135 |
| 返回跨墙且行区间有门 | 1.6152054795 | 0.8922935290 |
| 返回跨墙且行区间无门 | 1.9533377522 | 1.1722847364 |
| 目标位于落点侧 | 0.3491780822 | 0.1366065439 |
| 目标位于墙列范围 | 1.3123287671 | 0.8827633277 |

划分仍为 `numpy.random.default_rng(7)` 的公开 100 万条原始顺序随机排列：验证 `[300000:350000]`，测试 `[350000:400000]`，其余 900,000 条训练。固定测试集在历史开发中被多次查看，不能视为从未使用的最终泛化评估。

此外，端口解析使用离线生成的 64 位完美哈希，同时验证名称长度和完整键，保持未知端口拒绝行为；没有采用仅信任哈希槽的宽松解析。两个程序均流式读取，输出保留原顺序，无答案缓存。

## 完整验证

在 `fpga-ziguang:latest`（Ubuntu 22.04.5、GCC 11.4.0）内编译，CPU 为本机 i9-14900HX，限制单 CPU 配额和 2 GiB 容器内存。最终采用 `-march=x86-64`，不要求 `-march=native` 生成的本机专有指令；使用 PGO 和 `-ffp-contract=off`。

| 检查 | 总分优先 | 高准确度 |
| --- | ---: | ---: |
| 公开 1M C++/Python 不一致数 | 0 | 0 |
| 同引脚零延时 | 1/1 | 1/1 |
| 10M 压测次数 | 5 | 5 |
| 10M CPU 范围，秒 | 2.968052～3.071520 | 12.935337～13.433629 |
| 10M CPU 平均，秒 | 3.028637 | 13.189198 |
| 进程峰值 RSS | 44,032 KiB | 44,800 KiB |
| 容器峰值内存 | 844,095,488 B | 843,706,368 B |
| 观察到的最大线程数 | 1 | 1 |
| 五次输出 SHA256 | 完全相同 | 完全相同 |
| ZIP 在无外部模型目录独立运行 | 通过 | 通过 |

每个 10M 请求文件由公开 1M 请求重复 10 次构造。每次压测的完整输出还与 1M 参考输出重复 10 次的 SHA256 比较。相同会话旧第四阶段对照复测 3 次，CPU 平均为 15.706811 秒/10M。LF、CRLF、BOM、无末尾换行、空行、端点顺序、非法表头/字段/端口等接口检查均通过。

独立审阅还对全部 496 个端口哈希和 300,000 组随机芯片内坐标的结构特征做了对照，没有发现不一致。这不能代替官方隐藏集验证。

## 提交产物和证据

| 产物 | 总分优先 | 高准确度 |
| --- | --- | --- |
| ZIP | `submission.zip` / `stage5_score_submission.zip` | `stage5_accuracy_submission.zip` |
| ELF | `bin/estimate` / `bin/estimate_stage5_score` | `bin/estimate_stage5_accuracy` |
| 生成目录 | `build/generated_s5_score`；默认同步到 `build/generated` | `build/generated_s5_accuracy` |
| ZIP 字节数 | 41,688,436 | 42,016,719 |
| ELF 字节数 | 57,726,920 | 58,437,424 |

总分优先 ELF SHA256：`81f9073a3ad0f1eafeb384612172e973871cec87c12152d73654bda3474b0154`。

默认 ZIP SHA256：`e6503167549cdd41aa3c7a1e4b0cfc2dd5e3ec7814b329ce885d54a219971785`。

高准确度 ELF SHA256：`52aff6c10ba5f63899fead002e69c504374a77d3a0301007d0cbf327cd3433b8`。

高准确度 ZIP SHA256：`e751aa3d8a74090fce6ce76813686d9739a39a924abfe9ee93296610b90ce4a4`。

核心证据为 `build/stage5_manifest.json`、`build/stage5_score_metrics.json`、`build/stage5_accuracy_metrics.json`，各候选的 `stage5_generated_s5_*_benchmark_10m.json`、`*_integration.json` 和 `stage5_*_standalone.json`。旧第四阶段包保留为 `stage4_submission.zip`。本 Git 仓库直接管理源码、模型参数、导出文件和关键验证证据；原工作目录另有 `stage5_source_archive.zip` 快照，但提交仓库时不必重复保存该压缩副本。

## 复现高准确度候选

默认候选的命令见 README。高准确度候选在相同主模型基础上运行：

```powershell
.\solution\.venv\Scripts\python.exe -u solution\tools\net_correction_experiment.py `
  --short-trees 25 --fast-trees 18 --cascade-trees 0 --l1-only `
  --output-dir net_correction_f18_c0
.\solution\.venv\Scripts\python.exe -u solution\tools\export_cpp.py `
  --model s3_topology_lr32_511l_25r --trees 25 --fast-trees 18 `
  --refine-distance 100 --fast-categories --packed-ports `
  --net-correction net_correction_f18_c0/metrics.json --output-dir generated_s5_accuracy
wsl -d Ubuntu-20.04 -- bash /mnt/f/EDA/solution/eda2026/solution/tools/build_candidate_wsl.sh generated_s5_accuracy pgo x86-64
```

`verify_cpp.py --all-rows` 核对全公开集；`validate_stage5_wsl.sh` 重做接口和压测；`check_stage5_packages_wsl.sh` 验证独立运行。`release_stage5.py` 检查证据后打包并同步默认产物。重新训练或编译后必须重新核验，不能沿用旧哈希作为新二进制的验证凭据。
