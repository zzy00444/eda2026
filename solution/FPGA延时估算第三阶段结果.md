# FPGA 延时估算第三阶段结果

记录日期：2026-10-19。

本报告为第三阶段历史快照，文中“当前默认”仅指记录时的状态。现行第七阶段默认见 `README.md`。2026-10-21 已清理旧包、旧二进制与运行缓存，以下旧产物路径不保证仍存在；30 分钟是本阶段开发目标，正式硬超时为单次 1 小时。

按最新目标，本阶段优先达到准确度 96 分以上，并保持单线程 10 亿查询 CPU 时间严格小于 30 分钟。当前候选的固定测试分数为 **96.0529**；在本机容器内，三次 1000 万条实测外推为 **23.82～25.77 分钟/10 亿条**。96.5 分目标尚未达到。

时间数据来自 1000 万条完整 CSV 读入、预测、写出测试，包含程序的用户态和内核态 CPU 时间。目前没有执行完整的 10 亿条终测；隐藏集分数和官方机器上的实际运行时间仍需正式评测确认，不能把外推值当作官方成绩。

## 最终候选与准确度

| 项目 | 结果 |
| --- | ---: |
| 模型 | `s3_topology_lr32_511l_25r` |
| 训练/验证/测试样本数 | 900,000 / 50,000 / 50,000 |
| 固定验证集分数 | 96.0678948 |
| 固定测试集分数 | 96.0528653 |
| 第二阶段固定测试分数 | 95.3849698 |
| 相对第二阶段提升 | +0.6678955 分 |
| 验证集/测试集 C++ 与 Python 不一致行数 | 0 / 0 |
| 全部公开 1,000,000 条 C++ 与 Python 不一致行数 | 0 |
| 同引脚零延时样本 | 1 条，正确 |

使用 `numpy.random.default_rng(7).permutation(1000000)`：验证集取 `[300000:350000]`，测试集取 `[350000:400000]`，其余 900,000 条训练。固定验证集和测试集标签不参与训练。开发过程中已多次查看这个固定测试集，因此它是本地回归指标，后续泛化判断应增加新的留出划分或使用官方隐藏集。

## 算法与速度优化

1. 保留由架构 Arc/Net 构造的周期最短路代价表。四个方向的势函数下界取最大值，作为补充特征，帮助识别短距离和长线首段带来的下界误差。
2. 添加 Block 的端点距离、行带、边缘距离和穿越方向等特征，共 73 个特征。类别交互特征与训练时的全量端口类别编码保持一致。
3. LightGBM 使用 L1 残差目标、学习率 0.32、最多 511 叶、25 棵树。曼哈顿距离小于 100 的查询使用全部 25 棵树，其他查询使用前 18 棵；判断只依赖输入特征。固定验证集约 20.56% 的查询走完整模型。
4. 类别编码使用直接映射；类别分裂使用共享位图；Gap Line 使用前缀和。根据架构表中两方向代价差的最大值，安全跳过不可能改善下界的额外查表。
5. 编译采用 GCC 的 PGO 优化代码布局。采样只读取输入查询，不读取延时答案，也不缓存查询预测结果。
6. 特征、阈值与累加使用 `double`，并使用 `-ffp-contract=off`，使特征算术和整数舍入与 Python 参考实现一致。

生成文件包含 25,525 个节点。程序运行时只执行编译进 ELF 的模型和架构表，无 Python、模型文件或 JSON 依赖，无公开查询答案缓存。

## 单线程、内存和输出稳定性

环境为 Windows + WSL2 `Ubuntu-20.04` + Docker Desktop，镜像 `fpga-ziguang:latest`，GCC 11.4.0，CPU 为 Intel Core i9-14900HX。容器限制：`--cpus=1 --memory=2g --memory-swap=2g`。

| 轮次 | 1000 万条 CPU 秒 | 1000 万条墙钟秒 | 外推 10 亿条 CPU 分钟 |
| --- | ---: | ---: | ---: |
| 1 | 14.480111 | 15.265417 | 24.1335 |
| 2 | 14.293921 | 14.944614 | 23.8232 |
| 3 | 15.464497 | 16.232650 | 25.7742 |

最慢一轮外推距 30 分钟约有 4.23 分钟余量。每轮输出 10,000,000 行，输入由公开 100 万条查询重复 10 次构造；三轮完整输出都与 100 万条参考输出的重复内容做 SHA256 校验。

- 观测到的估算程序线程数：1。
- 最大进程 RSS：44,288 KiB，约 43.25 MiB。
- 容器峰值内存：879,222,784 字节，约 838.5 MiB，包含容器内文件缓存等开销。
- 三轮输出 SHA256：`e7b1b1c925e3520e6992c1b994be88e00bc6c644c763f53e9a3b14d60b98a834`。
- 完整记录：`build/stage3_benchmark_10m.json`。

`--cpus=1` 限制容器的 CPU 配额；程序自身没有创建计算线程。测量辅助脚本在独立进程中取样线程数、计算哈希，其 CPU 时间不计入估算程序的 CPU 时间。

## 分组结果与下一步空间

| 固定测试集曼哈顿距离 | 条数 | 分数 |
| --- | ---: | ---: |
| 0～9 | 152 | 70.2061 |
| 10～29 | 1,073 | 81.6309 |
| 30～99 | 9,038 | 93.3825 |
| 100～299 | 26,285 | 96.7738 |
| 300～999 | 13,452 | 97.8807 |

下一步争取 96.5 时，应优先改善短距离的可达路径代价和局部绕行特征，再考虑更复杂的模型。短距离精确局部查表可以只覆盖小位移和首尾端口类型，并通过架构生成，避免依赖重复公开查询。单纯增加大树已经可以使离线分数超过 96.5，但现有实现的时间明显超标，不能作为满足本阶段约束的提交。

后续候选仍必须同时满足分数和严格小于 30 分钟的时间条件。建议以最慢一轮外推不超过 26 分钟作为开发控制线，给数据分布和机器波动保留余量。

## 提交与归档

- 当前默认提交包：`submission.zip`；同内容的阶段快照：`stage3_submission.zip`。
- 当前默认 ELF：`bin/estimate`；阶段快照：`bin/estimate_stage3`。
- ZIP 仅包含 `bin/estimate`，保留 Linux 执行权限，大小为 42,022,269 字节。
- ELF SHA256：`d9824002a0b599968a042ae4c4633dc5fc70ae94073595155caff79990150143`。
- ZIP SHA256：`544c6ec924e672d7feab9ac1523d025b72fff7403057edc41b2dc69fac53fddd`。
- 第二阶段原提交包保留为 `stage2_submission.zip`；原 `stage2_source_archive.zip` 保持不变。
- 本阶段源码与构建材料：`stage3_source_archive.zip`，文件清单和哈希见 `build/stage3_archive_manifest.json`。
- LF、CRLF、BOM、无结尾换行、空行、顺序、同点零延时和非法输入检查均通过，见 `build/stage3_integration.json`。
- ZIP 在不包含模型/架构文件的临时目录中独立运行通过，见 `build/stage3_standalone.json`。

## 复现

以下命令从 `F:\EDA` 执行，需要原始公开数据目录、现有 Python 虚拟环境和启用 WSL 集成的 Docker Desktop：

```powershell
# 若架构表缺失，先离线重新生成。
.\solution\.venv\Scripts\python.exe -u solution\tools\architecture.py

# 重新训练。不要复用来自不同数据/旧特征定义的 stage3_features.pkl。
.\solution\.venv\Scripts\python.exe -u solution\tools\topology_experiment.py `
  --leaves 511 --rounds 25 --learning-rate 0.32 --prefix s3_topology_lr32

# 导出 double 模型、类别位图、18/25 棵树门控和安全下界判断。
.\solution\.venv\Scripts\python.exe -u solution\tools\export_cpp.py `
  --model s3_topology_lr32_511l_25r --trees 25 --fast-trees 18 `
  --refine-distance 100 --fast-categories --output-dir generated_s3_final_pgo

# 在镜像内进行 PGO 构建和两次 100 万条基准。
wsl -d Ubuntu-20.04 -- bash /mnt/f/EDA/solution/tools/build_candidate_wsl.sh generated_s3_final_pgo pgo

# 核对固定分数和全部公开行的预测一致性。
.\solution\.venv\Scripts\python.exe -u solution\tools\verify_cpp.py `
  --result solution/build/generated_s3_final_pgo_result.csv `
  --generated-dir generated_s3_final_pgo --name stage3_final --all-rows
```

PGO 构建使用 WSL 项目 `/home/zzy/fpga-delay-solution/request.csv` 作为执行路径采样输入，该文件应是公开 CSV 去掉答案列后、保持行顺序的 `From,To` 查询文件。构建脚本输出的候选 ELF 位于 WSL 项目的 `candidates/generated_s3_final_pgo/estimate`。`build_wsl.sh` 的常规 LTO 构建不包含 PGO，不能用它的产物冒充本报告测量的 PGO 二进制。

训练依赖：Python 3.12.4、NumPy 1.26.4、pandas 2.2.2、SciPy 1.13.1、LightGBM 4.6.0。PGO 最终编译主要参数：`-O3 -march=native -ffp-contract=off -fprofile-use=profile -fprofile-correction -DNDEBUG -std=c++17`。由于采用 `-march=native`，本次时间与二进制适配以当前本机 CPU 为准，正式评测应核对目标 CPU 的指令兼容性。
