# EDA 2026 赛题交接

更新于 2026-10-21。当前默认已从第七阶段切换为第八阶段 `displacement_cal16`。本轮[计划](docs/优化推进计划.md)的 P0～P5 已完成；[执行记录](docs/完整位移模型优化记录.md)包含方法、正负结果与参数重建命令。

## 入口与产物

- [根 README](README.md)：环境、数据、当前构建、验收及发布命令。
- [solution/README](solution/README.md)：赛题规则、接口和算法。
- 当前源码：`solution/src/displacement.cpp`；精确参考：`displacement_oracle.cpp`。
- 当前默认：`solution/bin/estimate`、`solution/submission.zip`；候选备份为 `estimate_displacement_cal16`、`displacement_cal16_submission.zip`。
- 指纹/发布依据：[第八阶段清单](docs/stage8_manifest.json)、[最终基准](docs/displacement_final_benchmark.json)、[验收汇总](docs/displacement_cal16_validation.json)。
- 冻结模型：`solution/build/displacement/oracle/`；新参数：`U_refit_q16.bin`、`M_refit.bin`、`calibration_train.bin`，均不入 Git。
- 第七阶段回退：`solution/bin/estimate_stage7_fold`、`solution/stage7_fold_submission.zip`；原清单/模型在 `solution/build/stage7_*`、`foldable_lr08_300r.txt` 和 `generated_s7_fold/`。

Git 保存源码、参数生成脚本及紧凑证据，排除公开数据、模型二进制、临时 CSV、ELF、ZIP 与大缓存。克隆后需导入外部模型并重建参数，不能只凭仓库里的报告直接发布。

## 当前实测

| 方案 | 固定测试准确度 | 本轮 10M CPU | 预计总分 |
| --- | ---: | ---: | ---: |
| 第七阶段 | 94.607402 | 1.343385 s | 94.566433 |
| 原始完整位移基线 | 97.595517 | 1.963007 s | 96.440575 |
| 第八阶段默认 | 97.650829 | 2.022604 s | 96.435160 |

新默认 ZIP 80,797,119 B；容器峰值 582,066,176 B；单线程；五次 10M 输出一致、重复内容核对、11+11 项接口/错误检查及独立 ZIP 通过。第八阶段准确度比第七阶段提高 3.043427 分；相对原始完整位移模型只小幅提高，预计总分在测时波动内接近。64 MB 分块相同输出且本轮较慢，选择 16 MB。

固定 rng7 切片已被历史开发反复查看；全公开分 97.685798 含训练行。新参数只用训练行拟合，验证集选型，测试集冻结后回归。没有执行完整 1B 或官方隐藏集，CPU 外推和预计总分不是官方成绩。先前批次的第七阶段 1.3498 秒以及 Pango 比较批次的 1.7310/2.3186 秒均保留作历史，不能与当前批次混算。

## 复测与发布

本机仓库为 `F:\EDA\solution\eda2026`，WSL 对应 `/mnt/f/EDA/solution/eda2026`。Windows Python 为 `F:\EDA\solution\.venv`；发行版 `Ubuntu-20.04`、镜像 `fpga-ziguang:latest`。本轮容器精确限额 2,000,000,000 B。先确认 Docker Desktop 的 WSL Integration 可用。

官方数据默认 `F:\EDA\EDA-PANGO-SRB-Arch-Delay\EDA-PANGO-SRB-Arch-Delay`，可用 `EDA_DATA_DIR` 指定。请求为带 `From,To` 表头的 `solution/build/local_runs/request_1m.csv`，新克隆可按 README 从答案裁两列生成。

外部 `F:\EDA\pango_delay_handover_v3` 在本轮中仍有人修改源码，后续构建必须用本仓库冻结的 `displacement.cpp`。模型已导入本地 `oracle/`；重新导入应核对 [displacement_import.json](docs/displacement_import.json) 的五个资产指纹。原始推理版本冻结于 `055c7d9`；当前代码增加了量化、诊断和输入校验。不要修改外部工程或原始赛题附件。

最短复测步骤见 README；完整参数重建在执行记录 P1～P3。重建/调参写到隔离候选后：`run` → Windows `analyze` → `package` → `validate`，再共同五轮 `benchmark`，最后 `release_stage8.py displacement_cal16`。发布器核对实际 ELF 中六段资产、源码、评分输出、五次压测、ZIP 和接口证据，不重训也不重新打包。所有门槛通过才复制为默认文件。

部分淘汰 ELF、重复 ZIP、结果 CSV 和 U-int8 缓存已清理，紧凑报告与再生脚本保留。重复基准前需重建被清理的 `displacement_cal64`；Pango 原始基线和第七阶段参照 ELF 仍保留。诊断 trace、几何和紧凑预测数组用于参数/泛化再生，未重新生成约 32 GB 原始稠密矩阵。

## 下一轮研究

1. 扩大位置分组精确参考：已有 8/8 官方案例匹配与 48 条保留坐标查询，但后者相对分 88.248625→88.050655，不能称全位置改善。先覆盖不同端口族、30～300 主体距离、短距离、墙门和边界，明确 Net 落点与解析项语义。
2. 分离完整位移低秩截断误差与位置规则误差，先评估合法路径模板的覆盖和候选独立代价，再训练可部署的选择器。逐样本 oracle 最优不等于在线收益。
3. 利用量化后约 19.2 MB ZIP 余量试小型位置/结构残差；继续用训练折拟合、验证选择、新位置分组检查，不以全公开标签校准证明泛化。
4. 准确度增益须抵消 CPU 扣分，同时实际检查包体、内存及单线程。当前 97.65 离 99 仍有明显差距；最终需官方 1B 与隐藏集确认真实成绩。

第七阶段在原表键加树/叶子及超调比特的负结果只约束那种表示；不能由此认定新模型也达到上限。历史比较及误差贡献见[实现对比与适配评估](solution/实现对比与适配评估.md)。
