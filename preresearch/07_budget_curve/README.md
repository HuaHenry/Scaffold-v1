# 实验 07：等总预算下的学习曲线

设计见 `plan/07_对照_预算对等.md`。回答一个问题：`state` 全部练操作，`process` 一半练建造、一半练操作；**每档总预算完全相等**，随预算增加，`process` 的拆除连锁预测是否从落后变成领先？不是单独增加 `process` 的训练量。

## 17 个固定档位与待检验预测

每条件 episode 取 **16、24、32、48、64、96、128、192、256、384、512、768、1024、1536、2048、3072、4096**；最后一档是每条件 **262,144 帧**，相当于旧曲线最大值的八倍。所有模型训练上限统一为 **12000 步 × batch 256**，每 500 步按固定验证集的一步 `cascade_acc` 选最佳 EMA（同分取较早步），而不是使用 FormBench 选模。`state` 每个 episode 是 64 帧操作；`process` 是 32 帧建造 + 32 帧操作，预算始终相等。

**要检验的假设**：`state` 先提升、之后趋于饱和；`process` 初期落后，随着预算增加反超。结果可能与预期不同；必须按实际曲线解释。自动从旧 `data_v3` 的 seed 0 生成独立 `data_v3_curve_4096/`；旧 256 episode 的全部四条件视频、场景和动作会逐项核对完全一致，旧 `data_v3` 与先前 `data_v3_curve_512` 不会覆盖。原 bench 若与新加入的场景重叠，就另生成去重 bench 供全部 17 档共同评测。形成视频只含 31 个可训练相邻帧，操作片段各跳过重置帧，因此每 episode 的实际转移数为 `state:56`、`process:31+28=59`。训练代码为 `process` 每个 batch 从形成和操作各抽 **128 条**，训练步数不变；`state` 的 256 条都来自操作。严格各半的是视频帧与每次优化采样份额，不是两条件不同种类的独特转移数。

## 环境和运行

依赖复用 `../01_pilot_validity/requirements.txt`，绘图另需 `matplotlib>=3.7`；训练需 PyTorch GPU。默认生成独立 4096 集并核对旧前缀；由于旧 `data_val_v3` 与新增训练场景可能重叠，默认在 `DATA/validation` 生成新的去重验证集，旧验证集不会改动。`EPISODES`（至少 4096）、`SOURCE`、`DATA`、`VAL`、`BENCH` 可覆盖输入，`WORKERS` 控制生成并行数；指定已存在且有重叠的 `VAL` 会明确拒绝。其他超参固定为 `lr=3e-4`、`change_weight=8`、`depth=8`、`unroll=1`。原实验的 6000 步末步检查点与本轮验证选模、12000 步、共享新 bench 不可拼接；全部 17 档重新训练和评测。固定场景数内验证精度后期下降提示过拟合，但验证的一步指标不等同于自回归 FormBench F1，实际改善须在全曲线评测后确认。

在本目录运行；**默认就是一个种子**，不必额外设置：

```bash
cd /data/huazhouqi/Scaffold-v1/Scaffold-v1/preresearch/07_budget_curve
GPUS=0,1,2,3 bash run_all.sh
```

先生成并校验 4096 episode，随后审计数据、运行 **17 档 × 2 条曲线 = 34 次训练**，输出全部分数和图。单种子可以用来描述预算曲线和观察是否交叉，但不能宣称不同初始化都会交叉。若以后想加种子，可指定同一运行名续跑，已完成的 seed 0 自动跳过：

```bash
FW_RUN=<上一次输出的运行名> SEEDS=0,1,2 GPUS=0,1,2,3 bash run_all.sh
```

默认单种子足以开始研究你关心的**曲线形状**；如果后来要论证微小差异也能跨种子稳定复现，可再增加种子并用独立去重确认 bench。支持 `STEPS=20` 等短训练只做工程冒烟，不能当实验结果；训练步数对所有档位相同，最大档平均每条 `state` 转移抽取约 13.4 次。续跑时保持同一 `FW_RUN`、数据、验证集、参数、bench；生成数据若已存在就核对并复用，不重新生成。`bash run_all.sh --tag expanded_4096` 会在新运行名中使用该标签；指定 `FW_RUN` 可续跑同一个运行。

分步命令（需要同一 `FW_RUN`；可通过 `python -m exp.new_run` 新建）：

```bash
export FW_RUN=$(python -m exp.new_run)
python -m exp.generate_data
DATA=../../game_code/output/data_v3_curve_4096
VAL="$DATA/validation"
BENCH=$(python -c 'from exp.shared import common; print(common.resolve(common.load_json(common.GAME_CODE/"output/data_v3_curve_4096/curve_source.json")["bench"]))')
python -m exp.audit --run "$FW_RUN" --data "$DATA" --val "$VAL" --bench "$BENCH"
python -m exp.make_jobs --run "$FW_RUN" --data "$DATA" --val "$VAL" --seeds 0
RUN_DIR=$(python -m exp.new_run --path)
python -m exp.run_queue --jobs "$RUN_DIR/jobs/curve_steps12000_seeds0.json" --gpus 0,1,2,3 --bench "$BENCH" --eval-after
python -m exp.analysis --run "$FW_RUN" --data "$DATA" --bench "$BENCH" --seeds 0 --steps 12000
```

`python -m exp.run_queue --jobs <任务清单> --dry-run` 只查看命令；每个程序有 `--help`。`audit` 将阻断训练与验证/bench 的场景泄漏；现有训练集中 256 个 episode 只有 255 种独特成品，内部重复 1 例会记录但不影响等预算对照。

## 结果位置和解释

结果位于 `experiments/preresearch/07_budget_curve/<运行名>/`：

- `game_code/output/data_v3_curve_4096/curve_source.json`：原始 256 集前缀对照与训练、验证、bench 文件哈希；`audit.json`：帧数、转移数、去重与输入指纹；
- `jobs/curve_steps12000_seeds0.json`：34 个可校验续跑的任务；`train/{state,process}_n<档位>_s0_*/`：验证最佳模型及选定步数、训练日志、逐题评测；
- `analysis/budget_curve_steps12000_seeds*/budget_curve.json`、`budget_curve.md`、`budget_curve.pdf/png`、`budget_difference.pdf/png`：四划分曲线、配对差值、区间与输入来源。

横轴是**每条件总训练帧**，增加的是场景数，不是优化步数。主要看 `id` 的 `cascade_f1`：`state` 的线是否趋平，`process` 能否从低于 `state` 到高于它。单种子差值区间只反映题目抽样误差，曲线从负到正可称“本次运行观察到交叉”，**不能据此宣称不同训练种子也会交叉**；多个种子可再复核稳健性。17 档×4 个划分的多重比较只有在多种子复核后再报告 Holm 判定。17 档未交叉只能说**本次所测预算范围没有确认**，也不能将曲线仍上升解释为数据饱和。旧 512 档曲线在两组 256→512 均继续上升；同一档里末步验证下降，不等于先前步数的自回归测试 F1 一定更高。`01` 默认训练不启用验证选模，不影响原实验。
