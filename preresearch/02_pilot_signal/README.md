# 预实验 02：形成缺口存在性与假设信号（v3 数据）

实现 `plan/02_预实验_形成缺口存在性与假设信号.md`。本实验回答三个问题：形成数据是否覆盖了更多局部变化（Q1）、只见成品的模型是否缺少形成侧能力（Q2）、四个条件在拆除级联考题上的单种子差异有多大（Q3）。**Q3 只用于下一步工作的分流，不是多种子效应结论。**训练、模型、动作编码、FormBench 评测、配对 bootstrap 和任务队列直接复用 `preresearch/01_pilot_validity/exp/`；本目录只实现 02 独有部分。

## 1. 目录

```text
preresearch/02_pilot_signal/
  README.md  run_all.sh
  exp/
    shared.py                    加载 01 的公共代码与结果登记接口
    new_run.py                   新建或选定 02 的一次运行
    make_jobs.py                 四条件各一个 seed 0 的训练任务
    run_queue.py                 复用 01 的多 GPU 训练、评测队列
    configs/pilot_decision.json  G1/G2/C 决策阈值及 bootstrap 配置
    analysis/coverage.py        Q1：3×3、5×5 局部规则覆盖；附录表 A2
    analysis/formation_probe.py Q2：教师强制单步探针；Q3：bench 信号；图 A1
    analysis/pilot_decision.py  G1/G2/S/C 分流与文字决策
    tests/test_signal.py        覆盖键、转移分类和决策边界测试
```

本目录不放训练结果；所有结果写入 `experiments/preresearch/02_pilot_signal/<运行名>/`。不覆盖 01 的代码、已生成数据或旧结果。

## 2. 环境与开跑前提

- 从**本目录**执行命令。使用 01 的依赖，并额外安装画图依赖：`pip install -r ../01_pilot_validity/requirements.txt 'matplotlib>=3.7'`。正式训练需要可用的 PyTorch GPU 环境；Q1 和测试可以使用 CPU。
- 默认读取 `game_code/output/data_v3/` 的四条件训练数据与 `bench.npz`，以及 `game_code/output/data_val_v3/` 的验证数据。可分别用 `FW_DATA`、`FW_VAL`、`FW_GAME_CODE`、`FW_RESULTS` 覆盖默认路径。
- 需要明确指定一次**实验 01 的 `pilot_validity.json` 绝对路径**，其中必须有四个划分的 MDE；代码按当前 `plan/01` 口径核对 V1 增益 ≥0.12 和 V2 通过，不直接把旧自动报告中的 V1 `pass:false` 当作前提失败。
- 为快速取得 02 的探索性结果，默认沿用 01 实际使用的参数：学习率 `3e-4`、变化格权重 `8`、深度 `8`、`unroll=1`，无需先做 18 组调参。`../01_pilot_validity/exp/configs/frozen.json` 目前仍标明“NOT tuned”，如实保留；沿用默认参数不等于已经完成计划中的验证集调参与冻结。如果以后更换参数，则不应直接用这次 01 的 MDE 比较新配置。
- `exp/configs/pilot_decision.json` 的阈值应在查看 02 的 bench 读数前固定，不能根据结果临时改阈值。

## 3. 一键运行

```bash
cd /data/huazhouqi/Scaffold-v1/Scaffold-v1/preresearch/02_pilot_signal
export PILOT01=/data/huazhouqi/Scaffold-v1/Scaffold-v1/experiments/preresearch/01_pilot_validity/20261104-130756/analysis/pilot_validity_steps6000_boot10000_cascade_f1/pilot_validity.json
FW_RUN=20261105-074946 GPUS=0,1,2,3 bash run_all.sh
```

流程：单元测试 → 创建 02 的运行文件夹 → 确保 32 episode、seed 1000 且与训练成品去重的四条件验证集存在 → 确认 01 前提与冻结配置 → Q1 覆盖分析 → 四条件各训练/评测一个种子 → Q2/Q3 → 自动决策。完整训练使用 `video` 协议、相同冻结配置、6000 步、batch 256，训练数据默认各 256 episode。**这些模型尚未由本目录的代码实际训练；执行一键脚本才会产生新的实验结果。**

一次执行对应 `experiments/preresearch/02_pilot_signal/<时间戳>[_标签]/` 的一个文件夹。`LABEL=xxx` 给新运行加后缀；中断后用 `FW_RUN=<已有运行名> bash run_all.sh` 继续，任务队列跳过已有 `done.json` 的训练。`STEPS` 可用于独立冒烟运行（例如 `LABEL=smoke STEPS=20`），冒烟结果不能用作正式分流。

## 4. 分步运行

以下命令都在 `preresearch/02_pilot_signal/` 内运行，要求指定已有的 01 分析结果；当前直接使用与 01 相同的默认参数。

```bash
python -m pytest -q -p no:cacheprovider exp/tests
export FW_RUN=$(python -m exp.new_run)   # 新建；续跑则设为已有运行名
RUN_DIR=$(python -m exp.new_run --path)

# 验证集不存在时，借用 01 的生成器（生成到 game_code/output/data_val_v3）
(cd ../01_pilot_validity && python -m exp.gen_val)

python -m exp.make_jobs --pilot01 "$PILOT01" --frozen ../01_pilot_validity/exp/configs/frozen.json
python -m exp.analysis.coverage --run "$FW_RUN"
python -m exp.run_queue --jobs "$RUN_DIR/jobs/pilot_signal_steps6000.json" --gpus 0,1,2,3 --eval-after
python -m exp.analysis.formation_probe --run "$FW_RUN" --steps 6000
python -m exp.analysis.pilot_decision --run "$FW_RUN" --pilot01 "$PILOT01"
```

- 只看训练清单：`python -m exp.run_queue --jobs "$RUN_DIR/jobs/pilot_signal_steps6000.json" --dry-run`。
- 若验证集已生成且 `overlap.json`、四条件验证数据均存在，**不要重复生成**，以免改变探针输入；一键脚本仅在 `train_state.npz` 不存在时生成。探针会核对保留了 32 个 episode 且验证成品与训练成品不重叠。
- 如需明确定位一次分析输入，`formation_probe` 可用 `--val`、`--bench`、`--data`；`pilot_decision` 可用 `--coverage`、`--probe` 指向两个分析结果**目录**。每个脚本都有 `--help`。正式 Q3 必须使用与 Q1 相同的 bench 和训练数据。

## 5. 四个任务与三个读数

| 任务 id | 条件 | 训练与评测 |
| --- | --- | --- |
| `state_s0` | 只看成品状态 | `video`、seed 0、6000 步，完整 FormBench |
| `process_s0` | 有序形成过程 | 同上 |
| `partial_s0` | 部分形成过程 | 同上 |
| `shuffle_s0` | 打乱形成顺序对照 | 同上 |

**Q1 覆盖度（CPU，无需训练）**：只用四条件 `video` 协议的有效相邻转移，对前景或背景发生变化的格子，把上一状态邻域、当前动作图邻域和下一状态中心格组成局部键；分别统计 3×3 和 5×5。再看真实 bench 轨迹中的键是否在每种条件的训练数据里出现。按四划分、第一步/后续步、全部变化/排除直接动作格的连锁变化汇总；3×3 另报整步无动作时的变化。v3 的复合干预**可能在后续步骤再次有动作**，覆盖计算按实际每一步的动作脚本处理，不能假定第一步以后全是 noop。`C` 使用**3×3、四划分与各步骤合并后的连锁变化覆盖度**，看 `process−state` 是否达到 0.02。

**Q2 形成侧探针（四条件同一验证转移）**：形成转移来自 `process` 验证集，其中“形成·放置”指真值变化全在动作格内，“形成·物理”指还有动作格外的变化；“操作”来自 `state` 验证集。模型每次输入**真实上一帧和该步动作**，只预测一步，不用自己的预测继续滚动。变化格准确率要求该格前景、背景同时预测正确；`trans_exact` 要求整帧全部正确。没有变化格的转移仍计入类别和 `trans_exact`，但不计入变化格准确率的分母；报告 `n`、`n_changed`、`n_no_change`。相同转移上的条件差做 10,000 次配对 bootstrap，区间不含训练种子的随机性。

**Q3 单种子 bench 信号**：对四条件 `eval_samples.npz` 用 01 的 `cascade_f1` 口径，把一道题的两个干预平均，再按划分计算 `process−state`、`process−partial`、`process−shuffle` 与 95% CI。`S` 还对照 01 在**同一划分**的 MDE。单种子 CI 只含考题抽样波动，不能证明训练条件的稳定优势，也不能据此选择超参或改造 bench。

## 6. 怎么看自动决策

阈值见 `exp/configs/pilot_decision.json`，正式定义见 `plan/02_预实验_形成缺口存在性与假设信号.md` §5：

- **G1**：任一形成类 `process−state` 变化格准确率 ≥0.15 且 95% CI 排除零；同时操作类差值的绝对值 ≤0.03。
- **G2**：任一形成类 `process−shuffle` ≥0.05 且 95% CI 排除零。
- **S+**：某划分 `process−state` 连锁 F1 差至少达到该划分的 MDE，且 CI 为正；**S0**：各划分差值绝对值均小于各自 MDE；**S−**：某划分 `state−process` 至少达到 MDE。不同划分同时出现明显正负差记 `S-mixed`；差值超过 MDE 但 CI 仍包含零且无法落入上述类别记 `S?`，不硬判成 S+。
- **C+**：3×3 的合并连锁覆盖度 `process−state ≥0.02`；否则 C0。

优先处理 G1 不成立，再处理 G2 不成立；若二者都成立，则 S+ 可继续做多种子验证，S0 重点解释“形成侧有缺口、拆除侧暂未检出”，S− 或 `S-mixed` 先排查操作数据量是否混杂。具体原因与下一步由 `pilot_decision.md` 给出。

## 7. 输出与目录

```text
experiments/preresearch/02_pilot_signal/<运行名>/
  run.json
  jobs/pilot_signal_steps6000.json
  train/{state,process,partial,shuffle}_s0_ep256_video_steps6000_.../
    meta.json  config.json  train_log.jsonl  ckpt.pt  done.json  eval.json  eval_samples.npz
  analysis/pilot_coverage_bench-data_v3/
    meta.json  pilot_coverage.json  table_A2.md
  analysis/pilot_probe_steps6000_boot10000_cascade_f1/
    meta.json  pilot_probe.json  pilot_signal.json  figs/figure_A1.pdf  figs/figure_A1.png
  analysis/pilot_decision_cascade_f1/
    meta.json  pilot_decision.json  pilot_decision.md
```

若重复单独运行分析，会新建带 `-2` 等后缀的目录，不覆盖旧结果。每个分析的 `meta.json` 和 JSON 都记录实际输入来源；任务查找限定在本次 `FW_RUN` 下，01 的 MDE 则由 `PILOT01` 显式指定。可先看 `pilot_decision.md` 的分流建议，再看 `pilot_probe.json` 的各类别分数、`table_A2.md` 的覆盖度和 `figs/figure_A1.*` 的柱状图。四条件的训练日志与逐题评测保留在各自 `train/` 目录。

## 8. 与计划文本需注意的口径

- `plan/02` 的示例路径 `runs/main/`、`results/`、`output/data_val/val_*.npz` 是旧布局；实际使用 `experiments/preresearch/02_pilot_signal/<运行名>/`，验证数据沿用 01 的 `data_val_v3/train_*.npz` 与同名动作文件。验证集生成用 01 的 `exp.gen_val`，未在 `game_code/run.py` 增加 `--gen-val`。
- 01 的旧自动 JSON 仍按当时的 V1 ≥0.15 判失败，而当前 `plan/01` 的增益门槛是 ≥0.12；02 根据原始均值核对当前口径。01 的 MDE 原始测量值仍按各划分读取，不因门槛变更而改变。当前参数来源是 01 使用的默认值，未做验证集调参；阈值与已用 bench 的来源要能追溯。
- 此前 v2 的形成物理占比只是先验参考，02 的分类完全以当前 v3 验证数据重算；不把先验数字写成实测结果。

## 9. 验证状态

已验证：02 单元测试 4 项通过；脚本通过 Bash 语法检查；真实 v3 验证数据可分出“形成·放置”752、“形成·物理”240、“操作”1792 个转移；真实 bench 的局部键提取和已有 01 逐样本评测数据读取做过只读冒烟检查。**尚未正式运行四条件训练、全量 Q1/Q2/Q3 与决策，不存在 02 的实测结论。**
