# 实验 03：四条件主实验（简单结构训练 → 复杂结构评测）

对应 `plan/03_主实验_四条件对比.md`，比较 `state / process / partial / shuffle` 的干预后连锁预测。当前实现固定 **81,920 episodes、5 个完整 epochs、训练种子 0**，四个条件分别从头独立训练。复用 `preresearch/01_pilot_validity/exp/` 的模型、训练和评测接口；本目录独立负责数据审计、四条件任务、队列和主表分析。

**只使用已经生成并冻结的数据，不生成或修补训练集、验证集、评测集；不复用其他实验的已训练模型。**当前是四条件单预算实验，不是实验 07 的完整预算曲线，也不是计划中三训练种子的最终版本。

## 1. 目录

```text
preresearch/03_main/
  README.md
  run_all.sh
  exp/
    shared.py          实验常量、预算和完成产物身份校验
    new_run.py         新建或选定一次独立运行
    audit.py           只读检查冻结数据、隔离、帧预算与内容对照
    make_jobs.py       四条件独立训练任务
    run_queue.py       单卡任务的多 GPU 并行训练、评测队列
    analysis.py        四条件主表、五组配对比较与图
    tests/test_main.py 小规模测试，不训练模型或生成正式数据
```

结果统一写入 `experiments/preresearch/03_main/<运行名>/`，不会写进代码目录，也不覆盖实验 01、02、07 的结果。

## 2. 环境与开跑前提

- 使用 `zhouqi` 环境。训练需要 GPU 版 PyTorch；依赖参考 `../01_pilot_validity/requirements.txt`，绘图另需 `matplotlib>=3.7`。已有环境满足时不必重复安装。
- Python 模块命令必须从 **`preresearch/03_main/`** 运行，避免加载其他实验的同名 `exp` 包。`run_all.sh` 会自动切换到自己的目录。
- 默认数据是项目根目录下 `game_code/output/data_v3_simple_train_hard_eval_81920/`，必须已有 `curve_source.json`、四条件训练 NPZ/动作 JSON、场景和训练 summary。
- 验证集和 bench 的路径从该数据的 `curve_source.json` 读取，不硬编码为别的实验路径；三个集合的终态哈希必须两两不重叠。
- 审计核对冻结来源和实际 SHA1、简单训练 profile、每条件 81,920 集、每集 64 帧、形成段 32 帧、操作片段 8 帧，以及 `p_mistake=0.0, scaffold=False`。
- bench 必须是完整的困难迁移考题：四划分各 128 场景，共 512 场景；每场景两个干预，场景唯一、排序有效，强干预连锁至少 6 格、传播深度至少 3，弱干预连锁至少 2 格。
- 数据缺失、来源改变或隔离检查失败时会停止，**不会自动重新生成数据或换考题**。可以用 `DATA` 指向另一份数据，但仍须满足相同冻结的 81,920 集协议，不能直接换成 163,840 集。

## 3. 一键运行

新建运行并训练：

```bash
conda activate zhouqi
cd /data/huazhouqi/Scaffold-v1/Scaffold-v1/preresearch/03_main
env -u FW_RUN GPUS=0,1,2,3 bash run_all.sh --tag four_conditions
```

流程：检查已有数据 → 单元测试 → 创建独立运行 → 只读数据审计 → 生成四任务清单 → 四卡并行训练与评测 → 汇总主表和图。

- `GPUS=0,1,2,3`：四个条件各占一张卡，并行运行；**不是一个模型使用四卡 DDP**。只有两张可用时可设 `GPUS=0,1`，任务会排队。
- `PY` 可指定 Python 解释器；`DATA` 可指定上述协议下的已有数据目录；`--tag` 给新运行加标签。
- 审计会读取、解压和校验大文件，队列启动前也会复核输入哈希；这些是数据检查，不是数据生成，部分阶段没有周期进度输出。
- 训练日志写入各任务的 `train.log`，评测日志写入 `eval.log`。终端主要打印运行路径及任务完成、失败消息；终端静默不代表没有训练。

## 4. 分步运行

以下命令均在 `preresearch/03_main/` 中执行：

```bash
conda activate zhouqi
cd /data/huazhouqi/Scaffold-v1/Scaffold-v1/preresearch/03_main
python -m pytest -q -p no:cacheprovider exp/tests
unset FW_RUN
export FW_RUN=$(python -m exp.new_run --label four_conditions)
RUN_DIR=$(python -m exp.new_run --path)

python -m exp.audit --run "$FW_RUN" \
  --data /data/huazhouqi/Scaffold-v1/Scaffold-v1/game_code/output/data_v3_simple_train_hard_eval_81920
python -m exp.make_jobs --run "$FW_RUN"
python -m exp.run_queue --run "$FW_RUN" --gpus 0,1,2,3
python -m exp.analysis --run "$FW_RUN" --boot 2000
```

`exp.make_jobs` 从本次 `audit.json` 读取输入，`exp.run_queue` 从本次 `jobs/main03.json` 读取任务；它们不支持旧计划中的位置参数 `main` 或 `--jobs/--eval-after`。只重做分析可在四个任务完成后单独执行最后一行，不重新训练。

查看训练进度，在同一 shell 中执行：

```bash
find "$RUN_DIR/train" -name train.log -exec tail -n 3 {} \;
```

各入口均可用 `python -m exp.<模块名> --help` 查看实际参数。没有 `--dry-run`、改种子或改预算的运行参数，不应把其他实验的命令直接复制进来。

## 5. 四个任务与实际训练预算

| 条件 | 训练内容 | 每集有效转移 | 总有效转移 | 总更新数 | 形成标签转移占比 |
| --- | --- | ---: | ---: | ---: | ---: |
| `state` | 成品及操作片段 | 56 | 4,587,520 | 89,600 | 0 |
| `process` | 有序形成过程＋操作片段 | 59 | 4,833,280 | 94,400 | 31/59，约 52.54% |
| `partial` | 未完工结构的独立重置操作片段，无连续形成转移 | 56 | 4,587,520 | 89,600 | 0 |
| `shuffle` | 与 `process` 相同的形成帧被打乱，操作段不变 | 59 | 4,833,280 | 94,400 | 31/59，但这些是非物理跳变 |

每条件原始帧数均为 **5,242,880**。五个完整 epochs 中，每条有效转移每轮恰好一次，保留尾 batch，不启用 `--balanced-phases` 替换采样。**等集数、等帧数、等轮次不代表等更新数或严格等计算量。**

- 模型：网格状态输入的 `GridFormer`，一步动作条件动力学预测，不是视频生成模型。
- 固定超参：batch 上限 256、学习率 `3e-4`、变化格权重 8、depth 8、unroll 1，四条件训练种子均为 `0`。
- 各条件使用同一独立验证集的 `state` 操作转移选模；每个完整 epoch 末按验证 `cascade_acc` 选择最佳 EMA，平分保留较早检查点。命令里的 `--val-every 500` 不改变完整 epoch 模式下的轮末验证。
- bench 不用于选检查点或调参。评测读取被选中的 `ckpt.pt`，不是默认使用第五轮模型。
- 内容审计检查 `process` 操作段与 `state` 的对应操作帧和动作一致，以及 `shuffle` 形成帧和动作确为同一形成序列的排列。

## 6. 续跑与数据复用

续跑同一个运行：

```bash
conda activate zhouqi
cd /data/huazhouqi/Scaffold-v1/Scaffold-v1/preresearch/03_main
FW_RUN=已有运行名 GPUS=0,1,2,3 bash run_all.sh
```

- 数据始终只读复用；重跑仍会校验输入，不重新生成数据。
- 已有 `done.json` 的任务必须通过配置、五轮覆盖、输入 SHA1、最佳检查点及其 SHA1 校验，才会跳过训练。
- 已完成训练但没有完整评测时，只补评测；已有评测需核对同一 bench 和同一检查点。
- **没有完成标记的半途训练会从头重新训练该条件，不能从半途 `ckpt.pt` 恢复优化器和随机状态。**这是已完成任务级续跑，不是真正的步级断点续训，重跑还会覆盖该未完成任务的训练日志。
- 同一运行的审计或任务清单不一致时会拒绝。想重新从头做四条件实验，应使用新运行名；不删除旧结果，不把其他实验的完成模型拷贝进来。
- 重复分析会新建带 `-2` 等后缀的分析目录，保留之前的图表。

## 7. 输出与读法

```text
experiments/preresearch/03_main/<运行名>/
  run.json
  audit.json
  jobs/main03.json
  train/<条件>_ep81920_video_steps<更新数>_bs256_lr0.0003_cw8_d8/
    meta.json  config.json  train.log  train_log.jsonl
    ckpt.pt  done.json
    eval.log  eval.json  eval_samples.npz
    failed.json                         仅失败时写入，任务成功后清除
  analysis/main03_ep81920_epochs5/
    meta.json  main03.json  main03.md
    four_conditions.png  four_conditions.pdf
    paired_differences.png  paired_differences.pdf
```

先看 `main03.md` 的四划分连锁 F1 表及配对差值，再看 `main03.json` 和逐场景 `eval_samples.npz`。分析必须取得本运行四个条件的完整、身份一致的结果；缺一个条件不能出完整主表，不自动导入实验 07 的成绩。

- 主指标：`cascade_f1`；辅助指标：`change_f1`、`structure_iou`、`cell_acc_mask`、`rank_acc`、`any_violation`，以及单步、四步的 `cascade_f1__s1/__s4`。
- 比较：`process−state`、`process−partial`、`process−shuffle`、`state−shuffle`、`partial−state`。
- 配对区间默认 bootstrap 2,000 次，先在场景内平均两个干预，再重采样同一划分的 128 场景；**只有种子 0，区间不含重复训练的随机波动**。
- `id` 标签代表困难结构迁移，不是训练分布内 ID。四划分为 `id/combo_ood/scale_ood/multi_ood`。
- 图是四条件柱状图和差值区间图，不是预算曲线；不能据此宣称所有模型架构、真实视频或全部物理任务均获益。

## 8. 与计划文本的区别及结论边界

- 计划写的是四条件 × 三训练种子及 `causal` 补充；当前代码仅四条件 × 种子 0 的 `video` 实验，不冒充已完成三种子或 `causal` 结果。
- 当前沿用已有预算实验配置和五轮协议，不执行验证集超参搜索；不能把默认参数沿用描述为已完成调参。
- 计划中的 `runs/main/`、`exp.make_jobs main`、`exp.aggregate` 是旧布局／命令；实际入口、路径以本 README 第 3–7 节为准。
- `partial/shuffle` 是重要对照，但一步模型不会读取完整建造历史；`shuffle` 破坏的是合法相邻转移，优势不能直接解释成掌握了长程建造顺序。
- 训练关闭脚手架和建造失误，不能用本次结果宣称这些机制已经被验证；`process` 减少操作片段的权衡也须保留。
- 没有自动检验实验 01、02 的“继续”结论，也没有自动完成计划全部验收门槛。是否开展本实验及后续如何解释，由研究决策与实际结果决定，不为匹配预期挑选指标。

## 9. 验证状态

2026-11-11 核对实际入口和参数，运行本目录测试：**11 项通过**；`run_all.sh` 通过 Bash 语法检查。测试覆盖四条件预算和结果隔离、完整 epoch 抽样、缺数据不生成、集合隔离、内容对照、完成标记／检查点／评测身份校验，以及四条件图表渲染。

本次仅补充说明文件并运行小规模测试，**未启动正式数据生成、主实验训练或真实 bench 评测，未修改实验代码与既有数据、结果**。单元测试通过不等于已完成真实大数据审计或四条件主实验。
