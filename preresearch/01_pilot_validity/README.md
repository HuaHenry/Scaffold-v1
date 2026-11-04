# 预实验 01：基准有效性与噪声底（v3 数据）

实现 `plan/01_预实验_基准有效性与噪声底.md`，并按 `plan/00_公共约定与环境.md` 实现它依赖的公共部分（编码、协议、指标、基线、GridFormer、训练、推演评测、队列、统计、前置检查）。**数据用 v3**（`game_code/output/data_v3`），不修改 `game_code/` 里的任何文件。

## 1. 目录

```
preresearch/01_pilot_validity/
  README.md  run_all.sh  requirements.txt  pytest.ini
  exp/
    common.py        路径、数据加载、video/causal 协议切片、动作编码（含 v3 复合干预）
    metrics.py       cascade_f1、change_f1、rank_acc、违规数等（复用 datasets.score_outcome）
    baselines.py     copy_last / apply_edit / random_physics / oracle_reseed
    model.py         GridFormer（约 14.5M 参数）
    train.py         训练入口（固定 6000 步，EMA，bf16）
    rollout.py       批量自回归推演、定点检测
    evaluate.py      评测入口（模型与基线共用同一套打分）
    gen_val.py       生成监控用验证集（32 episode，seed 1000，与训练成品去重）
    run_queue.py     多 GPU 任务队列（done.json 跳过、失败写 failed.json）
    make_jobs.py     任务清单：pilot_validity（12 个）、pilot_compose（检查 E）、tune（18 组调参）
    aggregate.py     配对 bootstrap、种子层级 bootstrap、Holm、A/A 划分枚举
    precheck.py      00 §13 前置检查 1–6
    analysis/pilot_validity.py   分析 A–E，验收 V1–V5，MDE 与建议，输出 Markdown 表
    analysis/freeze.py           00 §13 第 7 项：从调参结果写 frozen.json
    configs/frozen.json          超参（目前是 00 §9 的默认值，未调参）
    tests/test_pilot.py          18 个 pytest
```

**所有运行结果都不写在本目录**，统一写到项目根目录下的 `experiments/preresearch/`（见第 9 节）；本目录只有代码和 `exp/jobs/`（任务清单）。

## 2. 环境

- 目标：Linux + 4×H200，Python ≥ 3.10，`torch ≥ 2.4`。
- 在本目录安装：`pip install -r requirements.txt`。`numpy` 同时被 `game_code` 使用。
- 需要 `game_code/output/data_v3/` 已存在（之前已生成）。如不在默认位置：`export FW_DATA=/path/to/data_v3`（也可用 `FW_GAME_CODE`、`FW_VAL` 覆盖）。
- 之后所有命令都在 `preresearch/01_pilot_validity/` 下执行（`exp/jobs/` 等代码侧相对路径以此为基准）。结果根目录 `experiments/preresearch/` 可用 `export FW_RESULTS=/其他路径` 改。

## 3. 一键运行

```bash
cd preresearch/01_pilot_validity
pip install -r requirements.txt
GPUS=0,1,2,3 bash run_all.sh                 # 完整流程，约 1–2 小时（训练 12 个任务 < 1 小时 + 基线评测）
RUN_TUNE=1 GPUS=0,1,2,3 bash run_all.sh      # 另外先做 00 §13 的一次性调参（18 组，30–60 分钟）
```

`run_all.sh` 的步骤：pytest → 生成验证集 → 前置检查 1/2/5 → （可选）调参 → 4 个基线评测 → 检查 E 的短训练 → 12 个训练任务（带评测）→ 前置检查 3/4/6 → 分析。任何一步失败会停下（`set -e`），已完成的任务有 `done.json`，重跑会自动跳过。

## 4. 分步运行

```bash
# 0. 测试
python -m pytest -q

# 1. 验证集（只用于训练时监控，不选模型）
python -m exp.gen_val                         # -> game_code/output/data_val_v3

# 2. 前置检查（00 §13）
python -m exp.precheck --items 1,2,5          # 编码往返与转移数、单 episode 过拟合、确定性
python -m exp.precheck --items 3,4,6 --run state_s0   # 稳定性、击穿 apply_edit、漂移（需先训练 + 评测）

# 3. 基线（每个基线新建一个 experiments/preresearch/01_pilot_validity/<时间戳>_baseline_<名称>_bench-data_v3/）
for b in copy_last apply_edit random_physics oracle_reseed; do
  python -m exp.evaluate --predictor $b --workers 8
done

# 4. 检查 E（形成转移占比，两个 300 步短训练）
python -m exp.make_jobs pilot_compose
python -m exp.run_queue --jobs exp/jobs/pilot_compose_steps6000.json --gpus 0,1

# 5. 12 个训练任务 + 评测
python -m exp.make_jobs pilot_validity
python -m exp.run_queue --jobs exp/jobs/pilot_validity_steps6000.json --gpus 0,1,2,3 --eval-after

# 6. 分析
python -m exp.analysis.pilot_validity          # 新建 <时间戳>_analysis_pilot_validity_... 目录
```

常用选项：

- 单个训练：`python -m exp.train --condition state --seed 0 [--episodes 64 --tag n64]   # 不给 --out 就自动新建带时间戳的结果目录`
- 单个评测：`python -m exp.evaluate --run state_s0 [--splits id,scale_ood] [--bench PATH] [--tag _b1]`（`--bench` 用于 MDE 在 0.04–0.08 时的第二份确认集）
- 只看任务不运行：`python -m exp.run_queue --jobs exp/jobs/pilot_validity_steps6000.json --dry-run`
- 每个脚本都有 `--help`；失败返回非零码。

## 5. 12 个任务

| 名称 | episode | 种子 | 结果目录（`experiments/preresearch/` 下） |
| --- | --- | --- | --- |
| `state_s0..s2` | 256 | 0,1,2 | `01_pilot_validity/<时间戳>_train_state_s{seed}_...`（与实验 03 共用，按 id 查找） |
| `state_s3..s5` | 256 | 3,4,5 | `01_pilot_validity/<时间戳>_train_state_s{seed}_...` |
| `state_n64_s0..s2` | 前 64 个 | 0,1,2 | `01_pilot_validity/<时间戳>_train_state_n{64,16}_s{seed}_...` |
| `state_n16_s0..s2` | 前 16 个 | 0,1,2 | `01_pilot_validity/<时间戳>_train_state_n{64,16}_s{seed}_...` |

全部为条件 `state`、协议 `video`、6000 步、batch 256。数据量不同则 epoch 数不同，这是有意的。

## 6. 怎么读结果

打开最新的 `experiments/preresearch/01_pilot_validity/<时间戳>_analysis_pilot_validity_.../` 里的 `pilot_validity.md`（附录表 A1）和 `pilot_validity.json`：

- **A 排序**：每个划分上 `copy_last`、`apply_edit`、`random_physics`、`state`、`oracle_reseed` 的均值与 95% CI。
- **B 阳性对照**：`n256−n64`、`n64−n16`、`n256−n16` 的差值与 CI，以及“每翻倍多少个点”。
- **C A/A**：6 个种子的 10 种 3+3 划分，报配对 bootstrap 与种子层级 bootstrap 各自“CI 排除 0”的比例。
- **D 噪声底**：`sigma_seed`、`se_boot`、`mde = 2.8·sqrt(se_boot² + 2·sigma_seed²/3)`；另给种子层级的 `mde_seed_level`。
- **E**：process/shuffle 每个 batch 的形成转移占比，期望 31/59 = 0.525。
- **验收 V1–V5**：写在 `pilot_validity.json` 的 `checks`，文字结论在 `json["recommendation"]`（3 种子够 / 需 5 种子 + 第二份 bench / 暂停）。阈值取自 plan 的 §5，代码里不可通过参数改。

决策依据：`MDE(id) ≤ 0.04` 用 3 种子；`0.04–0.08` 用 5 种子并再生成第二份 bench（`python run.py --data-v3 --seed 1 --out output_b1`，取其 `bench.npz`，评测时 `--bench`）；`> 0.08` 暂停。

## 7. 与 plan 不同或需要你注意的地方

1. **v3 干预是复合的**：`{kind, steps:[{t, actions}]}`。动作编码对每个 tick 单独编码；评测推演时第 t 步使用该步的动作图，之后全部 noop。定点步数从干预最后一步 `t_last` 之后开始找。
2. **`cascade_f1` 的下限被抬高**：定义是“预测和真值都没有连锁变化记 1.0”。v3 bench 的干预 0 总有连锁，但较弱的干预 1 约一半没有连锁，所以对两个干预取平均时 `copy_last` 的 `cascade_f1` 约 0.25（`id` 上实测 0.254），而不是接近 0；只看干预 0 时它是 0。这个底会压缩各方法之间的差距，V1 的 0.15 阈值要在这个底之上解读。分析脚本提供 `--cascade-metric cascade_f1_nonempty`（只在有真值连锁的干预上聚合）作对照。正式的 V1–V5 默认仍按 plan 的定义。
3. **`frozen.json` 目前是默认值**（lr 3e-4、权重 8、深度 8），没有调参。如要按 00 §13 第 7 项调参：`RUN_TUNE=1` 或 `make_jobs tune` + `exp.analysis.freeze`（只用验证集，不碰 bench）。
4. **验证集由 `exp.gen_val` 生成**，没有扩展 `game_code/run.py`（00 §2 里的 `--gen-val`、`--gen-train` 变体接口是实验 05–08 用的，实验 01 不需要，未实现）。`--episodes N` 在 `train.py` 里是取前 N 个 episode，等价于 plan 说的前缀。
5. 前置检查 6（漂移）用评测结果里的 `exact`（`final_fp`）与 `exact__32` 对比。
6. 同种子的 bf16 GPU 训练前 200 步 loss 可能因非确定性 kernel 超过 1e-4（检查 5 在 CPU 上差为 0）。若失败，需要设 `CUBLAS_WORKSPACE_CONFIG` 和 `torch.use_deterministic_algorithms`，再评估是否值得。

## 8. 已验证与未验证

已验证（在没有 GPU 的开发机上，CPU）：

- `pytest` 18 个测试通过；`game_code` 原有 23 个测试仍通过。
- 四个基线在 v3 bench 上的数值与 `audit.json` 里的基线数值逐项一致（`cell_acc_mask`、`structure_iou`、`change_f1`、`cascade_f1`、`exact`，干预 0）。全量 512 样本评测：`oracle_reseed` 约 6 秒（8 进程）。
- 前置检查 1 通过（动作编码往返；转移数 state 14336、partial 14336、process 15104、shuffle 15104/7168 与 plan 一致）；检查 5 在 CPU 上通过。
- 训练 → 队列 → 评测 → 分析全链路在小模型 + 合成数据上跑通；`--log-batch-composition` 的形成占比在 50 步里约 0.526（期望 0.525）。
- 合成评测数据上，分析脚本给出 10 种 A/A 划分、MDE 和建议。

**未验证**：

- **没有在 GPU 上跑过**，也没有训练过真实的 14M GridFormer，所以 V1–V5、MDE 都还没有真实数值；耗时、显存是估计。
- 前置检查 2（单 episode 过拟合 ≥ 0.99）在 CPU 上用极小模型（`--small`，0.3M 参数）只得到 0.53，未通过。这不能说明真实模型失败，必须在 GPU 上用完整模型重跑 `python -m exp.precheck --items 2`。若仍低于 0.99，先查损失权重、动作编码和学习率，不要继续跑 12 个任务。
- 前置检查 3、4、6 要等 `state_s0` 训练 + 评测完成后才能真实检验。

## 9. 结果放在哪里、怎么命名

根目录：`<项目根>/experiments/preresearch/`（`FW_RESULTS` 可改）。规则的正式版在 `plan/00_公共约定与环境.md` §11.1。

```
experiments/preresearch/
  00_common/                      前置检查、调参（各预实验共用）
  01_pilot_validity/              本实验
    20261104-173012_train_state_s0_ep256_video_steps6000_bs256_lr0.0003_cw8_d8/
    20261104-173012_train_state_n16_s0_ep16_video_steps6000_bs256_lr0.0003_cw8_d8/
    20261104-181530_baseline_copy_last_bench-data_v3/
    20261104-190244_analysis_pilot_validity_steps6000_boot10000_cascade_f1/
```

- 文件夹名：`<YYYYMMDD-HHMMSS>_<类型>_<名称>_<设置>`。时间戳在最前面，所以按名字排序就是时间顺序。
- 类型：`train`（训练 + 它自己的评测）、`baseline`、`analysis`、`precheck`、`freeze`。
- 设置：训练 = `ep<episode数>_<协议>_steps<步数>_bs<batch>_lr<学习率>_cw<变化格权重>_d<深度>`（unroll>1 时加 `_unr<K>`）；基线 = `bench-<bench 所在目录>`，部分评测会加 `_first<N>` 或划分名；分析 = `steps.._boot.._<cascade 指标>`。
- 每个结果目录都有 `meta.json`（逻辑 id、类型、设置、创建时间）。分析脚本按 **逻辑 id + 设置** 在整个结果树里找“最新的、已完成的”结果（例如 `state_s0` 且 `steps == 6000`），不依赖文件夹名里的时间戳，所以实验 03 可以直接复用本实验的 `state_s0..s2`。分析输出的 `meta.json` 和 `pilot_validity.json` 里的 `inputs` 记录了这次用了哪些文件夹。
- 一个训练目录里：`config.json`、`train_log.jsonl`、`ckpt.pt`、`done.json`、`eval.json`、`eval_samples.npz`、`train.log`、`eval.log`、`meta.json`（失败时有 `failed.json`）。基线目录里：`eval.json`、`eval_samples.npz`、`meta.json`。分析目录里：`pilot_validity.json`、`pilot_validity.md`、`meta.json`。前置检查目录里：`precheck.json`（以及检查 2、5 的临时训练子目录）。
- 时间戳在 `make_jobs` 生成任务清单时确定，同一批 12 个任务共用一个时间戳。任务清单 `exp/jobs/<实验>_steps<步数>.json` 已存在时会直接复用，所以**中断后重跑同一条 `run_queue` 命令会续跑同一批目录**（已有 `done.json` 的跳过）。想另起一批新目录：`python -m exp.make_jobs pilot_validity --new`。
- 短步数的冒烟测试（如 `--steps 20`）会生成自己的目录和任务清单文件，不会被分析当成正式结果（分析只取 `steps == --steps` 的结果，默认 6000）。
- 部分评测（`--limit`、`--splits`）写成 `eval_first3.json` 之类带后缀的文件，不会覆盖完整的 `eval.json`；基线的部分评测另建目录，且分析会忽略。
