# 实验 09：物理诊断、长程推演与建造推演

本实验只评测实验 03 的冻结模型，不训练新模型、不修改已有数据或权重。入口是 `preresearch.09_physics_rollout.exp.run`，一次运行完成三项评测和图表导出。

**现在怎么跑：先执行第 1 节的小样本检查，再执行第 2 节的完整数据评测。两条命令都已显式使用当前已有的 `seed=0` 四个模型，不会因为默认要求三个训练重复而报缺失错误。**

## 1. 先跑小样本检查

必须从项目根目录运行，不是在实验 09 目录里直接执行 `run.py`。

已验证可用的 Python 是 `/root/miniconda3/envs/zhouqi/bin/python`，环境包含 PyTorch、NumPy、Matplotlib 和 pytest。系统默认 `python` 不一定有这些依赖，下面直接使用完整路径。

```bash
cd /data/huazhouqi/Scaffold-v1/Scaffold-v1

/root/miniconda3/envs/zhouqi/bin/python -m preresearch.09_physics_rollout.exp.run \
  --source-run 20261111-044057_four_conditions \
  --seeds 0 \
  --run physics09_smoke_v1 \
  --device auto \
  --batch 8 \
  --limit-per-split 2 \
  --formation-limit 8 \
  --boot 200
```

这条命令评测：
- 四类 FormBench 测试各 2 个场景，共 8 个场景，每场景 2 个干预；
- 同一批场景的 128 步无动作推演；
- 前 8 个验证形成片段的 31 步建造推演；
- State、Process、Partial、Shuffle 四个冻结模型及真实引擎对照。

默认自动使用 CUDA；没有 CUDA 时使用 CPU。CPU 也能运行，但模型推演会更慢。

结果目录：

```text
/data/huazhouqi/Scaffold-v1/Scaffold-v1/experiments/preresearch/09_physics_rollout/physics09_smoke_v1/
```

小样本结果是流程检查，不是完整实验结果。看到 `qualified: false` 不等于程序没有跑完，具体见第 5 节。

## 2. 跑当前四个模型的完整数据评测

确认小样本命令能跑通后，使用新的 `--run` 名称，去掉两个样本数量限制：

```bash
cd /data/huazhouqi/Scaffold-v1/Scaffold-v1

/root/miniconda3/envs/zhouqi/bin/python -m preresearch.09_physics_rollout.exp.run \
  --source-run 20261111-044057_four_conditions \
  --seeds 0 \
  --run physics09_full_seed0_v1 \
  --device auto \
  --batch 32 \
  --boot 2000
```

该命令使用全部 512 个 FormBench 场景、全部指定验证集形成片段，自动生成统计结果和图表。这里的“完整”指完整评测数据；目前只有一次训练重复，仍不满足计划中的三次独立训练重复要求。

- 当前已核实的模型来源：`experiments/preresearch/03_main/20261111-044057_four_conditions/`。
- 模型是实验 03 通过验证集选出的 EMA 权重。
- benchmark 默认读取该来源的 `audit.json`，当前路径为 `game_code/output/data_v3_simple_train_hard_eval_81920/bench.npz`。
- 建造验证集默认同样读取 `audit.json`，当前目录为 `game_code/output/data_v3_simple_train_hard_eval_81920/validation/`。不会静默换成另一份 `output/data_val`。
- 建造脚本统一取 Process 验证数据的第 1 至 31 帧动作，四条件使用同一份脚本。没有 teacher forcing，也不会使用 Shuffle 的乱序形成片段。

如要明确指定另一份正序 Process 验证集，使用 `--val /验证集绝对目录`，并采用新的 `--run` 名称。程序会校验形状、形成阶段、连续性和空地形初态，并记录文件身份；这会改变建造探针的评测数据。

## 3. 中断后续跑、只评测、只重新画图

### 中断后续跑

重新执行原命令，保持同一个 `--run`、相同来源和相同评测参数即可。已完整写入且文件校验通过的模型/探针结果会跳过，未完成部分重新计算。

改变来源、评测样本、保存轨迹设置或评测代码后，应使用新的 `--run` 名称。不要删除来源校验文件来绕过检查。`--batch` 和设备可以调整，它们不改变缓存中的评测协议。

### 先只评测，不进行统计和画图

在第 2 节命令末尾增加 `--mode evaluate`。评测完成后，再执行下面的分析命令。

### 仅重新统计和画图，不再推演模型

```bash
cd /data/huazhouqi/Scaffold-v1/Scaffold-v1

/root/miniconda3/envs/zhouqi/bin/python -m preresearch.09_physics_rollout.exp.run \
  --run physics09_full_seed0_v1 \
  --mode analysis \
  --boot 2000
```

分析模式从已有 `source.json` 读取来源，不需要再指定 `--source-run`、`--seeds` 或设备；必须已经完成四条件和引擎对照的三项评测。

## 4. 输出在哪里、先看什么

以下路径均相对于 `experiments/preresearch/09_physics_rollout/<run>/`：

| 文件或目录 | 内容 |
|---|---|
| `source.json` | 模型、数据、代码身份，评测步数、样本 ID 和运行协议 |
| `audit.json` | 真值轨迹检查、引擎建造 exact、无动作引擎漂移和验收状态 |
| `audit_intervention.npz` | FormBench 真值逐步检测计数及场景 ID |
| `audit_formation.npz`、`audit_contexts.npz` | 建造真值和初态的检测记录 |
| `raw/<condition>_seed0/` | 每个模型三项评测的逐场景、逐干预、逐步指标及最终预测网格 |
| `raw/oracle_reseed/` | 新随机流下真实引擎的同协议评测 |
| `analysis/results.json` | 全部指标、分测试汇总、配对差值、训练标准差与场景区间 |
| `analysis/tables/endpoints.csv` | 终点指标、标准差、场景区间和 Process 相对其他条件的差值 |
| `analysis/tables/curves.csv` | 每步指标均值和训练标准差 |
| `analysis/tables/endpoints.tex` | 可用于排版的终点汇总表 |
| `analysis/figures/fig8.pdf`、`fig8.png` | 六面板总览图 |
| `analysis/figures/intervention.pdf`、`drift.pdf`、`formation.pdf` | 各探针的详细曲线，同时导出 PNG |
| `analysis/complete.json` | 导出文件身份及 `qualified` 状态 |

建议先看 `audit.json`，再看 `fig8.pdf` 和 `endpoints.csv`。`--mode evaluate` 不生成 `analysis/`；`--no-plots` 不生成图。

默认只保存逐步指标和最终预测网格。如需所有预测网格，首次运行时增加 `--save-trajectories`，每个探针旁会生成 `*_trajectories.npz`，磁盘占用会显著增加。不能在旧 run 上直接切换此选项。

## 5. 验收和结果解释：必须注意

### 现有检测器不能把合法下落直接判为非法物理

`check_violations` 检查的是应当静止的状态。正在合法下落的沙子或结构，在中间帧也可能被计为“浮空”。真实数据小样本检查已经观察到：部分中间帧计数非零，最终帧为零。

因此，目前计划中“真值每步违规数恒为零”的验收条件不成立。程序会：
1. 保存真实轨迹逐步检测结果，图中保留真值对照；
2. 在 `audit.json` 标记该项验收未通过；
3. 默认打印警告后继续评测，保留可分析的结果；
4. 不把这些中间帧检测计数直接当作非法动力学的证明。

**不要为了消除警告修改真值或检测结果。** 若增加 `--strict-audit`，程序在真值/引擎验收失败时会停止，不会开始学习模型推演；当前数据已存在上述问题，首次跑通流程不要增加这个选项。

### `qualified: false` 与“未跑完”不同

`qualified` 只有在验收通过、至少三个配对训练重复且没有样本截断时才为真。以下情况会为假：
- 当前 `--seeds 0` 的单次训练诊断；
- 使用 `--limit-per-split` 或 `--formation-limit`；
- 真值/引擎验收未通过。

是否完成导出，应同时检查 `analysis/complete.json` 是否生成及日志是否有异常。不要仅凭 `qualified: false` 判断运行失败，也不要把它改成 true。

### 指标含义

- 质量误差使用已有 `metrics.mass_error`：两材料层中，各非空材料格数与同一步真值格数差的绝对值之和。合法编辑和背景清理不被假定为严格质量守恒。
- 违规分类计数可能重叠；`struct_on_loose` 是 `floating_struct` 的子集，总和不是不重复的违规格数。
- 场景区间与独立训练模型间的标准差分开报告。只有一个训练重复时，标准差为空，不伪造为 0。
- 配对差值方向是 Process 减对照。对于违规、质量误差和逐格误差，负数更好；对于准确率、F1、IoU、exact，正数更好。
- 建造推演对 Process 的训练暴露天然有利，是能力探针，不是 H1 的独立证据。

## 6. 参数和常见问题

| 参数 | 用法 |
|---|---|
| `--run` | 必填，只填 run 名称，不填输出绝对路径 |
| `--source-run` | 已审计的实验 03 run 名称或绝对目录，可重复指定 |
| `--seeds 0` | 当前已有的四个模型；不写此项会默认要求 `0 1 2` |
| `--device auto` | 自动选择 CUDA/CPU；也可指定 `cpu` 或 `cuda:0` |
| `--batch 32` | 模型推演批量；显存不足时降低为 8 或 4 |
| `--limit-per-split 2` | 每类测试截取前 2 个场景，仅用于小样本诊断 |
| `--formation-limit 8` | 截取前 8 个形成片段，仅用于小样本诊断 |
| `--mode all/evaluate/analysis` | 全流程 / 仅评测 / 仅分析，默认 `all` |
| `--boot 2000` | 场景 bootstrap 次数，不改变预测轨迹 |
| `--save-trajectories` | 额外保存全部预测网格 |
| `--no-plots` | 跳过画图，仍导出统计和表格 |
| `--strict-audit` | 真值/引擎验收失败即停止，见第 5 节 |

- **`No module named torch/numpy/matplotlib`**：使用第 1 节的完整 Python 路径，不使用缺依赖的系统 Python。
- **`Unpaired source models. Missing=...`**：当前模型只有 seed 0，必须显式写 `--seeds 0`。程序不自动训练缺失模型。
- **`CUDA requested but unavailable`**：改用 `--device auto` 或 `--device cpu`。
- **`Existing experiment 09 run has different sources/protocol`**：换一个新的 `--run` 名称，不覆盖旧结果。
- **`Cached result identity changed`**：缓存文件已缺失或变动，应检查文件完整性，不绕过校验。
- **文件身份、协议或 benchmark 检查失败**：确认使用原始实验 03 已完成目录和对应数据，不能把实验 06、08 或五任务 rerun 权重冒充为 03 来源。

后续若准备好了三次独立训练的模型，可以重复传入它们真实的 `--source-run` 目录并使用 `--seeds 0 1 2`。每个 seed 必须包含四个条件，训练/验证/benchmark 文件身份必须一致。也可用 `--models` JSON 显式选择，字段为 `models=[{condition, seed, path, source_run}, ...]`；它与 `--source-run` 不能同时使用。当前没有核实其他两个 seed 的来源，因此此处不提供虚构目录。

## 7. 自动测试

```bash
cd /data/huazhouqi/Scaffold-v1/Scaffold-v1

/root/miniconda3/envs/zhouqi/bin/python -m pytest \
  preresearch/09_physics_rollout/tests/test_protocol.py -q
```

此前已通过 15 项测试，包括固定步数、建造动作对齐、无 teacher forcing、引擎对照、配对统计、缓存身份和 Fig8 导出。真实数据小样本与冻结 State 模型两步接口也已检查。完整模型评测尚未由助手启动。
