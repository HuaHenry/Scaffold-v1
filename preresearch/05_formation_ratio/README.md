# 实验05：形成监督采样比例

## 1. 做什么，为什么改方案

研究问题：**总训练预算相同时，把更多监督样本分配给结构建造过程，能否提高模型对结构变化的预测能力？**

本版本保留原始物理轨迹和现有模型，不再把建造过程压进更少帧。每次训练更新，从“形成转移池”和“成品操作转移池”按指定比例抽样。这里的比例是**训练监督样本的占比，不是序列中的帧数，也不是损失权重**。

原计划见 [`plan/05_消融_形成过程占比.md`](../../plan/05_消融_形成过程占比.md)。旧压缩方案在冻结数据上失败：重新安排建造时间改变沙丘成品，且F16有136个场景无法在11个建造槽中完整编码动作。用户确认改为监督比例实验，因此本版本不再执行物理重放、生成剂量数据变体或过滤场景。

**协议版本为2。必须新建运行，不能续跑旧的 `20261111-111326_ratio_f16_f48`。旧失败目录、缓存和实验03结果原样保留。**

## 2. 五档设置与可比预算

所有档位都使用同一份冻结的 `process` 数据：81,920条原始序列，每条原本包含32帧形成和4个8帧操作片段。数据文件仍然是64帧，但本实验**不再让每个剂量对应不同的64帧序列**。

- 形成池：每条31个合法相邻帧转移，共2,539,520个。
- 操作池：每条28个合法相邻帧转移，共2,293,760个。
- 相邻帧转移始终取原轨迹的 `t−1 → t`及原动作；不跳帧、不跨操作片段重置边界。
- 所有档位的场景、形成池和四个操作片段完全相同。变化的只是两个池被训练采样的频率。

默认batch为256，每个batch严格按整数配额分别有放回抽样，再打乱合并后的样本。所有档位统一94,400次更新：

| 形成监督比例 | 每batch形成/操作样本 | 总更新数 | 总监督抽样次数 | 形成抽样次数 | 操作抽样次数 |
|---:|---:|---:|---:|---:|---:|
| 0% | 0 / 256 | 94,400 | 24,166,400 | 0 | 24,166,400 |
| 12.5% | 32 / 224 | 94,400 | 24,166,400 | 3,020,800 | 21,145,600 |
| 25% | 64 / 192 | 94,400 | 24,166,400 | 6,041,600 | 18,124,800 |
| 50% | 128 / 128 | 94,400 | 24,166,400 | 12,083,200 | 12,083,200 |
| 75% | 192 / 64 | 94,400 | 24,166,400 | 18,124,800 | 6,041,600 |

94,400次更新选取了03旧 `process`训练的计算量级作为默认锚点，但这里是**固定更新、有放回采样，不是完整5轮**。不保证每条转移恰好出现，也不称为完整epochs。

### 为什么五档都要重新训练

**包括0%和50%在内，五档全部从头训练，不复用03旧模型作为剂量端点。**

- 新0%只训练原 `process`的四个操作片段；03的 `state`有八个操作片段，支持集不同。
- 新50%每batch精确一半形成、一半操作；03 `process`的自然形成转移比例为31/59，约52.54%，并且采用完整epochs。
- 03旧模型的预算和验证时点也与新协议不同，因此不能混进等预算曲线。

本版本训练种子固定为0。三种子验收尚未完成，不能用场景bootstrap替代它。

### 共同训练设置

| 设置 | 值 |
|---|---|
| 模型 | 原GridFormer，隐藏维度384、深度8、6个注意力头 |
| 输入与目标 | 原前景、背景和当前动作图 → 下一帧前景、背景 |
| 训练协议 | 原始物理单步 `video`，unroll=1 |
| 优化器 | AdamW，betas=(0.9,0.95)，矩阵参数权重衰减0.05，梯度裁剪1 |
| 学习率 | 3e-4，预热300步，余弦衰减至3e-5；所有档位共用进度 |
| 损失与EMA | 变化格权重8，EMA最高衰减0.999 |
| 选模 | 每500步及最后一步，在同一独立验证集评估EMA；按连锁准确率选最佳，平分取较早checkpoint |
| 测试 | 同一冻结困难bench，512个场景，四个划分各128个 |

原始动作仍用现有单格动作token编码。审计会报告源数据已有的同格动作覆盖限制，不额外合并动作。当前冻结来源的全量检查发现同帧同格冲突为0。

## 3. 环境和冻结来源

```bash
conda activate zhouqi
cd /data/huazhouqi/Scaffold-v1/Scaffold-v1/preresearch/05_formation_ratio
```

使用仓库已有的Python、NumPy、PyTorch和Matplotlib；小规模测试还需要pytest。已核验的环境为Python 3.11、NumPy 2.4.6、PyTorch 2.8.0。

本目录复用同一仓库中 `01_pilot_validity`的模型、损失、单步数据加载和评测，以及 `03_main`/`04_ood_generalization`的来源校验、统计辅助函数，不能单独复制出去运行。

默认冻结来源：

```text
experiments/preresearch/03_main/20261111-044057_four_conditions/
game_code/output/data_v3_simple_train_hard_eval_81920/
```

训练数据、验证集和bench实际路径从03的 `audit.json`读取，并校验SHA1。缺文件、来源身份变化或旧运行协议不匹配都会拒绝执行，不重新生成或修复源数据。

## 4. 推荐运行方式

### 4.1 先只读检查

```bash
env -u FW_RUN bash run_all.sh --dry-run
```

检查03冻结来源，打印五档预算；不创建目录，不写审计，不生成数据，不训练。哈希检查可能需要一些时间。数据池和动作检查在下一步执行。

### 4.2 准备新运行，只审计和保存作业

```bash
env -u FW_RUN bash run_all.sh \
  --tag supervision_ratio \
  --stage prepare
```

准备阶段读取原始数据的数组头、阶段标签、重置边界、动作和场景，核对全部合法转移及池指纹，保存 `source.json`、`audit.json`和任务列表。

**不生成任何训练变体，不创建 `data/`、`cache/`或 `assemble/`，不启动训练。** 不再有旧方案的16.4GB磁盘缓存。

终端打印 `run=<运行名>`。后续使用该运行名：

```bash
RUN='<准备阶段打印的运行名>'
bash run_all.sh --run "$RUN" --stage train --gpus 0,1,2,3
```

五个新任务，每个使用一张GPU；四张卡时先执行四档，剩下一档排队。每档训练结束后评测；全部成功后自动生成分析。不要把旧失败运行名放入 `RUN`。

### 4.3 一键准备、训练、评测和分析

```bash
env -u FW_RUN bash run_all.sh \
  --tag supervision_ratio \
  --gpus 0,1,2,3
```

默认 `--stage all`。此命令会实际开始训练；只想检查或准备时必须使用 `--dry-run`或 `--stage prepare`。

### 4.4 只重新分析

```bash
bash run_all.sh --run "$RUN" --stage analyze --boot 2000 --seed 12345
```

`--seed`控制场景bootstrap，不修改训练种子0。每次分析创建新目录，不覆盖之前报告。

### 4.5 如需缩小共同更新预算

可以为**新运行**显式指定更少更新，例如：

```bash
env -u FW_RUN bash run_all.sh \
  --tag supervision_ratio_steps12000 \
  --steps 12000 \
  --stage prepare
```

五档仍共享同一更新数、batch和验证规则。运行一旦绑定，不能改变更新预算或比例；续跑默认继承记录中的预算。

默认比例无需传参。需要显式子集时使用例如 `--ratios 0,0.25,0.5,0.75`；比例必须升序、不重复，包含0和0.5，并满足 `ratio × 256`为整数。不接受旧 `--levels 0,16,32,48`，也不把旧F标签静默换算成监督比例。

## 5. 输出与报告

默认结果位置：

```text
experiments/preresearch/05_formation_ratio/<运行名>/
├── run.json
├── source.json
├── pipeline.lock
├── audit.json
├── jobs/formation_ratio.json
├── train/
│   ├── r000_seed0/                 # 0%，每batch形成0个样本
│   ├── r032_seed0/                 # 12.5%，每batch形成32个样本
│   ├── r064_seed0/                 # 25%
│   ├── r128_seed0/                 # 50%
│   └── r192_seed0/                 # 75%
└── analysis/supervision_ratio_boot2000_seed12345/
    ├── meta.json
    ├── formation_ratio.json
    ├── formation_ratio.md
    └── figs/dose_response.png、dose_response.pdf
```

每个训练目录保存 `config.json`、`train.log`、`train_log.jsonl`、`ckpt.pt`、`done.json`、`eval.log`、`eval.json`、`eval_samples.npz`和 `eval_complete.json`；失败时记录 `failed.json`。

- `source.json`绑定协议版本、来源、预算和哈希。
- `audit.json`记录完整场景分布、动作类型、共享形成/操作池大小与指纹，以及每档计划监督暴露量。
- `done.json`记录实际更新数、形成/操作样本数、实际比例、最佳验证步数及checkpoint哈希。
- `formation_ratio.md/.json`报告各划分的连锁F1、变化F1、变化区域准确率、干预排序准确率和规则违反率。
- 报告给出各档相对0%/50%的场景配对差值及95%bootstrap区间、Spearman、线性斜率、均值是否单调和观察到的最佳比例。
- `dose_response.png/.pdf`画四类测试的“监督比例 → 连锁F1”曲线；误差带是场景不确定性，不是种子标准差。

## 6. 续跑、资源与常见错误

对已准备好的新版本运行，重新执行训练阶段即可：

```bash
bash run_all.sh --run "$RUN" --stage train --gpus 0,1,2,3
```

- 完成模型先校验配置、池身份、总更新和实际比例，再复用；不会覆盖无效完成结果。
- 本版本不支持优化器步级恢复。中断的训练从第0步重跑该档，其他已完成档位保留。
- 已完成训练但评测中断：没有 `eval_complete.json`时只重做推理，不重训。完成标记存在却文件哈希变化时拒绝覆盖。
- 所有独立阶段共用运行锁，训练器另有单任务锁；结果路径中的软链接被拒绝。
- GPU编号不能重复；`--gpus cpu`可用于小规模接口检查，不建议正式训练用CPU。
- `FW_RUN`会影响运行名，新运行命令使用 `env -u FW_RUN`防止误用旧值。
- `GPUS`提供默认GPU列表，`PY`可指定解释器。`FW_RESULTS`若已设置会替换默认结果根目录，来源检查和续跑要保持一致。

**资源**：准备阶段不解压大帧数组到磁盘，但会读取大型动作JSON和较小的phase/cont数组。正式训练沿用既有全量数据加载器：每个任务将原始前景、背景和预编码动作图放在自己的GPU上，数据本体约24.7GB，另需模型、优化器、验证和激活空间；多卡任务也会各自加载数据，需足够主机内存。没有额外帧缓存不等于没有内存开销。

| 错误 | 处理 |
|---|---|
| retired frame-compression protocol | 不要改旧 `source.json`；新建 `supervision_ratio`运行。 |
| unrecognized arguments: `--levels`、`--workers`、`--shard-size`等 | 这些是旧生成方案参数，新版本不再需要；使用上面的新命令。 |
| source / sampling budget changed | 恢复正确来源和原预算，或新建运行；不要改完成标记。 |
| loaded transition pools differ | 停止，检查冻结数据和审计身份；不过滤或替换场景。 |
| CUDA out of memory / 主机内存不足 | 查看 `train.log`；释放资源或减少并行任务数，不擅自删减数据或改变batch。 |
| missing/uncommitted evaluation | 先完成对应新运行的训练与评测，再分析。 |

队列每30秒显示日志进度，可用 `--progress-every`调整；实际训练更新和验证细节在 `train.log`和 `train_log.jsonl`。

## 7. 结果能说明什么

- 相同总监督量下，形成占比增加必然减少操作抽样量。本实验衡量训练预算分配，不是“操作训练量不变、额外加入形成数据”。
- 保留原轨迹消除了压缩改变物理时间的混杂，但有放回采样改变每类样本的重复暴露次数，不能宣称所有转移等次数覆盖。
- 五档种子0结果只是一组训练结果。场景配对bootstrap不能证明跨初始化稳定性。
- `id`是当前困难bench的陌生结构测试，不是真正训练分布内测试。
- 先升后降、平坦和下降都是允许结果；最佳比例只作描述，不用困难bench反向选checkpoint。
- 这是对原实验05方案的明确修订，不称为完成了原“五档F帧、三种子、旧端点直接复用”的验收。

## 8. 小规模验证

新协议98项小规模回归全部通过，覆盖严格比例采样、共同更新预算、原始转移边界、来源隔离、五档微型CPU训练、EMA验证选模、checkpoint内部身份和有限权重、真实评测元数据、队列续跑和配对统计。

真实03来源只读检查通过；全部81,920条原始process数据的转移池、场景、动作审计通过，池大小为2,539,520个形成转移和2,293,760个操作转移。真实旧失败运行的拒绝续跑检查通过，其 `source.json`未变。本次没有启动正式训练，临时测试输出仅在 `/tmp`且已自动清理。

重跑测试时，将临时输出限制在 `/tmp`并自动清理：

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
from pathlib import Path
import subprocess
import sys
import tempfile

with tempfile.TemporaryDirectory(prefix="scaffold05-v2-tests-", dir="/tmp") as temporary:
    result = subprocess.run([
        sys.executable, "-m", "pytest", "exp/tests/test_ratio.py", "-q",
        "-p", "no:cacheprovider", "--basetemp", str(Path(temporary) / "pytest"),
    ])
raise SystemExit(result.returncode)
PY
```

禁用pytest缓存及Python字节码，测试不在项目根目录生成临时文件夹。

## 9. 参数与代码入口

| 主入口参数 | 默认值或要求 |
|---|---|
| `--source-run` | 默认上述03来源；续跑继承已记录来源 |
| `--run` | 新协议运行名，也可用 `FW_RUN`；不接受完整路径 |
| `--tag` | `supervision_ratio`，自动新建运行的标签 |
| `--ratios` | `0,0.125,0.25,0.5,0.75`，取值0–1，不是百分数整数 |
| `--steps` | 每档94,400次更新，所有档位相同；续跑继承原值 |
| `--stage` | `all`；可选 `prepare`、`train`、`analyze` |
| `--dry-run` | 只读来源检查，不创建运行或训练 |
| `--gpus` | `GPUS`或 `0,1,2,3` |
| `--progress-every` | 队列刷新间隔30秒 |
| `--boot` | 2000次场景bootstrap |
| `--seed` | bootstrap种子12345，不改变训练种子0 |

`exp/main.py`编排流水线，`exp/audit.py`审计原始池，`exp/shared.py`校验预算和身份，`exp/train_ratio.py`执行比例训练，`exp/make_jobs.py`/`exp/run_queue.py`处理作业，`exp/analysis.py`生成报告，`exp/new_run.py`提供运行名工具。`exp/generate_data.py`仅保留退役提示，不再生成数据。

```bash
bash run_all.sh --help
```
