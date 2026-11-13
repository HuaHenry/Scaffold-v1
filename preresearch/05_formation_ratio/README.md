# 实验05：形成监督采样比例

## 1. 做什么，为什么改方案

研究问题：**总训练预算相同时，把更多监督样本分配给结构建造过程，能否提高模型对结构变化的预测能力？**

本版本保留原始物理轨迹和现有模型，不再把建造过程压进更少帧。每次训练更新，从“形成转移池”和“成品操作转移池”按指定比例抽样。这里的比例是**训练监督样本的占比，不是序列中的帧数，也不是损失权重**。

原计划见 [`plan/05_消融_形成过程占比.md`](../../plan/05_消融_形成过程占比.md)。旧压缩方案在冻结数据上失败：重新安排建造时间改变沙丘成品，且F16有136个场景无法在11个建造槽中完整编码动作。用户确认改为监督比例实验，因此本版本不再执行物理重放、生成剂量数据变体或过滤场景。

**当前协议版本为3：只评测训练全部完成时第94,400步的EMA模型。验证分数只记录，不再用于选择早期模型。**

**现有 `20261111-115405_supervision_ratio`只保存了验证分数最好的早期模型，没有保存训练最后一步的权重。这是旧代码的遗漏，不能靠重新评测恢复；需新建运行重新训练。** 新版会保存最后一步的原模型和EMA，评测使用最后一步EMA。旧结果、旧失败运行和实验03原样保留。

## 2. 11档设置与可比预算

默认比例为 **0%、10%、20%、30%、40%、50%、60%、70%、80%、90%、100%**。含两端共11档。0%只抽操作样本，100%只抽形成样本；其他档混合抽样。新建11档独立运行，不在旧五档运行中追加任务。

旧五档最终模型运行 `20261112-125456_supervision_ratio_final`按其已记录的比例继续处理，不改档位、不覆盖权重。

所有档位都使用同一份冻结的 `process` 数据：81,920条原始序列，每条原本包含32帧形成和4个8帧操作片段。数据文件仍然是64帧，但本实验**不再让每个剂量对应不同的64帧序列**。

- 形成池：每条31个合法相邻帧转移，共2,539,520个。
- 操作池：每条28个合法相邻帧转移，共2,293,760个。
- 相邻帧转移始终取原轨迹的 `t−1 → t`及原动作；不跳帧、不跨操作片段重置边界。
- 所有档位的场景、形成池和四个操作片段完全相同。变化的只是两个池被训练采样的频率。

默认batch为256，所有档位统一94,400次更新、24,166,400次监督抽样。10%平均对应25.6个形成样本，不能在一个batch内抽小数个样本；代码按累计目标安排每步25或26个，默认完整训练后的总比例精确为10%。其他十分位同样处理，不改变batch或总更新数。

| 形成监督比例 | 每步形成样本范围 | 形成抽样次数 | 操作抽样次数 |
|---:|---:|---:|---:|
| 0% | 0 | 0 | 24,166,400 |
| 10% | 25–26 | 2,416,640 | 21,749,760 |
| 20% | 51–52 | 4,833,280 | 19,333,120 |
| 30% | 76–77 | 7,249,920 | 16,916,480 |
| 40% | 102–103 | 9,666,560 | 14,499,840 |
| 50% | 128 | 12,083,200 | 12,083,200 |
| 60% | 153–154 | 14,499,840 | 9,666,560 |
| 70% | 179–180 | 16,916,480 | 7,249,920 |
| 80% | 204–205 | 19,333,120 | 4,833,280 |
| 90% | 230–231 | 21,749,760 | 2,416,640 |
| 100% | 256 | 24,166,400 | 0 |

每步有放回抽取两个池，再打乱合并的batch。若自定义极短预算导致总样本不能按比例整除，累计数量向下取整，误差不到一个样本；日志和完成记录显示实际比例，不静默声称精确。

94,400次更新选取了03旧 `process`训练的计算量级作为默认锚点，但这里是**固定更新、有放回采样，不是完整5轮**。不保证每条转移恰好出现，也不称为完整epochs。

### 为什么11档都要重新训练

**包括0%和50%在内，11档全部从头训练，不复用03旧模型作为剂量端点。**

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
| 保存与评测 | 只保存训练全部结束时的checkpoint，评测其中最后一步EMA；每500步和最后一步的验证分数仅作日志 |
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

检查03冻结来源，打印11档预算；不创建目录，不写审计，不生成数据，不训练。哈希检查可能需要一些时间。数据池和动作检查在下一步执行。

### 4.2 准备新运行，只审计和保存作业

```bash
env -u FW_RUN bash run_all.sh \
  --tag supervision_ratio_11levels_final \
  --stage prepare
```

准备阶段读取原始数据的数组头、阶段标签、重置边界、动作和场景，核对全部合法转移及池指纹，保存 `source.json`、`audit.json`和任务列表。

**不生成任何训练变体，不创建 `data/`、`cache/`或 `assemble/`，不启动训练。** 不再有旧方案的16.4GB磁盘缓存。

终端打印 `run=<运行名>`。后续使用该运行名：

```bash
RUN='<准备阶段打印的运行名>'
bash run_all.sh --run "$RUN" --stage train --gpus 0,1,2,3
```

11个新任务，每个使用一张GPU；四卡最多同时训练四档，其余排队，总共三批。旧任务若还在占卡，先等它完成再启动，不重复占用显存。每档训练结束后评测；全部成功后自动生成分析。不要把旧失败运行名放入 `RUN`。

### 4.3 一键准备、训练、评测和分析

```bash
env -u FW_RUN bash run_all.sh \
  --tag supervision_ratio_11levels_final \
  --ratios 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 \
  --gpus 0,1,2,3
```

默认 `--stage all`。此命令会实际开始训练；只想检查或准备时必须使用 `--dry-run`或 `--stage prepare`。

### 4.4 独立评测最后模型

训练完成后，可以只执行评测，不启动训练：

```bash
bash evaluate_final.sh --run "$RUN" --gpus 2,3
```

脚本先检查全部11档都已保存最后一步模型，再执行推理并生成分析。默认GPU为2、3；`--no-analysis`只评测，`--dry-run`只读检查末轮权重、不推理或写文件。已经完成且哈希匹配的末轮评测会复用。

**不能把旧运行 `20261111-115405_supervision_ratio`传给这个脚本获取末轮成绩：旧权重没有保存，脚本会明确拒绝，不偷偷使用早期模型，也不会自动重训。**

仅重新分析新版已完成结果：

```bash
bash run_all.sh --run "$RUN" --stage analyze --boot 2000 --seed 12345
```

`--seed`控制场景bootstrap，不修改训练种子0。每次分析创建新目录，不覆盖之前报告。

### 4.5 如需缩小共同更新预算

可以为**新运行**显式指定更少更新，例如：

```bash
env -u FW_RUN bash run_all.sh \
  --tag supervision_ratio_11levels_final_steps12000 \
  --steps 12000 \
  --stage prepare
```

11档仍共享同一更新数、batch和验证规则。运行一旦绑定，不能改变更新预算或比例；续跑默认继承记录中的预算。

默认比例为0至1、间隔0.1，共11档。可显式传入 `--ratios 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1`。子集仍须升序、不重复，包含0和0.5；不再要求单batch配额整除，由累计配额保证总量。旧五档运行沿用其记录，不自动扩展。旧 `--levels`等帧数参数不接受。

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
│   ├── r000_seed0/                 # 0%
│   ├── p010_seed0/                 # 10%
│   ├── p020_seed0/                 # 20%
│   ├── p030_seed0/                 # 30%
│   ├── p040_seed0/                 # 40%
│   ├── r128_seed0/                 # 50%
│   ├── p060_seed0/                 # 60%
│   ├── p070_seed0/                 # 70%
│   ├── p080_seed0/                 # 80%
│   ├── p090_seed0/                 # 90%
│   └── r256_seed0/                 # 100%
└── analysis/supervision_ratio_boot2000_seed12345/
    ├── meta.json
    ├── formation_ratio.json
    ├── formation_ratio.md
    └── figs/dose_response.png、dose_response.pdf
```

每个训练目录保存 `config.json`、`train.log`、`train_log.jsonl`、`ckpt.pt`、`done.json`、`eval.log`、`eval.json`、`eval_samples.npz`和 `eval_complete.json`；失败时记录 `failed.json`。

- `source.json`绑定协议版本、来源、预算和哈希。
- `audit.json`记录完整场景分布、动作类型、共享形成/操作池大小与指纹，以及每档计划监督暴露量。
- `done.json`记录实际更新数、形成/操作样本数、实际比例、`checkpoint_step=94400`、`selection=last`及checkpoint哈希；最佳验证步数仅保留作日志。
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
| old run saved only validation-best weights | 旧训练末轮权重没有保存；不能仅重新评测，需新建运行重新训练。旧结果保留。 |
| retired checkpoint/frame-compression protocol | 不要改旧 `source.json`；新建 `supervision_ratio_11levels_final`运行。 |
| unrecognized arguments: `--levels`、`--workers`、`--shard-size`等 | 这些是旧生成方案参数，新版本不再需要；使用上面的新命令。 |
| source / sampling budget changed | 恢复正确来源和原预算，或新建运行；不要改完成标记。 |
| loaded transition pools differ | 停止，检查冻结数据和审计身份；不过滤或替换场景。 |
| CUDA out of memory / 主机内存不足 | 查看 `train.log`；释放资源或减少并行任务数，不擅自删减数据或改变batch。 |
| missing/uncommitted evaluation | 先完成对应新运行的训练与评测，再分析。 |

队列每30秒显示日志进度，可用 `--progress-every`调整；实际训练更新和验证细节在 `train.log`和 `train_log.jsonl`。

## 7. 结果能说明什么

- 相同总监督量下，形成占比增加必然减少操作抽样量。本实验衡量训练预算分配，不是“操作训练量不变、额外加入形成数据”。
- 保留原轨迹消除了压缩改变物理时间的混杂，但有放回采样改变每类样本的重复暴露次数，不能宣称所有转移等次数覆盖。
- 11档种子0结果只是一组训练结果。场景配对bootstrap不能证明跨初始化稳定性。
- `id`是当前困难bench的陌生结构测试，不是真正训练分布内测试。
- 先升后降、平坦和下降都是允许结果；最佳比例只作描述，不用困难bench反向选checkpoint。
- 这是对原实验05方案的明确修订，不称为完成了原“五档F帧、三种子、旧端点直接复用”的验收。

## 8. 小规模验证

本次140项小规模回归全部通过，包含11档微型CPU训练、完整94,400步的精确累计配额、0%和100%端点、最后一步模型保存与评测、旧五档协议保持兼容、独立任务和报告曲线检查。真实11档来源只读检查通过；旧五档运行的来源、任务列表及最后checkpoint验收仍通过。本次没有启动正式训练。

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
| `--tag` | `supervision_ratio_11levels_final`，自动新建运行的标签 |
| `--ratios` | `0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1`，共11档，取值0–1 |
| `--steps` | 每档94,400次更新，所有档位相同；续跑继承原值 |
| `--stage` | `all`；可选 `prepare`、`train`、`analyze` |
| `--dry-run` | 只读来源检查，不创建运行或训练 |
| `--gpus` | `GPUS`或 `0,1,2,3` |
| `--progress-every` | 队列刷新间隔30秒 |
| `--boot` | 2000次场景bootstrap |
| `--seed` | bootstrap种子12345，不改变训练种子0 |

`evaluate_final.sh`/`exp/evaluate_final.py`只评测训练完成时的模型，不启动训练；`exp/main.py`编排流水线，`exp/audit.py`审计原始池，`exp/shared.py`校验预算和身份，`exp/train_ratio.py`执行比例训练，`exp/make_jobs.py`/`exp/run_queue.py`处理作业，`exp/analysis.py`生成报告，`exp/new_run.py`提供运行名工具。`exp/generate_data.py`仅保留退役提示，不再生成数据。

```bash
bash run_all.sh --help
```
