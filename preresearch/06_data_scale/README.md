# 实验06：数据量曲线

目标：观察训练数据增加时，四条件的连锁F1如何变化，以及 `process − state` 的提升是否缩小、保持或扩大。

**直接使用实验03的冻结数据，不生成新数据；所有模型从头训练，统一评测训练结束时的EMA。**

对应计划：`../../plan/06_消融_数据量曲线.md`。

## 1. 当前默认协议

| 项目 | 默认值 |
|---|---|
| 来源03运行 | `20261111-044057_four_conditions` |
| 训练数据 | `game_code/output/data_v3_simple_train_hard_eval_81920` |
| 验证数据 | 上述目录的 `validation`，各档不变 |
| 测试数据 | 上述目录的 `bench.npz` 和 `bench.json`，512场景，各划分128场景 |
| 数据量N | 5,120 / 10,240 / 20,480 / 40,960 / 81,920 |
| 条件 | `state / process / partial / shuffle`，含义沿用03 |
| 训练种子 | `0,1,2` |
| 每格更新数 | 94,400 |
| batch及抽样 | 256，合法video转移中均匀有放回抽样 |
| 每格抽样总数 | 24,166,400 |
| 模型 | dim384、depth8、heads6 |
| 优化 | lr0.0003→0.00003、warmup300、AdamW、wd0.05 |
| 其他 | change_weight8、unroll1、EMA0.999 |
| 验证 | 每500步和末步；只记录，不选模型 |
| 评测checkpoint | 最后第94,400步EMA，保留最后原模型及EMA |
| 任务数 | 5档 × 4条件 × 3种子 = **60个新任务** |
| 统计 | 种子先按场景平均，再做10,000次配对场景bootstrap；种子标准差单独报告 |

预算对齐05当前的等更新设置，不沿用旧计划的32～512集与6,000步。

**只复用03的数据，不复用03模型。** 03采用五个完整epochs和验证最佳模型，更新数随条件不同，与本实验的等更新、最终EMA协议不一致。05的比例模型也不进入本实验曲线。

## 2. 数据前缀与预算

每档四条件均取同一数据来源的 `[0,N)`，不换episode顺序。小档是大档的嵌套前缀；训练种子只改变模型初始化和抽样，不重新选择前缀。

以下不同成品数由真实03数据只读预检得到：

| N | 不同成品结构 | 重复episode | state/partial转移数 | process/shuffle转移数 |
|---:|---:|---:|---:|---:|
| 5120 | 4663 | 457 | 286720 | 302080 |
| 10240 | 8776 | 1464 | 573440 | 604160 |
| 20480 | 16287 | 4193 | 1146880 | 1208320 |
| 40960 | 30115 | 10845 | 2293760 | 2416640 |
| 81920 | 55464 | 26456 | 4587520 | 4833280 |

N是episode数，不是不同成品数。每个episode中，`state/partial`有56条合法相邻转移，`process/shuffle`有59条。不跨操作片段重置边界；保留 `shuffle` 的乱序形成转移，不能改为只抽 `cont=True` 的causal对照。

不复制或压缩轨迹，不重生成验证/测试数据，也不生成 `data512` 或大体积前缀缓存。全文件SHA1继承03审计，各档额外记录episode范围、场景清单摘要、不同成品数、重复数和类别计数。

## 3. 环境

沿用已安装的 `zhouqi` 环境及公共模型、物理规则和评测代码，不需要安装新依赖。

```bash
conda activate zhouqi
cd /data/huazhouqi/Scaffold-v1/Scaffold-v1/preresearch/06_data_scale
```

两份shell入口均可从其他目录用完整路径调用；脚本会自行进入实验目录。默认用当前 `python`，也可设置 `PY` 指定已有环境的Python。

## 4. 启动命令

### 4.1 先只读检查

```bash
env -u FW_RUN bash run_all.sh --dry-run
```

核对真实03输入SHA1、数组尺寸、数据档位和预算。**不创建运行目录、不写任务、不训练、不推理。** 输入变化、N超范围、重复档位或非法种子都会拒绝，不允许加载器静默截断N。

### 4.2 四卡完整运行

```bash
env -u FW_RUN bash run_all.sh \
  --tag data_scale_final \
  --gpus 0,1,2,3
```

默认60个任务，每张卡运行一个独立模型，其余排队；不是单个模型四卡DDP。每格完成训练后立即评测，全部完成后自动分析。启动时打印运行名和结果目录。

### 4.3 先跑单种子版本

若想先得到一版曲线，可用独立新运行，仅训练种子0，共20个任务：

```bash
env -u FW_RUN bash run_all.sh \
  --tag data_scale_seed0_final \
  --seeds 0 \
  --gpus 0,1,2,3
```

报告会明确只有一个训练种子，不报告种子标准差，不能称完成默认三种子实验。后续三种子版本要新建运行，不能直接修改旧运行的任务集合。

### 4.4 分阶段执行与续跑

先准备来源、审计及任务表，不训练：

```bash
env -u FW_RUN bash run_all.sh --stage prepare --tag data_scale_final
```

把终端打印的运行名替换下面的 `<运行名>`：

```bash
bash run_all.sh --run <运行名> --stage train --gpus 0,1,2,3
bash run_all.sh --run <运行名> --stage analyze
```

`train`阶段完成后也会自动分析。续跑继承该运行记录的数据来源、档位、种子和步数，不必再次传这些参数。

- 已有有效 `train_complete.json`：复用最终checkpoint。
- 已有有效 `eval_complete.json`：复用评测，不再次推理。
- 训练中断但尚未提交完成：配置必须匹配，然后从第0步重新训练，**不恢复优化器**。
- 评测中断且未提交：保留已完成训练，重新评测。
- 已提交结果的文件或身份改变：报错，不自动覆盖。
- 队列失败：记录 `failed.json` 并非零退出，不自动产生完整分析。
- 同一运行同时启动两个流水线：由 `pipeline.lock` 拒绝。

### 4.5 独立纯评测

```bash
bash evaluate_final.sh --run <运行名> --gpus 0,1,2,3
```

**启动任何推理前，先核验全部记录任务的最终checkpoint。** 缺任一模型整体停止；不会重训、回退验证最佳模型或只评测部分条件。已提交且身份匹配的评测会直接复用。

只读核验最终checkpoint：

```bash
bash evaluate_final.sh --run <运行名> --dry-run
```

只评测，不生成分析：

```bash
bash evaluate_final.sh --run <运行名> --gpus 0,1,2,3 --no-analysis
```

## 5. 可调参数

| 参数 | 含义 |
|---|---|
| `--source-run` | 数据来源03运行名或路径；默认使用当前冻结运行，不依赖其模型成绩 |
| `--run` | 续跑或分析的已准备运行名；默认可读取 `FW_RUN` |
| `--tag` | 新运行标签，默认 `data_scale_final` |
| `--sizes` | 递增且不重复的episode数，至少两档，范围1～81920 |
| `--seeds` | 递增、不重复、非负的训练种子；默认 `0,1,2` |
| `--steps` | 每格统一更新数；默认94400，只能给新运行改预算 |
| `--gpus` | 独立任务使用的GPU编号，默认 `0,1,2,3`；支持 `cpu` 用于小规模调试 |
| `--progress-every` | 队列读取日志并打印进度的间隔秒数，默认30 |
| `--boot` | 场景bootstrap次数，默认10000 |
| `--seed` | bootstrap种子，默认12345；**不是训练种子** |
| `--allow-incomplete` | 仅用于 `--stage analyze`，列明缺失并仅使用完整共同种子 |

例如三档、单种子的简化新运行：

```bash
env -u FW_RUN bash run_all.sh \
  --tag data_scale_3sizes_seed0 \
  --sizes 5120,20480,81920 \
  --seeds 0 \
  --gpus 0,1,2,3
```

该配置为12个任务，不是默认60任务。所有改动均写入新运行；旧运行不能更换来源、档位、种子、模型配置或更新数。

## 6. 输出目录

```text
experiments/preresearch/06_data_scale/<运行名>/
  run.json
  source.json
  audit.json
  pipeline.lock
  jobs/data_scale.json
  train/<条件>_n<N>_seed<种子>/
    config.json
    meta.json
    train.log
    train_log.jsonl
    training.lock
    ckpt.pt
    done.json
    train_complete.json
    eval.log
    eval.json
    eval_samples.npz
    eval_complete.json
    failed.json                  # 失败时存在；成功后移除
  analysis/data_scale_boot10000_seed12345/
    meta.json
    data_scale.json
    data_scale.md
    figs/data_scale.png
    figs/data_scale.pdf
    figs/process_state_gain.png
    figs/process_state_gain.pdf
```

重复生成分析时使用新编号目录，不覆盖旧分析。`train_scale.py`是队列内部入口；常规运行请使用 `run_all.sh`，以保留整条流水线的互斥锁。

## 7. 分析如何计算

- 指标：连锁F1（`cascade_f1`）、变化F1（`change_f1`）、关注区格准确率（`cell_acc_mask`）、排序准确率（`rank_acc`）、违规率（`any_violation`）。前四项越高越好，违规率越低越好。
- 每个场景的两个干预先按公共指标定义合并，再对训练种子求平均；之后配对抽样场景，输出95%区间。
- 每个N输出 `process − state/partial/shuffle`，以及这些差值各自的种子标准差。
- 拟合 `F1 = a + b·log2(N)`：b表示数据量每翻倍的变化；另报斜率差与最大/最小N的增益差。
- `data_scale.json`保存精确值、各种子均值、区间、标准差及训练/验证准确率；Markdown给出便于阅读的表格。
- 默认要求完整任务集合。显式不完整分析时，只使用在**所有N和四条件上完整**的共同训练种子；缺失格和未纳入的已完成格都会列出。没有完整共同种子则停止，损坏的已提交结果不能被忽略。

```bash
bash run_all.sh --run <运行名> --stage analyze --allow-incomplete
```

这只生成有明确缺失说明的分析，不会补训练或改动已记录任务。

## 8. 结论与资源限制

- 固定的是更新数，不是完整epochs。数据增大时，每条合法转移的平均重复抽样次数下降；不能把曲线变化都归因于数据量本身。
- 当前仅一个固定数据前缀顺序；训练种子不是不同数据子集种子。N还包含重复成品，需结合审计表解释。
- 当前 `id` 也是陌生困难结构测试，不是训练分布内的简单ID。
- 区间是场景区间，不是跨训练种子的置信区间。单种子结果不能声称跨种子稳定。
- 非单调曲线应如实报告。验证只监控；训练/验证的转移类型不同，准确率差值不能单独证明过拟合。
- 增益缩小或持续，只支持当前范围内的观察，不能单凭本实验证明“仅数据效率”或“额外信息”。

默认60个大模型任务，四卡约15批；请先确认其他训练已释放GPU。旧小数据计划的分钟级耗时估计不适用。

原始公共加载器先解压全量NPZ，再切前N集，因此小N也可能产生较高CPU内存峰值；四卡并行会叠加。全量任务还需要较大的数据显存及模型训练显存，代码不自动改变batch或缩短预算来规避OOM。准备阶段只读取哈希、数组头和场景清单，不解压整套帧数据。

## 9. 验证

已在 `zhouqi` 环境通过97项测试，包括：

- 五档预算和60个互不冲突的新任务；非法N、种子和GPU编号拒绝。
- 冻结来源、嵌套前缀、转移预算及重置边界。
- 四条件 × 两档 × 两种子的16格真实微型CPU训练。
- 最终EMA保存、配置/权重/预算身份核验、已完成训练复用。
- 纯评测全部预检、不自动训练、只读入口不写入。
- 不完整共同种子的处理与损坏结果拒绝。
- 真实CPU EMA推演及指标从预测状态复核。
- 种子平均、配对差值、log2斜率、种子标准差、JSON/Markdown及PNG/PDF图。

正式03数据的只读来源与五档预算预检已通过；**没有启动正式GPU训练，也没有完成60个任务的实际实验评测**。

复现测试：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  -q -p no:cacheprovider exp/tests
```
