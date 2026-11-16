# 实验08：噪声形成轨迹

目标：在成品结构、后续操作、验证和测试场景保持相同的条件下，比较干净形成轨迹与两档噪声形成轨迹的预测效果。

**只使用实验03的原始结构任务，不是多任务主实验。先生成配对数据，再准备运行、训练评测，最后单独分析。**

**本实验默认评测最后一步EMA，不采用03的验证最佳模型；94,400步为固定更新预算，不是五轮完整遍历。**

## 1. 直接怎么跑

沿用其他预实验的 `zhouqi` 环境。以下命令按顺序在同一个终端执行；示例中的数据版本和运行名应使用尚未占用的名称。

```bash
conda activate zhouqi
cd /data/huazhouqi/Scaffold-v1/Scaffold-v1
export PYTHONDONTWRITEBYTECODE=1

DATA08=/data/huazhouqi/Scaffold-v1/Scaffold-v1/experiments/preresearch/08_noisy_formation/data_v1
RUN=noisy_seed0_v1
```

### 1.1 只读检查03来源

```bash
bash preresearch/08_noisy_formation/generate_data.sh \
  --out "$DATA08" --dry-run
```

核验03冻结来源和数据量范围，不创建派生数据、不训练。这个检查不代表噪声生成已经通过验收。

### 1.2 CPU生成配对噪声数据

```bash
bash preresearch/08_noisy_formation/generate_data.sh \
  --out "$DATA08" \
  --source-run 20261111-044057_four_conditions \
  --episodes 81920 \
  --noise-seed 0 \
  --max-fallback 0.1 \
  --workers 8
```

默认8个CPU进程。生成、压缩、写盘和哈希校验全部完成后才打印 `[08 data] complete`；采样进度100%不等于完成。

不合格数据会保留供诊断，但禁止训练。不要跳过验收或删除诊断文件后强行使用。

### 1.3 准备六个任务

```bash
bash preresearch/08_noisy_formation/run_all.sh \
  prepare --run "$RUN" --data "$DATA08" \
  --episodes 81920 --steps 94400
```

只写入冻结协议和任务清单，不训练。六个任务为四个原始条件和两个噪声条件，均为训练种子0。

可选的运行核验：

```bash
python -m preresearch.08_noisy_formation.exp.main audit --run "$RUN"
```

`audit`核验运行设计、数据和代码身份，不代表六个模型已经训练完毕。

### 1.4 四卡训练并评测

```bash
bash preresearch/08_noisy_formation/run_all.sh \
  run --run "$RUN" --gpus 0,1,2,3 --progress-every 30
```

每卡一个独立模型，六个任务排队；不是单模型四卡DDP。两卡改为 `--gpus 0,1`，单卡改为 `--gpus 0`。以 `--gpus`指定实际卡号，队列会给子进程重新设置 `CUDA_VISIBLE_DEVICES`。

每个任务训练结束后立即评测。**`run`不会自动生成分析，必须继续执行下一步。**

### 1.5 单独生成分析

```bash
python -m preresearch.08_noisy_formation.exp.analysis \
  --run "$RUN" --bootstrap 10000 --bootstrap-seed 12345
```

结果默认在：

`experiments/preresearch/08_noisy_formation/noisy_seed0_v1/analysis/`

主要查看 `table.csv`、`results.json` 和 `dose_response.svg`。当前代码不生成Markdown报告或PNG/PDF图。

两份shell入口也可从任意目录通过绝对路径调用，会自行进入项目根目录。默认使用当前 `python`；可用 `PY`指定已有环境的Python解释器。

**注意：`run_all.sh`必须传入 `prepare / run / evaluate / audit` 子命令，裸运行不会自动执行全流程。`--run`只能是简单运行名，不能传目录路径，也不自动读取 `FW_RUN`。**

## 2. 当前默认协议

| 项目 | 默认值 |
|---|---|
| 来源03运行 | `20261111-044057_four_conditions` |
| 原始训练数据 | `game_code/output/data_v3_simple_train_hard_eval_81920` |
| 验证数据 | 原始数据目录的 `validation`，所有条件相同 |
| 测试数据 | 原始数据目录的 `bench.npz`及`bench.json`，512场景、每组128场景 |
| 每个条件的数据量 | 81,920集，同一 `[0,N)` 前缀 |
| 原始条件 | `state / process / partial / shuffle` |
| 噪声条件 | `process_noisy_low / process_noisy_high` |
| 名义噪声概率 | clean：0；low：0.1；high：0.3 |
| 训练种子 | 仅0 |
| 更新数 | 每任务94,400步 |
| batch及抽样 | 256；合法video转移中均匀、有放回抽样 |
| 每任务累计监督抽样 | 24,166,400次 |
| 模型 | 公共`GridFormer`，dim384、depth8、heads6 |
| 优化 | lr0.0003→0.00003、warmup300、AdamW、wd0.05 |
| 其他 | change_weight8、unroll1、EMA0.999、日志每50步 |
| 验证 | 每500步及末步，仅监控，不选模型 |
| 评测模型 | 最后一步EMA，保留最后原模型及EMA |
| 默认任务数 | 6个；默认全部从头训练 |

`state/partial`每集56条合法转移；`process/shuffle/两个noisy`每集59条。默认不做形成/操作各半抽样，process的形成转移比例为31/59，约52.54%。不跨操作片段重置边界。

**复用的是03数据和审计，不是03模型。** 03使用五轮完整无放回遍历、验证最佳EMA，且state/partial只有89,600步；不能把03报告成绩当作08的等设置基线。

## 3. 噪声改了什么

- 只改形成阶段；原始地形和形成终态相同，索引32起的操作状态及动作逐项沿用03，`cont/phase`直接复制。
- 对形成帧1～18中符合条件的原始place动作按概率注入延迟、位置偏移或暂时遗漏。
- 暂时遗漏不是永久删除：动作会延后补做；21～27帧进行合法修复，28～31帧无动作、按真实物理演化。
- 含`dune/pond`的episode不注入噪声，但仍保留在数据中，不过滤这些场景。
- 0.1和0.3是名义动作触发概率，不是实际物理变化比例；档位固定，CLI不能修改。

生成后必须同时满足：

1. 实际物理转移比例：clean < low < high。
2. high的实际物理转移比例至少30%。
3. 两个noisy的回退率均严格小于 `min(0.1, max_fallback)`。

因此调大`--max-fallback`也不能放宽至10%以上。是否合格及配对审计保存在派生数据的`manifest.json`。

## 4. 续跑、纯评测与复用06

### 4.1 中断后续跑

```bash
bash preresearch/08_noisy_formation/run_all.sh \
  run --run noisy_seed0_v1 --gpus 0,1,2,3
```

- 已完成且身份匹配的训练和评测直接复用。
- 中断训练没有有效完成标记：从第0步重训，**不恢复优化器或中间步数**。
- 仅评测中断：保留训练模型，重新评测。
- 单任务失败记录`queue/<任务>/failed.json`；其他任务可能继续，最终队列非零退出。
- 同一运行的并发流水线由`pipeline.lock`拒绝。
- 数据生成中断后，非空且未完成的生成目录不自动续生成；保留诊断并换新数据版本。
- 完成后的数据请求、运行设计及源码身份不可变；要改N、步数、来源等，请用新运行名。数据目录与运行目录必须分离。

### 4.2 只评测，不训练

```bash
bash preresearch/08_noisy_formation/run_all.sh \
  evaluate --run noisy_seed0_v1 --gpus 0,1,2,3
```

推理前先核验全部六个最终checkpoint，缺任一模型就停止；不会补训练、回退到验证最佳模型或静默只测部分条件。此命令也不会自动分析。

### 4.3 可选：只读复用06的四个基线

为一个新运行执行：

```bash
bash preresearch/08_noisy_formation/run_all.sh \
  prepare --run noisy_reuse06_seed0_v1 --data "$DATA08" \
  --episodes 81920 --steps 94400 \
  --reuse-scale-run <已完成且配置匹配的06运行名>
```

将占位符替换成实际06运行名后再执行。要求06中对应N、种子0的四个基线都已完成，来源、哈希、步数、batch及训练配置与08一致；不满足则拒绝。

复用时只训练两个noisy任务；四个基线模型留在06原目录，全部评测输出写入08，不覆盖06成绩。

## 5. 参数

| 阶段 | 参数 | 含义及默认值 |
|---|---|---|
| 生成 | `--out` | 必填，派生数据目录 |
| 生成 | `--source-run` | 默认03冻结运行`20261111-044057_four_conditions` |
| 生成 | `--episodes` | 生成集数，默认81920，不超过03总量 |
| 生成 | `--noise-seed` | 噪声种子，默认0；不是训练种子 |
| 生成 | `--workers` | CPU生成进程数，默认8 |
| 生成 | `--max-fallback` | 回退率限制，默认0.1，实际仍要求严格小于10% |
| 生成 | `--dry-run` | 只读检查来源和N，不生成、不训练 |
| prepare | `--run / --data` | 必填，简单运行名及合格派生数据目录 |
| prepare | `--episodes / --steps` | 默认81920/94400；N不得超过派生数据集数 |
| prepare | `--reuse-scale-run` | 可选，只读复用匹配的06四基线 |
| run/evaluate | `--run` | 必填，已准备的运行名 |
| run/evaluate | `--gpus` | 默认`0,1,2,3`；不允许重复编号；支持`cpu`调试 |
| run/evaluate | `--progress-every` | 进度打印间隔，默认30秒，须为有限正数 |
| analysis | `--run` | 必填，六任务均已完成的运行名 |
| analysis | `--bootstrap` | 配对场景抽样次数，默认10000 |
| analysis | `--bootstrap-seed` | 统计种子，默认12345；不是训练种子 |

小规模调试可减少生成与prepare的`--episodes`并缩短`--steps`，但仍须通过噪声验收；小样本不保证合格，也不能冒充正式结果。08没有`--stage`、`--tag`、`--boot`、`--allow-incomplete`或可调训练种子参数。

## 6. 输出目录

默认结果根为`experiments/preresearch`，公共`FW_RESULTS`可覆盖此根目录。

```text
experiments/preresearch/08_noisy_formation/
  data_v1/
    manifest.json
    complete.json
    process_noisy_low/
      train_process.npz
      train_process_actions.json
      train_summary.json
      episode_audit.json
    process_noisy_high/
      ...
  noisy_seed0_v1/
    run.json
    protocol.json
    jobs.json
    run_complete.json
    pipeline.lock
    train/<variant>_seed0/
      config.json
      train_log.jsonl
      ckpt.pt
      done.json
      train_complete.json
    evaluate/<variant>_seed0/
      eval.json
      eval_samples.npz
      eval_complete.json
    queue/<variant>_seed0/
      train.log
      eval.log
    analysis/
      results.json
      table.csv
      dose_response.svg
      complete.json
```

原始四条件不复制进派生数据目录，继续引用03。`run_complete.json`表示协议与任务清单已提交，**不是全部训练完成**。

分析固定写入同一`analysis/`。相同参数及身份重复分析会复用；已有完成标记后更改bootstrap参数或产物身份会拒绝覆盖，不会自动建立新编号目录。

## 7. 如何看结果

- 指标：连锁F1（`cascade_f1`）、变化F1（`change_f1`）、关注区准确率（`cell_acc_mask`）、排序准确率（`rank_acc`）、违规率（`any_violation`）。前四项越高越好，违规率越低越好。
- 报告全部场景、各测试分组和`n_changed >= 3`子集；该子集只是变化量代理，不是精确的支撑拆除标签。
- JSON包含两个noisy分别减去四个原始基线的差值，以及high减low的差值和95%配对场景区间。
- `dose_response.svg`画clean、low、high的连锁F1，横轴为名义噪声概率；无论是否递增都如实保留。
- 仅一个训练种子；场景区间不代表跨训练种子稳定性。全部六任务必须有有效训练和评测产物，不支持缺失格分析。
- 当前`id`也属于困难结构测试，不是简单训练分布内测试。噪声改善或变差只能支持当前结构任务与预算下的观察。
- 评测沿用01/03公共32步推演和评分代码，默认主要指标取预测的首次稳定状态；不要直接与多任务主实验的固定第32步指标混用。

## 8. 资源与检查

无需增加08专用依赖；沿用公共环境。公共依赖文件为`preresearch/01_pilot_validity/requirements.txt`。08依赖01训练评测、03审计、04统计、05指标校验、06辅助代码和仓库物理模块，不能单独复制运行。

生成需较多CPU内存、临时磁盘和压缩写盘时间。公共训练加载器会解压数据并将训练帧和动作放到设备，正式N需要较高内存与显存；多卡并行会叠加资源占用。代码不会自动缩batch或预算规避OOM，运行前先确认卡上没有冲突任务。

已有小规模测试可执行：

```bash
cd /data/huazhouqi/Scaffold-v1/Scaffold-v1
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  -q -p no:cacheprovider preresearch/08_noisy_formation/tests
```

真实03来源的前512集噪声探针默认跳过；需要时单独启用，不生成正式派生数据、不训练，但会读取原始数据并使用8个CPU进程：

```bash
RUN_08_REAL_SOURCE=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  -q -s -p no:cacheprovider \
  preresearch/08_noisy_formation/tests/test_frozen_source.py
```

本README依据当前入口与源码编写；本次补文档未启动正式数据生成、GPU训练或评测，也未执行上述测试。
