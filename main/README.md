# 多任务主实验

这里只实现主实验03的扩展。**当前正式执行精简为5个代表任务：T01、W01、W02、G03、G07，只分别训练四条件模型，全部seed0，共20个模型。** 完整17任务及混训仍作为可选功能保留，不进入本次五任务队列。

数据生成和训练完全分开。当前每任务使用前10,240集、每模型94,400步、batch256、形成比例0.4、最终EMA；属于固定step、有放回抽样方案，不称为完整训练5轮。设置继承原运行的冻结协议，05/06选择依据保留。

## 当前五任务版：复制已完成模型，继续训练剩余模型

| 类型 | 任务 | 当前迁移时状态 |
|---|---|---|
| 结构干预 | T01：原结构基准 | 四条件训练、评测已完成，复制复用 |
| 简单平地 | W01：平地落水 | 四条件训练、评测已完成，复制复用 |
| 容器释放 | W02：容器侧壁放水 | 四条件训练、评测已完成，复制复用 |
| 坡面运动 | G03：坡面移沙 | 四条件需要新训练 |
| 狭缝释放 | G07：狭缝落沙 | 四条件需要新训练 |

按任务内容和计算预算选代表任务，不按测试分数筛选。结构、水、沙均有覆盖，但不宣称覆盖全部地形。17任务的数据保留，不重新生成、不删除。原运行已停止，当前可复制 **12个已完成模型及其第32步评测**，只新训练 **8个模型**。以后完成数以 `--stage status` 为准。

### 为什么用专用入口

新入口为 `main/run_subset.sh`，实现位于 `scripts/main_subset.py`。单训模型只依赖本任务数据及设置，因此缩小汇总任务集合不会改变其训练语义；混训模型不适用这种复用。

模型的 `config.json`、`done.json`、`ckpt.pt`、`train_complete.json`、已完成评测全部普通复制，原字节和原哈希不变，不用符号链接或硬链接，不修改旧目录。新目录中 `protocol.json`、`jobs.json`、`run.json` **逐字节保留原17任务训练协议快照**，用于验证模型原身份；它们不是五任务执行清单。

**实际五任务执行清单与复制来源在 `subset.json`，由 `subset_complete.json` 冻结。专用入口只执行其中20个单训模型，分析只纳入这5个任务。** 不改旧Python训练源码或数据身份，避免引入兼容性绕过。入口自身的源码哈希也冻结，之后修改入口需新建运行或保留对应版本。

**不要对五任务目标目录运行 `main/run_main.sh` 或直接调用 `main.training.main`。** 原完整入口读的是17任务快照，可能排队68个模型；shell入口现在会拒绝子集目录并提示改用专用入口。训练与分析均使用下面的 `run_subset.sh`。

### 第一步：迁移准备（只复制，不启动GPU训练）

项目根目录执行：

```bash
cd /data/huazhouqi/Scaffold-v1/Scaffold-v1
export PY=/usr/bin/python3
export OLD_RUN=/data/huazhouqi/Scaffold-v1/Scaffold-v1/experiments/main/runs/main_single_n10240_r04_seed0_v1
export RUN=/data/huazhouqi/Scaffold-v1/Scaffold-v1/experiments/main/runs/main_five_n10240_r04_seed0_v1

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  bash main/run_subset.sh --stage prepare --source-run "$OLD_RUN" --run "$RUN" \
  --tasks T01,W01,W02,G03,G07
```

准备会核验源数据和模型，普通复制已完成训练及已完成评测；未完成训练目录、未提交评测、失败记录不复制。复制中断可用同一命令继续，目标文件不匹配会拒绝，不自动覆盖损坏结果。已有成功迁移可幂等核验。源流水线或相关训练/评测锁仍被占用时拒绝迁移。

本次代理已执行上述准备并验证后，你无需再准备，直接执行下一步。

### 第二步：真正启动训练（只新训缺失模型）

```bash
PY=/usr/bin/python3 bash /data/huazhouqi/Scaffold-v1/Scaffold-v1/main/run_subset.sh \
  --stage train \
  --run /data/huazhouqi/Scaffold-v1/Scaffold-v1/experiments/main/runs/main_five_n10240_r04_seed0_v1 \
  --gpus 0,1,2,3 --progress-every 30
```

**这一条才启动GPU训练。** 已完成的12个模型和评测通过校验后直接跳过；只训练G03、G07的8个模型，每卡一个模型，训练结束自动评测。若之后停止并再执行同一命令，已提交模型仍跳过；被强停且无完成标记的模型从step0重训，不恢复优化器。不会启动其他12个任务，不启动混训，也不会重新生成数据。

按当前main每模型约4.3～4.4小时，8个新模型四卡约两批，**纯训练参考约8.6～8.8小时**，另加数据核验、加载、验证和评测；不同任务速度可能变化，不承诺精确结束时间。降低N没有减少94,400更新预算。

### 查看状态、纯评测、五任务分析

```bash
PY=/usr/bin/python3 bash main/run_subset.sh --stage status --run "$RUN"
PY=/usr/bin/python3 bash main/run_subset.sh --stage evaluate --run "$RUN" --gpus 0,1,2,3
PY=/usr/bin/python3 bash main/run_subset.sh --stage analyze --run "$RUN" --boot 10000 --bootstrap-seed 12345
```

`status`显示20格完成数及剩余训练数。纯评测先检查选中20个最终模型，缺失就停止，不启动训练。分析也要求所有选中模型/评测齐全，输出 `analysis/subset_boot10000_seed12345/` 的 `results.json`、`results.md` 与PNG/PDF图；后续重复分析使用新后缀目录。报告记录原协议、复制来源、五任务分析视图和完整身份，不将旧17任务平均值沿用为5任务结果。

下面各节描述完整任务注册、数据接口及原完整运行方式；当前执行以上五任务命令即可。

## 1. 目录结构

```text
main/
  README.md
  requirements.txt
  shared.py
  generate_data.sh           # 只生成/导入数据
  run_main.sh                # prepare/train/evaluate/analyze；不生成数据
  tasks/                    # 17任务注册、互斥边界、实际物理生成器
  data/                     # 生成、导入03、内存映射存储、冻结审计
  training/                 # 固定setting、任务表、采样、单训/混训、GPU队列
  evaluation/               # 统一32步评测、配对统计、表格和图
  tests/                    # 微型真实物理、CPU训练与评测测试
  extensions/               # 后续消融/预算/长程扩展的位置与接口约定

experiments/main/            # 仅存放实验结果，不放源码
  datasets/<数据版本>/       # 运行时生成，忽略版本管理
  runs/<运行名>/             # 运行时产生，忽略版本管理
  generate_<时间>.log        # 数据生成日志
```

每个模块目录均有自己的README。源码已迁移到项目根目录的 `main`，入口包名为 `main`；数据、训练结果和日志留在 `experiments/main`。项目 `.gitignore` 忽略整个实验结果目录，源码正常跟踪，不影响旧预实验结果。

## 2. 任务集：17个平行任务

- **T01：当前03预实验任务完整保留**。简单结构训练、困难结构操作评测，保留原冻结数据来源。不再把它里面的塔、桥、房屋分别算成新任务。
- **W01～W08：8个水任务**。
- **G01～G08：8个沙任务**。

新任务只包含本类材料及静态地形，评测是对应流动/堆积，不包含T01的结构破坏组合。水、沙各有平地、独立容器、单坡、连级平台、凹坑、孤立挡板、狭缝隔挡、孤立高台八个独立生成模板。

具体边界见 [tasks/README.md](tasks/README.md)。这里的“17任务”是17个不同且互斥指定的场景模板，不是17种物理机制。不存在子任务标签或混合组合标签。

## 3. 完全统一的评测

全部任务：**当前状态＋操作脚本 → 预测第32个物理tick的前景/背景状态**。

全部报告相同的两个metric：

1. `change_f1`：主指标，预测变化格子的F1。
2. `cell_acc_mask`：初态/真实终态定义的关注区中，两层材料均正确的格子比例。

每场景保留两个操作推演，先平均两个干预，再平均训练种子，最后配对场景bootstrap。没有水量、出料率或修复成功率等专属metric。

**统一使用第32步，不是第一个固定点。** T01原始测试目标保留，但这里重新评测模型；旧03/05的first-fixed-point成绩不会直接填入新主表。当前core含随机更新顺序，数据生成的模拟种子可复现；模型只看状态和动作，不读模拟随机状态，成绩反映这个现有接口的限制。

## 4. 环境

复用现有 `zhouqi` 环境和公共GridFormer/损失/物理代码，不修改游戏规则：

```bash
conda activate zhouqi
cd /data/huazhouqi/Scaffold-v1/Scaffold-v1/main
```

依赖范围见 `requirements.txt`，不需要为现有环境重装。脚本自行切回项目根目录，因此可以从任意工作目录使用完整路径调用。可用 `PY=/已有环境/bin/python` 指定Python。

## 5. 数据生成：单独运行

**可以先生成较大的M，等06结果后再选训练N；只要N≤M，就不需要重新生成数据。** 生成端 `--episodes M` 是每任务保存数量，训练prepare的 `--episodes N` 是每任务实际使用数量，两者可以不同。

以下以M=81920覆盖06全部五档；仅训练数组约1.53TiB，先确认磁盘足够。也可降低M，但之后不能选超过M的训练量。R示例不是已经确定的最优比例。

```bash
export M=81920
export DATA=/data/huazhouqi/Scaffold-v1/Scaffold-v1/experiments/main/datasets/m81920_v1
```

只读预检来源和存储估计：

```bash
bash generate_data.sh --out "$DATA" --episodes "$M" --dry-run
```

CPU生成17任务数据，默认每10秒打印进度；四卡还在训练时建议只用2个CPU生成进程：

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  bash generate_data.sh --out "$DATA" --episodes "$M" \
  --val-episodes 32 --test-scenes 128 --seed 20261113 --workers 2 --progress-every 10
```

进度包含任务/划分、完成数量和百分比、速度、已用时间、当前阶段ETA及已提交任务总数。新任务样本剩余时间是随实测速率更新的粗估，不含T01导入、写盘和审计；这些阶段单独显示。没有足够数据估算时显示“估算中”，最后仍需等待“整体完成，冻结校验通过”。

- T01流式导入当前03冻结数据的前M集，逐位保留帧和编码后的动作，不重新生成来源。验证使用实际冻结的完整32集，测试保留完整512场景及原四划分。验证summary中的部分shape字段陈旧，上限以真实NPZ数组头为准，不静默截断。
- 新16任务由现有core动作和tick生成，保存真实形成轨迹、四条件训练数据、独立验证和测试。
- 新任务使用明确的空间迁移划分：训练的模板锚点x为4～22，验证25～30，测试34～地图允许范围。**这不是训练分布内ID测试。** 各任务训练、验证、测试最终状态哈希无交集；训练内部允许重复，实际不同终态数和重复数均记录。
- `--workers`只控制按任务CPU并行，不改变种子或数据身份。一个任务内部顺序固定。
- 超过最大重试仍无法获得合法独立场景时失败，不放宽物理规则或偷偷接受重叠。
- 已提交任务可复用；留下未提交的半生成任务时拒绝自动删除或覆盖，请保留旧目录并选择新版本。

数据完成后，切到项目根目录执行单独全量审计：

```bash
cd /data/huazhouqi/Scaffold-v1/Scaffold-v1
python -m main.data.audit --data "$DATA"
```

## 6. 固定setting：准备主实验，不训练

等05/06结果后选择N和R；下面以N=20480、R=0.4示例，不是已确定的最优设置。训练只取每任务 `[0,N)`，单训和混训的四条件使用同一前缀，验证/测试不变。换N要创建新run，复用同一DATA即可。

```bash
cd /data/huazhouqi/Scaffold-v1/Scaffold-v1/main
export N=20480
export R=0.4
export RUN=/data/huazhouqi/Scaffold-v1/Scaffold-v1/experiments/main/runs/main_n${N}_v1
bash run_main.sh --stage prepare --run "$RUN" --data "$DATA" \
  --episodes "$N" --formation-ratio "$R" \
  --selection-note "记录05/06或开发集如何选择N和R，并注明是否使用过T01测试成绩" \
  --modes both --mixed-budget per_task
```

可以加 `--dry-run`只读核验，不创建运行目录。

### 四条件

| 条件 | 训练数据 | 抽样 |
|---|---|---|
| `state` | 完整对象的8个操作片段 | 56条操作转移中均匀有放回抽样 |
| `process` | 32帧有序形成＋4个操作片段 | 每任务内部按R分配形成/操作抽样 |
| `partial` | 形成中间状态稳定后，各自的8个操作片段 | 操作转移均匀有放回抽样 |
| `shuffle` | 同一32帧形成打乱＋相同4个操作片段 | 与process相同的R，不删乱序video转移 |

`process/shuffle`必须使用同一R。R需严格在0～1之间；端点属于后续消融，不将缺少一类监督的设置当作四条件正式主实验。其余条件不伪造形成占比。

每个模型默认dim384、depth8、heads6、batch256、lr0.0003→0.00003、warmup300、wd0.05、EMA0.999、change_weight8。单任务默认94,400更新，可在prepare用 `--steps`显式修改。验证每500步及末步，只监控，不选最佳checkpoint。

### 单训与混训预算

- `--modes single`：仅每任务分别训练。
- `--modes mixed`：仅全部任务混训。
- `--modes both`：两者都有，默认。
- `--mixed-budget per_task`：混训更新数＝单训更新数×任务数，保证每任务平均抽样量一致，默认。
- `--mixed-budget total`：混训和一个单训模型总更新数一致，每任务平均训练量下降；不据此直接作公平单训/混训优劣结论。

任务份额按累计轮转配额均衡，各任务内部形成比例也使用累计整数配额。不能整批随机选一个任务导致长期份额不同。混训的学习率进度按它自身总更新数计算。

**默认只训练一个种子0，不需要传 `--seeds`。** 显式指定时也只允许一个非负种子，多种子配置会直接拒绝。17任务、四条件、both：**68个单训模型＋4个混训模型＝72个模型**。默认per_task混训每个模型1,604,800更新，资源量很大，建议先小规模核验。

## 7. 训练、纯评测和分析

训练只消费已冻结数据，不生成任何任务：

```bash
bash run_main.sh --stage train --run "$RUN" --gpus 0,1,2,3
```

每张卡运行一个独立模型，不是单模型四卡DDP。每个模型训练完成后立即评测对应任务；混训模型评测全部任务。训练完成后需显式执行分析，不自动选择setting。

独立纯评测：

```bash
bash run_main.sh --stage evaluate --run "$RUN" --gpus 0,1,2,3
```

启动任何推理前先检查所有模型的最终权重，缺任何模型整体停止，**不重训、不回退验证best**。已提交且身份匹配的评测复用。

结果分析：

```bash
bash run_main.sh --stage analyze --run "$RUN" --boot 10000 --bootstrap-seed 12345
```

报告按单训/混训分别给每任务四条件表、process相对state/partial/shuffle差值与95%场景区间、种子标准差、等任务权重总平均，以及PNG/PDF图。默认全部条件、种子和评测齐全才分析，不用缺失值冒充0分。

## 8. 安全与复用边界

- 来源、任务注册表、源码、数据文件、设置、任务表、模型和评测都绑定冻结身份。
- 续跑只传运行目录、阶段、GPU等执行参数；再次传N/R/种子/训练步数等prepare参数会拒绝，不静默忽略。
- 训练中断从第0步重新训练，不恢复优化器；已提交最终模型不会重训。
- 已提交评测文件修改或损坏会报错，不自动覆盖；未提交评测可以重跑。
- 同一流水线、训练任务和评测任务有互斥锁；拒绝输出路径符号链接以及与来源/数据重叠。
- 修改源码后旧数据和旧运行会拒绝复用。这是有意的严格复现约束，修改实现应新建数据/运行，或保留冻结源码版本。实际加载模块的路径也受核验，拒绝外部 `FW_PROJECT/FW_GAME_CODE` 覆盖；03审计、04统计及公共模型/规则均在依赖身份中。
- 本次迁移只移动源码和文档，没有移动或改写数据/模型记录。迁移前检查旧目录未发现 `datasets/`、`runs/`，因此没有对旧冻结身份增加例外。若另在其他路径生成过旧版main数据，请保留它们；新入口不会自动改写其哈希或假装身份兼容，需新建数据/运行或单独核对迁移兼容。
- T01生成请求绑定源审计和输入哈希，同一路径换来源不能继续复用；完整导出数据用于训练时不需要重新读取原始03来源。
- 所有任务已经完成、但manifest或整体完成标记提交中断时，可在同一请求下恢复最终提交，不重生成任务；仅残留 `pipeline.lock` 的prepare也可重试。
- T01测试已用于05/06选setting时，只能作为锚点/开发结果，不宣称全新未使用测试。新增任务独立测试不参与设置选择。

## 9. 存储和运行提醒

NPY内存映射避免把全部任务一次搬进GPU，也避免每个batch重复解压全量NPZ，但会增加磁盘占用与随机读取开销。

仅训练数组每任务每集约1.15MiB：N=5120、17任务约98GiB；N=81920约1.53TiB，另加验证、bench、元数据及模型结果。`--dry-run`会打印数组规模估计；请先确认磁盘及I/O能力，不要直接盲目生成全量。

不同模板的真实不同终态数量可能小于N，不能把重复episode称作N个独特场景。大规模生成及GPU吞吐尚需正式运行校准。

## 10. 后续扩展

后续形成比例、数据量、预算对齐、长程推演等统一放 `extensions/` 下独立模块和独立运行，参见 [extensions/README.md](extensions/README.md)。当前不实现这些消融，也不让它们进入主实验任务表。

测试与验证范围见 [tests/README.md](tests/README.md)。所有正式数据生成和GPU训练都由你后续单独启动。
