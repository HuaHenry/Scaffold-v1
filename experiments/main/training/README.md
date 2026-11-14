# 单训、混训与四条件协议

- `main.py`：prepare/train/evaluate/analyze独立阶段。
- `protocol.py`：冻结N、R、任务、种子、预算、源码和数据身份，生成任务表。
- `sampler.py`：任务均衡累计配额，任务内部形成/操作累计配额，阶段内有放回抽样。
- `train.py`：公共GridFormer和加权CE，保存最后原模型及EMA。
- `queue.py`：独立GPU任务并行、日志进度、失败标记、纯评测全量预检。

从 `experiments/main` 执行 `bash run_main.sh --help`；数据生成必须已经单独完成。prepare必填 `--data --run --episodes --formation-ratio --selection-note`。其余默认seeds0,1,2、steps94400、modes both、mixed-budget per_task。

`process/shuffle`按相同R抽样；state/partial仅操作。形成转移1～31；混合操作转移33～39、41～47、49～55、57～63，保留shuffle乱序转移。R需严格在0～1之间。

## 任务与预算

单训：每任务四条件×种子。混训：全部任务四条件×种子，任务按样本位置轮转，整个训练期每任务样本数相差最多1。每任务的形成数等于累计 `floor(该任务样本数 × R)`，不采用固定错误整数比例。

默认per_task令混训steps＝单训steps×任务数，保持每任务抽样量。total令混训总steps与一个单训模型一致，不保持每任务暴露。两种预算都在四条件内部一致，报告必须说明预算；混训不是跨任务迁移验证。

## 结果结构

```text
<run>/
  protocol.json
  run.json
  jobs.json
  pipeline.lock
  train/<single或mixed>_<任务或all>_<条件>_seed<种子>/
    config.json
    train.log
    train_log.jsonl
    ckpt.pt
    done.json
    train_complete.json
    eval.log
    eval/<任务>/...
    failed.json              # 失败时存在，成功后移除
  analysis/...
```

最终完成标记绑定配置、done和checkpoint哈希；校验最后步数、总抽样数、每任务形成/操作配额、模型cfg、raw/EMA严格加载和有限权重。验证只监控，不选best。已完成单作业入口也先核验训练/验证/bench输入哈希，不因复用模型而跳过数据检查；prepare仅残留pipeline锁文件时允许重试。

续跑不接受prepare参数。中断训练按相同配置从step0重训，不恢复优化器；已有最终模型复用。纯评测必须全部模型预检成功后才启动任何子进程。常规运行使用外层入口以持有pipeline锁，`train.py`是内部单任务入口。

所有任务数组通过CPU memory map按batch读取，不把全量混训数据放GPU；真实GPU吞吐需要实测，磁盘随机读取可能成为瓶颈。
