"""Per-task four-condition tables and equal-task summaries, no cross-task transfer claims."""
from importlib import import_module
from pathlib import Path

import numpy as np

from ..shared import CONDITIONS, METRICS, file_hash, load_json, write_json
from ..training.protocol import verify_run
from ..training.train import validate_done
from .evaluate import validate_eval

stats = import_module("preresearch.04_ood_generalization.exp.statistics")


def summarize(samples, p, boot=10000, seed=12345):
    if boot <= 0 or seed < 0:
        raise ValueError("positive bootstrap count and nonnegative bootstrap seed required")
    report = {"protocol": p, "bootstrap": boot, "bootstrap_seed": seed, "by_mode": {}}
    seeds = p["training_seeds"]
    for mode in p["modes"]:
        task_report = {}
        for task in p["tasks"]:
            block = {}
            for metric in METRICS:
                raw = {c: np.stack([samples[(mode, task, c, s)][metric].mean(axis=1) for s in seeds]) for c in CONDITIONS}
                if len({v.shape for v in raw.values()}) != 1 or not all(np.isfinite(v).all() for v in raw.values()):
                    raise ValueError("task/seed scores must share finite scene pairing")
                values = {c: v.mean(axis=0) for c, v in raw.items()}
                means = {}
                for c in CONDITIONS:
                    m = stats.estimate(values[c], boot, seed, (mode, task, metric, c))
                    m["seed_std"] = float(raw[c].mean(axis=1).std(ddof=1)) if len(seeds) > 1 else None
                    means[c] = m
                contrasts = {}
                for ref in ("state", "partial", "shuffle"):
                    v = stats.estimate(values["process"] - values[ref], boot, seed, (mode, task, metric, ref))
                    v["seed_std"] = float((raw["process"] - raw[ref]).mean(axis=1).std(ddof=1)) if len(seeds) > 1 else None
                    contrasts[f"process_minus_{ref}"] = v
                block[metric] = {"means": means, "contrasts": contrasts}
            task_report[task] = block
        macro = {metric: {c: float(np.mean([task_report[t][metric]["means"][c]["mean"] for t in p["tasks"]]))
                          for c in CONDITIONS} for metric in METRICS}
        report["by_mode"][mode] = {"tasks": task_report, "equal_task_macro": macro}
    return report


def markdown(report):
    p = report["protocol"]
    lines = ["# 多任务主实验", "", "全部任务采用固定32步状态预测；统一change_f1与cell_acc_mask。", "",
             "单任务训练是核心验证，混训为补充；不是训练任务A后迁移到陌生任务B。", "",
             f"形成比例：{p['formation_ratio']}；每任务N={p['episodes_per_task']}；训练种子：{p['training_seeds']}。", "",
             f"设置选择记录：{p['selection_note']}", ""]
    for mode, block in report["by_mode"].items():
        lines += [f"## {mode}", ""]
        for metric in METRICS:
            lines += [f"### {metric}", "", "| 任务 | state | process | partial | shuffle | process−state | 95%场景区间 |",
                      "|---|---:|---:|---:|---:|---:|---|"]
            for task, result in block["tasks"].items():
                value = result[metric]
                delta = value["contrasts"]["process_minus_state"]
                ci = delta["ci"]
                lines.append(f"| {task} | " + " | ".join(f"{value['means'][c]['mean']:.4f}" for c in CONDITIONS)
                             + f" | {delta['mean']:+.4f} | " + (f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "—") + " |")
            lines.append("| 等任务权重平均 | " + " | ".join(f"{block['equal_task_macro'][metric][c]:.4f}" for c in CONDITIONS) + " | — | — |")
    lines += ["", "## 解释限制", "",
              "- 先平均同场景两个干预，再平均训练种子，最后配对场景bootstrap；种子标准差保存在JSON。",
              "- 不同任务分别计算区间；总平均按任务等权，不将不同任务的场景混作同一分布。",
              "- 没有缺失格或损坏格被当作零分。默认全部完成才分析，不按测试成绩选择模型或比例。",
              "- T01沿用当前03原始测试场景；若它们参与05/06设置选择，T01只能作锚点/开发结果，不能称为全新未使用测试。新增任务测试与其训练/验证最终状态隔离。",
              "- 所有任务改用第32步终态作统一评测，历史预实验的first-fixed-point结果不冒充本次成绩。",
              f"- 混训预算模式为{p['mixed_budget']}；只有明确控制训练量才能直接比较单训和混训。",
              "- 17项是17个互斥生成模板，不是17种物理机制；水/沙对应模板平行，不进行跨机制组合。", ""]
    return "\n".join(lines)


def plot(report, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    for mode, block in report["by_mode"].items():
        tasks = list(block["tasks"])
        fig, axes = plt.subplots(2, 1, figsize=(max(10, len(tasks) * .65), 7), constrained_layout=True)
        x = np.arange(len(tasks))
        for ax, metric in zip(axes, METRICS):
            for i, c in enumerate(CONDITIONS):
                y = [block["tasks"][t][metric]["means"][c]["mean"] for t in tasks]
                ax.bar(x + (i - 1.5) * .2, y, width=.2, label=c)
            ax.set_xticks(x, tasks)
            ax.set(ylabel=metric, ylim=(0, 1), title=mode)
            ax.legend(ncol=4)
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{mode}.{ext}", dpi=200)
        plt.close(fig)


def analyze(run, boot=10000, seed=12345):
    run = Path(run)
    p, _ = verify_run(run)
    samples, identities = {}, {}
    for row in load_json(run / "jobs.json"):
        out = run / "train" / row["id"]
        validate_done(out, p, row)
        for task in row["tasks"]:
            values = validate_eval(out, p, row, task)
            samples[(row["mode"], task, row["condition"], row["seed"])] = values
            identities[f"{row['id']}/{task}"] = {"training": file_hash(out / "train_complete.json"),
                                                "evaluation": file_hash(out / "eval" / task / "complete.json")}
    report = summarize(samples, p, boot, seed)
    report["identities"] = identities
    base = run / "analysis" / f"main_boot{boot}_seed{seed}"
    out, suffix = base, 1
    while out.exists():
        suffix += 1
        out = base.with_name(f"{base.name}-{suffix}")
    out.mkdir(parents=True)
    write_json(out / "results.json", report)
    (out / "results.md").write_text(markdown(report), encoding="utf-8")
    plot(report, out / "figs")
    print(f"analysis: {out}")
    return out
