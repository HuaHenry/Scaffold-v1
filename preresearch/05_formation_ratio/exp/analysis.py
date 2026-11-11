"""Equal-update supervision dose curves with paired-scene uncertainty."""
import argparse
from importlib import import_module

import numpy as np

from .audit import verify_audit
from .make_jobs import jobs
from .shared import (EPISODES, EXP_NAME, MODE, RATIOS, STEPS, budget, common, dose_key, metrics,
                     resolve_file, run_lock, run_path, validate_done, validate_eval,
                     validate_metrics, validate_ratios, write_json)

stats = import_module("preresearch.04_ood_generalization.exp.statistics")
METRICS = ("cascade_f1", "change_f1", "cell_acc_mask", "rank_acc", "any_violation")


def average_ranks(values):
    values = np.asarray(values, float)
    result = np.empty(len(values))
    for v in np.unique(values):
        mask = values == v
        result[mask] = int((values < v).sum()) + (int(mask.sum()) - 1) / 2
    return result


def spearman(x, y):
    x, y = average_ranks(x), average_ranks(y)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def slope_distribution(matrix, x, boot, seed, key):
    matrix = np.asarray(matrix, float)
    matrix = matrix[:, np.isfinite(matrix).all(axis=0)]
    if matrix.shape[1] == 0:
        return {"n_cases": 0, "slope_per_unit_fraction": None, "ci": None, "spearman": None}
    center = x - x.mean()
    per_scene = (center / (center @ center)) @ matrix
    result = stats.estimate(per_scene, boot, seed, key)
    return {"n_cases": result["n_cases"], "slope_per_unit_fraction": result["mean"], "ci": result["ci"],
            "spearman": spearman(x, matrix.mean(axis=1)),
            "note": "descriptive ratio trend; paired scene CI, not across-seed significance"}


def summarize(samples, split_ids, splits, boot=2000, seed=12345, ratios=RATIOS, steps=STEPS):
    ratios = validate_ratios(ratios)
    if set(samples) != set(ratios) or boot <= 0 or seed < 0:
        raise ValueError("all selected ratios, positive bootstrap count and nonnegative seed required")
    split_ids = np.asarray(split_ids)
    if (split_ids.ndim != 1 or tuple(splits) != tuple(common.SPLITS)
            or not np.isin(split_ids, np.arange(len(splits))).all()):
        raise ValueError("invalid benchmark split order or scene IDs")
    for ratio in ratios:
        for metric in METRICS:
            values = np.asarray(samples[ratio][metric])
            if values.shape != split_ids.shape:
                raise ValueError("supervision dose scores must align with benchmark scenes")
            if metric != "rank_acc" and not np.isfinite(values).all():
                raise ValueError("non-ranking scores must be finite for every scene")
    report = {"settings": {"ratios": list(ratios), "episodes": EPISODES, "steps": steps,
                           "epochs": None, "mode": MODE, "training_seeds": [0], "bootstrap": boot,
                           "bootstrap_seed": seed, "seed_std": None, "full_five_dose_design": ratios == RATIOS,
                           "all_doses_retrained": True, "reuse_main_endpoints": False,
                           "uncertainty": "paired scenes, not training seed variability"},
              "budgets": {dose_key(r): budget(r, steps) for r in ratios}, "by_split": {}}
    x = np.array(ratios)
    for i, split in enumerate(splits):
        mask = split_ids == i
        block = {"n_scenes": int(mask.sum()), "metrics": {}}
        for metric in METRICS:
            matrix = np.stack([samples[r][metric][mask] for r in ratios])
            matrix = matrix[:, np.isfinite(matrix).all(axis=0)]
            means = {dose_key(r): stats.estimate(matrix[j], boot, seed, (split, metric, r)) for j, r in enumerate(ratios)}
            contrasts = {}
            for ref in (0.0, 0.5):
                index = ratios.index(ref)
                contrasts[f"vs_{dose_key(ref)}"] = {dose_key(r): stats.estimate(matrix[j] - matrix[index], boot, seed,
                    (split, metric, ref, r)) for j, r in enumerate(ratios) if r != ref}
            scores = np.array([means[dose_key(r)]["mean"] if means[dose_key(r)]["mean"] is not None else np.nan for r in ratios])
            block["metrics"][metric] = {"ratios": means, "contrasts": contrasts,
                "trend": slope_distribution(matrix, x, boot, seed, (split, metric, "slope")),
                "descriptive_best_ratio": float(ratios[int(np.nanargmin(scores) if metric == "any_violation" else np.nanargmax(scores))]) if np.isfinite(scores).any() else None,
                "monotonic_increasing_means": bool(np.all(np.diff(scores) >= 0)) if np.isfinite(scores).all() else None}
        report["by_split"][split] = block
    return report


def load_results(name, audited):
    samples, sources, identities = {}, {}, {}
    bench = common.load_bench(resolve_file(audited["inputs"]["bench"]))
    ids = bench["split_ids"].copy()
    rows = jobs(name, audited)
    for row in rows:
        ratio = row["formation_ratio"]
        path = resolve_file(row["out"])
        done = validate_done(path, row, audited)
        if not (path / "eval_complete.json").exists():
            raise ValueError("dose evaluation is not committed; complete this run's training/evaluation queue")
        validate_eval(path, audited, done)
        with np.load(path / "eval_samples.npz", allow_pickle=False) as z:
            raw = {k: z[k] for k in (*METRICS, "split_ids", "rank_valid", "n_cascade_true", "pred_fg", "pred_bg")}
        if len(ids) != 512 or not np.array_equal(ids, raw["split_ids"]):
            raise ValueError("all supervision doses must use the same 512 scenes in the same order")
        validate_metrics(raw, bench)
        samples[ratio] = {metric: metrics.sample_level(raw, metric) for metric in METRICS}
        sources[dose_key(ratio)] = common.rel(path)
        identities[dose_key(ratio)] = {n: common.sha1_file(path / n) for n in
                                      ("config.json", "done.json", "ckpt.pt", "eval.json", "eval_samples.npz", "eval_complete.json")}
    return samples, ids, sources, identities


def markdown(report):
    ratios = report["settings"]["ratios"]
    design = "默认五档" if report["settings"]["full_five_dose_design"] else "显式子集"
    lines = ["# 实验05：形成监督采样比例", "", f"本次为{design}；形成监督比例：" + "、".join(f"{r:.1%}" for r in ratios) + "。", "",
             "所有档位使用同一份冻结的原始process轨迹和相同的合法相邻帧转移。每个batch分别从形成池、操作池有放回抽样，不压缩建造过程，也不改物理时间。", "",
             "五档全部从头训练，包括0%和50%。实验03旧模型不进入等预算曲线；本次比例不是F帧数，也不是原五轮训练的换算。", "",
             "## 1. 统一预算与实际监督量", "", "| 形成监督比例 | 每batch形成/操作 | 更新数 | 监督样本总数 | 形成样本数 | 操作样本数 |",
             "|---:|---:|---:|---:|---:|---:|"]
    for r in ratios:
        b = report["budgets"][dose_key(r)]
        lines.append(f"| {r:.1%} | {b['formation_per_batch']}/{b['operation_per_batch']} | {b['steps']} | {b['samples']} | {b['formation_samples']} | {b['operation_samples']} |")
    lines += ["", "更新数、batch、学习率进度和验证时点一致。有放回采样不代表完整覆盖全部转移，不能描述为完整epochs。", "", "## 2. 各类测试", ""]
    for metric in METRICS:
        lines += [f"### {metric}", "", "| 测试 | " + " | ".join(f"{r:.1%}" for r in ratios) + " |", "|---|" + "---:|" * len(ratios)]
        for split, result in report["by_split"].items():
            means = result["metrics"][metric]["ratios"]
            lines.append(f"| {split} | " + " | ".join("—" if means[dose_key(r)]["mean"] is None else f"{means[dose_key(r)]['mean']:.3f}" for r in ratios) + " |")
    lines += ["", "## 3. 连锁F1趋势", "", "| 测试 | Spearman | 线性斜率 | 95%场景区间 | 均值单调增加 | 观察到最佳比例 |", "|---|---:|---:|---|---|---:|"]
    for split, block in report["by_split"].items():
        row = block["metrics"]["cascade_f1"]
        trend, ci = row["trend"], row["trend"]["ci"]
        fmt = lambda v: "—" if v is None else f"{v:.3f}"
        interval = "—" if ci is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
        best = row["descriptive_best_ratio"]
        lines.append(f"| {split} | {fmt(trend['spearman'])} | {fmt(trend['slope_per_unit_fraction'])} | {interval} | {row['monotonic_increasing_means']} | " + ("—" if best is None else f"{best:.1%}") + " |")
    lines += ["", "斜率按监督比例0–1计算；Spearman描述均值次序，最佳比例只是描述性结果，不反向用测试集选模。", "",
              "## 4. 相对0%和50%的配对连锁F1差值", "", "| 测试 | 对比 | 差值 | 95%场景区间 |", "|---|---|---:|---|"]
    for split, block in report["by_split"].items():
        for ref in (0.0, 0.5):
            for r in ratios:
                if r == ref:
                    continue
                value = block["metrics"]["cascade_f1"]["contrasts"][f"vs_{dose_key(ref)}"][dose_key(r)]
                ci = value["ci"]
                interval = "—" if ci is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
                lines.append(f"| {split} | {r:.1%} − {ref:.1%} | " + ("—" if value["mean"] is None else f"{value['mean']:+.3f}") + f" | {interval} |")
    lines += ["", "![形成监督比例与连锁F1](figs/dose_response.png)", "", "## 5. 解释边界", "",
              "本次训练种子为0，场景bootstrap区间不能替代跨种子稳定性，不报告种子标准差。`id`仍是陌生结构困难测试。", "",
              "相同监督总量下，增加形成样本必然减少操作样本。该实验衡量训练预算如何分配，不把收益解释为只增加形成、操作训练量不变。", "",
              "原始process中的四个操作片段在所有档位相同；0%不等于03的八片段state条件，50%也不等于03的自然31/59混合比例。", "",
              "不重新分组原动作，沿用现有单格动作编码；原数据已有的同格动作覆盖限制在审计中如实记录。", ""]
    return "\n".join(lines)


def plot(report, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    ratios = report["settings"]["ratios"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, (split, result) in zip(axes.flat, report["by_split"].items()):
        scores = result["metrics"]["cascade_f1"]["ratios"]
        y = [scores[dose_key(r)]["mean"] if scores[dose_key(r)]["mean"] is not None else np.nan for r in ratios]
        low = [scores[dose_key(r)]["ci"][0] if scores[dose_key(r)]["ci"] else np.nan for r in ratios]
        high = [scores[dose_key(r)]["ci"][1] if scores[dose_key(r)]["ci"] else np.nan for r in ratios]
        x = np.array(ratios) * 100
        ax.plot(x, y, "o-", color="#c66d56")
        ax.fill_between(x, low, high, color="#c66d56", alpha=.15)
        ax.set_xticks(x, [f"{r:.1%}" for r in ratios])
        ax.set(title=split, xlabel="Formation supervision fraction", ylabel="Cascade F1", ylim=(0, 1))
        ax.grid(alpha=.2)
    fig.suptitle("Equal updates; original trajectories; paired-scene CI; training seed 0", fontsize=9)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"dose_response.{suffix}", dpi=300)
    plt.close(fig)


def analyze(name, boot=2000, seed=12345):
    audited = verify_audit(name)
    samples, ids, sources, identities = load_results(name, audited)
    splits = common.load_json(resolve_file(audited["inputs"]["bench"]).with_suffix(".json"))["splits"]
    p = audited["protocol"]
    report = summarize(samples, ids, splits, boot, seed, p["ratios"], p["steps"])
    report["sources"] = sources
    report["source_population"] = audited["pools"]
    out = common.new_result_dir(EXP_NAME, "analysis", "supervision_ratio", f"boot{boot}_seed{seed}", run=name)
    common.write_meta(out, "analysis", "supervision_ratio", report["settings"], inputs=audited["source"], models=identities)
    write_json(out / "formation_ratio.json", report)
    (out / "formation_ratio.md").write_text(markdown(report), encoding="utf-8")
    plot(report, out / "figs")
    print(f"[05] supervision analysis: {common.rel(out)}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args(argv)
    if a.boot <= 0 or a.seed < 0:
        ap.error("bootstrap count positive, seed nonnegative")
    with run_lock(run_path(a.run)):
        analyze(a.run, a.boot, a.seed)


if __name__ == "__main__":
    main()
