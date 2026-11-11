"""Five-dose paired-scene analysis; no seed standard deviation from one training seed."""
import argparse
from importlib import import_module

import numpy as np

from .audit import verify_audit
from .make_jobs import jobs
from .shared import (EPISODES, EXP_NAME, LEVELS, budget, common, metrics, resolve_file,
                     run_lock, run_path, validate_done, validate_eval, validate_levels, validate_metrics, write_json)

stats = import_module("preresearch.04_ood_generalization.exp.statistics")
METRICS = ("cascade_f1", "change_f1", "cell_acc_mask", "rank_acc", "any_violation")


def average_ranks(values):
    values = np.asarray(values, float)
    result = np.empty(len(values))
    for v in np.unique(values):
        mask = values == v
        below = int((values < v).sum())
        result[mask] = below + (int(mask.sum()) - 1) / 2
    return result


def spearman(x, y):
    x, y = average_ranks(x), average_ranks(y)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def slope_distribution(matrix, x, boot, seed, key):
    matrix = np.asarray(matrix, float)
    common_cases = np.isfinite(matrix).all(axis=0)
    matrix = matrix[:, common_cases]
    if matrix.shape[1] == 0:
        return {"n_cases": 0, "slope_per_unit_fraction": None, "ci": None, "spearman": None}
    center = x - x.mean()
    coefficients = center / (center @ center)
    per_scene = coefficients @ matrix
    result = stats.estimate(per_scene, boot, seed, key)
    means = matrix.mean(axis=1)
    return {"n_cases": result["n_cases"], "slope_per_unit_fraction": result["mean"],
            "ci": result["ci"], "spearman": spearman(x, means),
            "note": "linear summary of selected doses, not proof of monotonicity; scene CI only"}


def summarize(samples, split_ids, splits, boot=2000, seed=12345, levels=LEVELS):
    levels = validate_levels(levels)
    if set(samples) != set(levels) or boot <= 0 or seed < 0:
        raise ValueError("all selected doses, a positive bootstrap count and nonnegative seed are required")
    split_ids = np.asarray(split_ids)
    if split_ids.ndim != 1 or tuple(splits) != tuple(common.SPLITS) or not np.isin(split_ids, np.arange(len(splits))).all():
        raise ValueError("invalid benchmark split order or scene IDs")
    if any(np.asarray(samples[F][metric]).shape != split_ids.shape for F in levels for metric in METRICS):
        raise ValueError("dose metrics must align with scene IDs")
    report = {"settings": {"levels": list(levels), "episodes": EPISODES, "epochs": 5,
                           "formation_mode": "compress", "training_seeds": [0], "bootstrap": boot,
                           "bootstrap_seed": seed, "seed_std": None, "full_five_dose_design": levels == LEVELS,
                           "uncertainty": "paired scenes, not training seed variability"},
              "budgets": {str(F): budget(F) for F in levels}, "by_split": {}}
    x = np.asarray(levels, float) / 64
    for i, split in enumerate(splits):
        selected = split_ids == i
        block = {"n_scenes": int(selected.sum()), "metrics": {}}
        for metric in METRICS:
            matrix = np.stack([samples[F][metric][selected] for F in levels])
            finite = np.isfinite(matrix).all(axis=0)
            matrix = matrix[:, finite]
            means = {str(F): stats.estimate(matrix[j], boot, seed, (split, metric, F)) for j, F in enumerate(levels)}
            contrasts = {}
            for ref in (0, 32):
                index = levels.index(ref)
                contrasts[f"vs_f{ref}"] = {str(F): stats.estimate(matrix[j] - matrix[index], boot, seed, (split, metric, ref, F))
                                            for j, F in enumerate(levels) if F != ref}
            scores = np.array([means[str(F)]["mean"] if means[str(F)]["mean"] is not None else np.nan for F in levels])
            block["metrics"][metric] = {"levels": means, "contrasts": contrasts,
                "trend": slope_distribution(matrix, x, boot, seed, (split, metric, "slope")),
                "descriptive_best_F": int(levels[int(np.nanargmin(scores) if metric == "any_violation" else np.nanargmax(scores))]) if np.isfinite(scores).any() else None,
                "monotonic_increasing_means": bool(np.all(np.diff(scores) >= 0)) if np.isfinite(scores).all() else None}
        report["by_split"][split] = block
    return report


def load_results(name, audited):
    samples, sources, identities = {}, {}, {}
    bench = common.load_bench(resolve_file(audited["inputs"]["bench"]))
    ids = bench["split_ids"].copy()
    rows = {row["F"]: row for row in jobs(name, audited)}
    for F in validate_levels(audited["protocol"]["levels"]):
        if F in (0, 32):
            path = resolve_file(audited["endpoints"][str(F)])
        else:
            path = common.resolve_out(rows[F]["out"])
            done = validate_done(path, rows[F], audited)
            if not (path / "eval_complete.json").exists():
                raise ValueError("dose evaluation is not committed; resume the training/evaluation queue")
            validate_eval(path, audited, done)
        with np.load(path / "eval_samples.npz", allow_pickle=False) as z:
            raw = {k: z[k] for k in (*METRICS, "split_ids", "rank_valid", "n_cascade_true", "pred_fg", "pred_bg")}
        if not np.array_equal(ids, raw["split_ids"]) or len(ids) != 512:
            raise ValueError("dose evaluations must use the same 512 scenes in the same order")
        validate_metrics(raw, bench)
        samples[F] = {metric: metrics.sample_level(raw, metric) for metric in METRICS}
        if any(values.shape != (512,) for values in samples[F].values()):
            raise ValueError("dose metrics have incorrect scene counts")
        sources[str(F)] = common.rel(path)
        identities[str(F)] = {n: common.sha1_file(path / n) for n in ("done.json", "ckpt.pt", "eval.json", "eval_samples.npz")}
    return samples, ids, sources, identities


def markdown(report):
    levels = tuple(report["settings"]["levels"])
    design = "完整五档" if report["settings"]["full_five_dose_design"] else "显式缩减设计，不是完整五档实验"
    lines = ["# 实验05：建造过程占比", "", f"本次：{design}；F={','.join(map(str, levels))}。", "",
             "同一套81,920个成品和操作事件，固定每条64帧、完整5轮；F为建造帧数。F=0/32复用已完成主实验，其他选定档位独立训练。", "",
             "建造越多，成品操作越少；形成时间也被重新切分。因此这不是只改变某一份训练数据权重的纯剂量实验，不能将时间压缩影响等同于形成信息价值。", "",
             "## 1. 各档预算", "", "| F | 建造帧比例 | 操作片段 | 有效转移/集 | 更新数 | 形成转移比例 |", "|---:|---:|---:|---:|---:|---:|"]
    for F in levels:
        b = budget(F)
        lines.append(f"| {F} | {b['formation_frames_fraction']:.1%} | {b['operation_clips']} | {b['pairs_per_episode']} | {b['steps']} | {b['formation_fraction']:.2%} |")
    lines += ["", "## 2. 各类测试", ""]
    for metric in METRICS:
        lines += [f"### {metric}", "", "| 测试 | " + " | ".join(f"F={F}" for F in levels) + " |", "|---|" + "---:|" * len(levels)]
        for split, result in report["by_split"].items():
            scores = result["metrics"][metric]["levels"]
            lines.append(f"| {split} | " + " | ".join("—" if scores[str(F)]["mean"] is None else f"{scores[str(F)]['mean']:.3f}" for F in levels) + " |")
    lines += ["", "## 3. 趋势与配对差值", "", "| 测试 | Spearman | 线性斜率 | 95%场景区间 | 均值单调增加 | 观察到最高F1的F |", "|---|---:|---:|---|---|---:|"]
    for split, result in report["by_split"].items():
        row = result["metrics"]["cascade_f1"]
        trend = row["trend"]
        fmt = lambda v: "—" if v is None else f"{v:.3f}"
        ci = trend["ci"]
        lines.append(f"| {split} | {fmt(trend['spearman'])} | {fmt(trend['slope_per_unit_fraction'])} | " + ("—" if ci is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]") + f" | {row['monotonic_increasing_means']} | {row['descriptive_best_F']} |")
    lines += ["", "斜率按建造帧比例0–1计算；Spearman描述选定档位均值的顺序关系，不是跨训练种子显著性检验。最高F仅是本次观察，不能用测试结果反向选训练设置。", "",
              "各档相对F=0/F=32的场景配对差值及区间见 `formation_ratio.json`。", "",
              "![建造帧比例与连锁F1](figs/dose_response.png)", "", "## 4. 解释边界", "",
              "本次只有训练种子0，不报告虚构的种子标准差。误差范围是场景采样的不确定性。`id`仍是陌生结构困难测试，不是真正训练分布内。", "",
              "数据审计要求同一完整成品、同一动作顺序，操作事件前缀完全一致；如果压缩后动作失败、成品改变或无法稳定，必须停止而不是过滤场景。", "",
              "先升后降、平坦或下降均是合法结果；不可为了符合先验强行拟合单调曲线。更多形成帧不等于更多操作练习，等帧与等轮次也不是等更新。", ""]
    return "\n".join(lines)


def plot(report, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    levels = report["settings"]["levels"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, (split, result) in zip(axes.flat, report["by_split"].items()):
        scores = result["metrics"]["cascade_f1"]["levels"]
        x = np.asarray(levels) / 64 * 100
        y = [scores[str(F)]["mean"] if scores[str(F)]["mean"] is not None else np.nan for F in levels]
        low = [scores[str(F)]["ci"][0] if scores[str(F)]["ci"] else np.nan for F in levels]
        high = [scores[str(F)]["ci"][1] if scores[str(F)]["ci"] else np.nan for F in levels]
        ax.plot(x, y, "o-", color="#c66d56")
        ax.fill_between(x, low, high, color="#c66d56", alpha=.15)
        ax.set_xticks(x, [f"{v:g}%\nF={F}" for v, F in zip(x, levels)])
        ax.set(title=split, xlabel="Formation frame fraction", ylabel="Cascade F1", ylim=(0, 1))
        ax.grid(alpha=.2)
    fig.suptitle("Paired-scene uncertainty; one training seed; equal frame/epoch, unequal update budget", fontsize=9)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"dose_response.{suffix}", dpi=300)
    plt.close(fig)


def analyze(name, boot=2000, seed=12345):
    audited = verify_audit(name)
    samples, ids, sources, identities = load_results(name, audited)
    bench = common.load_json(resolve_file(audited["inputs"]["bench"]).with_suffix(".json"))
    report = summarize(samples, ids, bench["splits"], boot, seed, audited["protocol"]["levels"])
    report["sources"] = sources
    out = common.new_result_dir(EXP_NAME, "analysis", "formation_ratio", f"boot{boot}_seed{seed}", run=name)
    common.write_meta(out, "analysis", "formation_ratio", report["settings"], inputs=audited["source"], models=identities)
    write_json(out / "formation_ratio.json", report)
    (out / "formation_ratio.md").write_text(markdown(report), encoding="utf-8")
    plot(report, out / "figs")
    print(f"[05] analysis: {common.rel(out)}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args(argv)
    if a.boot <= 0 or a.seed < 0:
        ap.error("bootstrap count must be positive and seed nonnegative")
    with run_lock(run_path(a.run)):
        analyze(a.run, a.boot, a.seed)


if __name__ == "__main__":
    main()
