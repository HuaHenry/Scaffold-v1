"""Existing-result analysis: split gaps, relative gains, difficulty and structural coverage."""
import argparse

import numpy as np

from .load import load_source
from .shared import (CONDITIONS, CONTRASTS, EXP_NAME, METRICS, bind_source, common, forbid_source_output,
                     output_path, write_json)
from .statistics import cluster_values, difference_of_differences, estimate, paired

BINS = (("0–1", 0, 2), ("2–4", 2, 5), ("5–9", 5, 10), ("≥10", 10, np.inf))
DEPTH_BINS = (("0–1", 0, 2), ("2–3", 2, 4), ("4–7", 4, 8), ("≥8", 8, np.inf))


def grouped_scores(values, mask, boot, seed, key):
    mask = np.asarray(mask, bool).copy()
    for c in CONDITIONS:
        mask &= np.isfinite(values[c])
    means = {c: estimate(values[c][mask], boot, seed, (key, c)) for c in CONDITIONS}
    contrasts = {f"{a}-{b}": paired(values[a], values[b], mask, boot, seed, (key, a, b)) for a, b in CONTRASTS}
    return {"n_cases": int(mask.sum()), "conditions": means, "contrasts": contrasts}


def existing_report(bundle, boot=2000, seed=12345):
    if boot <= 0 or seed < 0:
        raise ValueError("bootstrap count must be positive and seed nonnegative")
    bench, group, features = bundle["bench"], bundle["group"], bundle["features"]
    ids = bench["split_ids"]
    names = bench["splits"]
    ref = ids == names.index("id")
    split_scores, gaps, relative = {}, {}, {}
    for metric in METRICS:
        values = {c: group[c][metric] for c in CONDITIONS}
        split_scores[metric], gaps[metric], relative[metric] = {}, {}, {}
        for i, name in enumerate(names):
            mask = ids == i
            split_scores[metric][name] = grouped_scores(values, mask, boot, seed, ("split", metric, name))
            if name == "id":
                continue
            gaps[metric][name], relative[metric][name] = {}, {}
            for c in CONDITIONS:
                gap = difference_of_differences(np.zeros(len(ids)), values[c], ref, mask, boot, seed, ("gap", metric, name, c))
                gaps[metric][name][c] = gap
            for a, b in CONTRASTS:
                relative[metric][name][f"{a}-{b}"] = difference_of_differences(
                    values[a], values[b], ref, mask, boot, seed, ("relative", metric, name, a, b))
    difficulty = {}
    for field, bins in (("n_changed", BINS), ("cascade", BINS), ("depth", DEPTH_BINS)):
        difficulty[field] = {}
        for i, name in enumerate(names):
            rows = []
            for label, low, high in bins:
                member = (bench[field] >= low) & (bench[field] < high) & (ids[:, None] == i)
                values = {c: cluster_values(bundle["raw"][c]["cascade_f1"], member) for c in CONDITIONS}
                rows.append({"bin": label, "n_interventions": int(member.sum()),
                             **grouped_scores(values, member.any(axis=1), boot, seed, ("difficulty", field, name, label))})
            difficulty[field][name] = rows
    structural = {}
    labels = {"contains_unseen_kind": (False, True), "composition_seen_in_training": (False, True),
              "held_pair_present": (False, True), "n_struct": tuple(sorted({r["n_struct"] for r in features}))}
    values = {c: group[c]["cascade_f1"] for c in CONDITIONS}
    for field, choices in labels.items():
        structural[field] = {}
        for i, name in enumerate(names):
            structural[field][name] = [{"value": choice,
                **grouped_scores(values, (ids == i) & np.array([r[field] == choice for r in features]),
                                 boot, seed, ("structure", field, name, choice))} for choice in choices]
    scales = {}
    for kind in ("tower", "bridge"):
        choices = sorted({r["max_parameter"][kind] for r in features if kind in r["max_parameter"]})
        scales[kind] = {"training_range": bundle["ranges"][kind], "status": "observational_not_controlled_sweep", "levels": []}
        for value in choices:
            mask = np.array([r["max_parameter"].get(kind) == value for r in features])
            scales[kind]["levels"].append({"value": value, "split_counts": {s: int((mask & (ids == i)).sum()) for i, s in enumerate(names)},
                **grouped_scores(values, mask, boot, seed, ("observed_scale", kind, value))})
    return {"settings": {"bootstrap": boot, "bootstrap_seed": seed, "training_seeds": [0],
                         "reference": "id", "reference_is_training_id": False,
                         "interval_scope": "paired scenes only; not training seed variance",
                         "difficulty_unit": "average selected interventions within each scene, then equally weighted scenes"},
            "h3_status": "not_tested_against_training_id", "split_scores": split_scores,
            "gaps_reference_minus_other": gaps, "relative_gains_other_minus_reference": relative,
            "difficulty": difficulty, "structural_groups": structural, "observed_scales": scales,
            "strict_component_control": "requires separately evaluated matched singleton scenes"}


def fmt(value):
    return "—" if value is None else f"{value:.3f}"


def ci_text(row):
    ci = row.get("ci")
    return "—" if ci is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def markdown(report, source):
    lines = ["# 实验04：结构泛化分析", "", f"来源：`{source}`。无需重新训练。", "",
             "## 1. 怎样解读", "",
             "参考划分 `id` 也是陌生结构困难测试，不是训练分布内测试。因此下面的差距与差的差描述不同迁移类型之间的差异，**不宣布真正 ID→OOD 的 H3 已被验证**。",
             "所有置信区间按相同场景配对；不同划分分别重采样。这里只有训练种子0，区间不包含重复训练波动。分组不足8场景仅作探索，空组不补零。", "",
             "## 2. 四划分成绩", ""]
    for metric, splits in report["split_scores"].items():
        lines += [f"### {metric}", "", "| 划分 | 有效场景 | state | process | partial | shuffle |", "|---|---:|---:|---:|---:|---:|"]
        for split, row in splits.items():
            lines.append(f"| {split} | {row['n_cases']} | " + " | ".join(fmt(row["conditions"][c]["mean"]) for c in CONDITIONS) + " |")
    lines += ["", "## 3. 参考划分减去其他划分：正值表示成绩下降", "", "| 指标 | 划分 | 条件 | 差距 | 95% CI |", "|---|---|---|---:|---|"]
    for metric, splits in report["gaps_reference_minus_other"].items():
        for split, rows in splits.items():
            for c, row in rows.items():
                lines.append(f"| {metric} | {split} | {c} | {fmt(row['difference'])} | {ci_text(row)} |")
    lines += ["", "## 4. 条件收益的差的差", "", "定义：`(process−对照)(其他划分) − (process−对照)(参考划分)`。正值表示相对收益更大；这不是物理机制的因果证明。", "",
              "| 指标 | 划分 | 比较 | 参考收益 | 其他收益 | 差的差 | 95% CI |", "|---|---|---|---:|---:|---:|---|"]
    for metric, splits in report["relative_gains_other_minus_reference"].items():
        for split, rows in splits.items():
            for pair, row in rows.items():
                lines.append(f"| {metric} | {split} | {pair} | {fmt(row['delta_reference'])} | {fmt(row['delta_other'])} | {fmt(row['difference'])} | {ci_text(row)} |")
    lines += ["", "## 5. 难度分层", "", "分别按真值全部变化格、排除动作直接覆盖格的连锁变化、物理稳定时间深度分组。深度不是结构高度。一个场景的两种干预仍归为一个抽样单位。", "",
              "| 难度 | 划分 | 分箱 | 场景 | 干预 | state | process | partial | shuffle |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for field, splits in report["difficulty"].items():
        for split, rows in splits.items():
            for row in rows:
                lines.append(f"| {field} | {split} | {row['bin']} | {row['n_cases']} | {row['n_interventions']} | " + " | ".join(fmt(row["conditions"][c]["mean"]) for c in CONDITIONS) + " |")
    lines += ["", "## 6. 结构覆盖分组", "", "`contains_unseen_kind`：包含训练中没出现的类别；`composition_seen_in_training`：完整类别组合是否出现过；`held_pair_present`：包含预留类别对；`n_struct`：实际组件个数。各组还混合其他难度因素，不能当成单因素因果效应。", "",
              "| 分组 | 划分 | 值 | 场景 | state | process | partial | shuffle |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for field, splits in report["structural_groups"].items():
        for split, rows in splits.items():
            for row in rows:
                lines.append(f"| {field} | {split} | {row['value']} | {row['n_cases']} | " + " | ".join(fmt(row["conditions"][c]["mean"]) for c in CONDITIONS) + " |")
    lines += ["", "## 7. 已有样本的尺度趋势", "", "塔按蓝图高度、桥按包含两端支撑的总跨度；内部空跨为总跨度减2。多组件场景取该类组件最大值，同一场景不重复计数。图包含混合结构，仅是观察性趋势；真正的单结构扫描由 `--with-probes` 完成。", "",
              "详细分数、区间、分组样本数和训练实际参数范围见 `ood_analysis.json`。", "",
              "![四条件迁移成绩](figs/split_scores.png)", "", "![相对参考划分的收益变化](figs/relative_gains.png)", "",
              "![按真值连锁大小分层](figs/difficulty_cascade.png)", "", "![已有场景的尺度](figs/observed_scales.png)", ""]
    return "\n".join(lines)


def plot(report, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    colors = dict(zip(CONDITIONS, ("#3d639e", "#c66d56", "#48876f", "#9278ab")))

    def save(fig, name):
        for suffix in ("png", "pdf"):
            fig.savefig(out / f"{name}.{suffix}", dpi=300)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for c in CONDITIONS:
        rows = list(report["split_scores"]["cascade_f1"].values())
        x = np.arange(len(rows))
        y = [r["conditions"][c]["mean"] for r in rows]
        ax.plot(x, y, "o-", label=c, color=colors[c])
    ax.set_xticks(np.arange(4), list(report["split_scores"]["cascade_f1"]))
    ax.set(ylabel="Cascade F1", ylim=(0, .5), title="Four transfer splits (id is not training ID)")
    ax.legend(frameon=False, ncol=4)
    ax.grid(alpha=.2)
    save(fig, "split_scores")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, (metric, splits) in zip(axes, report["relative_gains_other_minus_reference"].items()):
        for j, pair in enumerate(f"{a}-{b}" for a, b in CONTRASTS):
            rows = [r[pair] for r in splits.values()]
            x = np.arange(len(rows)) + (j - 1) * .12
            y = np.array([r["difference"] if r["difference"] is not None else np.nan for r in rows])
            ax.plot(x, y, "o", label=pair)
            for xx, row in zip(x, rows):
                if row["ci"] is not None:
                    ax.vlines(xx, *row["ci"], linewidth=1)
        ax.axhline(0, color="gray", linewidth=1)
        ax.set_xticks(np.arange(3), list(splits), rotation=25)
        ax.set(title=metric, ylabel="Gain minus hard-reference gain")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    save(fig, "relative_gains")
    for field, splits in report["difficulty"].items():
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        for ax, (split, rows) in zip(axes.flat, splits.items()):
            for c in CONDITIONS:
                y = [r["conditions"][c]["mean"] if r["conditions"][c]["mean"] is not None else np.nan for r in rows]
                ax.plot(np.arange(len(rows)), y, "o-", color=colors[c], label=c)
            ax.set_xticks(np.arange(len(rows)), [f"{r['bin']}\nn={r['n_cases']}" for r in rows])
            ax.set(title=split, xlabel=f"True {field}", ylabel="Cascade F1", ylim=(0, .5))
            ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=8)
        save(fig, f"difficulty_{field}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, (kind, result) in zip(axes, report["observed_scales"].items()):
        for c in CONDITIONS:
            ax.plot([r["value"] for r in result["levels"]], [r["conditions"][c]["mean"] for r in result["levels"]], "o-", color=colors[c], label=c)
        low, high = result["training_range"]["generator_base_range"]
        ax.axvspan(low, high, color="gray", alpha=.1, label="Training parameter support")
        ax.set(title=f"{kind}: mixed-scene observations", xlabel=result["training_range"]["parameter"], ylabel="Cascade F1", ylim=(0, .5))
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    save(fig, "observed_scales")


def run_analysis(bundle, run, boot, seed):
    out = output_path(run)
    forbid_source_output(out, bundle["source"], bundle["audited"])
    bind_source(out, bundle["manifest"])
    report = existing_report(bundle, boot, seed)
    result = common.new_result_dir(EXP_NAME, "analysis", "ood", f"boot{boot}_seed{seed}", run=run)
    common.write_meta(result, "analysis", "ood", report["settings"], inputs=bundle["manifest"])
    write_json(result / "ood_analysis.json", report)
    write_json(result / "case_features.json", bundle["features"])
    (result / "ood_analysis.md").write_text(markdown(report, common.rel(bundle["source"])), encoding="utf-8")
    plot(report, result / "figs")
    print(f"[04] existing-result analysis: {common.rel(result)}", flush=True)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-run", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345, help="bootstrap seed, not training seed")
    args = ap.parse_args(argv)
    run_analysis(load_source(args.source_run), args.run, args.boot, args.seed)


if __name__ == "__main__":
    main()
