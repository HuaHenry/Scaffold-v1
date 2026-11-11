"""Summarize scale sweeps and matched single-component comparisons."""
import numpy as np

from .evaluate import evaluation_identity, load_cached
from .shared import CONDITIONS, CONTRASTS, METRICS, common, metrics, write_json
from .statistics import difference_of_differences, estimate, paired


def read_probe(run_dir, name, batch):
    data = run_dir / "data" / name
    done = common.load_json(data / "complete.json")
    if done["status"] != "ready":
        return None
    bench = common.load_bench(data / "bench.npz")
    samples = {}
    for c in (*CONDITIONS, "oracle_reseed"):
        expected = evaluation_identity(run_dir, name, c, batch)[0]
        out = run_dir / "evaluate" / name / c
        load_cached(out, expected)
        with np.load(out / "eval_samples.npz") as z:
            samples[c] = {key: z[key].copy() for key in (*METRICS, "exact")}
    return bench, samples, common.load_json(data / "coverage.json")


def summarize_scale(bench, samples, coverage, boot, seed):
    groups = bench["meta"]["groups"]
    report = {"coverage": coverage, "comparability": "same generator and removal-pair selection across sizes; outcomes conditioned on stable unique scenes",
              "scale_fields": {"tower": "blueprint height h", "bridge": "total support-to-support span; inner gap = span - 2"},
              "material_limit": "bridge span > 10 forces brick beam; this is not a pure material-controlled scale experiment",
              "by_kind": {}}
    for kind in ("tower", "bridge"):
        ref = np.array([r["kind"] == kind and r["within_training_parameter_support"] for r in groups])
        rows = []
        for value in sorted({r["value"] for r in groups if r["kind"] == kind}):
            mask = np.array([r["kind"] == kind and r["value"] == value for r in groups])
            row = {"value": value, "n_cases": int(mask.sum()), "within_training_parameter_support": value <= (7 if kind == "tower" else 8), "metrics": {}}
            for metric in METRICS:
                values = {c: metrics.sample_level(samples[c], metric) for c in CONDITIONS}
                scores = {c: estimate(values[c][mask], boot, seed, ("scale", kind, value, metric, c)) for c in CONDITIONS}
                contrasts = {f"{a}-{b}": paired(values[a], values[b], mask, boot, seed, ("scale", kind, value, metric, a, b)) for a, b in CONTRASTS}
                relative = {}
                if not row["within_training_parameter_support"]:
                    for a, b in CONTRASTS:
                        relative[f"{a}-{b}"] = difference_of_differences(values[a], values[b], ref, mask, boot, seed, ("scale_relative", kind, value, metric, a, b))
                row["metrics"][metric] = {"conditions": scores, "contrasts": contrasts,
                                          "relative_to_within_parameter_support": relative}
            oracle = metrics.sample_level(samples["oracle_reseed"], "exact")
            row["oracle_exact"] = float(oracle[mask].mean())
            rows.append(row)
        report["by_kind"][kind] = rows
    return report


def component_family_scores(samples, families, metric):
    combined, single = [], []
    for family in families:
        if metric == "rank_acc":
            raise ValueError("full-scene ranking is not comparable to isolated components with noop substitutions")
        full = samples[metric][family["combined"]]
        alone = np.array([samples[metric][family["singles"][owner], k] for k, owner in enumerate(family["owners"])])
        combined.append(float(full.mean()))
        single.append(float(alone.mean()))
    return np.asarray(combined), np.asarray(single)


def summarize_components(bench, samples, coverage, boot, seed):
    families = coverage["families"]
    positive = []
    for family in families:
        full = bench["cascade"][family["combined"]]
        alone = np.array([bench["cascade"][family["singles"][owner], k] for k, owner in enumerate(family["owners"])])
        positive.append(bool((full > 0).all() and (alone > 0).all()))
    report = {"coverage": coverage, "unit": "matched composition family; average both interventions on their targeted singleton",
              "ranking": "not compared because non-target components use noop",
              "limit": "isolated at the original coordinates with unchanged materials and geometry; imported composite rows may be disjoint side-by-side rather than mechanically interacting",
              "metrics": {}}
    for metric in ("cascade_f1", "cell_acc_mask"):
        values = {c: component_family_scores(samples[c], families, metric) for c in CONDITIONS}
        rows = {}
        for subset, mask in (("all_matched_families", np.ones(len(families), bool)),
                             ("both_sides_nonempty_cascade", np.asarray(positive, bool))):
            scores, contrasts = {}, {}
            for c, (full, alone) in values.items():
                scores[c] = {"single": estimate(alone[mask], boot, seed, ("component", metric, subset, c, "single")),
                             "combined": estimate(full[mask], boot, seed, ("component", metric, subset, c, "combined")),
                             "single_minus_combined": paired(alone, full, mask, boot, seed, ("component", metric, subset, c, "gap"))}
            for a, b in CONTRASTS:
                full_a, alone_a = values[a]
                full_b, alone_b = values[b]
                changes = (full_a - full_b) - (alone_a - alone_b)
                contrasts[f"{a}-{b}"] = estimate(changes[mask], boot, seed, ("component_gain", metric, subset, a, b))
            rows[subset] = {"n_families": int(mask.sum()), "conditions": scores, "relative_gains_combined_minus_single": contrasts}
        report["metrics"][metric] = rows
    return report


def plot_scale(report, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, (kind, rows) in zip(axes, report["by_kind"].items()):
        for c in CONDITIONS:
            scores = [r["metrics"]["cascade_f1"]["conditions"][c] for r in rows]
            x = [r["value"] for r in rows]
            y = [r["mean"] for r in scores]
            ax.plot(x, y, "o-", label=c)
            low = [r["ci"][0] if r["ci"] else np.nan for r in scores]
            high = [r["ci"][1] if r["ci"] else np.nan for r in scores]
            ax.fill_between(x, low, high, alpha=.1)
        ax.axvspan(4, 7 if kind == "tower" else 8, color="gray", alpha=.1)
        ax.set(title=f"{kind}: unique stable scenes", xlabel="height h" if kind == "tower" else "total span", ylabel="Cascade F1", ylim=(0, 1))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Shading: pointwise scene CI; gray: training parameter support", fontsize=9)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"scale_sweep.{suffix}", dpi=300)
    plt.close(fig)


def summarize_probes(run_dir, result_dir, names, batch, boot, seed):
    reports = {}
    lines = ["# 实验04：额外评测探针", "", "只使用现有检查点，不训练模型。新探针数据与原训练、验证和测试集分开保存。", ""]
    for name in names:
        loaded = read_probe(run_dir, name, batch)
        if loaded is None:
            reports[name] = {"status": "no_unique_valid_cases", "coverage": common.load_json(run_dir / "data" / name / "coverage.json")}
            lines += [f"## {name}", "", "没有足够的唯一、稳定、无重叠场景，已记录失败原因；不以零分代替缺失结果。", ""]
            continue
        bench, samples, coverage = loaded
        if name == "scale":
            report = summarize_scale(bench, samples, coverage, boot, seed)
            plot_scale(report, result_dir / "figs")
            lines += ["## 单结构尺度扫描", "", "塔高4–16、桥总跨度4–14，每设置尝试生成指定数量的独立、唯一、稳定场景。灰色区间是训练参数范围，不代表完整结构分布内。桥跨度大于10时强制砖梁，材料变化会影响尺度趋势。", "",
                      "| 类别 | 尺度 | 场景 | Oracle整图准确率 | state F1 | process F1 | partial F1 | shuffle F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for kind, rows in report["by_kind"].items():
                for row in rows:
                    scores = row["metrics"]["cascade_f1"]["conditions"]
                    vals = ["—" if scores[c]["mean"] is None else f"{scores[c]['mean']:.3f}" for c in CONDITIONS]
                    lines.append(f"| {kind} | {row['value']} | {row['n_cases']} | {row['oracle_exact']:.3f} | " + " | ".join(vals) + " |")
            lines += ["", "![单结构尺度扫描](figs/scale_sweep.png)", ""]
        else:
            report = summarize_components(bench, samples, coverage, boot, seed)
            lines += ["## 相同组件：单独出现与组合出现", "", "从原组合考题提取相同位置、材料和几何的组件，保留其受到的干预。作用于其他组件的干预替换为noop。比较时只取干预目标组件，不将其他组件的noop成绩作为单结构掌握的证据。", "",
                      "同时报告两侧都有连锁变化的子集，避免无变化样本的F1=1掩盖困难；组合场景可能只是并排摆放，不能自动称为机械交互。", "",
                      "| 子集 | 家族数 | 条件 | 单组件F1 | 组合F1 | 单组件−组合 |", "|---|---:|---|---:|---:|---:|"]
            for subset, row in report["metrics"]["cascade_f1"].items():
                for c, scores in row["conditions"].items():
                    vals = ["—" if scores[k]["mean"] is None else f"{scores[k]['mean']:.3f}" for k in ("single", "combined", "single_minus_combined")]
                    lines.append(f"| {subset} | {row['n_families']} | {c} | " + " | ".join(vals) + " |")
            lines += ["", "不比较单组件替换noop后的排序准确率。单组件若本身预测不佳，不应把组合错误全部归因于组合泛化。", ""]
        reports[name] = report
    lines += ["## 统计与覆盖限制", "", "区间为场景／组件家族配对区间，不是训练种子方差。有效样本不足8的分组只作探索；尺寸、材料、终态唯一性或无法稳定可能导致某些尺度没有样本。详细接受率和排除原因见 `data/*/coverage.json`。结果不合并进旧测试分数，不根据探针挑选检查点。", ""]
    write_json(result_dir / "probe_analysis.json", reports)
    (result_dir / "probe_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    return reports
