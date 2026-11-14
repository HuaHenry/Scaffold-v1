"""Log2 data curves: average training seeds per scene before paired bootstrap."""
import argparse
from importlib import import_module

import numpy as np

from .run_queue import checked_jobs
from .shared import (CONDITIONS, EXP_NAME, METRICS, common, metrics, resolve_file,
                     protocol, run_lock, run_path, validate_done, validate_eval, verify_audit, write_json)

stats = import_module("preresearch.04_ood_generalization.exp.statistics")


def seed_std(values):
    values = np.asarray(values, float)
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def summarize(samples, ids, sizes, seeds, steps, boot=10000, seed=12345):
    if boot <= 0 or seed < 0:
        raise ValueError("positive bootstrap count and nonnegative seed required")
    design = protocol(sizes, seeds, steps)
    sizes, seeds = tuple(design["sizes"]), tuple(design["training_seeds"])
    expected = {(n, c, s) for n in sizes for c in CONDITIONS for s in seeds}
    if set(samples) != expected:
        raise ValueError("all selected size/condition/seed cells must be present")
    ids = np.asarray(ids)
    if ids.ndim != 1 or not np.isin(ids, np.arange(len(common.SPLITS))).all():
        raise ValueError("invalid benchmark split IDs")
    for key in expected:
        for metric in METRICS:
            values = np.asarray(samples[key][metric], float)
            if values.shape != ids.shape or np.isinf(values).any():
                raise ValueError("scores must align with benchmark scenes")
            if metric != "rank_acc" and not np.isfinite(values).all():
                raise ValueError("non-ranking scores must be finite")
            finite = values[np.isfinite(values)]
            if np.any((finite < 0) | (finite > 1)):
                raise ValueError("scores outside [0,1]")
    x = np.log2(sizes)
    center = x - x.mean()
    w = center / (center @ center)
    report = {"settings": {"sizes": list(sizes), "training_seeds": list(seeds), "steps": steps,
        "batch": 256, "sampling": "uniform_with_replacement", "epochs": None,
        "bootstrap": boot, "bootstrap_seed": seed, "checkpoint_selection": "last", "weights": "ema",
        "all_cells_retrained": True, "reuse_main_endpoints": False,
        "uncertainty": "paired scene bootstrap after seed averaging; seed std reported separately"}, "by_split": {}}
    for i, split in enumerate(common.SPLITS):
        block = {"n_scenes": int((ids == i).sum()), "metrics": {}}
        for metric in METRICS:
            raw = {c: np.stack([np.stack([samples[(n, c, s)][metric][ids == i] for s in seeds]) for n in sizes])
                   for c in CONDITIONS}
            finite = np.logical_and.reduce([np.isfinite(v).all(axis=(0, 1)) for v in raw.values()])
            raw = {c: v[:, :, finite] for c, v in raw.items()}
            averaged = {c: v.mean(axis=1) for c, v in raw.items()}
            curves, contrasts, slopes = {}, {}, {}
            for c in CONDITIONS:
                curves[c] = {}
                for j, n in enumerate(sizes):
                    value = stats.estimate(averaged[c][j], boot, seed, (split, metric, c, n))
                    value["seed_std"] = seed_std(raw[c][j].mean(axis=1)) if finite.any() else None
                    value["per_seed_means"] = {str(s): float(raw[c][j, k].mean()) for k, s in enumerate(seeds)} if finite.any() else {}
                    curves[c][str(n)] = value
                per_scene_slope = w @ averaged[c]
                value = stats.estimate(per_scene_slope, boot, seed, (split, metric, c, "slope"))
                value["seed_std"] = seed_std(np.einsum("n,nsk->sk", w, raw[c]).mean(axis=1)) if finite.any() else None
                value["intercept"] = float((averaged[c].mean(axis=0) - x.mean() * per_scene_slope).mean()) if finite.any() else None
                slopes[c] = value
            for reference in ("state", "partial", "shuffle"):
                delta = averaged["process"] - averaged[reference]
                raw_delta = raw["process"] - raw[reference]
                per_n = {}
                for j, n in enumerate(sizes):
                    value = stats.estimate(delta[j], boot, seed, (split, metric, "process", reference, n))
                    value["seed_std"] = seed_std(raw_delta[j].mean(axis=1)) if finite.any() else None
                    per_n[str(n)] = value
                slope = stats.estimate(w @ delta, boot, seed, (split, metric, reference, "slope_delta"))
                slope["seed_std"] = seed_std(np.einsum("n,nsk->sk", w, raw_delta).mean(axis=1)) if finite.any() else None
                endpoints = stats.estimate(delta[-1] - delta[0], boot, seed, (split, metric, reference, "endpoint_gain_change"))
                endpoints["seed_std"] = seed_std((raw_delta[-1] - raw_delta[0]).mean(axis=1)) if finite.any() else None
                contrasts[f"process_minus_{reference}"] = {"by_size": per_n, "slope_difference": slope,
                                                           "largest_minus_smallest_gain": endpoints}
            block["metrics"][metric] = {"n_cases": int(finite.sum()), "curves": curves,
                "contrasts": contrasts, "slopes_per_doubling": slopes}
        report["by_split"][split] = block
    return report


def load_results(name, audited, allow_incomplete=False):
    rows = checked_jobs(name, audited)
    missing = [r["id"] for r in rows if not (resolve_file(r["out"]) / "train_complete.json").exists()
               or not (resolve_file(r["out"]) / "eval_complete.json").exists()]
    if missing and not allow_incomplete:
        raise ValueError(f"{len(missing)} cells incomplete; finish queue or explicitly use --allow-incomplete")
    complete_rows, diagnostics = [], {}
    for row in rows:
        out = resolve_file(row["out"])
        if not (out / "train_complete.json").exists():
            if (out / "eval_complete.json").exists():
                raise ValueError("committed evaluation exists without committed training")
            continue
        done = validate_done(out, row, audited)
        diagnostics[row["id"]] = {k: done[k] for k in ("val", "train_raw", "train_ema", "seconds", "samples_seen")}
        if not (out / "eval_complete.json").exists():
            continue
        validate_eval(out, audited, done)
        complete_rows.append(row)
    p = audited["protocol"]
    usable = [s for s in p["training_seeds"] if
              all(r["id"] not in missing for r in rows if r["seed"] == s)]
    if not usable:
        raise ValueError("no training seed has a complete curve across every condition and size")
    ignored = [r["id"] for r in complete_rows if r["seed"] not in usable]
    bench = common.load_bench(resolve_file(audited["inputs"]["bench"]))
    ids = bench["split_ids"]
    samples, identities = {}, {}
    for row in complete_rows:
        if row["seed"] not in usable:
            continue
        out = resolve_file(row["out"])
        with np.load(out / "eval_samples.npz", allow_pickle=False) as z:
            if not np.array_equal(z["split_ids"], ids):
                raise ValueError("different benchmark scene order")
            samples[(row["episodes"], row["condition"], row["seed"])] = {
                m: metrics.sample_level({m: z[m]}, m) for m in METRICS}
        identities[row["id"]] = {k: common.sha1_file(out / k) for k in
            ("config.json", "done.json", "ckpt.pt", "train_complete.json", "eval.json", "eval_samples.npz", "eval_complete.json")}
    return samples, ids, usable, missing, ignored, identities, diagnostics


def fmt(value, signed=False):
    if value is None:
        return "—"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def interval(value):
    return "—" if value is None else f"[{value[0]:+.4f}, {value[1]:+.4f}]"


def markdown(report):
    p = report["settings"]
    sizes = p["sizes"]
    lines = ["# 实验06：数据量曲线", "", "复用实验03的冻结数据，同一前N集用于四个条件；不生成新数据，不复用旧模型。", "",
        f"每格{p['steps']:,}次更新，batch=256；评测最终EMA。训练种子：{p['training_seeds']}。", "",
        f"完整计划：{report['planned_training_seeds']}；缺失格数：{len(report['missing_cells'])}；因种子曲线不完整而未纳入的已完成格数：{len(report['ignored_completed_cells'])}。", "",
        "## 数据规模", "", "| N | 不同成品结构数 | 重复episode数 |", "|---:|---:|---:|"]
    for n in sizes:
        a = report["prefixes"][str(n)]
        lines.append(f"| {n} | {a['unique_final_structures']} | {a['duplicate_episodes']} |")
    for metric in METRICS:
        lines += ["", f"## {metric}", "", "| 测试 | 条件 | " + " | ".join(str(n) for n in sizes) + " |",
                  "|---|---|" + "---:|" * len(sizes)]
        for split, block in report["by_split"].items():
            for c in CONDITIONS:
                values = block["metrics"][metric]["curves"][c]
                lines.append(f"| {split} | {c} | " + " | ".join(fmt(values[str(n)]["mean"]) for n in sizes) + " |")
    lines += ["", "## 连锁F1：process − state", "", "| 测试 | N | 差值 | 95%场景区间 | 种子标准差 |", "|---|---:|---:|---|---:|"]
    for split, block in report["by_split"].items():
        for n in sizes:
            v = block["metrics"]["cascade_f1"]["contrasts"]["process_minus_state"]["by_size"][str(n)]
            lines.append(f"| {split} | {n} | {fmt(v['mean'], True)} | {interval(v['ci'])} | {fmt(v['seed_std'])} |")
    lines += ["", "## 数据量每翻倍的连锁F1变化", "", "拟合 F1 = a + b·log2(N)。b只是当前预算下的曲线概括，不要求曲线单调。", "",
              "| 测试 | 条件/对比 | b或b差值 | 95%场景区间 | 种子标准差 |", "|---|---|---:|---|---:|"]
    for split, block in report["by_split"].items():
        result = block["metrics"]["cascade_f1"]
        values = {**result["slopes_per_doubling"], "process − state": result["contrasts"]["process_minus_state"]["slope_difference"]}
        for label, v in values.items():
            lines.append(f"| {split} | {label} | {fmt(v['mean'], True)} | {interval(v['ci'])} | {fmt(v['seed_std'])} |")
    lines += ["", "![数据量曲线](figs/data_scale.png)", "", "![process相对state的增益](figs/process_state_gain.png)", "",
              "## 完成状态与解释边界", "",
              "- 区间先对同场景的训练种子求平均，再进行配对场景bootstrap；两个干预不作为独立场景。种子标准差单独报告，只有一个种子时不报告标准差。",
              "- 若允许不完整分析，只纳入在全部N和四个条件上完整的共同种子，避免不同档位使用不同种子。没有完整共同种子时停止分析。",
              "- 缺失或损坏不能当作低分；损坏的已提交结果会报错，不因允许不完整分析而忽略。",
              "- N是episode数，不是不同成品数；前缀嵌套且共用冻结来源，但只评测一个固定前缀顺序。",
              "- 固定更新数而不是固定完整epochs。N增大时每条转移被抽到的平均次数下降；曲线同时受数据规模和固定训练预算影响。",
              "- state/partial每集56条合法video转移，process/shuffle每集59条；保留shuffle的乱序形成转移，不改为causal筛选。",
              "- 验证分数只用于监控，不选最佳模型。训练/验证准确率记录在JSON中，但训练包含的转移类型不同，差值不能单独证明过拟合。",
              "- 当前id也是陌生困难结构测试。增益缩小或持续只能支持当前范围内的观察，不能直接证明形成过程是否提供额外信息。",
              "", "缺失格：" + ("、".join(report["missing_cells"]) or "无"), "",
              "未纳入的已完成格：" + ("、".join(report["ignored_completed_cells"]) or "无"), ""]
    return "\n".join(lines)


def plot(report, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    x = np.array(report["settings"]["sizes"])
    for kind in ("data_scale", "process_state_gain"):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        for ax, (split, block) in zip(axes.flat, report["by_split"].items()):
            result = block["metrics"]["cascade_f1"]
            curves = result["curves"] if kind == "data_scale" else {
                "process - state": result["contrasts"]["process_minus_state"]["by_size"]}
            for label, curve in curves.items():
                values = [curve[str(n)] for n in x]
                y = [v["mean"] if v["mean"] is not None else np.nan for v in values]
                low = [v["ci"][0] if v["ci"] else np.nan for v in values]
                high = [v["ci"][1] if v["ci"] else np.nan for v in values]
                line, = ax.plot(x, y, "o-", label=label)
                ax.fill_between(x, low, high, color=line.get_color(), alpha=.12)
            ax.set_xscale("log", base=2)
            ax.set_xticks(x, [str(n) for n in x])
            ax.set(title=split, xlabel="Training episodes", ylabel="Cascade F1" if kind == "data_scale" else "Cascade F1 difference")
            if kind != "data_scale":
                ax.axhline(0, color="gray", linewidth=.8)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle("Equal updates; final EMA; paired scene CI after seed averaging", fontsize=10)
        for suffix in ("png", "pdf"):
            fig.savefig(out / f"{kind}.{suffix}", dpi=250)
        plt.close(fig)


def analyze(name, boot=10000, seed=12345, allow_incomplete=False):
    audited = verify_audit(name)
    samples, ids, usable, missing, ignored, identities, diagnostics = load_results(name, audited, allow_incomplete)
    p = audited["protocol"]
    report = summarize(samples, ids, p["sizes"], usable, p["steps"], boot, seed)
    report.update(planned_training_seeds=p["training_seeds"], missing_cells=missing,
                  ignored_completed_cells=ignored, complete_design=not missing,
                  prefixes=audited["prefixes"], training_diagnostics=diagnostics)
    out = common.new_result_dir(EXP_NAME, "analysis", "data_scale", f"boot{boot}_seed{seed}", run=name)
    common.write_meta(out, "analysis", "data_scale", report["settings"], inputs=audited["source"], models=identities)
    write_json(out / "data_scale.json", report)
    (out / "data_scale.md").write_text(markdown(report), encoding="utf-8")
    plot(report, out / "figs")
    print(f"[06] analysis: {common.rel(out)}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--allow-incomplete", action="store_true")
    a = ap.parse_args(argv)
    with run_lock(run_path(a.run)):
        analyze(a.run, a.boot, a.seed, a.allow_incomplete)


if __name__ == "__main__":
    main()
