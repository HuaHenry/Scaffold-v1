"""Four-condition main table and paired scene-level comparisons."""
import argparse
from pathlib import Path

import numpy as np

from .audit import verify_inputs
from .shared import (CONDITIONS, EPISODES, EXP_NAME, METRICS, budget, common, metrics,
                     validate_completed, validate_evaluation)

COMPARISONS = (("process", "state"), ("process", "partial"), ("process", "shuffle"),
               ("state", "shuffle"), ("partial", "state"))


def load_results(run):
    run = Path(run)
    audited = common.load_json(run / "audit.json")
    paths = verify_inputs(audited)
    with np.load(paths["bench"]) as z:
        split_ids = z["split_ids"].copy()
    groups, sources = {}, {}
    for c in CONDITIONS:
        out = common.find_latest(c, "train", EXP_NAME, "eval_samples.npz",
                                 {"steps": budget(c)["steps"], "epochs": 5, "episodes": EPISODES}, run.name)
        if out is None:
            raise FileNotFoundError(f"main experiment missing completed evaluation: {c}")
        done = validate_completed(out, c, audited)
        validate_evaluation(out, audited, done)
        with np.load(out / "eval_samples.npz") as z:
            if not np.array_equal(z["split_ids"], split_ids):
                raise ValueError(f"{c}: benchmark cases or case ordering differ")
            values = {m: metrics.sample_level({m: z[m]}, m) for m in METRICS}
        if any(v.shape != (512,) for v in values.values()) or not np.isfinite(values["cascade_f1"]).all():
            raise ValueError(f"{c}: invalid or incomplete full-benchmark metrics")
        groups[c] = values
        sources[c] = common.rel(out)
    return groups, split_ids, sources, audited


def summarize(groups, split_ids, n_boot=2000):
    if set(groups) != set(CONDITIONS) or n_boot <= 0:
        raise ValueError("all four conditions and a positive bootstrap count are required")
    if any(groups[c]["cascade_f1"].shape != split_ids.shape for c in CONDITIONS):
        raise ValueError("all conditions must evaluate the same paired benchmark cases")
    report = {}
    for i, split in enumerate(common.SPLITS):
        mask = split_ids == i
        n = int(mask.sum())
        if n != 128:
            raise ValueError("each hard benchmark split must contain 128 cases")
        rng = np.random.default_rng(2027 + i)
        sample_indices = rng.integers(n, size=(n_boot, n))
        means = {c: {m: metrics.masked_mean(groups[c][m][mask])[0] for m in METRICS} for c in CONDITIONS}
        comparisons = {}
        for left, right in COMPARISONS:
            diff = groups[left]["cascade_f1"][mask] - groups[right]["cascade_f1"][mask]
            if not np.isfinite(diff).all():
                raise ValueError("cascade comparisons contain undefined scores")
            draws = diff[sample_indices].mean(axis=1)
            comparisons[f"{left}-{right}"] = {"difference": float(diff.mean()),
                                               "paired_case_95ci": np.percentile(draws, (2.5, 97.5)).tolist()}
        report[split] = {"conditions": means, "comparisons": comparisons, "cases": n}
    return report


def markdown(report, audited, sources):
    lines = ["# Main experiment 03: simple training to complex evaluation", "",
             "Four conditions, 81920 episodes and 5242880 original frames per condition; five complete "
             "no-replacement epochs. Checkpoints selected on the same independent validation set only.",
             "The split named `id` is hard transfer, not training-distribution ID.",
             "", "| Condition | Frames | Valid pairs | Updates | Formation transition fraction |",
             "|---|---:|---:|---:|---:|"]
    for c in CONDITIONS:
        b = budget(c)
        lines.append(f"| {c} | {b['frames']} | {b['pairs']} | {b['steps']} | {b['formation_fraction']:.4f} |")
    lines += ["", "All conditions share original scene/frame budgets and epoch count, not necessarily update counts. "
              "For shuffle the formation-labelled pairs are nonphysical jumps, not ordered construction.", "",
              "## Cascade F1", "", "| Split | state | process | partial | shuffle |",
              "|---|---:|---:|---:|---:|"]
    for split, row in report.items():
        lines.append("| " + split + " | " + " | ".join(f"{row['conditions'][c]['cascade_f1']:.3f}"
                                                         for c in CONDITIONS) + " |")
    lines += ["", "## Paired cascade comparisons", "",
              "Intervals resample the same benchmark scenes in both conditions.", "",
              "| Split | Contrast | Difference | Paired case 95% CI |", "|---|---|---:|---:|"]
    for split, row in report.items():
        for pair, value in row["comparisons"].items():
            a, b = value["paired_case_95ci"]
            lines.append(f"| {split} | {pair} | {value['difference']:+.3f} | [{a:+.3f}, {b:+.3f}] |")
    for split, row in report.items():
        lines += ["", f"## {split}: all metrics", "",
                  "| Condition | " + " | ".join(METRICS) + " |", "|---|" + "---:|" * len(METRICS)]
        for c in CONDITIONS:
            lines.append("| " + c + " | " + " | ".join(f"{row['conditions'][c][m]:.3f}" for m in METRICS) + " |")
    lines += ["", "## Inputs", "", *[f"- {k}: `{v}`" for k, v in audited["inputs"].items()],
              *[f"- {c}: `{path}`" for c, path in sources.items()], ""]
    return "\n".join(lines)


def plot(report, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    x = np.arange(4)
    for j, c in enumerate(CONDITIONS):
        ax.bar(x + (j-1.5)*.19, [r["conditions"][c]["cascade_f1"] for r in report.values()], width=.18, label=c)
    ax.set_xticks(x, list(report), rotation=30, ha="right")
    ax.set(ylabel="Cascade F1 (hard transfer)", ylim=(0, 1))
    ax.legend(ncol=4, frameon=False)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"four_conditions.{suffix}", dpi=240)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, (split, row) in zip(axes.flat, report.items()):
        pairs = list(row["comparisons"])
        values = [row["comparisons"][pair] for pair in pairs]
        y = [v["difference"] for v in values]
        errors = [[v["difference"]-v["paired_case_95ci"][0] for v in values],
                  [v["paired_case_95ci"][1]-v["difference"] for v in values]]
        ax.errorbar(np.arange(len(pairs)), y, yerr=np.maximum(errors, 0), fmt="o", capsize=4)
        ax.axhline(0, lw=1, color="gray")
        ax.set_xticks(np.arange(len(pairs)), pairs, rotation=45, ha="right", fontsize=8)
        ax.set(title=split, ylabel="Paired cascade F1 difference")
        ax.grid(axis="y", alpha=.2)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"paired_differences.{suffix}", dpi=240)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args(argv)
    run = common.RESULTS_ROOT / EXP_NAME / a.run
    groups, split_ids, sources, audited = load_results(run)
    report = summarize(groups, split_ids, a.boot)
    out = common.new_result_dir(EXP_NAME, "analysis", "main03", "ep81920_epochs5", run=a.run)
    common.write_meta(out, "analysis", "main03", {"episodes": EPISODES, "epochs": 5}, inputs=sources)
    common.dump_json({"by_split": report, "budgets": {c: budget(c) for c in CONDITIONS},
                      "inputs": audited["inputs"], "hashes": audited["hashes"], "sources": sources}, out / "main03.json")
    (out / "main03.md").write_text(markdown(report, audited, sources), encoding="utf-8")
    plot(report, out)
    print(common.rel(out))


if __name__ == "__main__":
    main()
