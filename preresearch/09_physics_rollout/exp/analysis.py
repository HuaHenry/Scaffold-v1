"""Paired scene statistics, independent-training variability, and measured plots."""
import csv
from pathlib import Path

import numpy as np

from .evaluation import cached
from .shared import (CONDITIONS, CONTRASTS, METRICS, common, read_npz, write_json)

FINAL_METRICS = {
    "intervention": ("any_violation", "violations", "mass_error", "cell_error", "cell_acc_mask", "cascade_f1", "exact"),
    "drift": ("changed_fraction", "state_changed", "ever_changed", "any_violation", "ever_violation", "mass_error"),
    "formation": ("cell_acc_all", "cell_acc_mask", "structure_iou", "exact", "any_violation", "mass_error"),
}


def summarize_cube(cube, seeds, draws):
    if cube.ndim != 3 or len(seeds) != len(cube) or cube.shape[1] == 0 or not np.isfinite(cube).all():
        raise ValueError("Statistics require finite paired [training repetitions,scenes,ticks] scores")
    per_seed = cube.mean(axis=1)
    scene = cube.mean(axis=0)
    final_draws = scene[:, -1][draws].mean(axis=1)
    mean_draws = scene.mean(axis=1)[draws].mean(axis=1)
    std = per_seed.std(axis=0, ddof=1) if len(seeds) > 1 else None
    return {"mean": per_seed.mean(axis=0), "std": std,
            "per_seed_mean": {str(seed): values for seed, values in zip(seeds, per_seed)},
            "n_training_runs": len(seeds), "n_scenes": cube.shape[1],
            "final_mean": float(scene[:, -1].mean()),
            "final_std": None if std is None else float(std[-1]),
            "final_scene_95ci": np.percentile(final_draws, (2.5, 97.5)),
            "time_mean": float(scene.mean()),
            "time_mean_scene_95ci": np.percentile(mean_draws, (2.5, 97.5))}


def summarize_task(raw, oracle, seeds, splits, n_boot, bootstrap_seed):
    reference = raw[(CONDITIONS[0], seeds[0])]
    ids, split_ids, ticks = reference["case_ids"], reference["split_ids"], reference["ticks"]
    n = len(ids)
    if len(np.unique(ids)) != n:
        raise ValueError("Scene/episode IDs must be unique")
    for values in (*raw.values(), oracle):
        if any(not np.array_equal(values[key], reference[key]) for key in ("case_ids", "split_ids", "ticks")):
            raise ValueError("Condition, repetition and oracle case/tick ordering must match")
        expected_shape = (n, 1 if np.all(split_ids == -1) or len(ticks) == 128 else 2, len(ticks))
        if any(values[key].shape != expected_shape for key in METRICS):
            raise ValueError("Raw metric shapes do not follow paired scene/intervention/tick axes")
    masks = {"all": np.ones(n, dtype=bool)}
    masks.update({name: split_ids == index for index, name in enumerate(splits) if (split_ids == index).any()})
    output = {"ticks": ticks, "groups": {}}
    for group_index, (group, mask) in enumerate(masks.items()):
        count = int(mask.sum())
        draws = np.random.default_rng(bootstrap_seed + group_index).integers(count, size=(n_boot, count))
        cubes = {condition: {key: np.stack([raw[(condition, seed)][key][mask].mean(axis=1) for seed in seeds])
                             for key in METRICS} for condition in CONDITIONS}
        conditions = {condition: {key: summarize_cube(cube, seeds, draws) for key, cube in values.items()}
                      for condition, values in cubes.items()}
        differences = {f"{left}-{right}": {key: summarize_cube(cubes[left][key] - cubes[right][key], seeds, draws)
                                         for key in METRICS} for left, right in CONTRASTS}
        reference_scores = {key: {"mean": oracle[key][mask].mean(axis=(0, 1)),
                                 "final_mean": float(oracle[key][mask, :, -1].mean())} for key in METRICS}
        output["groups"][group] = {"conditions": conditions, "paired_differences": differences,
                                   "oracle_reseed": reference_scores, "n_scenes": count}
    return output


def load_results(out, source, tasks):
    scores = {}
    for task in tasks:
        raw = {}
        for row in source["models"]:
            path = out / "raw" / f"{row['condition']}_seed{row['seed']}" / f"{task}.npz"
            identity = {"protocol": source, "task": task, "model": row}
            if not cached(path, identity):
                raise FileNotFoundError(f"Missing completed result: {path}")
            raw[(row["condition"], row["seed"])] = read_npz(path)
        path = out / "raw/oracle_reseed" / f"{task}.npz"
        if not cached(path, {"protocol": source, "task": task, "model": "oracle_reseed"}):
            raise FileNotFoundError(path)
        scores[task] = raw, read_npz(path)
    return scores


def tables(report, out):
    out.mkdir(parents=True, exist_ok=True)
    fields = ("task", "group", "condition", "metric", "mean", "training_std", "scene_ci_low", "scene_ci_high", "n_scenes", "n_training_runs")
    with (out / "endpoints.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for task, data in report["tasks"].items():
            for group, row in data["groups"].items():
                for condition, metrics in {**row["conditions"], **row["paired_differences"]}.items():
                    for name in FINAL_METRICS[task]:
                        value = metrics[name]
                        writer.writerow((task, group, condition, name, value["final_mean"], value["final_std"],
                                         *value["final_scene_95ci"], value["n_scenes"], value["n_training_runs"]))
    with (out / "curves.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("task", "group", "condition", "metric", "tick", "mean", "training_std"))
        for task, data in report["tasks"].items():
            for group, row in data["groups"].items():
                for condition, metrics in row["conditions"].items():
                    for key, values in metrics.items():
                        for index, tick in enumerate(data["ticks"]):
                            writer.writerow((task, group, condition, key, tick, values["mean"][index],
                                             None if values["std"] is None else values["std"][index]))
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Probe & Condition & Violation (\%) & Mass error & Cell error (\%) & Exact (\%)\\", r"\midrule"]
    for task, data in report["tasks"].items():
        row = data["groups"]["all"]
        for condition in (*CONDITIONS, "oracle_reseed"):
            scores = row["oracle_reseed"] if condition == "oracle_reseed" else row["conditions"][condition]
            values = []
            for key, factor in (("any_violation", 100), ("mass_error", 1), ("cell_error", 100), ("exact", 100)):
                value = f"{scores[key]['final_mean'] * factor:.2f}"
                if scores[key].get("final_std") is not None:
                    value += r" $\pm$ " + f"{scores[key]['final_std'] * factor:.2f}"
                values.append(value)
            lines.append(task.capitalize() + " & " + condition.replace("_", r"\_") + " & " +
                         " & ".join(values) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "endpoints.tex").write_text("\n".join(lines) + "\n")


def plot(report, out, truth):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    colors = {"state": "#52738F", "process": "#D45A39", "partial": "#2B887A", "shuffle": "#9A80AD"}
    panels = {"intervention": (("any_violation", "Rest-rule detector (%)", 100), ("mass_error", "Material-count error", 1), ("cell_error", "Cell error (%)", 100)),
              "drift": (("changed_fraction", "Changed cells (%)", 100), ("ever_changed", "Ever changed scenes (%)", 100), ("ever_violation", "Ever detector-positive scenes (%)", 100)),
              "formation": (("cell_acc_all", "Cell accuracy (%)", 100), ("structure_iou", "Structure IoU (%)", 100), ("exact", "Exact product (%)", 100))}
    for task, data in report["tasks"].items():
        groups = list(data["groups"])
        fig, axes = plt.subplots(len(groups), 3, figsize=(12, 2.8 * len(groups)), squeeze=False, constrained_layout=True)
        for row_index, group in enumerate(groups):
            row = data["groups"][group]
            for ax, (key, ylabel, factor) in zip(axes[row_index], panels[task]):
                for condition, values in row["conditions"].items():
                    mean = np.asarray(values[key]["mean"]) * factor
                    ax.plot(data["ticks"], mean, label=condition.capitalize(), color=colors[condition], lw=1.8)
                    if values[key]["std"] is not None:
                        sd = np.asarray(values[key]["std"]) * factor
                        ax.fill_between(data["ticks"], mean - sd, mean + sd, color=colors[condition], alpha=.14)
                ax.plot(data["ticks"], np.asarray(row["oracle_reseed"][key]["mean"]) * factor,
                        color="#333333", ls="--", lw=1.3, label="Oracle reseed")
                if task == "intervention" and key == "any_violation":
                    mask = np.ones(len(truth["split_ids"]), dtype=bool) if group == "all" else truth["split_ids"] == report["splits"].index(group)
                    curve = truth["any_violation"][mask].mean(axis=(0, 1)) * factor
                    ax.plot(data["ticks"], curve, color="black", ls=":", label="Recorded truth")
                ax.set(xlabel="Prediction tick", ylabel=ylabel, title=f"{task.capitalize()} / {group}")
                ax.grid(alpha=.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), frameon=False)
        subtitle = "Shading: training-run standard deviation" if len(report["training_seeds"]) > 1 else "One training repetition; no training-variance band"
        fig.suptitle(subtitle, fontsize=11)
        for suffix in ("pdf", "png"):
            fig.savefig(out / f"{task}.{suffix}", dpi=220)
        plt.close(fig)
    if set(report["tasks"]) == {"intervention", "drift", "formation"}:
        summary_panels = (("intervention", "any_violation", "Rest-rule detector (%)", 100),
                          ("intervention", "mass_error", "Material-count error", 1),
                          ("intervention", "cell_error", "Cell error (%)", 100),
                          ("drift", "changed_fraction", "No-action changed cells (%)", 100),
                          ("formation", "cell_acc_all", "Construction cell accuracy (%)", 100),
                          ("formation", "exact", "Exact constructed product (%)", 100))
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        for ax, (task, key, ylabel, factor) in zip(axes.flat, summary_panels):
            data = report["tasks"][task]
            row = data["groups"]["all"]
            for condition, scores in row["conditions"].items():
                mean = np.asarray(scores[key]["mean"]) * factor
                ax.plot(data["ticks"], mean, color=colors[condition], label=condition.capitalize())
                if scores[key]["std"] is not None:
                    sd = np.asarray(scores[key]["std"]) * factor
                    ax.fill_between(data["ticks"], mean - sd, mean + sd, color=colors[condition], alpha=.14)
            ax.plot(data["ticks"], np.asarray(row["oracle_reseed"][key]["mean"]) * factor,
                    color="#333333", ls="--", label="Oracle reseed")
            if task == "intervention" and key == "any_violation":
                ax.plot(data["ticks"], truth["any_violation"].mean(axis=(0, 1)) * factor,
                        color="black", ls=":", label="Recorded truth")
            ax.set(xlabel="Prediction tick", ylabel=ylabel)
            ax.grid(alpha=.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), frameon=False)
        for suffix in ("pdf", "png"):
            fig.savefig(out / f"fig8.{suffix}", dpi=240)
        plt.close(fig)


def analyze(out, source, tasks, n_boot=2000, bootstrap_seed=2027, make_plots=True):
    if n_boot <= 0:
        raise ValueError("Bootstrap count must be positive")
    all_scores = load_results(out, source, tasks)
    report = {"version": source["version"], "training_seeds": source["seeds"], "splits": source["splits"],
              "protocol": source, "tasks": {}, "n_boot": n_boot, "bootstrap_seed": bootstrap_seed,
              "statistics": "Average paired interventions within each scene; training means/std use independent models. Scene CIs resample scenes after averaging training repetitions, not training-run uncertainty.",
              "difference_direction": "Process minus reference; negative is better for violation, mass and cell errors.",
              "formation_interpretation": "An exposure-favored ability probe, not evidence for H1 or history-conditioned reasoning."}
    for index, (task, (raw, oracle)) in enumerate(all_scores.items()):
        report["tasks"][task] = summarize_task(raw, oracle, source["seeds"], source["splits"], n_boot, bootstrap_seed + 100 * index)
    report["audit"] = common.load_json(out / "audit.json")
    write_json(out / "analysis/results.json", report)
    tables(report, out / "analysis/tables")
    if make_plots:
        plot(report, out / "analysis/figures", read_npz(out / "audit_intervention.npz"))
    artifacts = [path for path in (out / "analysis").rglob("*") if path.is_file() and path.name != "complete.json"]
    write_json(out / "analysis/complete.json", {"qualified": report["audit"]["qualified"],
               "hashes": {str(path.relative_to(out / "analysis")): common.sha1_file(path) for path in artifacts}})
    return report
