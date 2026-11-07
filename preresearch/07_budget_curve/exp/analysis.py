"""Plot budget curves, paired process-state differences, and tested crossing decisions."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from .shared import CONDITIONS, EXP_NAME, aggregate, common, metrics
from .generate_data import input_hashes


def paired_draws(state, process, n_boot, seed):
    """Same sampled seed numbers and bench cases in both conditions and at all budget levels."""
    levels = tuple(sorted(state))
    n_seeds, n_cases = state[levels[0]].shape
    rng = np.random.default_rng(seed)
    sidx = rng.integers(n_seeds, size=(n_boot, n_seeds))
    cidx = rng.integers(n_cases, size=(n_boot, n_cases))
    draws = {}
    for n in levels:
        a, b = state[n], process[n]
        draws[n] = (a[sidx[:, :, None], cidx[:, None, :]].mean((1, 2)),
                    b[sidx[:, :, None], cidx[:, None, :]].mean((1, 2)))
    return draws


def interval(draws):
    return [float(v) for v in np.percentile(draws, [2.5, 97.5])]


def summarize(state, process, n_boot=2000, seed=12345):
    levels = tuple(sorted(state))
    for n in levels:
        if state[n].shape != process[n].shape or state[n].shape != state[levels[0]].shape:
            raise ValueError("all paired levels need the same seeds and cases")
        if not np.isfinite(state[n]).all() or not np.isfinite(process[n]).all():
            raise ValueError("undefined score in paired budget data")
    counts = state[levels[0]].shape
    d = {n: process[n] - state[n] for n in levels}
    draws = paired_draws(state, process, n_boot, seed)
    result = {"n_seeds": counts[0], "n_cases": counts[1],
              "uncertainty": "paired seeds and paired bench scenes" if counts[0] > 1 else "scenes only; one seed is exploratory",
              "levels": {}, "interactions": {}, "crossings": []}
    for n in levels:
        sa, pr = draws[n]
        gap = pr - sa
        result["levels"][str(n)] = {"total_frames_per_condition": 64*n,
            "state": float(state[n].mean()), "process": float(process[n].mean()),
            "diff": float(d[n].mean()), "state_ci": interval(sa), "process_ci": interval(pr),
            "diff_ci": interval(gap), "p_two_sided": float(min(1, 2*min((gap <= 0).mean(), (gap >= 0).mean())))}
    from .shared import aggregate
    holm = aggregate.holm([result["levels"][str(n)]["p_two_sided"] for n in levels]) if counts[0] > 1 else [False]*len(levels)
    for n, sig in zip(levels, holm):
        result["levels"][str(n)]["holm_significant"] = bool(sig)
    for low, high in zip(levels, levels[1:]):
        delta = (draws[high][1] - draws[high][0]) - (draws[low][1] - draws[low][0])
        lo, hi = result["levels"][str(low)], result["levels"][str(high)]
        inter = {"diff_change": float(d[high].mean()-d[low].mean()), "ci": interval(delta)}
        key = f"{low}->{high}"
        result["interactions"][key] = inter
        if lo["diff"] < 0 < hi["diff"]:
            supported = bool(counts[0] > 1 and lo["diff_ci"][1] < 0 and hi["diff_ci"][0] > 0
                             and inter["ci"][0] > 0 and lo["holm_significant"] and hi["holm_significant"])
            result["crossings"].append({"between": [low, high], "status": "supported" if supported else "descriptive_only"})
    result["conclusion"] = ("supported_crossing" if any(x["status"] == "supported" for x in result["crossings"])
                             else "descriptive_crossing_only" if result["crossings"] else "no_crossing_in_measured_range")
    if {16, 64, 256, 512}.issubset(levels):
        result["growth_per_doubling"] = {
            cond: {"early_16_to_64": float((group[64].mean() - group[16].mean()) / 2),
                   "late_256_to_512": float(group[512].mean() - group[256].mean())}
            for cond, group in (("state", state), ("process", process))}
    if {512, 4096}.issubset(levels):
        result.setdefault("growth_per_doubling", {})
        for cond, group in (("state", state), ("process", process)):
            result["growth_per_doubling"].setdefault(cond, {})["extended_512_to_4096"] = float(
                (group[4096].mean() - group[512].mean()) / 3)
    return result


def correct_all_splits(by_split):
    """Holm-correct the full family of budget levels times four splits."""
    rows = [row for report in by_split.values() for row in report["levels"].values()]
    if by_split["id"]["n_seeds"] < 2:
        return by_split
    for row, flag in zip(rows, aggregate.holm([row["p_two_sided"] for row in rows])):
        row["holm_significant"] = bool(flag)
    for report in by_split.values():
        for crossing in report["crossings"]:
            low, high = (report["levels"][str(n)] for n in crossing["between"])
            interaction = report["interactions"][f"{crossing['between'][0]}->{crossing['between'][1]}"]
            supported = (low["diff_ci"][1] < 0 and high["diff_ci"][0] > 0
                         and interaction["ci"][0] > 0 and low["holm_significant"] and high["holm_significant"])
            crossing["status"] = "supported" if supported else "descriptive_only"
        if report["crossings"]:
            report["conclusion"] = ("supported_crossing" if any(c["status"] == "supported" for c in report["crossings"])
                                    else "descriptive_crossing_only")
    return by_split


def load_scores(run, seeds, data, bench, steps=None, epochs=None):
    audit = common.load_json(run / "audit.json")
    expected_protocol = {"mode": "full_epochs", "epochs": epochs, "batch": 256} if epochs else None
    if audit.get("training_protocol") != expected_protocol:
        raise ValueError("analysis training protocol differs from audited run")
    if audit["inputs"]["data"] != common.rel(data) or audit["inputs"]["bench"] != common.rel(bench):
        raise ValueError("training data or bench differs from the audited sources")
    val = common.resolve(audit["inputs"]["val"])
    if "hashes" in audit and (audit["hashes"]["data"] != input_hashes(data)
            or audit["hashes"]["val"] != input_hashes(val)
            or audit["hashes"]["bench"] != {p.name: common.sha1_file(p) for p in
                                               (bench, bench.with_suffix(".json"))}):
        raise ValueError("training, validation or bench files changed since audit")
    bench_sha = common.sha1_file(bench)
    with np.load(bench) as z:
        split_ids = z["split_ids"].copy()
    scores = {sp: {c: {} for c in CONDITIONS} for sp in common.SPLITS}
    auxiliary = {sp: {c: {} for c in CONDITIONS} for sp in common.SPLITS}
    sources = {}
    expected_config = None
    for n in sorted(map(int, audit["levels"])):
        for c in CONDITIONS:
            arr = []
            extra = {metric: [] for metric in ("change_f1", "any_violation")}
            for seed in seeds:
                rid = f"{c}_n{n}_s{seed}"
                expected_steps = (epochs * math.ceil(audit["levels"][str(n)]["conditions"][c]["video_pairs"] / 256)
                                  if epochs is not None else steps)
                d = common.find_latest(rid, "train", EXP_NAME, "eval_samples.npz", {"steps": expected_steps}, run.name)
                if d is None or not (d / "done.json").is_file():
                    raise FileNotFoundError(f"missing evaluated {rid} in {run}")
                info = common.load_json(d / "config.json")
                args = info["args"]
                cfg = {k: args[k] for k in ("batch", "lr", "change_weight", "depth", "unroll", "protocol")}
                cfg["epochs" if epochs is not None else "steps"] = epochs if epochs is not None else steps
                if expected_config is None:
                    expected_config = cfg
                elif expected_config != cfg:
                    raise ValueError(f"{rid}: unequal training parameters")
                if args["condition"] != c or args["seed"] != seed or args["episodes"] != n or args["data"] != common.rel(data):
                    raise ValueError(f"{rid}: wrong condition, seed, episode count, or data directory")
                if (args.get("epochs") != epochs or args["steps"] != expected_steps
                        or bool(args["balanced_phases"]) != (epochs is None and c == "process")):
                    raise ValueError(f"{rid}: incorrect training or phase-sampling protocol")
                done = common.load_json(d / "done.json")
                expected_form = (31 / 59 if epochs is not None else .5) if c == "process" else 0.0
                if abs(done["form_frac_mean"] - expected_form) > 1e-6:
                    raise ValueError(f"{rid}: incorrect formation share")
                if epochs is not None:
                    pairs_count = audit["levels"][str(n)]["conditions"][c]["video_pairs"]
                    if (done.get("epochs") != epochs or done.get("steps") != expected_steps
                            or done.get("n_pairs") != pairs_count or done.get("samples_seen") != epochs * pairs_count
                            or done.get("steps_per_epoch") != expected_steps // epochs):
                        raise ValueError(f"{rid}: did not complete each training transition {epochs} times")
                if "hashes" in audit:
                    if (not args["select_best_val"] or args["val"] != common.rel(val)
                            or args["val_every"] != 500 or done["selection"] != "val_cascade_acc"
                            or done["best_step"] is None or not 0 < done["best_step"] <= expected_steps
                            or (epochs is not None and done["best_step"] % done["steps_per_epoch"] != 0)
                            or info["data_sha1"] != {k: audit["hashes"]["data"][k] for k in info["data_sha1"]}
                            or info["val_sha1"] != {k: audit["hashes"]["val"][k] for k in info["val_sha1"]}
                            or done["ckpt_sha1"] != common.sha1_file(d / "ckpt.pt")):
                        raise ValueError(f"{rid}: training, validation or selected checkpoint differs")
                ev = common.load_json(d / "eval.json")
                if ev["bench"] != common.rel(bench) or common.sha1_file(common.resolve(ev["bench"])) != bench_sha:
                    raise ValueError(f"{rid}: wrong bench")
                if "hashes" in audit and (ev.get("checkpoint_sha1") != done["ckpt_sha1"]
                        or ev.get("bench_sha1") != bench_sha
                        or ev.get("bench_json_sha1") != audit["hashes"]["bench"][bench.with_suffix(".json").name]):
                    raise ValueError(f"{rid}: evaluation used a different checkpoint or bench")
                with np.load(d / "eval_samples.npz") as z:
                    if not np.array_equal(split_ids, z["split_ids"]):
                        raise ValueError(f"{rid}: different bench order")
                    score = metrics.sample_level({"cascade_f1": z["cascade_f1"]}, "cascade_f1")
                    for metric in extra:
                        extra[metric].append(metrics.sample_level({metric: z[metric]}, metric))
                arr.append(score)
                sources[rid] = common.rel(d)
            stack = np.stack(arr)
            for i, sp in enumerate(common.SPLITS):
                mask = split_ids == i
                scores[sp][c][n] = stack[:, mask]
                auxiliary[sp][c][n] = {metric: float(np.stack(rows)[:, mask].mean()) for metric, rows in extra.items()}
    return scores, auxiliary, sources, expected_config


def markdown(by_split, auxiliary, sources, config, budgets=None):
    lines = ["# Experiment 07_02: full-epoch budget curves" if budgets else
             "# Experiment 07: equal-total-budget learning curves", ""]
    if budgets:
        lines += ["Each valid transition is trained exactly once per epoch; process formation:operation is 31:28.",
                  "The x-axis is training frames; optimization updates grow with dataset size.",
                  "", f"Training configuration: `{config}`", "",
                  "| episodes | frames / condition | state / process updates | state / process training samples |",
                  "|---:|---:|---:|---:|"]
        for n, row in budgets.items():
            lines.append(f"| {n} | {row['frames']} | {row['state_steps']} / {row['process_steps']} | "
                         f"{row['state_samples']} / {row['process_samples']} |")
        lines.append("")
    else:
        lines += ["Each process batch is 50% formation, 50% operation; state batches are 100% operation.",
                  "The x-axis grows the number of scenes, not the number of optimization steps.",
                  "", f"Training configuration: `{config}`", ""]
    for sp, result in by_split.items():
        lines += [f"## {sp}", "", f"Seeds: {result['n_seeds']}; scenes: {result['n_cases']}; {result['uncertainty']}.",
                  "", "| frames / condition | state F1 | process F1 | process−state | 95% CI | Holm |",
                  "|---:|---:|---:|---:|---:|---|"]
        for row in result["levels"].values():
            a, b = row["diff_ci"]
            lines.append(f"| {row['total_frames_per_condition']} | {row['state']:.3f} | {row['process']:.3f} | "
                         f"{row['diff']:+.3f} | [{a:+.3f}, {b:+.3f}] | {row['holm_significant']} |")
        lines += ["", "Auxiliary readouts (means across paired seeds and cases):", "",
                  "| frames | state change F1 | process change F1 | state any violation | process any violation |",
                  "|---:|---:|---:|---:|---:|"]
        for n in result["levels"]:
            state, process = auxiliary[sp]["state"][int(n)], auxiliary[sp]["process"][int(n)]
            lines.append(f"| {64*int(n)} | {state['change_f1']:.3f} | {process['change_f1']:.3f} | "
                         f"{state['any_violation']:.3f} | {process['any_violation']:.3f} |")
        lines += ["", "Crossing: " + result["conclusion"], "",
                  "Per-doubling gains, early 16→64 versus late 256→512: " + str(result.get("growth_per_doubling", "not measured")),
                  "Adjacent differences: " + ", ".join(f"{k}={v['diff_change']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]"
                                                   for k, v in result["interactions"].items()), ""]
    lines += ["## Inputs", "", *[f"- `{k}`: `{v}`" for k, v in sources.items()], ""]
    return "\n".join(lines)


def plot(by_split, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, (sp, result) in zip(axes.flat, by_split.items()):
        rows = list(result["levels"].values())
        x = [r["total_frames_per_condition"] for r in rows]
        for c, color in (("state", "#3d639e"), ("process", "#c66d56")):
            ax.plot(x, [r[c] for r in rows], "o-", color=color, label=c)
        ax.set(xscale="log", title=sp, xlabel="Training frames per condition", ylabel="Cascade F1", ylim=(0, 1))
        ax.set_xticks(x, [str(v) for v in x])
        ax.grid(alpha=.2)
    axes.flat[0].legend(frameon=False)
    fig.savefig(out / "budget_curve.pdf")
    fig.savefig(out / "budget_curve.png", dpi=240)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    rows = list(by_split["id"]["levels"].values())
    x = [r["total_frames_per_condition"] for r in rows]
    y = [r["diff"] for r in rows]
    error = [[r["diff"]-r["diff_ci"][0] for r in rows], [r["diff_ci"][1]-r["diff"] for r in rows]]
    ax.errorbar(x, y, yerr=error, fmt="o-", color="#3d639e", capsize=4)
    ax.axhline(0, color="#c66d56", lw=1)
    ax.set(xscale="log", xlabel="Training frames per condition", ylabel="Process − state cascade F1 (id)")
    ax.set_xticks(x, [str(v) for v in x])
    ax.grid(alpha=.2)
    fig.savefig(out / "budget_difference.pdf")
    fig.savefig(out / "budget_difference.png", dpi=240)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--data", type=Path, default=common.GAME_CODE / "output/data_v3_curve_4096")
    ap.add_argument("--bench", type=Path, default=None)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args(argv)
    if a.steps is not None and a.epochs is not None:
        ap.error("--steps and --epochs are mutually exclusive")
    if a.epochs is not None and a.epochs <= 0:
        ap.error("--epochs must be positive")
    if a.epochs is None:
        a.steps = 12000 if a.steps is None else a.steps
    seeds = tuple(int(s) for s in a.seeds.split(","))
    run = common.RESULTS_ROOT / EXP_NAME / a.run
    data = common.resolve(a.data)
    bench = common.resolve(a.bench or common.load_json(run / "audit.json")["inputs"]["bench"])
    scores, auxiliary, sources, config = load_scores(run, seeds, data, bench, a.steps, a.epochs)
    result = correct_all_splits({sp: summarize(scores[sp]["state"], scores[sp]["process"], a.boot, 12345+i)
                                 for i, sp in enumerate(common.SPLITS)})
    budgets = None
    if a.epochs is not None:
        budgets = {str(n): {"frames": 64*n, "state_pairs": 56*n, "process_pairs": 59*n,
                            "state_samples": 56*n*a.epochs, "process_samples": 59*n*a.epochs,
                            "state_steps": a.epochs*math.ceil(56*n/256),
                            "process_steps": a.epochs*math.ceil(59*n/256),
                            "process_formation_fraction": 31/59}
                   for n in sorted(scores["id"]["state"])}
    protocol = f"epochs{a.epochs}" if a.epochs is not None else f"steps{a.steps}"
    out = common.new_result_dir(EXP_NAME, "analysis", "budget_curve", f"{protocol}_seeds{'-'.join(map(str,seeds))}", run=a.run)
    inputs = {"train": sources, "data": common.rel(data), "bench": common.rel(bench), "bench_sha1": common.sha1_file(bench)}
    common.write_meta(out, "analysis", "budget_curve", {"steps": a.steps, "epochs": a.epochs, "seeds": seeds, "boot": a.boot}, inputs=inputs)
    common.dump_json({"by_split": result, "budgets": budgets, "auxiliary": auxiliary,
                      "training": config, "inputs": inputs}, out / "budget_curve.json")
    (out / "budget_curve.md").write_text(markdown(result, auxiliary, sources, config, budgets), encoding="utf-8")
    plot(result, out)
    print(common.rel(out))
    print("id:", result["id"]["conclusion"], "crossings:", result["id"]["crossings"])


if __name__ == "__main__":
    main()
