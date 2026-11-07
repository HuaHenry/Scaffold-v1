"""Q2 teacher-forced formation probe, Q3 seed-0 benchmark signal, and Figure A1.

python -m exp.analysis.formation_probe --run <02-run> --steps 6000 [--pilot01 PATH]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..shared import EXP_NAME, aggregate, common, evaluate, metrics

CLASSES = ("formation_placement", "formation_physics", "operation")


def classify(phase, changed, act):
    """Disjoint exhaustive formation classes; return None for non-operation transitions in the wrong source."""
    if phase == 1:
        return "operation"
    return "formation_physics" if np.any(changed & (act == 0)) else "formation_placement"


def validation_pairs(val):
    result = {}
    for cond in ("process", "state"):
        d = common.load_condition(val, cond)
        pairs = common.make_pairs(val, cond, "video")
        if not np.all(d["cont"][pairs[:, 0], pairs[:, 1]]):
            raise ValueError(f"{cond} contains a discontinuous video pair")
        for e, t in pairs:
            e, t = int(e), int(t)
            phase = int(d["phase"][e, t])
            if (cond == "process" and phase != 0) or (cond == "state" and phase != 1):
                continue
            act = common.encode_actions(d["actions"][e][t])
            changed = (d["fg"][e, t-1] != d["fg"][e, t]) | (d["bg"][e, t-1] != d["bg"][e, t])
            category = classify(phase, changed, act)
            result.setdefault(category, []).append((e, t, cond, act))
    return result


def check_validation(val, data):
    overlap = common.load_json(Path(val) / "overlap.json")
    if overlap["kept"] != 32:
        raise ValueError("probe requires 32 validation episodes per condition")
    train = {x["final_hash"] for x in common.load_json(Path(data) / "train_scenes.json")}
    held = {x["final_hash"] for x in common.load_json(Path(val) / "train_scenes.json")}
    if train & held:
        raise ValueError("validation final scenes overlap training final scenes")


def probe(models, val, batch=128, n_boot=aggregate.N_BOOT):
    pairs = validation_pairs(val)
    arrays = {cond: common.load_condition(val, cond, with_actions=False) for cond in ("process", "state")}
    report = {"by_class": {}, "comparisons": {}, "n_boot": n_boot,
              "note": "one seed per condition; paired transition bootstrap excludes training-seed uncertainty"}
    for category in CLASSES:
        entries = pairs.get(category, [])
        if not entries:
            raise ValueError(f"no validation transitions in {category}")
        src = "state" if category == "operation" else "process"
        if any(c != src for _, _, c, _ in entries):
            raise ValueError(f"mixed source for {category}")
        d = arrays[src]
        n = len(entries)
        scores = {}
        for condition, model in models.items():
            change_acc = np.full(n, np.nan)
            exact = np.zeros(n)
            for start in range(0, n, batch):
                group = entries[start:start + batch]
                e, t = np.array([(e, t) for e, t, _, _ in group]).T
                fg, bg = d["fg"][e, t-1], d["bg"][e, t-1]
                target_fg, target_bg = d["fg"][e, t], d["bg"][e, t]
                action = np.stack([item[3] for item in group])
                device = next(model.parameters()).device
                with torch.no_grad():
                    pf, pb = model.step(torch.from_numpy(fg).long().to(device),
                                        torch.from_numpy(bg).long().to(device),
                                        torch.from_numpy(action).long().to(device))
                correct = (pf.cpu().numpy() == target_fg) & (pb.cpu().numpy() == target_bg)
                changed = (fg != target_fg) | (bg != target_bg)
                num = changed.sum((1, 2))
                indices = np.arange(start, start + len(group))
                change_acc[indices] = np.divide((correct & changed).sum((1, 2)), num,
                                                 out=np.full(len(group), np.nan), where=num > 0)
                exact[indices] = correct.all((1, 2))
            scores[condition] = {"change_acc": change_acc, "trans_exact": exact}
        valid = int(np.isfinite(scores["state"]["change_acc"]).sum())
        report["by_class"][category] = {"n": n, "n_changed": valid, "n_no_change": n - valid,
                                         "by_condition": {c: {m: float(np.nanmean(v)) for m, v in s.items()}
                                                          for c, s in scores.items()}}
        report["comparisons"][category] = {
            f"process-{other}": {m: aggregate.paired_bootstrap(scores["process"][m], scores[other][m], n_boot)
                                  for m in ("change_acc", "trans_exact")}
            for other in ("state", "partial", "shuffle")}
    return report


def bench_signal(folders, bench_path, n_boot=aggregate.N_BOOT):
    bench = common.load_bench(bench_path)
    sample = {}
    for cond, folder in folders.items():
        with np.load(folder / "eval_samples.npz") as z:
            if not np.array_equal(z["split_ids"], bench["split_ids"]):
                raise ValueError(f"{cond} evaluation is not the full same FormBench")
            sample[cond] = metrics.sample_level({"cascade_f1": z["cascade_f1"]}, "cascade_f1")
        info = common.load_json(folder / "eval.json")
        if info["bench"] != common.rel(bench_path):
            raise ValueError(f"{cond} evaluated on a different bench")
    result = {"by_split": {}, "note": "one seed per condition: sample-only CI, not evidence of a seed-level effect"}
    for i, sp in enumerate(common.SPLITS):
        sel = bench["split_ids"] == i
        result["by_split"][sp] = {
            "n": int(sel.sum()), "means": {c: float(np.mean(v[sel])) for c, v in sample.items()},
            "comparisons": {f"process-{c}": aggregate.paired_bootstrap(sample["process"][sel], sample[c][sel], n_boot)
                            for c in ("state", "partial", "shuffle")}}
    return result


def figure_a1(probe_report, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    x = np.arange(len(CLASSES))
    colors = ("#4263a5", "#cb6953", "#64a092", "#d6ab54")
    for i, (cond, color) in enumerate(zip(common.CONDITIONS, colors)):
        values = [probe_report["by_class"][c]["by_condition"][cond]["change_acc"] for c in CLASSES]
        ax.bar(x + (i - 1.5)*0.19, values, width=0.18, label=cond, color=color)
    ax.set_xticks(x, ["Formation: placement", "Formation: physics", "Operation"])
    ax.set_ylabel("Changed-cell accuracy (fg and bg both correct)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=.2)
    ax.legend(ncol=4, frameon=False, loc="upper center")
    (out / "figs").mkdir(exist_ok=True)
    fig.savefig(out / "figs" / "figure_A1.pdf")
    fig.savefig(out / "figs" / "figure_A1.png", dpi=300)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="02 run that contains the four complete seed-0 evaluations")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--val", type=Path, default=common.DEFAULT_VAL)
    ap.add_argument("--data", type=Path, default=common.DEFAULT_DATA)
    ap.add_argument("--bench", type=Path, default=common.DEFAULT_BENCH)
    ap.add_argument("--n-boot", type=int, default=aggregate.N_BOOT)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    data, val, bench = common.resolve(a.data), common.resolve(a.val), common.resolve(a.bench)
    check_validation(val, data)
    folders = {}
    for c in common.CONDITIONS:
        folder = common.find_latest(f"{c}_s0", "train", EXP_NAME, "eval_samples.npz", {"steps": a.steps}, a.run)
        if not folder or not (folder / "done.json").is_file():
            raise FileNotFoundError(f"missing completed/evaluated {c}_s0 in {EXP_NAME}/{a.run}")
        folders[c] = folder
    models = {c: evaluate.load_run(folder, torch.device("cuda" if torch.cuda.is_available() else "cpu"))[0]
              for c, folder in folders.items()}
    result = probe(models, val, n_boot=a.n_boot)
    signal = bench_signal(folders, bench, a.n_boot)
    inputs = {**{c: common.rel(d) for c, d in folders.items()}, "data": common.rel(data),
              "val": common.rel(val), "bench": common.rel(bench)}
    result["inputs"] = signal["inputs"] = inputs
    out = common.resolve_out(a.out) if a.out else common.new_result_dir(EXP_NAME, "analysis", "pilot_probe",
                            f"steps{a.steps}_boot{a.n_boot}_cascade_f1", run=a.run)
    out.mkdir(parents=True, exist_ok=True)
    common.write_meta(out, "analysis", "pilot_probe", {"steps": a.steps, "n_boot": a.n_boot,
                                                        "metric": "cascade_f1"}, inputs=inputs)
    common.dump_json(result, out / "pilot_probe.json")
    common.dump_json(signal, out / "pilot_signal.json")
    figure_a1(result, out)
    print(common.rel(out))


if __name__ == "__main__":
    main()
