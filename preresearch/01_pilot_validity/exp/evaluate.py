"""Evaluation on FormBench (plan 00 section 10): learned models and baselines go through the same scoring.

python -m exp.evaluate --run state_s0 --splits all [--bench PATH]      # run id or result-folder path
python -m exp.evaluate --predictor copy_last                            # -> new timestamped baseline folder

A trained run's evaluation is written into the run's own folder (eval.json, eval_samples.npz);
a baseline gets its own folder experiments/preresearch/<exp>/<timestamp>_baseline_<name>_<bench>/.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from . import common, metrics, baselines
from .common import HORIZON, resolve
from .rollout import fixed_point_step

SCORE_KEYS = metrics.PER_INTERVENTION + ("cascade_f1_nonempty",)


def score_bench(bench: dict, tf: np.ndarray, tb: np.ndarray, with_violations: bool = True) -> dict:
    """tf, tb: uint8 [N,2,32,H,W] predicted trajectories. Returns per-sample arrays ([N,2] or [N])."""
    N = tf.shape[0]
    fp = fixed_point_step(tf, tb, bench["ctx_fg"], bench["ctx_bg"], bench["t_last"])
    keys32 = ("cell_acc_mask", "change_f1", "cascade_f1", "exact", "structure_iou")
    s = {k: np.full((N, 2), np.nan) for k in SCORE_KEYS}
    s.update({f"{k}__32": np.full((N, 2), np.nan) for k in keys32})
    for h in ("s1", "s4"):
        s.update({f"change_f1__{h}": np.full((N, 2), np.nan), f"cascade_f1__{h}": np.full((N, 2), np.nan)})
    rate, rate32 = np.zeros((N, 2)), np.zeros((N, 2))
    pred_fg = np.zeros((N, 2) + tf.shape[3:], dtype=np.uint8)
    pred_bg = np.zeros_like(pred_fg)
    for i in range(N):
        S = (bench["ctx_fg"][i], bench["ctx_bg"][i])
        for k in range(2):
            A = bench["A"][i, k]
            T = lambda t: (bench["fut_fg"][i, k, t], bench["fut_bg"][i, k, t])
            P = lambda t: (tf[i, k, t], tb[i, k, t])
            Pfp = P(int(fp[i, k]))
            sc = metrics.score_sample(Pfp, T(HORIZON - 1), S, A, with_violations)
            for key in metrics.PER_INTERVENTION:
                s[key][i, k] = sc[key]
            if sc["has_cascade"]:
                s["cascade_f1_nonempty"][i, k] = sc["cascade_f1"]
            sc32 = metrics.score_sample(P(HORIZON - 1), T(HORIZON - 1), S, A, False)
            for key in keys32:
                s[f"{key}__32"][i, k] = sc32[key]
            for h, t in (("s1", 0), ("s4", 3)):
                sh = metrics.short_scores(P(t), T(t), S, A)
                s[f"change_f1__{h}"][i, k] = sh["change_f1"]
                s[f"cascade_f1__{h}"][i, k] = sh["cascade_f1"]
            rate[i, k] = metrics.change_rate(S, Pfp)
            rate32[i, k] = metrics.change_rate(S, P(HORIZON - 1))
            pred_fg[i, k], pred_bg[i, k] = Pfp
    valid = bench["rank_valid"]
    ra = np.array([metrics.rank_acc(*rate[i], *bench["labels"][i]) for i in range(N)])
    ra32 = np.array([metrics.rank_acc(*rate32[i], *bench["labels"][i]) for i in range(N)])
    s["rank_acc"] = np.where(valid, ra, np.nan)
    s["rank_acc__32"] = np.where(valid, ra32, np.nan)
    s["rate"], s["fp_step"] = rate, fp.astype(np.int16)
    s["split_ids"], s["rank_valid"] = bench["split_ids"], valid
    s["pred_fg"], s["pred_bg"] = pred_fg, pred_bg
    return s


def _agg(samples: dict, sel: np.ndarray, only_k0: bool = False) -> dict:
    out = {}
    for key, v in samples.items():
        if key in ("pred_fg", "pred_bg", "split_ids", "rank_valid", "rate", "fp_step"):
            continue
        v = v[sel]
        x = metrics.sample_level({key: v[:, :1] if only_k0 and v.ndim == 2 else v}, key)
        m, n = metrics.masked_mean(x)
        out[key] = {"mean": m, "n": n}
    return out


def summarize(samples: dict, bench: dict) -> dict:
    N = len(samples["split_ids"])
    names = bench["splits"]
    res = {"n_samples": N, "overall": _agg(samples, np.ones(N, bool)),
           "overall_k0": _agg(samples, np.ones(N, bool), only_k0=True), "by_split": {}, "by_scene": {}}
    for si, name in enumerate(names):
        sel = samples["split_ids"] == si
        if sel.any():
            res["by_split"][name] = _agg(samples, sel)
            res["by_split"][name + "@k0"] = _agg(samples, sel, only_k0=True)
    kinds = [set(sk.split("+")) for sk in bench["scene_kinds"]]
    for kd in sorted(set().union(*kinds)):
        sel = np.array([kd in ks for ks in kinds])
        res["by_scene"][kd] = _agg(samples, sel)
    for n in sorted({len(ks) for ks in kinds}):
        res["by_scene"][f"n_struct={n}"] = _agg(samples, np.array([len(ks) == n for ks in kinds]))
    return res


# --------------------------------------------------------------------------- model predictions
def load_run(run_dir: Path, device):
    import torch
    from .model import build_model
    ck = torch.load(run_dir / "ckpt.pt", map_location=device)
    m = build_model(ck["cfg"]).to(device)
    m.load_state_dict(ck["ema"])
    return m.eval(), ck


def predict_bench_model(model, bench: dict, device, batch: int = 512):
    import torch
    from .rollout import bench_action_maps, rollout_maps
    N = bench["ctx_fg"].shape[0]
    tf = np.zeros((N, 2, HORIZON) + bench["ctx_fg"].shape[1:], dtype=np.uint8)
    tb = np.zeros_like(tf)
    for k in range(2):
        for a in range(0, N, batch):
            sl = slice(a, min(N, a + batch))
            fg0 = torch.from_numpy(bench["ctx_fg"][sl]).long().to(device)
            bg0 = torch.from_numpy(bench["ctx_bg"][sl]).long().to(device)
            maps = torch.from_numpy(bench_action_maps([bench["ivs"][i][k] for i in range(sl.start, sl.stop)])).to(device)
            f, b = rollout_maps(model, fg0, bg0, maps, HORIZON)
            tf[sl, k], tb[sl, k] = f.cpu().numpy().astype(np.uint8), b.cpu().numpy().astype(np.uint8)
    return tf, tb


def evaluate_run(run_dir: Path, bench_path, splits, limit, tag, batch, device_name) -> dict:
    from .train import pick_device
    dev = pick_device(device_name)
    model, ck = load_run(run_dir, dev)
    bench = common.select_bench(common.load_bench(bench_path), splits, limit)
    t0 = time.time()
    tf, tb = predict_bench_model(model, bench, dev, batch)
    samples = score_bench(bench, tf, tb)
    summary = summarize(samples, bench)
    out = run_dir / f"eval{tag}.json"
    common.dump_json({"run": common.rel(run_dir), "bench": common.rel(bench_path),
                      "bench_sha1": common.sha1_file(bench_path),
                      "bench_json_sha1": common.sha1_file(Path(bench_path).with_suffix(".json")),
                      "checkpoint_sha1": common.sha1_file(run_dir / "ckpt.pt"),
                      "seconds": time.time() - t0, **summary}, out)
    np.savez_compressed(run_dir / f"eval_samples{tag}.npz", **samples)
    return summary


def evaluate_predictor(name, bench_path, splits, limit, out_dir: Path, workers: int) -> dict:
    bench = common.select_bench(common.load_bench(bench_path), splits, limit)
    tf, tb = baselines.predict_bench(name, bench, workers)
    samples = score_bench(bench, tf, tb)
    summary = summarize(samples, bench)
    out_dir.mkdir(parents=True, exist_ok=True)
    common.dump_json({"predictor": name, "bench": common.rel(bench_path), **summary}, out_dir / "eval.json")
    np.savez_compressed(out_dir / "eval_samples.npz", **samples)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate a trained run or a baseline on FormBench.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="run id (e.g. state_s0 = newest finished run) or a result-folder path")
    g.add_argument("--predictor", choices=baselines.NAMES)
    ap.add_argument("--steps", type=int, default=6000, help="when --run is an id: training steps of the run to pick (0 = any)")
    ap.add_argument("--bench", type=Path, default=common.DEFAULT_BENCH)
    ap.add_argument("--splits", default="all", help="all | comma list of id,combo_ood,scale_ood,multi_ood")
    ap.add_argument("--limit", type=int, default=None, help="at most N samples per split (quick checks)")
    ap.add_argument("--out", type=Path, default=None, help="baseline result folder (default: new timestamped folder)")
    ap.add_argument("--exp", default=common.EXP_NAME, help="experiment folder under the results root (baselines)")
    ap.add_argument("--tag", default="", help="suffix of eval json, e.g. _b1 for a second bench")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4, help="processes for engine-based baselines")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    bench = resolve(a.bench)
    if a.run:
        run = common.resolve_run(a.run, steps=a.steps)
        # partial evaluations never overwrite the full eval.json / eval_samples.npz that the analysis reads
        tag = a.tag or (f"_first{a.limit}" if a.limit else "") + ("" if a.splits == "all" else "_" + a.splits.replace(",", "-"))
        s = evaluate_run(run, bench, a.splits, a.limit, tag, a.batch, a.device)
        where = run
    else:
        label = common.bench_label(bench)
        sett = f"bench-{label}" + (f"_first{a.limit}" if a.limit else "") + ("" if a.splits == "all" else f"_{a.splits}")
        where = common.resolve_out(a.out) if a.out else common.new_result_dir(a.exp, "baseline", a.predictor, sett)
        where.mkdir(parents=True, exist_ok=True)
        common.write_meta(where, "baseline", a.predictor, {"bench": label, "splits": a.splits, "limit": a.limit})
        s = evaluate_predictor(a.predictor, bench, a.splits, a.limit, where, a.workers)
    for sp, m in s["by_split"].items():
        if "@" not in sp:
            print(f"{sp:10s} cascade_f1 {m['cascade_f1']['mean']:.3f}  change_f1 {m['change_f1']['mean']:.3f}  "
                  f"rank_acc {m['rank_acc']['mean']:.3f}  any_violation {m['any_violation']['mean']:.3f}")
    print(f"result: {common.rel(where)}")


if __name__ == "__main__":
    sys.exit(main())
