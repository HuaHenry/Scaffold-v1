"""Analysis of experiment 01 (plan 01 section 3, A-E) and the pre-registered checks V1-V5 (section 5).

python -m exp.analysis.pilot_validity [--steps 6000]

Inputs are looked up by logical id inside the CURRENT run folder (FW_RUN, else the newest run; the newest finished
result with the matching settings wins):
  runs      state_s0..s5, state_n64_s0..2, state_n16_s0..2   (kind=train, settings.steps == --steps, with eval_samples.npz)
  baselines copy_last, apply_edit, random_physics, oracle_reseed  (kind=baseline, same bench, all splits, no --limit)
  check E   compose_process, compose_shuffle
Output: a NEW folder <results root>/01_pilot_validity/<run>/analysis/pilot_validity_<settings>/ with
pilot_validity.json, pilot_validity.md and the list of input folders that were used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .. import aggregate as agg
from .. import common
from ..common import SPLITS
from ..metrics import sample_level

METRICS = ("cascade_f1", "change_f1", "rank_acc")
BASELINES = ("copy_last", "apply_edit", "random_physics", "oracle_reseed")
SEEDS = [0, 1, 2, 3, 4, 5]
EXPECTED_FORMATION_SHARE = 31 / 59
CASCADE_KEY = "cascade_f1"       # or "cascade_f1_nonempty": ignore interventions without any true cascade


def _load_npz(path) -> dict:
    z = np.load(path)
    d = {k: z[k] for k in z.files}
    if CASCADE_KEY != "cascade_f1":
        d["cascade_f1"] = d[CASCADE_KEY]
    return d


STEPS = 6000                   # training steps of the runs to analyse (set from --steps)
INPUTS: dict = {}              # logical id -> result folder name that was used (written to the output folder)


def run_id(n: int, seed: int) -> str:
    return f"state_s{seed}" if n == 256 else f"state_n{n}_s{seed}"


def this_run():
    """The analysis reads only the results of the current run folder (a re-run never picks up old results)."""
    return common.current_run(common.EXP_NAME, create=False)


def find_run(n: int, seed: int):
    return common.find_latest(run_id(n, seed), "train", common.EXP_NAME, "eval_samples.npz", {"steps": STEPS}, this_run())


def find_baseline(name: str):
    return common.find_latest(name, "baseline", common.EXP_NAME, "eval_samples.npz",
                              {"bench": common.bench_label(common.DEFAULT_BENCH), "splits": "all", "limit": None},
                              this_run())


def load_runs(n: int, seeds) -> dict:
    out = {}
    for s in seeds:
        d = find_run(n, s)
        if d is not None:
            out[s] = _load_npz(d / "eval_samples.npz")
            INPUTS[run_id(n, s)] = d.name
    return out


def metric_matrix(runs: dict, metric: str, sel: np.ndarray) -> np.ndarray:
    """[n_seeds, n_selected] sample-level metric."""
    return np.stack([sample_level(s, metric)[sel] for s in runs.values()])


def fmt(m: dict, d=3) -> str:
    return f"{m['mean']:.{d}f} [{m['lo']:.{d}f}, {m['hi']:.{d}f}]"


def analyze(split_ids: np.ndarray, boot: int = agg.N_BOOT) -> dict:
    sels = {sp: split_ids == i for i, sp in enumerate(SPLITS)}
    sels["all"] = np.ones_like(split_ids, dtype=bool)
    INPUTS.clear()
    R: dict = {"missing": []}

    base = {}
    for b in BASELINES:
        d = find_baseline(b)
        if d is not None:
            base[b] = _load_npz(d / "eval_samples.npz")
            INPUTS[b] = d.name
        else:
            R["missing"].append(f"baseline {b} (run: python -m exp.evaluate --predictor {b})")
    r256 = load_runs(256, SEEDS)
    r64, r16 = load_runs(64, (0, 1, 2)), load_runs(16, (0, 1, 2))
    for name, r, want in (("n256", r256, 6), ("n64", r64, 3), ("n16", r16, 3)):
        if len(r) < want:
            R["missing"].append(f"{name}: {len(r)}/{want} evaluated runs")

    # ---- A: ordering
    A: dict = {}
    for sp, sel in sels.items():
        A[sp] = {}
        for m in METRICS:
            row = {}
            for b, s in base.items():
                row[b] = agg.boot_mean(sample_level(s, m)[sel], boot)
            if r256:
                row["state"] = agg.boot_mean(agg.seed_mean(list(r256.values()), m)[sel], boot)
            A[sp][m] = row
    R["A_ordering"] = A

    # ---- B: positive control (seeds 0-2 of each size)
    B: dict = {}
    r3 = {s: v for s, v in r256.items() if s in (0, 1, 2)}
    sizes = {256: r3, 64: r64, 16: r16}
    pairs = (("n256-n64", 256, 64, 2), ("n64-n16", 64, 16, 2), ("n256-n16", 256, 16, 4))
    for sp, sel in sels.items():
        B[sp] = {}
        for m in METRICS:
            B[sp][m] = {}
            for name, hi, lo, doublings in pairs:
                if sizes[hi] and sizes[lo]:
                    a_ = agg.seed_mean(list(sizes[hi].values()), m)[sel]
                    b_ = agg.seed_mean(list(sizes[lo].values()), m)[sel]
                    r = agg.paired_bootstrap(a_, b_, boot)
                    r["points_per_doubling"] = 100 * r["diff"] / doublings
                    B[sp][m][name] = r
    R["B_positive_control"] = B

    # ---- C: A/A (6 seeds of n256, all 10 splits into 3 + 3)
    parts = agg.aa_partitions(SEEDS, 3)
    R["C_aa"] = {"partitions": [[list(a_), list(b_)] for a_, b_ in parts], "n_partitions": len(parts)}
    D: dict = {}
    if len(r256) == 6:
        C = {}
        for sp, sel in sels.items():
            C[sp] = {}
            for m in METRICS:
                M = metric_matrix(r256, m, sel)                         # [6, n]
                paired, seedlvl = [], []
                for ga, gb in parts:
                    a_, b_ = M[list(ga)], M[list(gb)]
                    with np.errstate(all="ignore"):
                        paired.append(agg.paired_bootstrap(np.nanmean(a_, 0), np.nanmean(b_, 0), boot))
                        seedlvl.append(agg.seed_level_bootstrap(a_, b_, min(boot, 2000)))
                C[sp][m] = {"paired_reject_rate": float(np.mean([p["excludes_0"] for p in paired])),
                            "seed_level_reject_rate": float(np.mean([p["excludes_0"] for p in seedlvl])),
                            "mean_abs_diff": float(np.mean([abs(p["diff"]) for p in paired])),
                            "se_boot": float(np.mean([p["se"] for p in paired])),
                            "se_seed_level": float(np.mean([p["se"] for p in seedlvl]))}
                seed_means = np.nanmean(M, axis=1)
                D.setdefault(sp, {})[m] = {"sigma_seed": float(np.std(seed_means, ddof=1)),
                                           "seed_means": seed_means.tolist(),
                                           "se_boot": C[sp][m]["se_boot"],
                                           "se_seed_level": C[sp][m]["se_seed_level"]}
                d = D[sp][m]
                d["mde"] = 2.8 * float(np.sqrt(d["se_boot"] ** 2 + 2 * d["sigma_seed"] ** 2 / 3))
                d["mde_seed_level"] = 2.8 * d["se_seed_level"]
        R["C_aa"]["by_split"] = C
    R["D_noise_mde"] = D

    # ---- E: formation share per batch
    R["E_formation_share"] = formation_share()
    R["checks"], R["recommendation"] = checks(R)
    R["inputs"] = dict(INPUTS)
    return R


def formation_share() -> dict:
    out = {"expected": EXPECTED_FORMATION_SHARE}
    for c in ("process", "shuffle"):
        d = common.find_latest(f"compose_{c}", "train", common.EXP_NAME, "done.json", None, this_run())
        log = d / "train_log.jsonl" if d else None
        if log is not None and log.exists():
            INPUTS[f"compose_{c}"] = d.name
            v = [json.loads(l)["form_frac"] for l in log.read_text().splitlines() if '"form_frac"' in l]
            out[c] = {"source": "train_log", "mean": float(np.mean(v)), "n_logs": len(v)}
        else:
            try:
                d = np.load(common.DEFAULT_DATA / f"train_{c}.npz")
                p = common.make_pairs(common.DEFAULT_DATA, c, "video")
                out[c] = {"source": "pairs (no training log found; run pilot_compose for the real check)",
                          "mean": float((d["phase"][p[:, 0], p[:, 1]] == 0).mean())}
            except Exception as e:                                       # data not available
                out[c] = {"source": f"unavailable: {e}"}
    return out


def checks(R: dict):
    V: dict = {}
    A, B, C, D = R["A_ordering"], R["B_positive_control"], R.get("C_aa", {}).get("by_split"), R["D_noise_mde"]
    # V1
    a = A["id"]["cascade_f1"]
    if all(k in a for k in ("random_physics", "apply_edit", "state", "oracle_reseed")):
        m = {k: v["mean"] for k, v in a.items()}
        c1 = m["random_physics"] < m["apply_edit"] + 0.05
        c2 = m["state"] - m["apply_edit"] >= 0.15
        c3 = m["state"] < m["oracle_reseed"]
        V["V1"] = {"pass": bool(c1 and c2 and c3), "random_lt_apply+0.05": bool(c1), "state-apply>=0.15": bool(c2),
                   "state<oracle": bool(c3), "means": m,
                   "on_fail": "back to 00 section 13 item 4; do not start the main experiment"}
    else:
        V["V1"] = {"pass": None, "note": "missing baselines or state runs"}
    # V2
    b = B["id"]["cascade_f1"]
    if all(k in b for k in ("n256-n16", "n256-n64", "n64-n16")):
        ok = b["n256-n16"]["excludes_0"] and b["n256-n16"]["diff"] > 0 and (
            (b["n256-n64"]["excludes_0"] and b["n256-n64"]["diff"] > 0) or
            (b["n64-n16"]["excludes_0"] and b["n64-n16"]["diff"] > 0))
        V["V2"] = {"pass": bool(ok), "diffs": {k: [v["diff"], v["lo"], v["hi"]] for k, v in b.items()},
                   "on_fail": "bench insensitive to more data: check if the bench is too easy, enlarge it; pause main"}
    else:
        V["V2"] = {"pass": None, "note": "missing n64 / n16 / n256 runs"}
    # V3, V4
    if C and "id" in C:
        rate = C["id"]["cascade_f1"]["paired_reject_rate"]
        V["V3"] = {"pass": bool(rate <= 0.20), "paired_reject_rate": rate,
                   "seed_level_reject_rate": C["id"]["cascade_f1"]["seed_level_reject_rate"],
                   "on_fail": "report a seed-level bootstrap in the main experiment, or use 5 seeds"}
        d = D["id"]["cascade_f1"]
        V["V4"] = {"pass": bool(d["sigma_seed"] <= 0.02 and d["mde"] <= 0.04), "sigma_seed": d["sigma_seed"],
                   "mde": d["mde"], "mde_seed_level": d["mde_seed_level"],
                   "on_fail": "see the MDE decision table"}
    else:
        V["V3"] = {"pass": None, "note": "need all 6 state seeds at n256"}
        V["V4"] = {"pass": None, "note": "need all 6 state seeds at n256"}
    # V5
    E = R["E_formation_share"]
    vals = [E[c]["mean"] for c in ("process", "shuffle") if "mean" in E.get(c, {})]
    real = all(E.get(c, {}).get("source") == "train_log" for c in ("process", "shuffle"))
    if vals:
        V["V5"] = {"pass": bool(all(abs(v - EXPECTED_FORMATION_SHARE) <= 0.03 for v in vals)),
                   "shares": {c: E[c].get("mean") for c in ("process", "shuffle")},
                   "provisional": not real, "on_fail": "make_pairs or the sampler has a bug; fix it"}
    else:
        V["V5"] = {"pass": None, "note": "no data"}
    # recommendation
    rec = "need the six n256 state runs before a recommendation"
    if "id" in D:
        mde = D["id"]["cascade_f1"]["mde"]
        if mde <= 0.04:
            rec = f"MDE(id)={mde:.3f} <= 0.04: 3 seeds are enough (as planned)"
        elif mde <= 0.08:
            rec = (f"MDE(id)={mde:.3f} in 0.04-0.08: use 5 seeds and generate a second bench "
                   f"(python run.py --data-v3 --seed 1 --out output_b1; evaluate with --bench) as confirmation set")
        else:
            rec = (f"MDE(id)={mde:.3f} > 0.08: pause; enlarge the bench or reduce evaluation variance "
                   f"(several oracle seeds per intervention), then redo this experiment")
        if V.get("V3", {}).get("pass") is False:
            rec += "; the paired bootstrap underestimates training noise: report the seed-level bootstrap"
    return V, rec


def table(R: dict) -> str:
    L = ["# Appendix table A1 - benchmark validity and noise floor (experiment 01)", ""]
    if R["missing"]:
        L += ["> missing inputs: " + "; ".join(R["missing"]), ""]
    for m in METRICS:
        L += [f"## A. Ordering - {m} (mean [95% CI], samples = both interventions)", "",
              "| method | " + " | ".join(SPLITS) + " |", "|---|" + "---|" * len(SPLITS)]
        for meth in (*BASELINES[:3], "state", BASELINES[3]):
            cells = [fmt(R["A_ordering"][sp][m][meth]) if meth in R["A_ordering"][sp][m] else "n/a" for sp in SPLITS]
            L.append(f"| {meth} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## B. Positive control - cascade_f1 difference (CI) and points per doubling", "",
          "| split | n256-n64 | n64-n16 | n256-n16 | per doubling (n256-n16) |", "|---|---|---|---|---|"]
    for sp in SPLITS:
        b = R["B_positive_control"][sp]["cascade_f1"]
        cells = [f"{b[k]['diff']:+.3f} [{b[k]['lo']:+.3f}, {b[k]['hi']:+.3f}]" if k in b else "n/a"
                 for k in ("n256-n64", "n64-n16", "n256-n16")]
        ppd = f"{b['n256-n16']['points_per_doubling']:.1f} pts" if "n256-n16" in b else "n/a"
        L.append(f"| {sp} | " + " | ".join(cells) + f" | {ppd} |")
    L += ["", "## C/D. A/A test, noise floor, MDE (cascade_f1)", "",
          "| split | sigma_seed | SE_boot | MDE | A/A reject (paired) | A/A reject (seed-level) | MDE (seed-level) |",
          "|---|---|---|---|---|---|---|"]
    for sp in SPLITS:
        d = R["D_noise_mde"].get(sp, {}).get("cascade_f1")
        c = R.get("C_aa", {}).get("by_split", {}).get(sp, {}).get("cascade_f1")
        if d and c:
            L.append(f"| {sp} | {d['sigma_seed']:.4f} | {d['se_boot']:.4f} | {d['mde']:.4f} | "
                     f"{c['paired_reject_rate']:.0%} | {c['seed_level_reject_rate']:.0%} | {d['mde_seed_level']:.4f} |")
        else:
            L.append(f"| {sp} | n/a | n/a | n/a | n/a | n/a | n/a |")
    E = R["E_formation_share"]
    L += ["", "## E. Formation share per batch", "",
          f"expected {E['expected']:.3f}; process {E.get('process', {}).get('mean', float('nan')):.3f} "
          f"({E.get('process', {}).get('source')}); shuffle {E.get('shuffle', {}).get('mean', float('nan')):.3f}",
          "", "## Pre-registered checks", "", "| check | result | detail |", "|---|---|---|"]
    for k, v in R["checks"].items():
        res = {True: "PASS", False: "FAIL", None: "n/a"}[v["pass"]]
        det = {kk: vv for kk, vv in v.items() if kk not in ("pass", "on_fail")}
        L.append(f"| {k} | {res} | {json.dumps(det, default=float)[:200]} |")
    L += ["", f"**Recommendation:** {R['recommendation']}", ""]
    return "\n".join(L)


def main(argv=None):
    global CASCADE_KEY, STEPS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=6000, help="analyse runs trained with this many steps")
    ap.add_argument("--n-boot", type=int, default=agg.N_BOOT)
    ap.add_argument("--cascade-metric", choices=("cascade_f1", "cascade_f1_nonempty"), default="cascade_f1",
                    help="cascade_f1 scores 1.0 when both prediction and truth have no cascade (plan 00 section 6)")
    ap.add_argument("--out", type=Path, default=None, help="result folder (default: <run>/analysis/...)")
    a = ap.parse_args(argv)
    CASCADE_KEY, STEPS = a.cascade_metric, a.steps
    ref = find_baseline("copy_last") or find_run(256, 0)
    if ref is None:
        raise SystemExit(f"no evaluation outputs found in run {this_run()!r} under {common.rel(common.RESULTS_ROOT)} "
                         f"(run exp.evaluate for the baselines and the runs first)")
    split_ids = np.load(ref / "eval_samples.npz")["split_ids"]
    R = analyze(split_ids, a.n_boot)
    out = common.resolve_out(a.out) if a.out else common.new_result_dir(
        common.EXP_NAME, "analysis", "pilot_validity", f"steps{a.steps}_boot{a.n_boot}_{a.cascade_metric}")
    out.mkdir(parents=True, exist_ok=True)
    common.write_meta(out, "analysis", "pilot_validity", {"steps": a.steps, "n_boot": a.n_boot,
                                                          "cascade_metric": a.cascade_metric}, inputs=R["inputs"])
    common.dump_json(R, out / "pilot_validity.json")
    (out / "pilot_validity.md").write_text(table(R), encoding="utf-8")
    print(table(R))
    print(f"\nresult: {common.rel(out)}\n  pilot_validity.json  pilot_validity.md")


if __name__ == "__main__":
    main()
