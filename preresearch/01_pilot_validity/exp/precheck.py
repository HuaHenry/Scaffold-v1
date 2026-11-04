"""Pre-flight checks (plan 00 section 13, items 1-6) that experiment 01 depends on.

python -m exp.precheck --items 1,2,5                      # no trained model needed
python -m exp.precheck --items 3,4,6 --run state_s0       # needs a finished 6000-step state run + baselines
Item 7 (the one-off tuning) is `make_jobs tune` + `python -m exp.analysis.freeze`.
Each call writes a new folder <results root>/00_common/<timestamp>_precheck_items<...>/precheck.json.
Exit code is non-zero if any selected check fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import common
from .common import resolve

TYPES = ("place", "remove", "interact")


def decode_token(tok: int) -> dict:
    v = tok - 1
    layer, rest = v % 2, v // 2
    return {"type": TYPES[rest // common.N_MAT], "material": rest % common.N_MAT, "layer": "back" if layer else "front"}


def check_1(data: Path) -> dict:
    res = {}
    # encoding round trip on real actions
    acts = common.load_condition(data, "process", 16)["actions"]
    bad = n = 0
    for ep in acts:
        for frame in ep:
            m = common.encode_actions(frame)
            last = {}
            for a in frame:
                if a["type"] in common.TYPE_IDX:
                    last[(a["x"], a["y"])] = a
            for (x, y), a in last.items():
                d = decode_token(int(m[y, x]))
                want_m = a["material"] if a["type"] == "place" else 0
                n += 1
                bad += int(d["type"] != a["type"] or d["material"] != want_m or d["layer"] != a["layer"])
            bad += int((m != 0).sum() != len(last))
    res["roundtrip"] = {"n": n, "bad": bad, "pass": bad == 0}
    # transition counts
    summ = common.load_summary(data)
    lay = summ["layout"]
    E = summ["conditions"]["state"]["episodes"]
    T = summ["conditions"]["state"]["frames_per_episode"]
    F, clip = lay["formation"], lay["clip"]
    exp_video = {"state": E * (T // clip) * (clip - 1), "partial": E * (T // clip) * (clip - 1),
                 "process": E * ((F - 1) + ((T - F) // clip) * (clip - 1)), "shuffle": E * ((F - 1) + ((T - F) // clip) * (clip - 1))}
    pairs = {}
    for c in common.CONDITIONS:
        pv = len(common.make_pairs(data, c, "video"))
        pc = len(common.make_pairs(data, c, "causal"))
        pairs[c] = {"video": pv, "expected_video": exp_video[c], "causal": pc,
                    "pass": pv == exp_video[c] and pc == int(np.load(data / f"train_{c}.npz")["cont"].sum())}
    if (E, T, F, clip) == (256, 64, 32, 8):                            # numbers printed in plan 00 section 5
        plan = {"state": (14336, 14336), "partial": (14336, 14336), "process": (15104, 15104), "shuffle": (15104, 7168)}
        for c, (v, k) in plan.items():
            pairs[c]["plan_numbers"] = [v, k]
            pairs[c]["pass"] &= pairs[c]["video"] == v and pairs[c]["causal"] == k
    res["pairs"] = pairs
    res["pass"] = res["roundtrip"]["pass"] and all(p["pass"] for p in pairs.values())
    return res


def _train(args, out):
    from . import train
    return train.main(args + ["--out", str(out)])


def check_2(data: Path, device: str, small: bool, work: Path) -> dict:
    extra = ["--depth", "2", "--dim", "96", "--heads", "2"] if small else []
    done = _train(["--condition", "state", "--episodes", "1", "--steps", "500", "--val-every", "0",
                   "--data", str(data), "--device", device, *extra], work / "check2_overfit")
    acc = done["train_raw"]["chg_acc"]
    return {"changed_cell_acc": acc, "threshold": 0.99, "pass": acc >= 0.99}


def check_5(data: Path, device: str, small: bool, work: Path) -> dict:
    logs = []
    extra = ["--depth", "2", "--dim", "96", "--heads", "2"] if small else []
    for i in range(2):
        _train(["--condition", "state", "--seed", "0", "--steps", "200", "--log-every", "10", "--val-every", "0",
                "--data", str(data), "--device", device, *extra], work / f"check5_determinism_{i}")
        logs.append([json.loads(l)["loss"] for l in
                     (work / f"check5_determinism_{i}" / "train_log.jsonl").read_text().splitlines()])
    diff = float(np.max(np.abs(np.array(logs[0]) - np.array(logs[1]))))
    return {"max_loss_diff_first_200": diff, "threshold": 1e-4, "pass": diff < 1e-4,
            "note": "bf16 on GPU may exceed 1e-4 through non-deterministic kernels; see plan 00 section 11"}


def check_3(run: Path, bench_path: Path, device: str) -> dict:
    import torch
    from .evaluate import load_run
    from .rollout import rollout_maps
    from .train import pick_device
    dev = pick_device(device)
    model, _ = load_run(run, dev)
    b = common.load_bench(bench_path)
    fg = torch.from_numpy(b["ctx_fg"]).long().to(dev)
    bg = torch.from_numpy(b["ctx_bg"]).long().to(dev)
    zero = torch.zeros((fg.shape[0], 1, common.H, common.W), dtype=torch.long, device=dev)
    f, g = [], []
    for i in range(0, len(fg), 256):
        o = rollout_maps(model, fg[i:i + 256], bg[i:i + 256], zero[i:i + 256], 32, stop_on_fixed_point=False)
        f.append(o[0][:, -1]); g.append(o[1][:, -1])
    f, g = torch.cat(f), torch.cat(g)
    same = ((f == fg).all(2).all(1) & (g == bg).all(2).all(1)).float().mean().item()
    return {"unchanged_fraction": same, "threshold": 0.99, "pass": same >= 0.99}


def check_4(run: Path) -> dict:
    ev = common.load_json(run / "eval.json")
    bp = common.find_latest("apply_edit", "baseline", common.EXP_NAME, "eval.json",
                            {"bench": common.bench_label(common.DEFAULT_BENCH), "splits": "all", "limit": None},
                            common.current_run(common.EXP_NAME, create=False))
    if bp is None:
        return {"pass": None, "note": "run: python -m exp.evaluate --predictor apply_edit (in the same run)"}
    state = ev["by_split"]["id"]["cascade_f1"]["mean"]
    base = common.load_json(bp / "eval.json")["by_split"]["id"]["cascade_f1"]["mean"]
    status = "ok (> 0.3)" if state > 0.3 else ("weak (0.15-0.3)" if state >= 0.15 else "STOP (< 0.15): check loss weight, action encoding, lr")
    return {"state_cascade_f1_id": state, "apply_edit_cascade_f1_id": base, "gain": state - base, "status": status,
            "pass": bool(state >= 0.15 and state - base >= 0.15)}


def check_6(run: Path) -> dict:
    """Drift: `exact` of final_fp (model's own fixed point) vs final_32, from the finished evaluation of the run."""
    ev = common.load_json(run / "eval.json")["overall"]
    fp, f32 = ev["exact"]["mean"], ev["exact__32"]["mean"]
    return {"exact_final_fp": fp, "exact_final_32": f32, "gap": fp - f32, "pass": abs(fp - f32) < 0.05,
            "on_fail": "add --unroll 2 for ALL conditions and redo item 4"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", default="1", help="comma list from 1..6")
    ap.add_argument("--data", type=Path, default=common.DEFAULT_DATA)
    ap.add_argument("--bench", type=Path, default=common.DEFAULT_BENCH)
    ap.add_argument("--run", default="state_s0", help="trained state run (id or folder) for items 3,4,6")
    ap.add_argument("--steps", type=int, default=6000, help="when --run is an id: training steps of the run to pick (0 = any)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--small", action="store_true", help="use a tiny model for items 2 and 5 (smoke test only)")
    ap.add_argument("--out", type=Path, default=None, help="result folder (default: <run>/precheck/<items>)")
    a = ap.parse_args(argv)
    items = {int(x) for x in a.items.split(",")}
    data, bench = resolve(a.data), resolve(a.bench)
    run = common.resolve_run(a.run, steps=a.steps) if items & {3, 4, 6} else None
    out_dir = common.resolve_out(a.out) if a.out else common.new_result_dir(
        common.EXP_NAME, "precheck", "items" + "-".join(map(str, sorted(items))),
        ("small_" if a.small else "") + (a.device if a.device != "auto" else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    common.write_meta(out_dir, "precheck", "items" + "-".join(map(str, sorted(items))), {"items": sorted(items), "small": a.small,
                                                                            "run": common.rel(run) if run else None})
    res = {}
    if 1 in items:
        res["1_encoding_and_pairs"] = check_1(data)
    if 2 in items:
        res["2_overfit_one_episode"] = check_2(data, a.device, a.small, out_dir)
    if 5 in items:
        res["5_determinism"] = check_5(data, a.device, a.small, out_dir)
    for i, fn in ((3, lambda: check_3(run, bench, a.device)), (4, lambda: check_4(run)),
                  (6, lambda: check_6(run))):
        if i in items:
            res[{3: "3_stability", 4: "4_beats_apply_edit", 6: "6_drift"}[i]] = fn()
    common.dump_json(res, out_dir / "precheck.json")
    print(f"result: {common.rel(out_dir)}")
    ok = True
    for k, v in sorted(res.items()):
        flag = {True: "PASS", False: "FAIL", None: "n/a"}[v.get("pass")]
        print(f"[{flag}] {k}: {json.dumps({kk: vv for kk, vv in v.items() if kk != 'pass'}, default=float)[:300]}")
        ok &= v.get("pass") is not False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
