"""Batched full-length inference and resumable per-model diagnostic artifacts."""
from importlib import import_module

import numpy as np
import torch

from .measurement import audit_summary, audit_trajectory, engine_script, score_trajectory
from .shared import (DRIFT_HORIZON, HORIZON, METRICS, PILOT, VIOLATIONS, common,
                     read_npz, write_json, write_npz)

rollout = import_module(f"{PILOT}.rollout")


def task_inputs(task, bench, formation):
    if task == "intervention":
        n = len(bench["split_ids"])
        scripts = [[common.iv_by_t(bench["ivs"][i][k]).get(t, []) for t in range(HORIZON)]
                   for i in range(n) for k in range(2)]
        return (bench["ctx_fg"], bench["ctx_bg"], scripts, bench["fut_fg"], bench["fut_bg"],
                bench["A"], bench["case_ids"], bench["split_ids"])
    if task == "drift":
        fg, bg = bench["ctx_fg"], bench["ctx_bg"]
        targets_f = np.broadcast_to(fg[:, None, None], (len(fg), 1, DRIFT_HORIZON, *fg.shape[1:]))
        targets_b = np.broadcast_to(bg[:, None, None], targets_f.shape)
        scripts = [[[] for _ in range(DRIFT_HORIZON)] for _ in fg]
        return fg, bg, scripts, targets_f, targets_b, None, bench["case_ids"], bench["split_ids"]
    if task == "formation":
        fg, bg = formation["fg"], formation["bg"]
        return (fg[:, 0], bg[:, 0], formation["actions"], fg[:, None, 1:], bg[:, None, 1:],
                None, formation["episode_ids"], np.full(len(fg), -1, dtype=np.int8))
    raise ValueError(f"Unknown task: {task}")


def predict_batches(model, fg, bg, scripts, horizon, batch, device):
    if batch <= 0 or horizon <= 0 or len(scripts) % len(fg):
        raise ValueError("Invalid batch, horizon or number of scripts")
    branches = len(scripts) // len(fg)
    for a in range(0, len(scripts), batch):
        end = min(len(scripts), a + batch)
        ids = np.arange(a, end) // branches
        initial_f = torch.as_tensor(fg[ids], dtype=torch.long, device=device)
        initial_b = torch.as_tensor(bg[ids], dtype=torch.long, device=device)
        maps = np.stack([[common.encode_actions(step) for step in script] for script in scripts[a:end]])
        maps = torch.as_tensor(maps, dtype=torch.long, device=device)
        if maps.shape[1] != horizon:
            raise ValueError("Every script must align with the requested horizon")
        if horizon == 31:
            predicted = rollout.rollout_script(model, initial_f, initial_b, maps, horizon)
            f, b = predicted["fg"], predicted["bg"]
        else:
            f, b = rollout.rollout_maps(model, initial_f, initial_b, maps, horizon, stop_on_fixed_point=False)
        yield a, f.cpu().numpy().astype(np.uint8), b.cpu().numpy().astype(np.uint8)


def evaluate_task(task, bench, formation, model=None, device="cpu", batch=32,
                  save_trajectories=False, oracle_seed=12345):
    fg0, bg0, scripts, target_f, target_b, masks, ids, splits = task_inputs(task, bench, formation)
    n, branches, ticks = target_f.shape[:3]
    values = {name: np.empty((n, branches, ticks), dtype=np.float64) for name in METRICS}
    values.update({"case_ids": ids, "split_ids": splits, "ticks": np.arange(1, ticks + 1)})
    values["pred_fg_final"] = np.empty((n, branches, *fg0.shape[1:]), dtype=np.uint8)
    values["pred_bg_final"] = np.empty_like(values["pred_fg_final"])
    trajectories = None
    if save_trajectories:
        trajectories = {"fg": np.empty(target_f.shape, dtype=np.uint8),
                        "bg": np.empty(target_b.shape, dtype=np.uint8)}
    if model is None:
        def oracle_batches():
            for index, script in enumerate(scripts):
                i = index // branches
                f, b, _ = engine_script(fg0[i], bg0[i], script, oracle_seed)
                yield index, f[None], b[None]
        predicted = oracle_batches()
    else:
        model.eval()
        predicted = predict_batches(model, fg0, bg0, scripts, ticks, batch, device)
    for start, pf, pb in predicted:
        for offset in range(len(pf)):
            i, k = divmod(start + offset, branches)
            result = score_trajectory(pf[offset], pb[offset], target_f[i, k], target_b[i, k],
                                      fg0[i], bg0[i], None if masks is None else masks[i, k])
            for name in METRICS:
                values[name][i, k] = result[name]
            values["pred_fg_final"][i, k], values["pred_bg_final"][i, k] = pf[offset, -1], pb[offset, -1]
            if trajectories is not None:
                trajectories["fg"][i, k], trajectories["bg"][i, k] = pf[offset], pb[offset]
        if (start + len(pf)) % 128 == 0 or start + len(pf) == len(scripts):
            print(f"[09] {task}: {start + len(pf)}/{len(scripts)} trajectories", flush=True)
    return values, trajectories


def audit_targets(bench, formation):
    truth = audit_trajectory(bench["fut_fg"], bench["fut_bg"], bench["ctx_fg"], bench["ctx_bg"])
    form = audit_trajectory(formation["fg"][:, None, 1:], formation["bg"][:, None, 1:],
                            formation["fg"][:, 0], formation["bg"][:, 0])
    contexts = audit_trajectory(bench["ctx_fg"][:, None, None], bench["ctx_bg"][:, None, None],
                                bench["ctx_fg"], bench["ctx_bg"])
    return {"intervention": truth, "formation": form, "contexts": contexts}, {
        "intervention": audit_summary(truth), "formation": audit_summary(form),
        "contexts": audit_summary(contexts),
        "detector_scope": "check_violations diagnoses rest states. Falling cells may trigger it in valid engine transients.",
        "count_scope": "Category counts may overlap: struct_on_loose is a subset of floating_struct. The sum is not unique cells.",
        "mass_error_scope": "Existing metrics.mass_error compares nonempty material counts across both layers to the same-tick target; legal edits/pruning are not assumed mass-conserving.",
    }


def cached(path, identity):
    marker = path.with_suffix(".json")
    if not path.exists() and not marker.exists():
        return False
    if not marker.is_file():
        return False
    done = common.load_json(marker)
    if done.get("identity") != identity:
        raise ValueError(f"Cached source/protocol differs: {marker}")
    for name, digest in done["files"].items():
        file = path.parent / name
        if not file.is_file() or common.sha1_file(file) != digest:
            raise ValueError(f"Cached result identity changed: {file}")
    return True


def save_task(path, identity, values, trajectories):
    write_npz(path, values)
    files = {path.name: common.sha1_file(path)}
    if trajectories is not None:
        extra = path.with_name(path.stem + "_trajectories.npz")
        write_npz(extra, trajectories)
        files[extra.name] = common.sha1_file(extra)
    write_json(path.with_suffix(".json"), {"identity": identity, "files": files})


def oracle_checks(results, exact_threshold=0.95):
    formation = results["formation"]
    drift = results["drift"]
    return {"formation_exact": float(formation["exact"][..., -1].mean()),
            "formation_exact_threshold": exact_threshold,
            "formation_probe_passed": bool(formation["exact"][..., -1].mean() >= exact_threshold),
            "noop_engine_static": bool(not drift["ever_changed"][..., -1].any()),
            "noop_engine_violation_free": bool(not drift["ever_violation"][..., -1].any()),
            "noop_max_changed_fraction": float(drift["changed_fraction"].max()),
            "formation_is_h1_evidence": False}
