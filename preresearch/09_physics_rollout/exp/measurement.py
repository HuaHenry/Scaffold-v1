"""Per-tick diagnostics; the existing violation detector describes states at rest."""
import numpy as np

from .shared import FORMATION_HORIZON, HORIZON, METRICS, VIOLATIONS, common, metrics
import datasets as D
import datasets_v3 as v3
from core import FormWorld, check_violations


def score_trajectory(fg, bg, target_fg, target_bg, initial_fg, initial_bg, action_mask=None):
    if fg.shape != bg.shape or target_fg.shape != fg.shape or target_bg.shape != fg.shape or fg.ndim != 3:
        raise ValueError("Prediction and target must be aligned [ticks,height,width] grids")
    if initial_fg.shape != fg.shape[1:] or initial_bg.shape != initial_fg.shape:
        raise ValueError("Initial grid shape differs from trajectory")
    if any(np.any((grid < 0) | (grid >= common.N_MAT)) for grid in (fg, bg, target_fg, target_bg)):
        raise ValueError("Unknown material identifier")
    mask = np.zeros_like(initial_fg, dtype=bool) if action_mask is None else action_mask
    result = {name: np.empty(len(fg), dtype=np.float64) for name in METRICS}
    ever_changed = ever_violation = False
    initial = (initial_fg, initial_bg)
    for t in range(len(fg)):
        predicted, target = (fg[t], bg[t]), (target_fg[t], target_bg[t])
        score = metrics.score_sample(predicted, target, initial, mask, violations=False)
        counts = check_violations(fg[t].copy(), bg[t].copy())
        score.update({key: float(counts[key]) for key in VIOLATIONS})
        score["violations"] = float(sum(counts.values()))
        score["any_violation"] = float(any(counts.values()))
        score["cell_error"] = 1 - score["cell_acc_all"]
        score["changed_fraction"] = float(metrics.changed(predicted, initial).mean())
        score["state_changed"] = float(metrics.changed(predicted, initial).any())
        ever_changed |= bool(score["state_changed"])
        ever_violation |= bool(score["any_violation"])
        score["ever_changed"], score["ever_violation"] = float(ever_changed), float(ever_violation)
        for name in METRICS:
            result[name][t] = score[name]
    return result


def engine_script(fg0, bg0, actions, seed=12345):
    world = FormWorld(seed=seed, grid=fg0.copy(), background=bg0.copy())
    fg, bg, applied, attempted = [], [], 0, 0
    for step in actions:
        for action in step:
            attempted += 1
            applied += int(world.apply_action(v3.to_action(action)))
        world.tick()
        world.prune_background()
        fg.append(world.grid.copy())
        bg.append(world.background.copy())
    return np.stack(fg), np.stack(bg), {"attempted_actions": attempted, "applied_actions": applied}


def load_formation(val, limit=None):
    with np.load(val / "train_process.npz", allow_pickle=False) as stream:
        fg, bg = stream["fg"], stream["bg"]
        phase, continuous = stream["phase"], stream["cont"]
        n = len(fg) if limit is None else min(limit, len(fg))
        if (n == 0 or fg.shape != bg.shape or fg.ndim != 4 or fg.shape[1] < HORIZON
                or fg.shape[2:] != (common.H, common.W)):
            raise ValueError("Invalid process validation grids")
        if phase.shape != fg.shape[:2] or continuous.shape != phase.shape:
            raise ValueError("Invalid formation phase or continuity shape")
        if not np.all(phase[:n, :HORIZON] == 0) or not np.all(continuous[:n, 1:HORIZON]):
            raise ValueError("Formation must be a continuous chronological process prefix")
        fg, bg = fg[:n, :HORIZON].copy(), bg[:n, :HORIZON].copy()
    source = common.load_json(val / "train_process_actions.json")
    if source.get("condition") != "process":
        raise ValueError("Only the shared ordered Process validation scripts may be used")
    actions = source["actions"][:n]
    if len(actions) != n or any(len(row) < HORIZON or row[0] for row in actions):
        raise ValueError("Formation frame 0 must have no action; destination frames carry actions")
    if not np.all(fg[:, 0] == D.terrain()) or np.any(bg[:, 0]):
        raise ValueError("Formation frame 0 must be empty dataset terrain")
    return {"fg": fg, "bg": bg, "actions": [row[1:HORIZON] for row in actions],
            "episode_ids": np.arange(n), "horizon": FORMATION_HORIZON}


def audit_trajectory(fg, bg, initial_fg, initial_bg):
    n, branches, ticks = fg.shape[:3]
    arrays = {name: np.zeros((n, branches, ticks), dtype=np.int32) for name in VIOLATIONS}
    arrays["changed_fraction"] = np.zeros((n, branches, ticks), dtype=np.float64)
    for i in range(n):
        for k in range(branches):
            for t in range(ticks):
                counts = check_violations(fg[i, k, t].copy(), bg[i, k, t].copy())
                for name in VIOLATIONS:
                    arrays[name][i, k, t] = counts[name]
                arrays["changed_fraction"][i, k, t] = metrics.changed(
                    (fg[i, k, t], bg[i, k, t]), (initial_fg[i], initial_bg[i])).mean()
    arrays["any_violation"] = np.maximum.reduce([arrays[name] > 0 for name in VIOLATIONS])
    return arrays


def audit_summary(arrays):
    counts = {name: int(arrays[name].sum()) for name in VIOLATIONS}
    return {"counts": counts, "all_ticks_zero": not any(counts.values()),
            "final_tick_zero": not bool(arrays["any_violation"][..., -1].any()),
            "any_violation_by_tick": arrays["any_violation"].mean(axis=(0, 1)).tolist(),
            "max_changed_fraction": float(arrays["changed_fraction"].max()),
            "final_changed_fraction": float(arrays["changed_fraction"][..., -1].mean())}
