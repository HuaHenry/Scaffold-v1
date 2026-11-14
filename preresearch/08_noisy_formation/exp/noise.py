"""State-aware action mistakes, 27 real build ticks and four real settling ticks."""
from copy import copy
from importlib import import_module

import numpy as np

from .shared import common

D = import_module("datasets")
core = import_module("core")
v3 = import_module("datasets_v3")


def execute(world, actions, strict=True):
    seen, applied = set(), []
    for a in actions:
        key = (a["x"], a["y"])
        if key in seen:
            raise ValueError("same-cell actions would overwrite the encoded token")
        seen.add(key)
        if not world.apply_action(v3.to_action(a)):
            if strict:
                raise ValueError("invalid noisy action")
            continue
        applied.append(dict(a))
    return finish_tick(world, applied)


def finish_tick(world, applied):
    before_f, before_b = world.grid.copy(), world.background.copy()
    world.tick()
    world.prune_background()
    changed = (before_f != world.grid) | (before_b != world.background)
    fallen = np.isin(before_f, list(core.STRUCT) + [core.SAND]) & (before_f != world.grid)
    return world.grid.copy(), world.background.copy(), applied, int(changed.sum()), int(fallen.sum())


def replay(fg, bg, actions, world_seed):
    world = D.new_world(world_seed)
    if not np.array_equal(world.grid, fg[0]) or not np.array_equal(world.background, bg[0]):
        raise ValueError("source initial terrain differs")
    physics, falling = 0, 0
    for t in range(1, 32):
        f, b, _, changed, falls = execute(world, actions[t])
        if not np.array_equal(f, fg[t]) or not np.array_equal(b, bg[t]):
            raise ValueError(f"source replay differs at frame {t}")
        physics += int(changed > 0)
        falling += falls
    if actions[0] or any(actions[t] for t in range(28, 32)):
        raise ValueError("formation must start at terrain and end with four no-action ticks")
    return {"physical_transitions": physics, "falling_cell_events": falling,
            "remove_actions": sum(a["type"] == "remove" for fr in actions[:32] for a in fr),
            "chunk_sizes": [len(fr) for fr in actions[1:28]],
            "actions": sum(len(fr) for fr in actions[:32])}


def repair_actions(world, fg, bg):
    result, used = [], set()

    def attempt(a):
        key = (a["x"], a["y"])
        if key in used:
            return False
        if a["type"] == "place" and a.get("layer", "front") == "front":
            trial = copy(world)
            trial.grid, trial.background = world.grid.copy(), world.background.copy()
            if not trial.apply_action(v3.to_action(a)):
                return False
            if a["material"] in core.LOAD and not trial.supported()[a["y"], a["x"]]:
                return False
            if a["material"] == core.LEAF and key not in trial.attached_leaves():
                return False
        if not world.apply_action(v3.to_action(a)):
            return False
        used.add(key)
        result.append(a)
        return True

    for layer, actual, target in (("front", world.grid, fg), ("back", world.background, bg)):
        for y, x in np.argwhere(actual != target):
            if actual[y, x] != core.EMPTY and actual[y, x] != target[y, x]:
                attempt(v3._a("remove", x, y, layer=layer))
    for layer, actual, target in (("front", world.grid, fg), ("back", world.background, bg)):
        while True:
            changed = False
            for y, x in sorted(np.argwhere((actual == core.EMPTY) & (target != core.EMPTY)),
                               key=lambda cell: (-cell[0], cell[1])):
                if actual[y, x] == core.EMPTY:
                    changed |= attempt(v3._a("place", x, y, int(target[y, x]), layer))
            if not changed:
                break
    return result


def derive(fg, bg, actions, world_seed, p, noise_seed=0, eligible=True, _baseline=None):
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("noise probability must be finite in [0,1]")
    baseline = replay(fg, bg, actions, world_seed) if _baseline is None else {
        k: _baseline[k] for k in ("physical_transitions", "falling_cell_events", "remove_actions", "chunk_sizes", "actions")}
    meta = {"probability": p, "eligible": eligible, "triggered": 0, "delay": 0, "offset": 0, "omit": 0,
            "candidate_actions": 0, "repair_actions": 0, "fallback": False, "fallback_reason": None,
            "effective_episode": False, "successful_offset_actions": 0, "skipped_invalid_actions": 0}
    if p == 0 or not eligible:
        return fg[:32].copy(), bg[:32].copy(), [list(fr) for fr in actions[:32]], {**meta, **baseline}
    rng = D.rng_for("experiment08", noise_seed, world_seed, str(p))
    world = D.new_world(world_seed)
    frames_f, frames_b, out = [world.grid.copy()], [world.background.copy()], [[]]
    pending, physics, falling = {}, 0, 0
    try:
        for t in range(1, 28):
            scheduled, used, wrong_actions = [], set(), []
            if t <= 20:
                candidates = list(pending.pop(t, [])) + [dict(a) for a in actions[t]]
                for a in candidates:
                    key = (a["x"], a["y"])
                    if key in used:
                        if t < 20:
                            pending.setdefault(t + 1, []).append(a)
                        continue
                    is_original = a in actions[t]
                    if is_original and a["type"] == "place" and t <= 18:
                        meta["candidate_actions"] += 1
                        if rng.random() < p:
                            kind = rng.choice(["delay", "offset", "omit"])
                            if kind in ("delay", "omit"):
                                distance = rng.randint(1, 6) if kind == "delay" else rng.randint(10, 20)
                                pending.setdefault(min(20, t + distance), []).append(a)
                                meta[kind] += 1
                                meta["triggered"] += 1
                                continue
                            positions = [(a["x"] + dx, a["y"] + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
                            rng.shuffle(positions)
                            grid = world.grid if a.get("layer", "front") == "front" else world.background
                            for x, y in positions:
                                if 0 <= x < D.DW and 0 <= y < D.DH and grid[y, x] == 0 and (x, y) not in used:
                                    wrong = {**a, "x": x, "y": y}
                                    scheduled.append(wrong)
                                    wrong_actions.append(wrong)
                                    used.add((x, y))
                                    pending.setdefault(min(20, t + 2), []).append(a)
                                    meta["offset"] += 1
                                    meta["triggered"] += 1
                                    break
                            else:
                                scheduled.append(a)
                                used.add(key)
                            continue
                    grid = world.grid if a.get("layer", "front") == "front" else world.background
                    if a["type"] == "place" and grid[a["y"], a["x"]] == a["material"]:
                        continue
                    if a["type"] == "place" and grid[a["y"], a["x"]] != 0:
                        if t < 20:
                            pending.setdefault(t + 1, []).append(a)
                        continue
                    scheduled.append(a)
                    used.add(key)
                f, b, applied, changed, falls = execute(world, scheduled, strict=False)
                meta["successful_offset_actions"] += sum(a in applied for a in wrong_actions)
                meta["skipped_invalid_actions"] += len(scheduled) - len(applied)
            else:
                if meta["triggered"] == 0:
                    return fg[:32].copy(), bg[:32].copy(), [list(fr) for fr in actions[:32]], {**meta, **baseline}
                scheduled = repair_actions(world, fg[31], bg[31])
                meta["repair_actions"] += len(scheduled)
                f, b, applied, changed, falls = finish_tick(world, scheduled)
            frames_f.append(f)
            frames_b.append(b)
            out.append(applied)
            physics += int(changed > 0)
            falling += falls
        for _ in range(4):
            f, b, applied, changed, falls = execute(world, [])
            frames_f.append(f)
            frames_b.append(b)
            out.append(applied)
            physics += int(changed > 0)
            falling += falls
        if not np.array_equal(world.grid, fg[31]) or not np.array_equal(world.background, bg[31]):
            raise ValueError("repair did not recover identical source final structure")
        final_f, final_b = D.settle_copy(world.grid, world.background, world_seed)
        if not np.array_equal(final_f, world.grid) or not np.array_equal(final_b, world.background):
            raise ValueError("noisy final frame is not already stable")
        if sum(core.check_violations(world.grid, world.background).values()):
            raise ValueError("noisy final structure violates game rules")
    except ValueError as exc:
        meta.update(fallback=True, fallback_reason=str(exc))
        return fg[:32].copy(), bg[:32].copy(), [list(fr) for fr in actions[:32]], {**meta, **baseline}
    frames_f, frames_b = np.stack(frames_f), np.stack(frames_b)
    meta.update(physical_transitions=physics, falling_cell_events=falling,
                effective_episode=not (np.array_equal(frames_f, fg[:32]) and np.array_equal(frames_b, bg[:32]) and out == actions[:32]),
                chunk_sizes=[len(fr) for fr in out[1:28]],
                remove_actions=sum(a["type"] == "remove" for fr in out for a in fr), actions=sum(map(len, out)))
    return frames_f, frames_b, out, meta
