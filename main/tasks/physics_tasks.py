"""Action-generated water/sand scenes; core physics is never modified."""
import math

import numpy as np

from ..shared import D, H, W, core, pack
from .registry import TASKS, validate_spec


def snapshot(world, actions=()):
    return world.grid.copy(), world.background.copy(), list(actions)


def tick(world, actions):
    applied = []
    seen = set()
    for a in actions:
        key = (a["x"], a["y"], a["layer"])
        if key in seen:
            raise ValueError("two same-cell actions cannot share one encoded frame")
        seen.add(key)
        action = core.Action(a["type"], a["x"], a["y"], a.get("material", 0), a.get("layer", "front"))
        if world.apply_action(action) and a["type"] != "noop":
            applied.append(a)
    world.tick()
    world.prune_background()
    return snapshot(world, applied)


def settle(fg, bg, seed, limit=256):
    world = core.FormWorld(seed=seed, grid=fg, background=bg)
    for _ in range(limit):
        changed, _ = world.tick()
        removed = world.prune_background()
        if not changed and not removed:
            return world.grid.copy(), world.background.copy()
    raise ValueError("scene failed to settle within generation limit")


def action(kind, x, y, material=0):
    return {"type": kind, "x": int(x), "y": int(y), "material": int(material), "layer": "front"}


def geometry(spec):
    x, width, height = spec["x"], spec["width"], spec["height"]
    template, floor = spec["template"], D.SURF
    cells, removals = {}, []
    if template == "pit":
        for y in range(floor, min(H - 2, floor + 2)):
            for xx in range(x, x + width):
                removals.append(action("remove", xx, y))
    elif template == "vessel":
        bottom = floor - 2
        for xx in (x, x + width - 1):
            for y in range(bottom, floor):
                cells[(xx, y)] = core.DIRT
        for xx in range(x, x + width):
            cells[(xx, bottom)] = core.BRICK
        for xx in (x, x + width - 1):
            for y in range(bottom - height, bottom):
                cells[(xx, y)] = core.DIRT
    elif template in ("ramp", "terraces"):
        for dx in range(width):
            elevation = max(1, height - (dx if template == "ramp" else dx // 3))
            for y in range(floor - elevation, floor):
                cells[(x + dx, y)] = core.DIRT
    elif template == "baffle":
        for y in range(floor - height, floor):
            cells[(x, y)] = core.DIRT
    elif template == "slit":
        top = floor - height
        for xx in range(W):
            if xx == x:
                continue
            for y in range(top, floor):
                cells[(xx, y)] = core.DIRT
        cells[(x, top)] = core.STONE
    elif template == "ledge":
        for xx in range(x, x + width):
            for y in range(floor - height, floor):
                cells[(xx, y)] = core.DIRT
    elif template != "flat":
        raise ValueError("unknown geometry")
    builds = [action("place", xx, y, m) for (xx, y), m in sorted(cells.items(), key=lambda item: (-item[0][1], item[0][0]))]
    return removals + builds


def sample_spec(task, seed, split, index, attempt=0):
    entry = TASKS[task]
    rng = D.rng_for("main-v1", task, seed, split, index, attempt)
    width = rng.randint(6, 10)
    height = rng.randint(3, 6)
    if entry["template"] == "ramp":
        width = height
    if entry["template"] == "terraces":
        width = rng.choice([9, 12, 15])
        height = width // 3
    intervals = {"train": (4, 22), "validation": (25, 30), "test": (34, W - width - 2)}
    if split not in intervals:
        raise ValueError("unknown data split")
    spec = {"task": task, "family": entry["family"], "template": entry["template"],
            "x": rng.randint(*intervals[split]), "width": width, "height": height,
            "fill": rng.randint(4, 12 if entry["template"] == "ledge" else 18),
            "seed": D.int_seed(task, seed, split, index, attempt)}
    return validate_spec(task, spec)


def pour_columns(spec):
    x, width, template = spec["x"], spec["width"], spec["template"]
    if template == "vessel":
        return list(range(x + 1, x + width - 1))
    if template in ("ramp", "terraces"):
        return list(range(x, x + min(3, width)))
    if template == "baffle":
        return [x - 1, x, x + 1]
    if template == "slit":
        return [x - 1, x, x + 1]
    if template == "ledge":
        return [x, x + width - 1]
    return list(range(x, x + width))


def formation(spec):
    world = D.new_world(spec["seed"])
    frames = [snapshot(world)]
    builds = geometry(spec)
    rows = sorted({a["y"] for a in builds}, reverse=spec["template"] != "pit")
    if len(rows) > 10:
        raise ValueError("geometry exceeds formation row budget")
    groups = [[a for a in builds if a["y"] == y] for y in rows]
    for j in range(10):
        actions = groups[j] if j < len(groups) else []
        snap = tick(world, actions)
        if len(snap[2]) != len(actions):
            raise ValueError("formation action rejected; retry a different scene")
        frames.append(snap)
    material = core.WATER if spec["family"] == "water" else core.SAND
    columns = pour_columns(spec)
    remaining = spec["fill"]
    rng = D.rng_for(spec["seed"], "pour")
    for j in range(6):
        rng.shuffle(columns)
        count = min(len(columns), math.ceil(remaining / max(1, 6 - j)))
        actions = []
        for xx in columns[:count]:
            ys = np.flatnonzero(world.grid[:, xx] != core.EMPTY)
            y = max(1, int(ys[0]) - 2)
            if world.grid[y, xx] == core.EMPTY:
                actions.append(action("place", xx, y, material))
        snap = tick(world, actions)
        if len(snap[2]) != len(actions):
            raise ValueError("formation pour action rejected")
        frames.append(snap)
        remaining -= len(snap[2])
    if remaining != 0:
        raise ValueError("formation could not pour the declared material count")
    if np.count_nonzero(world.grid == material) != spec["fill"]:
        raise ValueError("formation material count differs from declared fill")
    for _ in range(15):
        frames.append(tick(world, []))
    if len(frames) != 32:
        raise ValueError("formation frame budget differs")
    return frames, world


def intervention(spec, fg, bg, seed):
    rng = D.rng_for("main-op", seed)
    template, x = spec["template"], spec["x"]
    if template == "vessel":
        bottom = D.SURF - 2
        if spec["family"] == "water":
            xx = rng.choice([x, x + spec["width"] - 1])
            acts = [action("remove", xx, y) for y in range(bottom - spec["height"], bottom) if fg[y, xx] != 0]
        else:
            xx, y = x + spec["width"] // 2, bottom
            acts = [action("remove", xx, y)] if fg[y, xx] != 0 else []
    elif template == "slit":
        y = D.SURF - spec["height"]
        acts = [action("remove", x, y)] if fg[y, x] != 0 else []
    else:
        columns = pour_columns(spec)
        xx = rng.choice(columns)
        ys = np.flatnonzero(fg[:, xx] != core.EMPTY)
        y = max(1, int(ys[0]) - rng.randint(2, 4))
        material = core.WATER if spec["family"] == "water" else core.SAND
        acts = [action("place", xx, y, material)] if fg[y, xx] == 0 else []
    return acts


def clip(fg, bg, acts, seed, length=8):
    world = core.FormWorld(seed=seed, grid=fg, background=bg)
    frames = [snapshot(world)]
    for t in range(1, length):
        frames.append(tick(world, acts if t == 1 else []))
    return frames


def episode(spec):
    form, world = formation(spec)
    fg, bg = settle(world.grid, world.background, spec["seed"])
    if not np.array_equal(fg, world.grid) or not np.array_equal(bg, world.background):
        raise ValueError("formation did not finish at rest; retry")
    ops = []
    for j in range(8):
        seed = D.int_seed(spec["seed"], "op", j)
        acts = intervention(spec, fg, bg, seed)
        ops += clip(fg, bg, acts, seed)
    rng = D.rng_for(spec["seed"], "shuffle")
    order = list(range(32))
    rng.shuffle(order)
    partial = []
    picks = []
    for j in range(8):
        k = rng.randint(1, 27)
        picks.append(k)
        seed = D.int_seed(spec["seed"], "partial", j)
        pf, pb = settle(form[k][0], form[k][1], seed)
        partial += clip(pf, pb, intervention(spec, pf, pb, seed), seed)
    arrays = {"state": pack(ops), "process": pack(form + ops[:32]),
              "shuffle": pack([form[k] for k in order] + ops[:32]), "partial": pack(partial)}
    metadata = {"spec": spec, "final_hash": D.state_hash(fg, bg), "shuffle_order": order,
                "partial_picks": picks, "formation_settled": True}
    return arrays, metadata, form, (fg, bg)


def bench_item(spec, fg, bg):
    maps, targets = [], []
    for j in range(2):
        seed = D.int_seed(spec["seed"], "bench", j)
        acts = intervention(spec, fg, bg, seed)
        world = core.FormWorld(seed=seed, grid=fg, background=bg)
        amap = np.zeros((32, H, W), np.uint8)
        for t in range(32):
            snap = tick(world, acts if t == 0 else [])
            amap[t] = pack([snap])["act"][0]
        maps.append(amap)
        targets.append((world.grid.copy(), world.background.copy()))
    return {"ctx_fg": fg, "ctx_bg": bg, "act": np.stack(maps),
            "target_fg": np.stack([v[0] for v in targets]), "target_bg": np.stack([v[1] for v in targets])}
