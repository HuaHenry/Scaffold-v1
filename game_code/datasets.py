"""FormWorld-2D dataset generator, v2 (Block & Bloom rule set).

What changed relative to v1
  * The world is the real game world: terrain (grass / dirt / stone / bedrock), 21 materials,
    TWO layers (foreground = load + collision, background = walls/windows/torches/ladders/flowers),
    two-cell doors, tick-level (animated) physics.
  * Scenes are buildings and landscapes: cottage, hut, watch-tower, bridge, arch, tower, tree,
    dune, garden, pond. Cottage/hut follow the in-game blueprint build order.
  * One frame = one physics tick (plus the actions applied before it), not one jump to equilibrium.
  * All four training conditions are built from the SAME scenes, the SAME frame budget and the
    SAME interventions; only the temporal structure of the formation segment differs.
  * Seeds are fixed strings, so output is reproducible across runs and machines.

Training conditions (T frames each, default T = 64)
  state    : 8 operation clips on the finished structure
  process  : 32 frames of continuous formation (empty terrain -> finished) + 4 clips on the finished structure
  shuffle  : exactly the process frames, but randomly permuted (no valid transitions) + the same 4 clips
  partial  : 8 clips that start from random intermediate (unfinished) structures; no formation transitions

An operation clip = 1 reset frame (the base structure, noop) + 1 intervention frame + 6 noop ticks.
`cont[t]` is True when frame t is a physical successor of frame t-1; False marks a scene reset / jump.

FormBench-2D: 4 splits (id, combo_ood, scale_ood, multi_ood). Each sample is a settled structure,
two candidate interventions (block removal / terrain dig) and, for each, a 32-tick ground-truth future.
Tasks: collapse prediction, counterfactual ranking (which of the two removals changes more),
and long-rollout physical consistency (check_violations on the final predicted state).
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np

from core import (EMPTY, GROUND, STONE, WOOD, LEAF, SAND, WATER, DIRT, GRASS, BRICK, GLASS,
                  ROOF, DOOR, WALL, TORCH, LADDER, FLOWER, DOORS, Action, FormWorld,
                  check_violations)
from render import render_layers, compose

# ----------------------------------------------------------------------------- constants
DW, DH = 56, 28                  # dataset grid (width, height)
SURF = DH - 4                    # grass row
FLOOR = SURF - 1                 # first air row above the grass
CELL = 2                         # pixels per cell in rendered frames
T_DEFAULT, F_FORM, CLIP, HORIZON, CONTEXT = 64, 32, 8, 32, 4
SETTLE = 4                       # formation frames reserved for settling at the end
N_STATE_CLIPS = 8                # clips per episode in the `state` / `partial` conditions
N_MIXED_CLIPS = 4                # clips per episode in `process` / `shuffle`
RANK_MARGIN = 0.03              # minimum label gap for a counterfactual pair to be rankable
SCHEMA = "formworld.dataset.v2"
CONDITIONS = ("state", "process", "partial", "shuffle")
STRUCT_EVENT = (STONE, WOOD, BRICK, ROOF)
TRAIN_KINDS = ["tower", "bridge", "cottage", "hut", "watchtower", "arch", "tree", "dune", "garden", "pond"]

SPLITS = {
    "id": {"scale": "base", "kinds": [[k] for k in TRAIN_KINDS]},
    "combo_ood": {"scale": "base", "kinds": [["cottage", "tree"], ["hut", "tower"], ["bridge", "dune"],
                                              ["watchtower", "pond"], ["arch", "garden"],
                                              ["cottage", "watchtower"]]},
    "scale_ood": {"scale": "big", "kinds": [["tower"], ["bridge"], ["arch"], ["cottage"], ["watchtower"]]},
    "multi_ood": {"scale": "base", "kinds": [["tower", "bridge", "tree"], ["cottage", "watchtower", "garden"],
                                              ["hut", "arch", "dune"], ["cottage", "tree", "pond"]]},
}


# ----------------------------------------------------------------------------- helpers
def terrain() -> np.ndarray:
    g = np.zeros((DH, DW), dtype=np.uint8)
    g[DH - 1, :] = GROUND
    g[DH - 2, :] = STONE
    g[DH - 3, :] = DIRT
    g[SURF, :] = GRASS
    return g


def new_world(seed: int, fg: np.ndarray | None = None, bg: np.ndarray | None = None) -> FormWorld:
    return FormWorld(seed=seed, grid=terrain() if fg is None else fg, background=bg)


def place(x: int, y: int, material: int, layer: str = "front") -> Action:
    return Action("place", int(x), int(y), int(material), layer)


def rng_for(*parts) -> random.Random:
    return random.Random("/".join(str(p) for p in parts))


def int_seed(*parts) -> int:
    return int(hashlib.sha256("/".join(str(p) for p in parts).encode()).hexdigest()[:8], 16)


def state_hash(fg: np.ndarray, bg: np.ndarray) -> str:
    return hashlib.sha1(fg.tobytes() + b"|" + bg.tobytes()).hexdigest()


def act_dict(a: Action) -> dict:
    return {"type": a.type, "x": int(a.x), "y": int(a.y), "material": int(a.material), "layer": a.layer}


# ----------------------------------------------------------------------------- scene specs
def sample_spec(kind: str, rng: random.Random, scale: str = "base") -> dict:
    big = scale == "big"
    if kind == "tower":
        return {"kind": kind, "width": 1, "h": rng.randint(10, 13) if big else rng.randint(4, 7),
                "mat": rng.choice([STONE, BRICK])}
    if kind == "bridge":
        span = rng.randint(10, 13) if big else rng.randint(4, 8)
        return {"kind": kind, "width": span, "span": span,
                "beam": BRICK if span > 10 else rng.choice([WOOD, BRICK])}
    if kind in ("cottage", "hut"):
        if kind == "hut":
            width, tall, n_win, beam = 6, 4, 1, BRICK
        elif big:
            width, tall, n_win, beam = rng.randint(13, 14), rng.randint(7, 8), rng.randint(3, 4), BRICK
        else:
            width, tall, n_win, beam = rng.randint(8, 10), rng.randint(4, 5), rng.randint(2, 3), WOOD
        ladder = rng.randint(4, width - 4) if big else None
        offsets = [o for o in range(2, width - 2) if o != ladder]
        windows = sorted(rng.sample(offsets, min(n_win, len(offsets))))
        return {"kind": kind, "width": width, "tall": tall, "beam": beam, "windows": windows,
                "ladder": ladder, "door_left": rng.random() < 0.5}
    if kind == "watchtower":
        return {"kind": kind, "width": 6, "h": rng.randint(11, 13) if big else rng.randint(6, 8)}
    if kind == "arch":
        span = rng.randint(9, 10) if big else rng.randint(5, 7)
        return {"kind": kind, "width": span, "span": span, "h": 5}
    if kind == "tree":
        return {"kind": kind, "width": 7, "h": rng.randint(4, 6)}
    if kind == "dune":
        return {"kind": kind, "width": 7, "n": rng.randint(10, 16)}
    if kind == "garden":
        lamps = [1, 6]
        free = [o for o in range(8) if o not in lamps]
        return {"kind": kind, "width": 8, "lamps": lamps, "flowers": sorted(rng.sample(free, rng.randint(3, 5)))}
    if kind == "pond":
        return {"kind": kind, "width": 4, "grains": rng.randint(8, 10)}
    raise ValueError(f"unknown scene kind: {kind}")


def _inward(cells: list[tuple[int, int, int]], x0: int, x1: int) -> list[tuple[int, int, int]]:
    """Order a horizontal beam from both supported ends toward the middle."""
    return sorted(cells, key=lambda c: (min(c[0] - x0, x1 - c[0]), c[0]))


def house_actions(spec: dict, x0: int) -> list[Action]:
    """Port of the in-game blueprint order: foundation, door, frame bottom-up, walls, details."""
    width, tall, floor = spec["width"], spec["tall"], FLOOR
    top = floor - tall
    door_x = x0 if spec["door_left"] else x0 + width - 1
    front: list[tuple[int, int, int]] = []
    found = [(x, floor, STONE) for x in range(x0, x0 + width)]
    for y in range(floor - 1, top, -1):
        for col in (x0, x0 + width - 1):
            if col == door_x and y >= floor - 2:
                continue
            front.append((col, y, BRICK))
    front += [(x, top, spec["beam"]) for x in range(x0, x0 + width)]
    for level in range(width // 2):
        front += [(x, top - 1 - level, ROOF) for x in range(x0 + level, x0 + width - level)]
    edge = lambda x: min(x - x0, x0 + width - 1 - x)
    ordered = sorted(found + front, key=lambda c: (-c[1], edge(c[0]), c[0]))
    foundation = [c for c in ordered if c[1] == floor and c[2] == STONE]
    others = [c for c in ordered if c not in foundation]
    acts = [place(*c) for c in foundation]
    acts.append(place(door_x, floor - 1, DOOR))
    acts += [place(*c) for c in others]
    for y in range(floor - 1, top, -1):
        acts += [place(x, y, WALL, "back") for x in range(x0 + 1, x0 + width - 1)]
    for o in spec["windows"]:
        acts += [place(x0 + o, floor - 2, GLASS, "back"), place(x0 + o, floor - 3, GLASS, "back")]
    for x in (x0 + 1, x0 + width - 2):
        acts.append(place(x, floor - 2, TORCH, "back"))
    if spec["ladder"] is not None:
        acts += [place(x0 + spec["ladder"], y, LADDER, "back") for y in range(floor - 1, top, -1)]
    return acts


def build_actions(spec: dict, x0: int) -> list[Action]:
    kind = spec["kind"]
    if kind == "tower":
        return [place(x0, FLOOR - i, spec["mat"]) for i in range(spec["h"])]
    if kind == "bridge":
        span, top = spec["span"], FLOOR - 4
        acts = []
        for y in range(FLOOR, top, -1):
            acts += [place(x0, y, STONE), place(x0 + span - 1, y, STONE)]
        acts += [place(x, y, spec["beam"]) for x, y, _ in _inward(
            [(x0 + i, top, 0) for i in range(span)], x0, x0 + span - 1)]
        return acts
    if kind in ("cottage", "hut"):
        return house_actions(spec, x0)
    if kind == "watchtower":
        h, c0 = spec["h"], x0 + 2
        yp = FLOOR - h
        acts = []
        for y in range(FLOOR, yp, -1):
            acts += [place(c0, y, STONE), place(c0 + 1, y, STONE)]
        deck = sorted(range(x0, x0 + 6), key=lambda x: (min(abs(x - c0), abs(x - (c0 + 1))), x))
        acts += [place(x, yp, WOOD) for x in deck]
        acts += [place(x0, yp - 1, BRICK), place(x0 + 5, yp - 1, BRICK), place(x0 + 1, yp - 1, TORCH, "back")]
        acts += [place(x0 + 4, y, LADDER, "back") for y in range(FLOOR, yp, -1)]
        return acts
    if kind == "arch":
        span, h = spec["span"], spec["h"]
        top, mid = FLOOR - h, x0 + spec["span"] // 2
        acts = []
        for y in range(FLOOR, top, -1):
            acts += [place(x0, y, STONE), place(x0 + span - 1, y, STONE), place(mid, y, STONE)]
        acts += [place(x, y, WOOD) for x, y, _ in _inward([(x0 + i, top, 0) for i in range(span)], x0, x0 + span - 1)]
        acts += [Action("remove", mid, y, STONE) for y in range(top + 1, FLOOR + 1)]
        return acts
    if kind == "tree":
        h, tx = spec["h"], x0 + 3
        ytop = FLOOR - h + 1
        acts = [place(tx, FLOOR - i, WOOD) for i in range(h)]
        crown = [(dx, dy) for dy in range(-2, 3) for dx in range(-3, 4)
                 if abs(dx) + abs(dy) <= 3 and not (dx == 0 and dy >= 0)]
        crown.sort(key=lambda c: (abs(c[0]) + abs(c[1]), c))
        acts += [place(tx + dx, ytop + dy, LEAF) for dx, dy in crown]
        return acts
    if kind == "dune":
        return [place(x0 + 3, SURF - 7, SAND) for _ in range(spec["n"])]
    if kind == "garden":
        acts = []
        for o in spec["lamps"]:
            acts += [place(x0 + o, FLOOR - i, WOOD) for i in range(3)]
            acts.append(place(x0 + o + 1, FLOOR - 2, TORCH, "back"))
        acts += [place(x0 + o, FLOOR, FLOWER, "back") for o in spec["flowers"]]
        return acts
    if kind == "pond":
        acts = [Action("remove", x0 + i, SURF + j, GRASS) for j in range(2) for i in range(4)]
        acts += [place(x0 + 1 + (n % 2), SURF - 3, WATER) for n in range(spec["grains"])]
        return acts
    raise ValueError(kind)


def sample_scene(rng: random.Random, kinds: list[str], scale: str) -> dict:
    specs = [sample_spec(k, rng, scale) for k in kinds]
    gaps = [rng.randint(2, 5) for _ in specs[1:]]
    total = sum(s["width"] for s in specs) + sum(gaps)
    x = rng.randint(2, DW - 2 - total)
    x0s = []
    for i, s in enumerate(specs):
        x0s.append(x)
        x += s["width"] + (gaps[i] if i < len(gaps) else 0)
    actions = [a for s, x0 in zip(specs, x0s) for a in build_actions(s, x0)]
    return {"kinds": kinds, "specs": specs, "x0s": x0s, "actions": actions, "scale": scale}


def settled_build(scene: dict, seed: int) -> FormWorld:
    """Reference construction: apply every action and let the world reach rest each time."""
    w = new_world(seed)
    for a in scene["actions"]:
        w.step(a)
    return w


# ----------------------------------------------------------------------------- interventions
def _cells(mask: np.ndarray) -> list[tuple[int, int]]:
    ys, xs = np.nonzero(mask)
    return list(zip(xs.tolist(), ys.tolist()))


def pick_event(fg: np.ndarray, bg: np.ndarray, rng: random.Random) -> Action:
    r = rng.random()
    if r < 0.55:
        c = _cells(np.isin(fg[:SURF], STRUCT_EVENT))
        if c:
            x, y = rng.choice(c)
            return Action("remove", x, y, STONE)
    elif r < 0.65:
        c = _cells(bg[:SURF] != 0)
        if c:
            x, y = rng.choice(c)
            return Action("remove", x, y, STONE, "back")
    elif r < 0.72:
        c = _cells(np.isin(fg[:SURF], list(DOORS)))
        if c:
            x, y = rng.choice(c)
            return Action("interact", x, y, DOOR)
    elif r < 0.80:
        c = [x for x in range(DW) if fg[SURF - 1, x] != 0]
        if c:
            return Action("remove", rng.choice(c), SURF, GRASS)
    for _ in range(30):
        x, y = rng.randrange(DW), SURF - 6
        if fg[y, x] == EMPTY:
            return Action("place", x, y, rng.choice([STONE, SAND, WATER, WOOD]))
    return Action("noop")


# ----------------------------------------------------------------------------- frame recording
def snap(w: FormWorld, acts: list[Action], cont: bool) -> tuple:
    return (w.grid.copy(), w.background.copy(), [act_dict(a) for a in acts], cont)


def tick_frame(w: FormWorld, acts: list[Action], cont: bool = True) -> tuple:
    for a in acts:
        w.apply_action(a)
    w.tick()
    w.prune_background()
    return snap(w, acts, cont)


def formation(actions: list[Action], seed: int, frames: int = F_FORM) -> tuple[list, list, FormWorld, int]:
    """Continuous build: frame 0 = bare terrain, then action chunks (one tick each), then settling ticks."""
    w = new_world(seed)
    out = [snap(w, [], False)]
    slots = frames - 1 - SETTLE
    size = max(1, -(-len(actions) // slots))
    chunks = [actions[i:i + size] for i in range(0, len(actions), size)]
    for chunk in chunks:
        out.append(tick_frame(w, chunk))
    while len(out) < frames:
        out.append(tick_frame(w, []))
    return out, chunks, w, size


def settle_copy(fg: np.ndarray, bg: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    w = FormWorld(seed=seed, grid=fg.copy(), background=bg.copy())
    w.step()
    return w.grid.copy(), w.background.copy()


def op_clip(fg: np.ndarray, bg: np.ndarray, event: Action, seed: int) -> list:
    w = FormWorld(seed=seed, grid=fg.copy(), background=bg.copy())
    frames = [snap(w, [], False)]
    for j in range(CLIP - 1):
        frames.append(tick_frame(w, [event] if j == 0 else []))
    return frames


def pack(frames: list, phase: list[int]) -> dict:
    return {
        "fg": np.stack([f[0] for f in frames]).astype(np.uint8),
        "bg": np.stack([f[1] for f in frames]).astype(np.uint8),
        "actions": [f[2] for f in frames],
        "cont": np.array([f[3] for f in frames], dtype=bool),
        "phase": np.array(phase, dtype=np.int8),
    }


def make_episode_set(seed: int, index: int, T: int = T_DEFAULT) -> dict:
    """Build one scene and derive the four training conditions from it."""
    F = F_FORM
    rng = rng_for(seed, "scene", index)
    kind = rng.choice(TRAIN_KINDS)
    scene = sample_scene(rng, [kind], "base")
    world_seed = int_seed(seed, "world", index)
    form, chunks, fworld, chunk_size = formation(scene["actions"], world_seed, F)

    final_fg, final_bg = settle_copy(fworld.grid, fworld.background, world_seed)
    settled = bool(np.array_equal(final_fg, fworld.grid) and np.array_equal(final_bg, fworld.background))

    erng = rng_for(seed, "events", index)
    events = [pick_event(final_fg, final_bg, erng) for _ in range(N_STATE_CLIPS)]

    def clips(base_fg, base_bg, evs, tag):
        out = []
        for j, ev in enumerate(evs):
            out += op_clip(base_fg, base_bg, ev, int_seed(seed, tag, index, j))
        return out

    n_ops = N_MIXED_CLIPS * CLIP
    assert F + n_ops == T and N_STATE_CLIPS * CLIP == T, "frame budgets must match across conditions"
    episodes = {}

    f = clips(final_fg, final_bg, events, "state")
    episodes["state"] = pack(f, [1] * len(f))

    ops = clips(final_fg, final_bg, events[:N_MIXED_CLIPS], "mixed")
    episodes["process"] = pack(form + ops, [0] * F + [1] * n_ops)

    prng = rng_for(seed, "shuffle", index)
    order = list(range(F))
    prng.shuffle(order)
    shuffled = [(form[i][0], form[i][1], form[i][2], False) for i in order]
    episodes["shuffle"] = pack(shuffled + ops, [0] * F + [1] * n_ops)
    episodes["shuffle"]["order"] = order

    qrng = rng_for(seed, "partial", index)
    pool = list(range(1, 1 + len(chunks)))
    picks = [qrng.choice(pool) for _ in range(N_STATE_CLIPS)]
    p_frames = []
    for j, k in enumerate(picks):
        base_fg, base_bg = settle_copy(form[k][0], form[k][1], int_seed(seed, "pbase", index, j))
        p_frames += op_clip(base_fg, base_bg, pick_event(base_fg, base_bg, qrng), int_seed(seed, "partial", index, j))
    episodes["partial"] = pack(p_frames, [1] * len(p_frames))
    episodes["partial"]["picks"] = picks

    meta = {"kind": kind, "specs": scene["specs"], "x0s": scene["x0s"], "n_actions": len(scene["actions"]),
            "chunk_size": chunk_size, "n_chunks": len(chunks), "formation_settled": bool(settled),
            "final_hash": state_hash(final_fg, final_bg)}
    return {"episodes": episodes, "meta": meta, "final": (final_fg, final_bg)}


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def generate_train(out: Path, episodes: int, T: int, seed: int, with_frames: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    cols = {c: {"fg": [], "bg": [], "cont": [], "phase": [], "actions": [], "extra": []} for c in CONDITIONS}
    metas, finals_fg, finals_bg = [], [], []
    for i in range(episodes):
        res = make_episode_set(seed, i, T)
        metas.append(res["meta"])
        finals_fg.append(res["final"][0])
        finals_bg.append(res["final"][1])
        for c in CONDITIONS:
            ep = res["episodes"][c]
            for key in ("fg", "bg", "cont", "phase", "actions"):
                cols[c][key].append(ep[key])
            cols[c]["extra"].append({k: ep[k] for k in ("order", "picks") if k in ep})
    summary = {}
    for c in CONDITIONS:
        arrays = {k: np.stack(cols[c][k]) for k in ("fg", "bg", "cont", "phase")}
        if with_frames:
            arrays["frames"] = render_layers(arrays["fg"], arrays["bg"], CELL)
        np.savez_compressed(out / f"train_{c}.npz", **arrays)
        (out / f"train_{c}_actions.json").write_text(
            json.dumps({"schema": SCHEMA, "condition": c, "scenes": [m["kind"] for m in metas],
                        "extra": _jsonable(cols[c]["extra"]), "actions": cols[c]["actions"]}), encoding="utf-8")
        summary[c] = {"episodes": episodes, "frames_per_episode": T, "total_frames": episodes * T,
                      "fg_shape": list(arrays["fg"].shape),
                      "continuous_transitions": int(arrays["cont"].sum()),
                      "reset_frames": int((~arrays["cont"]).sum()),
                      "formation_frames": int((arrays["phase"] == 0).sum())}
    np.savez_compressed(out / "train_finals.npz", fg=np.stack(finals_fg), bg=np.stack(finals_bg))
    (out / "train_scenes.json").write_text(json.dumps(_jsonable(metas)), encoding="utf-8")
    summary_all = {"schema": SCHEMA, "seed": seed, "grid": [DH, DW], "cell": CELL,
                   "layout": {"formation": F_FORM, "clip": CLIP, "state_clips": N_STATE_CLIPS,
                              "mixed_clips": N_MIXED_CLIPS}, "conditions": summary}
    (out / "train_summary.json").write_text(json.dumps(summary_all, indent=2), encoding="utf-8")
    return summary_all


# ----------------------------------------------------------------------------- FormBench-2D
def change_mask(a_fg, a_bg, b_fg, b_bg) -> np.ndarray:
    """Cells that matter: anything built above the surface in either state, or changed anywhere."""
    above = np.zeros((DH, DW), dtype=bool)
    above[:SURF] = (a_fg[:SURF] != 0) | (a_bg[:SURF] != 0) | (b_fg[:SURF] != 0) | (b_bg[:SURF] != 0)
    changed = (a_fg != b_fg) | (a_bg != b_bg)
    return above | changed


def change_rate(a_fg, a_bg, b_fg, b_bg) -> float:
    mask = change_mask(a_fg, a_bg, b_fg, b_bg)
    changed = ((a_fg != b_fg) | (a_bg != b_bg))
    return float(changed[mask].mean()) if mask.any() else 0.0


def rollout(fg: np.ndarray, bg: np.ndarray, action: Action, seed: int) -> dict:
    w = FormWorld(seed=seed, grid=fg.copy(), background=bg.copy())
    fgs, bgs, settled_at = [], [], None
    for t in range(HORIZON):
        if t == 0:
            w.apply_action(action)
        changed, _ = w.tick()
        removed = w.prune_background()
        fgs.append(w.grid.copy())
        bgs.append(w.background.copy())
        if settled_at is None and not changed and not removed:
            settled_at = t
    return {"fg": np.stack(fgs), "bg": np.stack(bgs), "settled_at": settled_at,
            "action": act_dict(action)}


def candidate_interventions(fg: np.ndarray, rng: random.Random, n: int) -> list[Action]:
    """Block removals that can matter: load-bearing blocks, soil under a structure, soil walling a pond."""
    pool = [Action("remove", x, y, STONE) for x, y in _cells(np.isin(fg[:SURF], STRUCT_EVENT))]
    pool += [Action("remove", x, SURF, GRASS) for x in range(DW) if fg[SURF - 1, x] != 0]
    soil = np.isin(fg, [GRASS, DIRT])
    wet = np.zeros_like(soil)
    wet[:, 1:] |= fg[:, :-1] == WATER
    wet[:, :-1] |= fg[:, 1:] == WATER
    wet[:-1] |= fg[1:] == WATER
    pool += [Action("remove", x, y, STONE) for x, y in _cells(soil & wet)]
    if not pool:
        pool = [Action("remove", x, y, STONE) for x, y in _cells(np.isin(fg[:SURF], [LEAF, SAND, WATER]))]
    rng.shuffle(pool)
    seen, picked = set(), []
    for a in pool:
        if (a.x, a.y) not in seen:
            seen.add((a.x, a.y))
            picked.append(a)
        if len(picked) == n:
            break
    return picked


def sample_bench_item(split: str, seed: int, i: int, attempt: int, exclude: set[str]) -> dict | None:
    rng = rng_for(seed, "bench", split, i, attempt)
    cfg = SPLITS[split]
    kinds = rng.choice(cfg["kinds"])
    scene = sample_scene(rng, kinds, cfg["scale"])
    w = settled_build(scene, int_seed(seed, "bench-world", split, i, attempt))
    fg, bg = w.grid.copy(), w.background.copy()
    if sum(check_violations(fg, bg).values()):
        return None
    h = state_hash(fg, bg)
    if h in exclude:
        return None
    runs = []
    for j, a in enumerate(candidate_interventions(fg, rng, 16)):
        r = rollout(fg, bg, a, int_seed(seed, "bench-roll", split, i, attempt, j))
        if r["settled_at"] is None:
            continue
        r["label"] = change_rate(fg, bg, r["fg"][-1], r["bg"][-1])
        r["n_changed"] = int(((fg != r["fg"][-1]) | (bg != r["bg"][-1])).sum())
        r["violations"] = int(sum(check_violations(r["fg"][-1], r["bg"][-1]).values()))
        if r["violations"] == 0:
            runs.append(r)
    if len(runs) < 2:
        return None
    best = max(((a, b) for a in range(len(runs)) for b in range(a + 1, len(runs))),
               key=lambda p: (abs(runs[p[0]]["label"] - runs[p[1]]["label"]), -p[0]))
    pair = sorted((runs[best[0]], runs[best[1]]), key=lambda r: -r["label"])   # pair[0] = the bigger effect
    valid = abs(pair[0]["label"] - pair[1]["label"]) >= RANK_MARGIN
    return {"fg": fg, "bg": bg, "pair": pair, "rank_valid": bool(valid), "scene": scene, "hash": h}


def generate_bench(out: Path, per_split: int, seed: int, exclude: set[str] | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    exclude = exclude or set()
    items, split_ids, kinds_list, skipped = [], [], [], {s: 0 for s in SPLITS}
    for sid, split in enumerate(SPLITS):
        for i in range(per_split):
            for attempt in range(40):
                item = sample_bench_item(split, seed, i, attempt, exclude)
                if item:
                    break
                skipped[split] += 1
            else:
                raise RuntimeError(f"could not sample {split} #{i}")
            items.append(item)
            split_ids.append(sid)
            kinds_list.append("+".join(item["scene"]["kinds"]))
    np.savez_compressed(
        out / "bench.npz",
        ctx_fg=np.stack([it["fg"] for it in items]), ctx_bg=np.stack([it["bg"] for it in items]),
        fut_fg=np.stack([[r["fg"] for r in it["pair"]] for it in items]),
        fut_bg=np.stack([[r["bg"] for r in it["pair"]] for it in items]),
        labels=np.array([[r["label"] for r in it["pair"]] for it in items], dtype=np.float32),
        settled_at=np.array([[r["settled_at"] for r in it["pair"]] for it in items], dtype=np.int16),
        n_changed=np.array([[r["n_changed"] for r in it["pair"]] for it in items], dtype=np.int16),
        violations=np.array([[r["violations"] for r in it["pair"]] for it in items], dtype=np.int16),
        rank_valid=np.array([it["rank_valid"] for it in items], dtype=bool),
        split_ids=np.array(split_ids, dtype=np.int8),
    )
    meta = {
        "schema": SCHEMA, "grid": [DH, DW], "cell": CELL, "context_frames": CONTEXT, "horizon": HORIZON,
        "splits": list(SPLITS), "split_definitions": {k: v["kinds"] for k, v in SPLITS.items()},
        "scale": {k: v["scale"] for k, v in SPLITS.items()},
        "tasks": {
            "collapse": "predict the settled state after the intervention; cell accuracy / structure IoU / change F1",
            "counterfactual": "rank the two interventions by change rate (only samples with rank_valid)",
            "long_rollout": "violations of check_violations() on the last predicted frame (ground truth = 0)",
        },
        "label": "fraction of mask cells that differ between the initial and final state; mask = built cells above the surface in either state plus any changed cell",
        "samples": len(items), "scene_kinds": kinds_list,
        "scenes": _jsonable([{"specs": it["scene"]["specs"], "x0s": it["scene"]["x0s"]} for it in items]),
        "interventions": [[r["action"] for r in it["pair"]] for it in items],
        "split_of_sample": [list(SPLITS)[s] for s in split_ids],
        "rejected_attempts": skipped,
    }
    (out / "bench.json").write_text(json.dumps(_jsonable(meta), indent=1), encoding="utf-8")
    return {"samples": len(items), "rejected_attempts": skipped}


# ----------------------------------------------------------------------------- metrics + audit
def score_outcome(pred_fg, pred_bg, true_fg, true_bg, start_fg, start_bg) -> dict:
    """Per-sample collapse metrics for a predicted settled state."""
    mask = change_mask(start_fg, start_bg, true_fg, true_bg)
    acc_all = float(((pred_fg == true_fg) & (pred_bg == true_bg)).mean())
    acc_mask = float(((pred_fg == true_fg) & (pred_bg == true_bg))[mask].mean()) if mask.any() else 1.0
    a, b = pred_fg[:SURF] != 0, true_fg[:SURF] != 0
    iou = float((a & b).sum() / (a | b).sum()) if (a | b).any() else 1.0
    pc = (pred_fg != start_fg) | (pred_bg != start_bg)
    tc = (true_fg != start_fg) | (true_bg != start_bg)
    tp, fp, fn = (pc & tc).sum(), (pc & ~tc).sum(), (~pc & tc).sum()
    f1 = float(2 * tp / max(1, 2 * tp + fp + fn)) if (tc.any() or pc.any()) else 1.0
    return {"cell_acc_all": acc_all, "cell_acc_mask": acc_mask, "structure_iou": iou, "change_f1": f1}


def audit(out: Path) -> dict:
    report: dict = {"schema": SCHEMA}
    scenes = json.loads((out / "train_scenes.json").read_text())
    kinds = {}
    for s in scenes:
        kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
    train = {"scene_counts": kinds, "formation_settled_rate": float(np.mean([s["formation_settled"] for s in scenes])),
             "mean_actions_per_scene": float(np.mean([s["n_actions"] for s in scenes])),
             "mean_chunk_size": float(np.mean([s["chunk_size"] for s in scenes])), "conditions": {}}
    summary = json.loads((out / "train_summary.json").read_text())["conditions"]
    finals = np.load(out / "train_finals.npz")
    hashes = {state_hash(f, b) for f, b in zip(finals["fg"], finals["bg"])}
    for c in CONDITIONS:
        d = np.load(out / f"train_{c}.npz")
        acts = json.loads((out / f"train_{c}_actions.json").read_text())["actions"]
        hist: dict[str, int] = {}
        for ep in acts:
            for fr in ep:
                for a in fr:
                    hist[a["type"]] = hist.get(a["type"], 0) + 1
        live = d["fg"] != 0
        material = np.bincount(d["fg"].ravel(), minlength=21)
        train["conditions"][c] = {**summary[c], "action_types": hist,
                                  "mean_nonempty_fg_cells": float(live.sum(axis=(2, 3)).mean()),
                                  "fg_material_hist": material.tolist(),
                                  "bg_nonzero_cells_per_frame": float((d["bg"] != 0).sum(axis=(2, 3)).mean())}
    # Conditions must share scenes / frame budgets: compare operation-phase frame counts and finals
    ref = {c: int(np.load(out / f"train_{c}.npz")["fg"].shape[1]) for c in CONDITIONS}
    train["same_frames_per_episode"] = len(set(ref.values())) == 1
    report["train"] = train

    meta = json.loads((out / "bench.json").read_text())
    b = np.load(out / "bench.npz")
    bench = {"samples": meta["samples"], "leak_with_train_finals": 0, "splits": {}}
    for sid, split in enumerate(meta["splits"]):
        idx = np.where(b["split_ids"] == sid)[0]
        copy_last, oracle = [], []
        for i in idx:
            start_fg, start_bg = b["ctx_fg"][i], b["ctx_bg"][i]
            true_fg, true_bg = b["fut_fg"][i, 0, -1], b["fut_bg"][i, 0, -1]
            if state_hash(start_fg, start_bg) in hashes:
                bench["leak_with_train_finals"] += 1
            copy_last.append(score_outcome(start_fg, start_bg, true_fg, true_bg, start_fg, start_bg))
            oracle.append(score_outcome(true_fg, true_bg, true_fg, true_bg, start_fg, start_bg))
        lab = b["labels"][idx]
        bench["splits"][split] = {
            "n": int(len(idx)), "mean_label": float(lab.mean()), "frac_label_gt_0": float((lab > 0).mean()),
            "rank_valid_rate": float(b["rank_valid"][idx].mean()),
            "mean_settled_at": float(b["settled_at"][idx].mean()),
            "frac_main_collapse": float((b["settled_at"][idx, 0] >= 1).mean()),
            "mean_cells_changed_main": float(b["n_changed"][idx, 0].mean()),
            "gt_violations": int(b["violations"][idx].sum()),
            "scene_counts": {k: meta["scene_kinds"].count(k) for k in sorted({meta["scene_kinds"][i] for i in idx})},
            "copy_last": {k: float(np.mean([m[k] for m in copy_last])) for k in copy_last[0]},
            "oracle": {k: float(np.mean([m[k] for m in oracle])) for k in oracle[0]},
        }
    report["bench"] = bench
    (out / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def generate_all(out: Path, episodes: int, T: int, per_split: int, seed: int, with_frames: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    train = generate_train(out, episodes, T, seed, with_frames)
    finals = np.load(out / "train_finals.npz")
    exclude = {state_hash(f, b) for f, b in zip(finals["fg"], finals["bg"])}
    bench = generate_bench(out, per_split, seed, exclude)
    report = audit(out)
    return {"train": train, "bench": bench, "audit": report}
