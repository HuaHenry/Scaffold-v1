"""FormWorld-2D dataset generator, v3 (richer scenes, deeper cascades, compound interventions).

Physics (core.py / physics.js) is NOT changed. v3 only changes what is generated:

  * Scenes   : 16 blueprints (the 10 of v2 + multistory, colonnade, aqueduct, fortress, stairs, granary),
               1-3 structures per training scene (4-5 in multi_ood), random mirroring, held-out kind pairs.
  * Builder  : random but physically valid build orders (structure by structure or interleaved), variable
               batch size per frame, temporary scaffolding, wrong-material rework, stray blocks, dropped
               blocks that fall and are cleared.  `p_mistake` controls the mistake rate.
  * Physics  : interventions are chosen to trigger deeper cascades (bearing chains, slabs that lose their
               pillar, water released from a channel, sand sliding off a deck, doors + ladders).
  * Actions  : an intervention is a compound object {"kind", "steps": [{"t", "actions": [...]}]}:
               single, multi (simultaneous removals), seq2 (second action while the collapse runs),
               place_remove, door (interact + remove), back (background layer), drop (place only).

Training conditions, frame budget and clip layout are identical to v2 (state / process / partial / shuffle,
T=64, F=32, clips of 8 frames); only the content changed.  Schema string: formworld.dataset.v3.
"""

from __future__ import annotations

import json
import os
import random
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from itertools import combinations
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic, strftime

import numpy as np

import datasets as D
from core import (EMPTY, STONE, WOOD, SAND, WATER, GRASS, BRICK, GLASS, ROOF, DOOR, WALL, TORCH,
                  LADDER, DOOR_TOP, OPEN_DOOR_TOP, DOORS, STRUCT, SPAN, Action, FormWorld, check_violations)
from datasets import (DW, DH, SURF, FLOOR, CLIP, HORIZON, CONTEXT, SETTLE, RANK_MARGIN, CONDITIONS,
                      CELL, STRUCT_EVENT, rng_for, int_seed, state_hash, act_dict)

SCHEMA = "formworld.dataset.v3"
T_DEFAULT, F_FORM = D.T_DEFAULT, D.F_FORM
NEW_KINDS = ["multistory", "colonnade", "aqueduct", "fortress", "stairs", "granary"]
KINDS = D.TRAIN_KINDS + NEW_KINDS
BIG_KINDS = ["tower", "bridge", "arch", "cottage", "watchtower", "multistory", "colonnade", "aqueduct",
             "fortress", "stairs"]
HELD_PAIRS = [("cottage", "tree"), ("hut", "tower"), ("bridge", "dune"), ("watchtower", "pond"),
              ("arch", "garden"), ("multistory", "granary"), ("fortress", "aqueduct"), ("colonnade", "stairs")]
HELD = {frozenset(p) for p in HELD_PAIRS}
DEFAULT_P = 0.08                  # default mistake rate of the training builder
MIN_CASCADE, MIN_DEPTH = 4, 2     # the larger intervention of a bench pair must reach this
BENCH_SPLITS = {
    "id": {"scale": "base", "n": ([1, 2, 3], [.30, .45, .25]), "gap": (2, 4), "held": False},
    "combo_ood": {"scale": "base", "gap": (2, 4), "held": True},
    "scale_ood": {"scale": "big", "n": ([1, 2], [.6, .4]), "gap": (2, 4), "held": False},
    "multi_ood": {"scale": "base", "n": ([4, 5], [.65, .35]), "gap": (1, 3), "held": False},
}
INTERVENTION_KINDS = ["single", "multi", "seq2", "place_remove", "door", "back", "drop"]
WRONG = {STONE: [WOOD, BRICK, ROOF], WOOD: [BRICK, ROOF], BRICK: [ROOF], ROOF: [BRICK]}   # span >= original


def P(x, y, m, layer="front", tag="build"):
    return (D.place(x, y, m, layer), tag)


# ----------------------------------------------------------------------------- blueprint specs
def sample_spec(kind: str, rng: random.Random, scale: str = "base") -> dict:
    big = scale == "big"
    if kind == "dune":
        return {"kind": kind, "width": 7, "n": rng.randint(12, 20) if big else rng.randint(10, 16)}
    if kind not in NEW_KINDS:
        return D.sample_spec(kind, rng, scale)
    if kind == "multistory":
        w = rng.randint(12, 14) if big else rng.randint(7, 9)
        nf = rng.randint(3, 4) if big else rng.randint(2, 3)
        lad = rng.randint(max(2, w - 8), min(w - 3, 7))
        need = max(lad, w - 1 - lad) - 1
        offs = [o for o in range(1, w - 1) if o != lad]
        return {"kind": kind, "width": w, "floors": nf, "ladder": lad, "scaffold": rng.random() < .5,
                "slab": rng.choice([m for m in (STONE, WOOD, BRICK) if SPAN[m] >= need]),
                "door_left": rng.random() < .5,
                "windows": [sorted(rng.sample(offs, rng.randint(1, 2))) for _ in range(nf)]}
    if kind in ("colonnade", "aqueduct"):
        n = (4 if kind == "colonnade" else rng.randint(3, 4)) if big else rng.randint(2, 3)
        p = rng.randint(5, 6) if big else rng.randint(4, 5)
        h = rng.randint(5, 6) if big else rng.randint(4, 5)
        feas = [m for m in (WOOD, BRICK, STONE) if SPAN[m] >= p // 2]
        spec = {"kind": kind, "width": n * p + 1, "n": n, "p": p, "h": h,
                "pillar": rng.choice([STONE, BRICK]), "beam": rng.choice(feas)}
        if kind == "colonnade":
            spec.update({"tiers": rng.choice([1, 2]), "h2": rng.randint(3, 4), "beam2": rng.choice(feas),
                         "scaffold": rng.random() < .5})
        else:
            spec.update({"cap": rng.choice([BRICK, STONE]), "grains": n * p - 1 - rng.randint(0, 2),
                         "scaffold": rng.random() < .5})
        return spec
    if kind == "fortress":
        w = rng.randint(18, 22) if big else rng.randint(12, 16)
        return {"kind": kind, "width": w, "wall_h": 5 if big else 4,
                "tower_h": rng.randint(10, 12) if big else rng.randint(7, 9), "gate": rng.randint(5, w - 6),
                "wall": rng.choice([STONE, BRICK]), "ladder": rng.random() < .5}
    if kind == "stairs":
        n = rng.randint(8, 9) if big else rng.randint(4, 6)
        mode = rng.choice(["asc", "pyr"])
        return {"kind": kind, "width": n if mode == "asc" else 2 * n - 1, "n": n, "mode": mode,
                "mat": rng.choice([STONE, BRICK])}
    if kind == "granary":
        w = 9 if big else rng.randint(6, 7)
        return {"kind": kind, "width": w, "ph": 5 if big else rng.randint(3, 4),
                "pillar": rng.choice([WOOD, STONE]), "scaffold": rng.random() < .5,
                "grains": rng.randint(20, 26) if big else rng.randint(10, 16)}
    raise ValueError(kind)


def _by_support(xs, anchors):
    """Order beam cells from the supported ends outwards: every prefix stays supported."""
    return sorted(xs, key=lambda x: (min(abs(x - a) for a in anchors), x))


def multistory_items(s, x0):
    w, nf, lad = s["width"], s["floors"], s["ladder"]
    left, right = x0, x0 + w - 1
    door_x = left if s["door_left"] else right
    out = [P(x, FLOOR, STONE) for x in sorted(range(x0, x0 + w), key=lambda x: (min(x - x0, right - x), x))]
    out.append(P(door_x, FLOOR - 1, DOOR))
    dist = lambda x: (x - x0) if x < x0 + lad else (right - x)
    for k in range(nf):
        base = FLOOR - 3 * k
        r1, r2, ys = base - 1, base - 2, base - 3
        for r in (r1, r2):
            for col in (left, right):
                if not (k == 0 and col == door_x):
                    out.append(P(col, r, BRICK))
        props = [x for x in (x0 + lad - 1, x0 + lad + 1)
                 if left < x < right] if (k == 0 and s["scaffold"]) else []
        out += [P(x, r, WOOD, tag="scaffold") for r in (r1, r2) for x in props]
        out += [P(x, ys, s["slab"]) for x in sorted([x for x in range(x0, x0 + w) if x != x0 + lad],
                                                    key=lambda x: (dist(x), x))]
        out += [(Action("remove", x, r, STONE), "scaffold") for r in (r2, r1) for x in props]
        for r in (r1, r2):
            out += [P(x, r, WALL, "back") for x in range(x0 + 1, x0 + w - 1)]
        for o in s["windows"][k]:
            out += [P(x0 + o, r1, GLASS, "back"), P(x0 + o, r2, GLASS, "back")]
        out.append(P(x0 + 1, r2, TORCH, "back"))
        out += [P(x0 + lad, y, LADDER, "back") for y in (r1, r2, ys)]
    ytop = FLOOR - 3 * nf
    for L in range(min(w // 2, 3)):
        row = ytop - 1 - L
        out += [P(x, row, ROOF) for x in range(x0 + L, x0 + w - L) if x != x0 + lad]
    return out


def colonnade_items(s, x0):
    n, p, h = s["n"], s["p"], s["h"]
    px = [x0 + i * p for i in range(n + 1)]
    xs = list(range(x0, x0 + n * p + 1))
    y1 = FLOOR - h
    out = [P(x, y, s["pillar"]) for y in range(FLOOR, y1, -1) for x in px]
    mids = [x0 + i * p + p // 2 for i in range(n)]
    if s["scaffold"]:
        out += [P(x, y, WOOD, tag="scaffold") for y in range(FLOOR, y1, -1) for x in mids]
    out += [P(x, y1, s["beam"]) for x in _by_support(xs, px)]
    if s["scaffold"]:
        out += [(Action("remove", x, y, STONE), "scaffold") for y in range(y1 + 1, FLOOR + 1) for x in mids]
    if s["tiers"] == 2:
        h2 = s["h2"]
        out += [P(x, y, s["pillar"]) for y in range(y1 - 1, y1 - h2 - 1, -1) for x in px]
        out += [P(x, y1 - h2 - 1, s["beam2"]) for x in _by_support(xs, px)]
    return out


def aqueduct_items(s, x0):
    n, p, h = s["n"], s["p"], s["h"]
    W = n * p + 1
    px = [x0 + i * p for i in range(n + 1)]
    yb = FLOOR - h
    out = [P(x, y, s["pillar"]) for y in range(FLOOR, yb, -1) for x in px]
    mids = [x0 + i * p + p // 2 for i in range(n)] if s["scaffold"] else []
    out += [P(x, y, WOOD, tag="scaffold") for y in range(FLOOR, yb, -1) for x in mids]
    out += [P(x, yb, s["beam"]) for x in _by_support(list(range(x0, x0 + W)), px)]
    out += [(Action("remove", x, y, STONE), "scaffold") for y in range(yb + 1, FLOOR + 1) for x in mids]
    out += [P(x, y, s["cap"]) for y in (yb - 1, yb - 2) for x in (x0, x0 + W - 1)]
    xm = x0 + W // 2
    out += [P(xm + (i % 2), yb - 4, WATER) for i in range(s["grains"])]
    return out


def fortress_items(s, x0):
    w, wh, th, wall, gx = s["width"], s["wall_h"], s["tower_h"], s["wall"], x0 + s["gate"]
    towers = {x0, x0 + 1, x0 + w - 2, x0 + w - 1}
    mat = lambda x: STONE if x in towers else wall
    out = [P(x, FLOOR, mat(x)) for x in range(x0, x0 + w)]
    out.append(P(gx, FLOOR - 1, DOOR))
    for y in range(FLOOR - 1, FLOOR - th, -1):
        for x in range(x0, x0 + w):
            if x not in towers and FLOOR - y >= wh:
                continue
            if x == gx and y in (FLOOR - 1, FLOOR - 2):
                continue
            out.append(P(x, y, mat(x)))
    out += [P(x, FLOOR - th, BRICK) for x in (x0, x0 + w - 1)]
    out += [P(x, FLOOR - wh, BRICK) for x in range(x0 + 2, x0 + w - 2, 2) if x != gx]
    out += [P(gx - 1, FLOOR - 2, TORCH, "back"), P(gx + 1, FLOOR - 2, TORCH, "back")]
    if s["ladder"]:
        out += [P(x0 + 1, y, LADDER, "back") for y in range(FLOOR - 1, FLOOR - th, -1)]
    return out


def stairs_items(s, x0):
    n = s["n"]
    hts = [i + 1 for i in range(n)] if s["mode"] == "asc" else [min(i, 2 * n - 2 - i) + 1 for i in range(2 * n - 1)]
    return [P(x0 + i, FLOOR - r, s["mat"]) for r in range(max(hts)) for i, h in enumerate(hts) if h > r]


def granary_items(s, x0):
    w, ph = s["width"], s["ph"]
    px = [x0, x0 + w - 1]
    out = [P(x, y, s["pillar"]) for y in range(FLOOR, FLOOR - ph, -1) for x in px]
    prop = x0 + w // 2 if s["scaffold"] else None
    if prop is not None:
        out += [P(prop, y, WOOD, tag="scaffold") for y in range(FLOOR, FLOOR - ph, -1)]
    out += [P(x, FLOOR - ph, WOOD) for x in _by_support(list(range(x0, x0 + w)), px)]
    if prop is not None:
        out += [(Action("remove", prop, y, STONE), "scaffold") for y in range(FLOOR - ph + 1, FLOOR + 1)]
    out += [P(x0 + w // 2 + (i % 2), FLOOR - ph - 5, SAND) for i in range(s["grains"])]
    return out


def build_items(spec: dict, x0: int) -> list:
    kind = spec["kind"]
    fn = {"multistory": multistory_items, "colonnade": colonnade_items, "aqueduct": aqueduct_items,
          "fortress": fortress_items, "stairs": stairs_items, "granary": granary_items}.get(kind)
    if fn:
        return fn(spec, x0)
    if kind == "dune":
        return [P(x0 + 3 + (i % 2), SURF - 7, SAND) for i in range(spec["n"])]
    return [(a, "build") for a in D.build_actions(spec, x0)]


def pick_kinds(rng, n, pool, avoid_held=True):
    for _ in range(60):
        ks = rng.sample(pool, n)
        if not avoid_held or not any(frozenset(c) in HELD for c in combinations(ks, 2)):
            return ks
    return None


def sample_scene(rng, kinds, scale, gap=(2, 4), mirror_p=.5, max_actions=None):
    specs = [sample_spec(k, rng, scale) for k in kinds]
    gaps = [rng.randint(*gap) for _ in specs[1:]]
    total = sum(s["width"] for s in specs) + sum(gaps)
    if total > DW - 4:
        return None
    x = rng.randint(2, DW - 2 - total)
    x0s, per, mir = [], [], []
    for i, s in enumerate(specs):
        x0s.append(x)
        items = build_items(s, x)
        m = rng.random() < mirror_p
        if m:
            cx = 2 * x + s["width"] - 1
            items = [(Action(a.type, cx - a.x, a.y, a.material, a.layer), t) for a, t in items]
        per.append(items)
        mir.append(m)
        x += s["width"] + (gaps[i] if i < len(gaps) else 0)
    n = sum(len(p) for p in per)
    if max_actions and n > max_actions:
        return None
    return {"kinds": list(kinds), "specs": specs, "x0s": x0s, "mirrored": mir, "items": per, "scale": scale,
            "n_actions": n, "actions": [a for p in per for a, _ in p]}


def sample_split_kinds(rng, split, profile="v3"):
    cfg = BENCH_SPLITS[split]
    if profile == "simple_to_complex" and split == "id":
        n = rng.choices([2, 3], weights=[.6, .4])[0]
        return pick_kinds(rng, n, KINDS)
    if split == "combo_ood":
        a, b = rng.choice(HELD_PAIRS)
        ks = [a, b] if rng.random() < .5 else [b, a]
        if profile == "simple_to_complex" or rng.random() < .5:
            rest = [k for k in KINDS if k not in ks]
            ks.insert(rng.randint(0, 2), rng.choice(rest))
        return ks
    n = rng.choices(*cfg["n"])[0]
    return pick_kinds(rng, n, BIG_KINDS if cfg["scale"] == "big" else KINDS)


# ----------------------------------------------------------------------------- builder script
def merge_structs(per, rng):
    if rng.random() < .55:
        order = list(range(len(per)))
        rng.shuffle(order)
        return [it for i in order for it in per[i]], "seq"
    queues, pos, seq = [list(p) for p in per], [0] * len(per), []
    while True:
        left = [i for i in range(len(per)) if pos[i] < len(queues[i])]
        if not left:
            return seq, "interleave"
        i = rng.choices(left, weights=[len(queues[j]) - pos[j] for j in left])[0]
        seq.append(queues[i][pos[i]])
        pos[i] += 1


def partition(items, S, rng):
    """Split a sequence into <= S frames. Forced cuts: explicit flags and two actions on one cell."""
    n = len(items)
    forced, seen = set(), set()
    for i, (a, _, cut) in enumerate(items):
        key = (a.x, a.y, a.layer)
        if key in seen:
            forced.add(i - 1)
            seen = set()
        seen.add(key)
        if cut:
            forced.add(i)
            seen = set()
    forced = {c for c in forced if 0 <= c < n - 1}
    if len(forced) + 1 > S:
        return None
    free = [i for i in range(n - 1) if i not in forced]
    k = min(len(free), S - 1 - len(forced))
    chosen = set()
    if k:
        step, jitter = n / (k + 1), n / (k + 1) * .45
        for j in range(k):
            want = round((j + 1) * step - 1 + rng.uniform(-jitter, jitter))
            avail = [i for i in free if i not in chosen]
            chosen.add(min(avail, key=lambda i: abs(i - want)))
    cuts = sorted(forced | chosen)
    frames, start = [], 0
    for c in cuts + [n - 1]:
        frames.append([(a, t) for a, t, _ in items[start:c + 1]])
        start = c + 1
    return frames


def build_script(scene, rng, S, p, use_scaffold=True):
    per = scene["items"]
    if not use_scaffold:
        per = [[it for it in s if it[1] != "scaffold"] for s in per]
    seq, mode = merge_structs(per, rng)
    items, n_m1 = [], 0
    for a, tag in seq:
        if p > 0 and tag == "build" and a.type == "place" and a.layer == "front" and a.material in WRONG \
                and n_m1 < 6 and rng.random() < p:
            items.append((Action("place", a.x, a.y, rng.choice(WRONG[a.material])), "mistake", rng.random() < .8))
            items.append((Action("remove", a.x, a.y, STONE), "fix", False))
            n_m1 += 1
        items.append((a, tag, False))
    frames = partition(items, S, rng)
    if frames is None:
        return None
    sizes = [len(f) for f in frames]
    frames += [[] for _ in range(S - len(frames))]
    used = set()
    for x0, s in zip(scene["x0s"], scene["specs"]):
        used.update(range(x0 - 1, x0 + s["width"] + 1))
    cols = [x for x in range(1, DW - 1) if x not in used]
    rng.shuffle(cols)
    q = min(1.0, p * 6)
    n_stray, n_drop = sum(rng.random() < q for _ in range(2)), sum(rng.random() < q for _ in range(2))
    stats = {"mistake": n_m1, "stray": 0, "drop": 0}
    for _ in range(n_stray):
        if cols and S > 6:
            x, f = cols.pop(), rng.randint(0, S - 3)
            f2 = f + rng.randint(1, 4)
            if f2 < S:
                frames[f].append((D.place(x, FLOOR, rng.choice([STONE, WOOD, BRICK])), "stray"))
                frames[f2].append((Action("remove", x, FLOOR, STONE), "clear"))
                stats["stray"] += 1
    for _ in range(n_drop):
        if cols and S > 10:
            x, d = cols.pop(), rng.randint(3, 8)
            f = rng.randint(0, S - 1 - d - 3)
            f2 = f + d + rng.randint(0, 2)
            if f2 < S:
                frames[f].append((D.place(x, FLOOR - d, rng.choice([STONE, WOOD, BRICK, SAND])), "drop"))
                frames[f2].append((Action("remove", x, FLOOR, STONE), "clear"))
                stats["drop"] += 1
    return {"frames": frames, "mode": mode, "sizes": sizes, "stats": stats}


MUST = {"build", "scaffold", "mistake", "fix"}


def run_formation(frames, world_seed, F):
    w = D.new_world(world_seed)
    out, ok = [D.snap(w, [], False)], True
    for fr in frames:
        applied = []
        for a, tag in fr:
            if w.apply_action(a):
                applied.append(a)
            elif tag in MUST:
                ok = False
        w.tick()
        w.prune_background()
        out.append(D.snap(w, applied, True))
    while len(out) < F:
        out.append(D.tick_frame(w, []))
    return out, w, ok


# ----------------------------------------------------------------------------- interventions
def to_action(d: dict) -> Action:
    return Action(d["type"], d.get("x", 0), d.get("y", 0), d.get("material", STONE), d.get("layer", "front"))


def _a(type_, x, y, material=STONE, layer="front"):
    return {"type": type_, "x": int(x), "y": int(y), "material": int(material), "layer": layer}


def mk(kind, *steps):
    return {"kind": kind, "steps": [{"t": int(t), "actions": list(acts)} for t, acts in steps]}


def iv_actions(iv):
    return [a for st in iv["steps"] for a in st["actions"]]


def action_cells(iv, fg):
    cells = set()
    for a in iv_actions(iv):
        x, y = a["x"], a["y"]
        cells.add((x, y))
        if a["type"] in ("remove", "interact") and 0 <= y < DH and int(fg[y, x]) in DOORS:
            by = y + 1 if int(fg[y, x]) in (DOOR_TOP, OPEN_DOOR_TOP) else y
            cells.update({(x, by), (x, by - 1)})
    return cells


def cells_of(mask):
    return D._cells(mask)


def static_cascade(fg, cells):
    g = fg.copy()
    for x, y in cells:
        g[y, x] = 0
    ok = FormWorld(grid=g).supported()
    return int((np.isin(g[:SURF], list(STRUCT)) & ~ok[:SURF]).sum())


def weighted_cell(fg, rng, pool, k=6):
    cands = rng.sample(pool, min(k, len(pool)))
    return rng.choices(cands, weights=[1 + static_cascade(fg, [c]) for c in cands])[0]


def struct_pool(fg):
    return cells_of(np.isin(fg[:SURF], STRUCT_EVENT))


def gen_single(fg, bg, rng, wseed=0):
    r = rng.random()
    if r < .78:
        pool = struct_pool(fg)
        if pool:
            x, y = weighted_cell(fg, rng, pool)
            return mk("single", (0, [_a("remove", x, y)]))
    elif r < .88:
        cols = [x for x in range(DW) if fg[SURF - 1, x] != 0]
        if cols:
            return mk("single", (0, [_a("remove", rng.choice(cols), SURF, GRASS)]))
    return gen_back(fg, bg, rng, wseed) or None


def gen_back(fg, bg, rng, wseed=0):
    pool = cells_of(bg[:SURF] != 0)
    if not pool:
        return None
    x, y = rng.choice(pool)
    return mk("back", (0, [_a("remove", x, y, STONE, "back")]))


def gen_multi(fg, bg, rng, wseed=0):
    pool = struct_pool(fg)
    if len(pool) < 3:
        return None
    k = rng.choice([2, 3])
    x, y = weighted_cell(fg, rng, pool)
    if rng.random() < .6:
        near = [c for c in pool if abs(c[0] - x) <= 4 and abs(c[1] - y) <= 4 and c != (x, y)]
        picks = [(x, y)] + rng.sample(near, min(k - 1, len(near)))
    else:
        picks = [(x, y)] + rng.sample([c for c in pool if c != (x, y)], k - 1)
    if len(picks) < 2:
        return None
    return mk("multi", (0, [_a("remove", px, py) for px, py in picks]))


def gen_seq2(fg, bg, rng, wseed=0):
    first = gen_single(fg, bg, rng)
    if not first or first["kind"] == "back":
        return None
    j = rng.randint(1, 4)
    w = FormWorld(seed=wseed, grid=fg.copy(), background=bg.copy())
    w.apply_action(to_action(first["steps"][0]["actions"][0]))
    for _ in range(j):
        w.tick()
        w.prune_background()
    pool = struct_pool(w.grid)
    if not pool:
        return None
    x, y = rng.choice(pool)
    second = _a("remove", x, y)
    if rng.random() < .3:
        air = [(cx, cy) for cx, cy in pool if cy > 2 and w.grid[cy - 1, cx] == EMPTY]
        if air:
            cx, cy = rng.choice(air)
            second = _a("place", cx, cy - 1, rng.choice([STONE, BRICK, WOOD, SAND]))
    return mk("seq2", (0, first["steps"][0]["actions"]), (j, [second]))


def gen_place_remove(fg, bg, rng, wseed=0):
    pool = struct_pool(fg)
    if not pool:
        return None
    x, y = weighted_cell(fg, rng, pool)
    solid = np.isin(fg[:SURF], list(STRUCT))
    edge = [(cx, cy) for cx, cy in cells_of(solid) if cy > 2 and fg[cy - 1, cx] == EMPTY]
    if not edge:
        return None
    cx, cy = rng.choice(edge)
    put = _a("place", cx, cy - 1, rng.choice([STONE, BRICK, WOOD, SAND, WATER]))
    rm = _a("remove", x, y)
    return mk("place_remove", *( [(0, [rm, put])] if rng.random() < .5 else [(0, [put]), (1, [rm])] ))


def gen_door(fg, bg, rng, wseed=0):
    doors = cells_of(np.isin(fg[:SURF], list(DOORS)))
    if not doors:
        return None
    dx, dy = rng.choice(doors)
    acts = [_a("interact", dx, dy, DOOR)]
    near = [c for c in struct_pool(fg) if abs(c[0] - dx) <= 6]
    if near and rng.random() < .7:
        x, y = weighted_cell(fg, rng, near)
        acts.append(_a("remove", x, y))
    return mk("door", (0, acts))


def gen_drop(fg, bg, rng, wseed=0):
    for _ in range(30):
        x = rng.randrange(DW)
        cols = np.nonzero(fg[:SURF, x])[0]
        y = (int(cols[0]) - 3) if len(cols) and rng.random() < .7 else SURF - 6
        if 0 <= y < SURF and fg[y, x] == EMPTY:
            return mk("drop", (0, [_a("place", x, y, rng.choice([STONE, SAND, WATER, WOOD, BRICK]))]))
    return None


GEN = {"single": gen_single, "multi": gen_multi, "seq2": gen_seq2, "place_remove": gen_place_remove,
       "door": gen_door, "back": gen_back, "drop": gen_drop}
TRAIN_W = {"single": .30, "multi": .18, "seq2": .14, "place_remove": .10, "door": .08, "back": .08, "drop": .12}


def pick_training_intervention(fg, bg, rng, wseed=0, profile="v3"):
    if profile == "simple_to_complex":
        pool = struct_pool(fg)
        if pool:
            choices = [cell for cell in pool if 1 <= static_cascade(fg, [cell]) <= 6]
            if not choices:
                choices = [cell for cell in pool if static_cascade(fg, [cell]) <= 6]
            x, y = rng.choice(choices or pool)
            return mk("single", (0, [_a("remove", x, y)]))
        return mk("noop", (0, [_a("noop", 0, 0)]))
    if profile != "v3":
        raise ValueError(f"unknown training profile: {profile}")
    kinds, ws = list(TRAIN_W), list(TRAIN_W.values())
    for _ in range(8):
        iv = GEN[rng.choices(kinds, weights=ws)[0]](fg, bg, rng, wseed)
        if iv:
            return iv
    return mk("noop", (0, [_a("noop", 0, 0)]))


def op_clip(fg, bg, iv, seed, length=CLIP):
    w = FormWorld(seed=seed, grid=fg.copy(), background=bg.copy())
    frames = [D.snap(w, [], False)]
    by_t = {}
    for st in iv["steps"]:
        by_t.setdefault(st["t"], []).extend(st["actions"])
    for j in range(length - 1):
        applied = [a for a in map(to_action, by_t.get(j, [])) if w.apply_action(a) and a.type != "noop"]
        w.tick()
        w.prune_background()
        frames.append(D.snap(w, applied, True))
    return frames


# ----------------------------------------------------------------------------- training episodes
def sample_train_scene(rng, max_actions, profile="v3"):
    if profile == "simple_to_complex":
        n = rng.choices([1, 2], weights=[.35, .65])[0]
        kinds = D.TRAIN_KINDS
    elif profile == "v3":
        n = rng.choices([1, 2, 3], weights=[.30, .45, .25])[0]
        kinds = KINDS
    else:
        raise ValueError(f"unknown training profile: {profile}")
    ks = pick_kinds(rng, n, kinds)
    return sample_scene(rng, ks, "base", (2, 4), max_actions=max_actions) if ks else None


def make_episode_set(seed: int, index: int, T: int = T_DEFAULT, F: int = F_FORM, n_clips: int | None = None,
                     n_state_clips: int | None = None, p_mistake: float = DEFAULT_P, scaffold: bool = True,
                     profile: str = "v3") -> dict:
    if profile not in ("v3", "simple_to_complex"):
        raise ValueError(f"unknown training profile: {profile}")
    n_clips = (T - F) // CLIP if n_clips is None else n_clips
    n_state = T // CLIP if n_state_clips is None else n_state_clips
    assert F + n_clips * CLIP == T and n_state * CLIP == T, "frame budgets must match across conditions"
    S = F - 1 - SETTLE
    cap = S * 9
    world_seed = int_seed(seed, "world", index)
    for attempt in range(60):
        rng = rng_for(seed, "scene", index, attempt)
        p = p_mistake if attempt < 30 else 0.0
        scene = sample_train_scene(rng, cap, profile) if profile != "v3" else sample_train_scene(rng, cap)
        if scene is None:
            continue
        script = build_script(scene, rng, S, p, scaffold and attempt < 45)
        if script is None:
            continue
        form, fworld, ok = run_formation(script["frames"], world_seed, F)
        if not ok:
            continue
        final_fg, final_bg = D.settle_copy(fworld.grid, fworld.background, world_seed)
        settled = bool(np.array_equal(final_fg, fworld.grid) and np.array_equal(final_bg, fworld.background))
        if not settled or sum(check_violations(final_fg, final_bg).values()):
            continue
        break
    else:
        raise RuntimeError(f"no valid training scene for seed={seed} index={index}")

    erng = rng_for(seed, "events", index)
    events = [pick_training_intervention(final_fg, final_bg, erng, int_seed(seed, "evw", index, j), profile)
              for j in range(n_state)]

    def clips(base_fg, base_bg, evs, tag):
        out = []
        for j, ev in enumerate(evs):
            out += op_clip(base_fg, base_bg, ev, int_seed(seed, tag, index, j))
        return out

    n_ops = n_clips * CLIP
    episodes = {}
    f = clips(final_fg, final_bg, events, "state")
    episodes["state"] = D.pack(f, [1] * len(f))
    ops = clips(final_fg, final_bg, events[:n_clips], "mixed")
    episodes["process"] = D.pack(form + ops, [0] * F + [1] * n_ops)
    prng = rng_for(seed, "shuffle", index)
    order = list(range(F))
    prng.shuffle(order)
    shuffled = [(form[i][0], form[i][1], form[i][2], False) for i in order]
    episodes["shuffle"] = D.pack(shuffled + ops, [0] * F + [1] * n_ops)
    episodes["shuffle"]["order"] = order

    qrng = rng_for(seed, "partial", index)
    pool = list(range(1, 1 + S))
    picks = [qrng.choice(pool) for _ in range(n_state)]
    p_frames, p_kinds = [], []
    for j, k in enumerate(picks):
        bfg, bbg = D.settle_copy(form[k][0], form[k][1], int_seed(seed, "pbase", index, j))
        iv = pick_training_intervention(bfg, bbg, qrng, int_seed(seed, "pw", index, j), profile)
        p_kinds.append(iv["kind"])
        p_frames += op_clip(bfg, bbg, iv, int_seed(seed, "partial", index, j))
    episodes["partial"] = D.pack(p_frames, [1] * len(p_frames))
    episodes["partial"]["picks"] = picks

    n_script = int(sum(len(fr[2]) for fr in form[1:]))
    meta = {"kind": "+".join(scene["kinds"]), "kinds": scene["kinds"], "n_struct": len(scene["kinds"]),
            "specs": scene["specs"], "x0s": scene["x0s"], "mirrored": scene["mirrored"], "mode": script["mode"],
            "n_actions": scene["n_actions"], "n_script_actions": n_script, "chunk_sizes": script["sizes"],
            "max_chunk": max(script["sizes"]), "builder": script["stats"], "attempt": attempt,
            "formation_settled": settled, "final_hash": state_hash(final_fg, final_bg),
            "event_kinds": [e["kind"] for e in events], "partial_event_kinds": p_kinds}
    return {"episodes": episodes, "meta": meta, "final": (final_fg, final_bg)}


def _duration(seconds):
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class GenerationProgress:
    def __init__(self, label, total=None, unit="items", interval=10.0):
        self.label, self.total, self.unit, self.interval = label, total, unit, interval
        self.completed = 0
        self._stop, self._lock = Event(), Lock()

    def _report(self, status="running"):
        with self._lock:
            elapsed = monotonic() - self.started
            details = f"elapsed={_duration(elapsed)}"
            if self.total is not None:
                rate = self.completed / elapsed if elapsed > 0 else 0
                percent = 100 * self.completed / self.total if self.total else 100
                eta = _duration((self.total - self.completed) / rate) if rate else "unknown"
                details = (f"{self.completed}/{self.total} {self.unit} ({percent:.1f}%) "
                           f"{details} rate={rate:.2f} {self.unit}/s ETA={eta}")
            print(f"[data {strftime('%Y-%m-%d %H:%M:%S')}] {self.label}: {status} {details}", flush=True)

    def _heartbeat(self):
        while not self._stop.wait(self.interval):
            self._report()

    def __enter__(self):
        self.started = monotonic()
        self._report("started")
        self._thread = Thread(target=self._heartbeat, daemon=True)
        self._thread.start()
        return self

    def advance(self, n=1):
        with self._lock:
            self.completed += n
            first = self.completed == n
        if first:
            self._report()

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        self._thread.join()
        self._report("failed" if exc_type else "done")


def _generate_results(worker, jobs, workers, label, unit):
    results = [None] * len(jobs)
    if workers <= 1:
        with GenerationProgress(label, len(jobs), unit) as progress:
            for i, job in enumerate(jobs):
                results[i] = worker(job)
                progress.advance()
    else:
        with ProcessPoolExecutor(workers) as ex:
            todo = iter(enumerate(jobs))
            pending = {ex.submit(worker, job): i for i, job in
                       (next(todo) for _ in range(min(len(jobs), 2 * workers)))}
            with GenerationProgress(label, len(jobs), unit) as progress:
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        results[pending.pop(future)] = future.result()
                        progress.advance()
                    for _ in done:
                        entry = next(todo, None)
                        if entry is not None:
                            i, job = entry
                            pending[ex.submit(worker, job)] = i
    return results


def _train_worker(args):
    return args[1], make_episode_set(*args)


def generate_train(out: Path, episodes: int, T: int, seed: int, workers: int = 0, **kw) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    F = kw.get("F", F_FORM)
    n_clips = (T - F) // CLIP if kw.get("n_clips") is None else kw["n_clips"]
    n_state = T // CLIP if kw.get("n_state_clips") is None else kw["n_state_clips"]
    profile = kw.get("profile", "v3")
    if profile not in ("v3", "simple_to_complex"):
        raise ValueError(f"unknown training profile: {profile}")
    jobs = [(seed, i, T, F, kw.get("n_clips"), kw.get("n_state_clips"), kw.get("p_mistake", DEFAULT_P),
             kw.get("scaffold", True)) + ((profile,) if profile != "v3" else ()) for i in range(episodes)]
    workers = workers or min(os.cpu_count() or 1, 10)
    results = dict(_generate_results(_train_worker, jobs, workers,
                                    f"generate train profile={profile} workers={workers} out={out}", "episodes"))
    cols = {c: {"fg": [], "bg": [], "cont": [], "phase": [], "actions": [], "extra": []} for c in CONDITIONS}
    metas, finals_fg, finals_bg = [], [], []
    with GenerationProgress("assemble training arrays", episodes, "episodes") as progress:
        for i in range(episodes):
            res = results[i]
            metas.append(res["meta"])
            finals_fg.append(res["final"][0])
            finals_bg.append(res["final"][1])
            for c in CONDITIONS:
                ep = res["episodes"][c]
                for key in ("fg", "bg", "cont", "phase", "actions"):
                    cols[c][key].append(ep[key])
                cols[c]["extra"].append({k: ep[k] for k in ("order", "picks") if k in ep})
            progress.advance()
    summary = {}
    for c in CONDITIONS:
        with GenerationProgress(f"stack train_{c} arrays"):
            arrays = {k: np.stack(cols[c][k]) for k in ("fg", "bg", "cont", "phase")}
        with GenerationProgress(f"compress/write train_{c}.npz"):
            np.savez_compressed(out / f"train_{c}.npz", **arrays)
        with GenerationProgress(f"write train_{c}_actions.json"):
            (out / f"train_{c}_actions.json").write_text(json.dumps(
                {"schema": SCHEMA, "condition": c, "scenes": [m["kind"] for m in metas],
                 "extra": D._jsonable(cols[c]["extra"]), "actions": cols[c]["actions"]}), encoding="utf-8")
        summary[c] = {"episodes": episodes, "frames_per_episode": T, "total_frames": episodes * T,
                      "fg_shape": list(arrays["fg"].shape), "continuous_transitions": int(arrays["cont"].sum()),
                      "reset_frames": int((~arrays["cont"]).sum()),
                      "formation_frames": int((arrays["phase"] == 0).sum())}
    with GenerationProgress("compress/write train_finals.npz and train_scenes.json"):
        np.savez_compressed(out / "train_finals.npz", fg=np.stack(finals_fg), bg=np.stack(finals_bg))
        (out / "train_scenes.json").write_text(json.dumps(D._jsonable(metas)), encoding="utf-8")
    summary_all = {"schema": SCHEMA, "seed": seed, "grid": [DH, DW], "cell": CELL,
                   "layout": {"formation": F, "clip": CLIP, "state_clips": n_state, "mixed_clips": n_clips},
                   "builder": {"p_mistake": kw.get("p_mistake", DEFAULT_P), "scaffold": kw.get("scaffold", True)},
                   "conditions": summary}
    if profile != "v3":
        summary_all["profile"] = profile
    (out / "train_summary.json").write_text(json.dumps(summary_all, indent=2), encoding="utf-8")
    return summary_all


# ----------------------------------------------------------------------------- FormBench v3
def rollout(fg, bg, iv, seed, horizon=HORIZON):
    w = FormWorld(seed=seed, grid=fg.copy(), background=bg.copy())
    by_t = {}
    for st in iv["steps"]:
        by_t.setdefault(st["t"], []).extend(st["actions"])
    t_last = max(by_t) if by_t else 0
    fgs, bgs, settled = [], [], None
    for t in range(horizon):
        for a in by_t.get(t, []):
            w.apply_action(to_action(a))
        changed, _ = w.tick()
        removed = w.prune_background()
        fgs.append(w.grid.copy())
        bgs.append(w.background.copy())
        if t >= t_last and not changed and not removed:
            settled = t
            break
    while len(fgs) < horizon:
        fgs.append(fgs[-1])
        bgs.append(bgs[-1])
    return {"fg": np.stack(fgs), "bg": np.stack(bgs), "settled_at": settled, "t_last": t_last, "iv": iv}


def settled_build(scene, seed):
    w = D.new_world(seed)
    for a in scene["actions"]:
        if not w.apply_action(a):
            return None
        w.step()
    return w


def bench_candidates(fg, bg, rng, wseed):
    counts = {"single": 5, "multi": 4, "seq2": 4, "place_remove": 3, "door": 2, "back": 2}
    out, seen = [], set()
    for kind, n in counts.items():
        for _ in range(n * 2):
            if sum(1 for iv in out if iv["kind"] == kind) >= n:
                break
            iv = GEN[kind](fg, bg, rng, wseed)
            if iv:
                key = json.dumps(iv["steps"], sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    out.append(iv)
    return out


def randomized_pair_order(pair, rng):
    return pair[::-1] if rng.getrandbits(1) else pair


def sample_bench_item(split, seed, i, attempt, exclude, profile="v3"):
    if profile not in ("v3", "simple_to_complex"):
        raise ValueError(f"unknown bench profile: {profile}")
    rng = rng_for(seed, "bench", split, i, attempt)
    cfg = BENCH_SPLITS[split]
    kinds = sample_split_kinds(rng, split, profile)
    if not kinds:
        return None
    scene = sample_scene(rng, kinds, cfg["scale"], cfg["gap"], max_actions=330)
    if scene is None:
        return None
    w = settled_build(scene, int_seed(seed, "bench-world", split, i, attempt))
    if w is None:
        return None
    fg, bg = w.grid.copy(), w.background.copy()
    if sum(check_violations(fg, bg).values()):
        return None
    h = state_hash(fg, bg)
    if h in exclude:
        return None
    runs = []
    for j, iv in enumerate(bench_candidates(fg, bg, rng, int_seed(seed, "bench-cand", split, i, attempt))):
        r = rollout(fg, bg, iv, int_seed(seed, "bench-roll", split, i, attempt, j))
        if r["settled_at"] is None:
            continue
        final_fg, final_bg = r["fg"][-1], r["bg"][-1]
        if sum(check_violations(final_fg, final_bg).values()):
            continue
        ch = (fg != final_fg) | (bg != final_bg)
        A = action_cells(iv, fg)
        r["label"] = D.change_rate(fg, bg, final_fg, final_bg)
        r["n_changed"] = int(ch.sum())
        r["cascade"] = int(sum(1 for y, x in zip(*np.nonzero(ch)) if (x, y) not in A))
        r["depth"] = r["settled_at"] - r["t_last"]
        r["violations"] = 0
        r["n_actions"] = len(iv_actions(iv))
        runs.append(r)
    if len(runs) < 2:
        return None
    pairs = [(a, b) for a in range(len(runs)) for b in range(len(runs)) if a != b
             and runs[a]["label"] >= runs[b]["label"]
             and runs[a]["cascade"] >= MIN_CASCADE and runs[a]["depth"] >= MIN_DEPTH]
    if not pairs:
        return None
    gap = lambda p: runs[p[0]]["label"] - runs[p[1]]["label"]
    if profile == "simple_to_complex":
        pairs = [p for p in pairs if runs[p[0]]["cascade"] >= 6 and runs[p[0]]["depth"] >= 3
                 and runs[p[1]]["cascade"] >= 2 and gap(p) >= RANK_MARGIN]
        if not pairs:
            return None
    same = [p for p in pairs if runs[p[0]]["n_actions"] == runs[p[1]]["n_actions"] and gap(p) >= RANK_MARGIN]
    valid = [p for p in pairs if gap(p) >= RANK_MARGIN]
    pool = same or valid or pairs
    best = max(pool, key=lambda p: (gap(p), -p[0], -p[1]))
    pair = randomized_pair_order([runs[best[0]], runs[best[1]]], rng)
    return {"fg": fg, "bg": bg, "pair": pair, "rank_valid": bool(gap(best) >= RANK_MARGIN), "scene": scene,
            "hash": h}


def _bench_worker(args):
    split, seed, i, exclude, *options = args
    profile = options[0] if options else "v3"
    for attempt in range(240 if profile == "simple_to_complex" else 80):
        item = sample_bench_item(split, seed, i, attempt, exclude, profile)
        if item:
            item["attempts"] = attempt
            return item
    raise RuntimeError(f"could not sample {split} #{i}")


def generate_bench(out: Path, per_split: int, seed: int, exclude=None, workers: int = 0,
                   profile: str = "v3") -> dict:
    if profile not in ("v3", "simple_to_complex"):
        raise ValueError(f"unknown bench profile: {profile}")
    out.mkdir(parents=True, exist_ok=True)
    exclude = set(exclude or ())
    jobs = [(split, seed, i, exclude) + ((profile,) if profile != "v3" else ())
            for split in BENCH_SPLITS for i in range(per_split)]
    workers = workers or min(os.cpu_count() or 1, 10)
    items = _generate_results(_bench_worker, jobs, workers,
                              f"generate bench profile={profile} workers={workers} out={out}", "samples")
    if profile == "simple_to_complex":
        used = set(exclude)
        with GenerationProgress("deduplicate hard bench", len(items), "samples") as progress:
            for j, item in enumerate(items):
                if item["hash"] in used:
                    split, _, i = jobs[j][:3]
                    for attempt in range(item["attempts"] + 1, 240):
                        candidate = sample_bench_item(split, seed, i, attempt, used, profile)
                        if candidate:
                            candidate["attempts"] = attempt
                            item = candidate
                            break
                    else:
                        raise RuntimeError(f"no unique hard bench scene for {split} #{i}")
                    items[j] = item
                used.add(item["hash"])
                progress.advance()
    split_ids = [list(BENCH_SPLITS).index(j[0]) for j in jobs]
    skipped = {s: int(sum(it["attempts"] for it, j in zip(items, jobs) if j[0] == s)) for s in BENCH_SPLITS}
    pair = lambda key: np.stack([[r[key] for r in it["pair"]] for it in items])
    with GenerationProgress("compress/write bench.npz"):
        np.savez_compressed(
            out / "bench.npz",
            ctx_fg=np.stack([it["fg"] for it in items]), ctx_bg=np.stack([it["bg"] for it in items]),
            fut_fg=pair("fg"), fut_bg=pair("bg"),
            labels=pair("label").astype(np.float32), settled_at=pair("settled_at").astype(np.int16),
            n_changed=pair("n_changed").astype(np.int16), cascade=pair("cascade").astype(np.int16),
            depth=pair("depth").astype(np.int16), violations=pair("violations").astype(np.int16),
            rank_valid=np.array([it["rank_valid"] for it in items], dtype=bool),
            split_ids=np.array(split_ids, dtype=np.int8))
    meta = {
        "schema": SCHEMA, "grid": [DH, DW], "cell": CELL, "context_frames": CONTEXT, "horizon": HORIZON,
        "splits": list(BENCH_SPLITS),
        "split_definitions": {"id": "1-3 structures, training kind distribution, no held-out pair",
                              "combo_ood": "contains a held-out kind pair: " + ", ".join("+".join(p) for p in HELD_PAIRS),
                              "scale_ood": "big variants, 1-2 structures",
                              "multi_ood": "4-5 structures, gaps 1-3"},
        "scale": {k: v["scale"] for k, v in BENCH_SPLITS.items()},
        "tasks": {
            "collapse": "predict the settled state after the intervention; cell accuracy / structure IoU / change F1 / cascade F1",
            "counterfactual": "rank the two interventions by change rate (only samples with rank_valid)",
            "long_rollout": "violations of check_violations() on the last predicted frame (ground truth = 0)"},
        "intervention_format": "{kind, steps:[{t, actions:[{type,x,y,material,layer}]}]}; actions of step t are applied before tick t",
        "label": "fraction of mask cells that differ between the initial and final state; mask = built cells above the surface in either state plus any changed cell",
        "min_cascade": MIN_CASCADE, "min_depth": MIN_DEPTH,
        "samples": len(items), "scene_kinds": ["+".join(it["scene"]["kinds"]) for it in items],
        "scenes": D._jsonable([{"kinds": it["scene"]["kinds"], "specs": it["scene"]["specs"],
                                "x0s": it["scene"]["x0s"], "mirrored": it["scene"]["mirrored"]} for it in items]),
        "interventions": [[r["iv"] for r in it["pair"]] for it in items],
        "split_of_sample": [list(BENCH_SPLITS)[s] for s in split_ids], "rejected_attempts": skipped}
    if profile != "v3":
        meta["profile"] = profile
        meta["split_definitions"]["id"] = "hard transfer: 2-3 structures from all 16 kinds; not in-distribution"
        meta["split_definitions"]["combo_ood"] = "held-out pair plus a third structure"
        meta["hard_requirements"] = {"main_cascade": 6, "main_depth": 3,
                                     "other_cascade": 2, "rank_margin": RANK_MARGIN}
    (out / "bench.json").write_text(json.dumps(D._jsonable(meta), indent=1), encoding="utf-8")
    return {"samples": len(items), "rejected_attempts": skipped}


# ----------------------------------------------------------------------------- audit
def _f1(pc, tc):
    tp, fp, fn = (pc & tc).sum(), (pc & ~tc).sum(), (~pc & tc).sum()
    return 1.0 if not (tc.any() or pc.any()) else float(2 * tp / max(1, 2 * tp + fp + fn))


def _mask_of(cells):
    m = np.zeros((DH, DW), dtype=bool)
    for x, y in cells:
        m[y, x] = True
    return m


def baseline_scores(fg, bg, iv, true_fg, true_bg, which, seed=12345):
    if which == "copy_last":
        pf, pb = fg, bg
    elif which == "apply_edit":
        w = FormWorld(grid=fg.copy(), background=bg.copy())
        for st in sorted(iv["steps"], key=lambda s: s["t"]):
            for a in st["actions"]:
                w.apply_action(to_action(a))
        pf, pb = w.grid, w.background
    else:
        r = rollout(fg, bg, iv, seed)
        pf, pb = r["fg"][-1], r["bg"][-1]
    A = _mask_of(action_cells(iv, fg))
    base = D.score_outcome(pf, pb, true_fg, true_bg, fg, bg)
    pc, tc = (pf != fg) | (pb != bg), (true_fg != fg) | (true_bg != bg)
    base["cascade_f1"] = _f1(pc & ~A, tc & ~A)
    base["exact"] = float(np.array_equal(pf, true_fg) and np.array_equal(pb, true_bg))
    base["cascade_cells"] = int((tc & ~A).sum())
    return base


def audit(out: Path) -> dict:
    report: dict = {"schema": SCHEMA}
    scenes = json.loads((out / "train_scenes.json").read_text())
    cnt = lambda it: {k: it.count(k) for k in sorted(set(it))}
    kinds_flat = [k for s in scenes for k in s["kinds"]]
    ev = [k for s in scenes for k in s["event_kinds"]]
    summary = json.loads((out / "train_summary.json").read_text())
    train = {"episodes": len(scenes), "n_struct": cnt([s["n_struct"] for s in scenes]),
             "kind_counts": cnt(kinds_flat), "distinct_scenes": len({s["final_hash"] for s in scenes}),
             "mirrored_rate": float(np.mean([m for s in scenes for m in s["mirrored"]])),
             "formation_settled_rate": float(np.mean([s["formation_settled"] for s in scenes])),
             "mean_actions_per_scene": float(np.mean([s["n_actions"] for s in scenes])),
             "mean_script_actions": float(np.mean([s["n_script_actions"] for s in scenes])),
             "max_chunk": {"mean": float(np.mean([s["max_chunk"] for s in scenes])),
                           "max": int(max(s["max_chunk"] for s in scenes))},
             "build_mode": cnt([s["mode"] for s in scenes]),
             "builder_events": {k: int(sum(s["builder"][k] for s in scenes)) for k in ("mistake", "stray", "drop")},
             "episodes_with_any_mistake": float(np.mean([sum(s["builder"].values()) > 0 for s in scenes])),
             "scaffold_episodes": float(np.mean([any(sp.get("scaffold") for sp in s["specs"]) for s in scenes])),
             "retry_attempts_mean": float(np.mean([s["attempt"] for s in scenes])),
             "event_kinds": cnt(ev), "conditions": {}}
    finals = np.load(out / "train_finals.npz")
    hashes = {state_hash(f, b) for f, b in zip(finals["fg"], finals["bg"])}
    for c in CONDITIONS:
        d = np.load(out / f"train_{c}.npz")
        acts = json.loads((out / f"train_{c}_actions.json").read_text())["actions"]
        hist, per_frame = {}, []
        for e, ep in enumerate(acts):
            for t, fr in enumerate(ep):
                per_frame.append(len(fr))
                for a in fr:
                    hist[a["type"]] = hist.get(a["type"], 0) + 1
        fg, cont, phase = d["fg"], d["cont"], d["phase"]
        prev = np.concatenate([fg[:, :1], fg[:, :-1]], 1)
        same = (fg == prev).all(axis=(2, 3))
        nchg = (d["fg"] != prev).sum(axis=(2, 3)) + (d["bg"] != np.concatenate([d["bg"][:, :1], d["bg"][:, :-1]], 1)).sum(axis=(2, 3))
        m = cont & (np.arange(fg.shape[1])[None] > 0)
        train["conditions"][c] = {**summary["conditions"][c], "action_types": hist,
                                  "mean_actions_per_frame": float(np.mean(per_frame)),
                                  "max_actions_per_frame": int(max(per_frame)),
                                  "identity_transition_rate": float((same & m).sum() / max(1, m.sum())),
                                  "mean_changed_cells_per_transition": float(nchg[m].mean()),
                                  "mean_nonempty_fg_cells": float((fg != 0).sum(axis=(2, 3)).mean())}
    report["train"] = train

    meta = json.loads((out / "bench.json").read_text())
    b = np.load(out / "bench.npz")
    bench = {"samples": meta["samples"], "leak_with_train_finals": 0, "splits": {}}
    rows = {"copy_last": {}, "apply_edit": {}, "oracle_reseed": {}}
    shortcut = []
    for sid, split in enumerate(meta["splits"]):
        idx = np.where(b["split_ids"] == sid)[0]
        main = b["labels"][idx].argmax(axis=1)
        acc = {k: [] for k in rows}
        for i, k_main in zip(idx, main):
            fg, bg = b["ctx_fg"][i], b["ctx_bg"][i]
            if state_hash(fg, bg) in hashes:
                bench["leak_with_train_finals"] += 1
            iv = meta["interventions"][i][k_main]
            for k in acc:
                acc[k].append(baseline_scores(fg, bg, iv, b["fut_fg"][i, k_main, -1], b["fut_bg"][i, k_main, -1], k))
            if b["rank_valid"][i]:
                n0, n1 = (len(iv_actions(meta["interventions"][i][j])) for j in (0, 1))
                shortcut.append(0.5 if n0 == n1 else float((n0 > n1) == (b["labels"][i, 0] > b["labels"][i, 1])))
        lab = b["labels"][idx]
        ivk = [meta["interventions"][i][j]["kind"] for i in idx for j in (0, 1)]
        cascade_main = b["cascade"][idx, main]
        bench["splits"][split] = {
            "n": int(len(idx)), "mean_label": float(lab.mean()), "rank_valid_rate": float(b["rank_valid"][idx].mean()),
            "mean_cascade_main": float(cascade_main.mean()),
            "frac_cascade_ge4": float((cascade_main >= 4).mean()),
            "frac_cascade_ge8": float((cascade_main >= 8).mean()),
            "frac_cascade_ge16": float((cascade_main >= 16).mean()),
            "mean_depth_main": float(b["depth"][idx, main].mean()), "mean_cells_changed_main": float(b["n_changed"][idx, main].mean()),
            "gt_violations": int(b["violations"][idx].sum()),
            "intervention_kinds": cnt(ivk), "main_kinds": cnt([meta["interventions"][i][k]["kind"] for i, k in zip(idx, main)]),
            "mean_structs": float(np.mean([len(meta["scenes"][i]["kinds"]) for i in idx])),
            "scene_counts": cnt([meta["scene_kinds"][i] for i in idx]),
            "baselines": {k: {m: float(np.mean([r[m] for r in v])) for m in v[0]} for k, v in acc.items()}}
    bench["shortcut_more_actions_rank_acc"] = float(np.mean(shortcut)) if shortcut else None
    report["bench"] = bench
    (out / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def generate_all(out: Path, episodes: int, T: int, per_split: int, seed: int, workers: int = 0, **kw) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    train = generate_train(out, episodes, T, seed, workers, **kw)
    finals = np.load(out / "train_finals.npz")
    exclude = {state_hash(f, b) for f, b in zip(finals["fg"], finals["bg"])}
    bench = generate_bench(out, per_split, seed, exclude, workers)
    return {"train": train, "bench": bench, "audit": audit(out)}
