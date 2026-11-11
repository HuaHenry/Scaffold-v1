"""Separate, reusable evaluation probes; never generate training data."""
from itertools import combinations
from pathlib import Path

import numpy as np

from .shared import VERSION, common, write_json, write_npz
import datasets as D
import datasets_v3 as v3
from core import Action, check_violations


def truth_run(fg, bg, iv, seed):
    r = v3.rollout(fg, bg, iv, seed)
    if r["settled_at"] is None or sum(check_violations(r["fg"][-1], r["bg"][-1]).values()):
        return None
    changed = (fg != r["fg"][-1]) | (bg != r["bg"][-1])
    cells = v3.action_cells(iv, fg)
    excluded = np.zeros_like(changed)
    for x, y in cells:
        excluded[y, x] = True
    r.update(label=D.change_rate(fg, bg, r["fg"][-1], r["bg"][-1]),
             n_changed=int(changed.sum()), cascade=int((changed & ~excluded).sum()),
             depth=int(r["settled_at"] - r["t_last"]), violations=0)
    return r


def stable(fg, bg, seed):
    if sum(check_violations(fg, bg).values()):
        return False
    r = v3.rollout(fg, bg, v3.mk("noop", (0, [])), seed)
    return np.array_equal(r["fg"][-1], fg) and np.array_equal(r["bg"][-1], bg)


def single_scene(kind, value, rng):
    spec = v3.sample_spec(kind, rng, "base")
    spec["h" if kind == "tower" else "span"] = int(value)
    if kind == "bridge":
        spec["width"] = int(value)
        spec["beam"] = v3.BRICK if value > 10 else spec["beam"]
    x = rng.randint(2, D.DW - 2 - spec["width"])
    mirror = bool(rng.getrandbits(1))
    items = v3.build_items(spec, x)
    if mirror:
        center = 2 * x + spec["width"] - 1
        items = [(Action(a.type, center - a.x, a.y, a.material, a.layer), tag) for a, tag in items]
    return {"kinds": [kind], "specs": [spec], "x0s": [x], "mirrored": [mirror],
            "actions": [a for a, _ in items]}


def removal_pair(fg, bg, seed, rng):
    runs = []
    for j, action in enumerate(D.candidate_interventions(fg, rng, 16)):
        iv = v3.mk("single", (0, [D.act_dict(action)]))
        run = truth_run(fg, bg, iv, D.int_seed(seed, "removal", j))
        if run:
            runs.append(run)
    pairs = [(a, b) for a, b in combinations(range(len(runs)), 2)
             if max(runs[a]["cascade"], runs[b]["cascade"]) >= 1
             and abs(runs[a]["label"] - runs[b]["label"]) > 0]
    if not pairs:
        return None
    a, b = max(pairs, key=lambda p: (abs(runs[p[0]]["label"] - runs[p[1]]["label"]), -p[0], -p[1]))
    return v3.randomized_pair_order([runs[a], runs[b]], rng)


def scale_probe(excluded, per_setting=32, seed=2027, max_attempts=512):
    items, coverage, used = [], [], set(excluded)
    for kind, values in (("tower", range(4, 17)), ("bridge", range(4, 15))):
        for value in values:
            accepted, rejected = 0, {"invalid_or_unstable": 0, "duplicate_or_overlap": 0, "no_pair": 0}
            attempts = 0
            while accepted < per_setting and attempts < max_attempts:
                rng = D.rng_for(seed, "04-scale", kind, value, attempts)
                scene = single_scene(kind, value, rng)
                world_seed = D.int_seed(seed, "04-scale-world", kind, value, attempts)
                world = v3.settled_build(scene, world_seed)
                attempts += 1
                if world is None or not stable(world.grid, world.background, world_seed):
                    rejected["invalid_or_unstable"] += 1
                    continue
                fg, bg = world.grid.copy(), world.background.copy()
                h = v3.state_hash(fg, bg)
                if h in used:
                    rejected["duplicate_or_overlap"] += 1
                    continue
                pair = removal_pair(fg, bg, world_seed, rng)
                if pair is None:
                    rejected["no_pair"] += 1
                    continue
                used.add(h)
                items.append({"fg": fg, "bg": bg, "pair": pair, "scene": scene,
                              "group": {"probe": "scale", "kind": kind, "value": value,
                                        "within_training_parameter_support": 4 <= value <= (7 if kind == "tower" else 8)}})
                accepted += 1
            coverage.append({"kind": kind, "value": value, "requested": per_setting,
                             "accepted": accepted, "attempts": attempts, "rejections": rejected,
                             "status": "complete" if accepted == per_setting else "insufficient_unique_valid_scenes"})
            print(f"[04] scale {kind}={value}: {accepted}/{per_setting} accepted, {attempts} attempts", flush=True)
    return items, coverage


def intervention_owners(scene, iv):
    owners = set()
    for action in v3.iv_actions(iv):
        matching = [j for j, (x, spec) in enumerate(zip(scene["x0s"], scene["specs"]))
                    if x <= action["x"] < x + spec["width"]]
        if len(matching) != 1:
            return None
        owners.add(matching[0])
    return next(iter(owners)) if len(owners) == 1 else None


def isolate_component(fg, bg, scene, j):
    x, width = scene["x0s"][j], scene["specs"][j]["width"]
    single_fg, single_bg = D.terrain(), np.zeros_like(bg)
    single_fg[:, x:x + width] = fg[:, x:x + width]
    single_bg[:, x:x + width] = bg[:, x:x + width]
    return single_fg, single_bg


def component_probe(bench, excluded, max_cases=32, seed=2027):
    items, groups, used = [], [], set(excluded)
    rejected = {"cross_component_action": 0, "single_overlap_or_unstable": 0, "unsettled_outcome": 0}
    candidate_ids = np.nonzero(bench["split_ids"] == bench["splits"].index("combo_ood"))[0]
    for i in candidate_ids:
        if len(groups) >= max_cases:
            break
        scene, ivs = bench["meta"]["scenes"][i], bench["ivs"][i]
        owners = [intervention_owners(scene, iv) for iv in ivs]
        if any(owner is None for owner in owners):
            rejected["cross_component_action"] += 1
            continue
        fg, bg = bench["ctx_fg"][i], bench["ctx_bg"][i]
        singles = [isolate_component(fg, bg, scene, j) for j in range(len(scene["kinds"]))]
        hashes = [v3.state_hash(f, b) for f, b in singles]
        if (len(set(hashes)) != len(hashes) or any(h in used for h in hashes)
                or any(not stable(f, b, seed) for f, b in singles)):
            rejected["single_overlap_or_unstable"] += 1
            continue
        family, family_items = len(groups), []
        composite = [truth_run(fg, bg, iv, D.int_seed(seed, "04-component", int(i), k)) for k, iv in enumerate(ivs)]
        if any(r is None for r in composite):
            rejected["unsettled_outcome"] += 1
            continue
        family_items.append({"fg": fg, "bg": bg, "pair": composite, "scene": scene,
                             "group": {"probe": "components", "family": family, "source_index": int(i), "role": "combined"}})
        good = True
        for j, (single_fg, single_bg) in enumerate(singles):
            interventions = [iv if owner == j else v3.mk("noop", (0, [])) for owner, iv in zip(owners, ivs)]
            pair = [truth_run(single_fg, single_bg, iv, D.int_seed(seed, "04-component", int(i), k))
                    for k, iv in enumerate(interventions)]
            if any(r is None for r in pair):
                good = False
                break
            single_scene_meta = {"kinds": [scene["kinds"][j]], "specs": [scene["specs"][j]],
                                 "x0s": [scene["x0s"][j]], "mirrored": [scene["mirrored"][j]]}
            family_items.append({"fg": single_fg, "bg": single_bg, "pair": pair, "scene": single_scene_meta,
                                 "group": {"probe": "components", "family": family, "source_index": int(i), "role": "single", "component": j}})
        if not good:
            rejected["unsettled_outcome"] += 1
            continue
        indices = list(range(len(items), len(items) + len(family_items)))
        groups.append({"family": family, "source_index": int(i), "combined": indices[0],
                       "singles": indices[1:], "owners": owners, "kinds": scene["kinds"]})
        used.update(hashes)
        items.extend(family_items)
    print(f"[04] matched components: {len(groups)}/{max_cases} families", flush=True)
    return items, {"requested": max_cases, "accepted": len(groups), "rejections": rejected,
                   "status": "complete" if len(groups) == max_cases else "insufficient_matchable_cases", "families": groups}


def pack_probe(items):
    def pairs(key):
        return np.stack([np.stack([r[key] for r in item["pair"]]) for item in items])
    arrays = {"ctx_fg": np.stack([it["fg"] for it in items]), "ctx_bg": np.stack([it["bg"] for it in items]),
              "fut_fg": pairs("fg"), "fut_bg": pairs("bg"), "labels": pairs("label").astype(np.float32),
              "n_changed": pairs("n_changed").astype(np.int16), "cascade": pairs("cascade").astype(np.int16),
              "depth": pairs("depth").astype(np.int16), "settled_at": pairs("settled_at").astype(np.int16),
              "violations": pairs("violations").astype(np.int16), "split_ids": np.zeros(len(items), np.int8),
              "rank_valid": np.array([abs(it["pair"][0]["label"] - it["pair"][1]["label"]) >= D.RANK_MARGIN for it in items])}
    meta = {"schema": v3.SCHEMA, "grid": [D.DH, D.DW], "horizon": D.HORIZON, "splits": ["probe"],
            "samples": len(items), "scene_kinds": ["+".join(it["scene"]["kinds"]) for it in items],
            "scenes": [{k: it["scene"][k] for k in ("kinds", "specs", "x0s", "mirrored")} for it in items],
            "interventions": [[r["iv"] for r in it["pair"]] for it in items],
            "groups": [it["group"] for it in items]}
    return arrays, meta


def verify_probe(directory, expected):
    directory = Path(directory)
    done = common.load_json(directory / "complete.json")
    if done["settings"] != expected:
        raise ValueError("probe settings or source changed; use a new run")
    for name, sha1 in done["hashes"].items():
        if common.sha1_file(directory / name) != sha1:
            raise ValueError("completed probe files changed; refusing regeneration")
    return done


def prepare_probe(bundle, run_dir, name, per_setting=32, seed=2027, max_attempts=512):
    if min(per_setting, max_attempts) <= 0 or seed < 0:
        raise ValueError("probe counts must be positive and seed nonnegative")
    directory = Path(run_dir) / "data" / name
    config = {"version": VERSION, "source": bundle["manifest"], "probe": name,
              "per_setting": per_setting, "seed": seed, "max_attempts": max_attempts}
    if (directory / "complete.json").exists():
        print(f"[04] reuse completed {name} evaluation data", flush=True)
        return directory, verify_probe(directory, config)
    if directory.exists():
        raise ValueError("partial probe exists; use a new run, not silent regeneration")
    excluded = {row["final_hash"] for row in bundle["training"]}
    val = common.resolve(bundle["audited"]["inputs"]["val"])
    excluded.update(row["final_hash"] for row in common.load_json(val / "train_scenes.json"))
    excluded.update(row["final_hash"] for row in bundle["features"])
    if name == "scale":
        items, coverage = scale_probe(excluded, per_setting, seed, max_attempts)
    elif name == "components":
        items, coverage = component_probe(bundle["bench"], excluded, per_setting, seed)
    else:
        raise ValueError("unknown evaluation probe")
    directory.mkdir(parents=True)
    write_json(directory / "coverage.json", coverage)
    if items:
        arrays, meta = pack_probe(items)
        write_npz(directory / "bench.npz", arrays)
        write_json(directory / "bench.json", meta)
    names = ["coverage.json"] + (["bench.npz", "bench.json"] if items else [])
    done = {"settings": config, "n_samples": len(items),
            "status": "ready" if items else "no_unique_valid_cases",
            "hashes": {name: common.sha1_file(directory / name) for name in names}}
    write_json(directory / "complete.json", done)
    return directory, done
