"""Read-only data audit for the independent four-condition main experiment."""
import argparse
from pathlib import Path

import numpy as np

from .shared import (BATCH, CONDITIONS, DEFAULT_DATA, EPISODES, EPOCHS, EXP_NAME,
                     bench_hashes, budget, check_separation, common, data_hashes, require_data)
from datasets_v3 import state_hash
import datasets as base


def verify_inputs(audited):
    paths = {k: common.resolve(v) for k, v in audited["inputs"].items()}
    if (data_hashes(paths["data"]) != audited["hashes"]["data"]
            or data_hashes(paths["val"]) != audited["hashes"]["val"]
            or bench_hashes(paths["bench"]) != audited["hashes"]["bench"]):
        raise ValueError("main-experiment input files changed since audit")
    return paths


def audit(data):
    data = Path(data)
    source = require_data(data)
    summary = common.load_summary(data)
    val, bench = common.resolve(source["val"]), common.resolve(source["bench"])
    hashes = {"data": data_hashes(data), "val": data_hashes(val), "bench": bench_hashes(bench)}
    if (source.get("data") != common.rel(data) or source.get("episodes") != EPISODES
            or source.get("profile") != "simple_to_complex"
            or source.get("data_sha1") != hashes["data"]
            or source.get("val_sha1") != hashes["val"] or source.get("bench_sha1") != hashes["bench"]):
        raise ValueError("enlarged dataset differs from its frozen simple-to-complex provenance")
    if (summary.get("profile") != "simple_to_complex" or summary.get("seed") != 0
            or summary.get("builder") != {"p_mistake": 0.0, "scaffold": False}
            or summary.get("layout") != {"formation": 32, "clip": 8, "state_clips": 8, "mixed_clips": 4}
            or any(summary["conditions"][c]["episodes"] != EPISODES for c in CONDITIONS)):
        raise ValueError("main 03 requires the frozen 81920-episode simple training dataset")
    scenes = common.load_json(data / "train_scenes.json")
    if len(scenes) != EPISODES or any(s["n_struct"] > 2 or not set(s["kinds"]).issubset(base.TRAIN_KINDS)
                                     or set(s["event_kinds"]) - {"single", "noop"} for s in scenes):
        raise ValueError("training must contain 81920 simple construction/removal scenes")
    training = {s["final_hash"] for s in scenes}
    validation = {s["final_hash"] for s in common.load_json(val / "train_scenes.json")}
    bm = common.load_json(bench.with_suffix(".json"))
    with np.load(bench) as z:
        benchmark = {state_hash(f, b) for f, b in zip(z["ctx_fg"], z["ctx_bg"])}
        if (bm.get("profile") != "simple_to_complex" or len(z["ctx_fg"]) != 512
                or len(benchmark) != 512 or not np.all(z["rank_valid"])
                or not np.all(np.max(z["cascade"], axis=1) >= 6)
                or not np.all(np.min(z["cascade"], axis=1) >= 2)
                or not np.all(np.max(z["depth"], axis=1) >= 3)
                or not np.array_equal(np.bincount(z["split_ids"], minlength=4), np.full(4, 128))):
            raise ValueError("benchmark must satisfy the frozen hard-transfer definition")
    check_separation(training, validation, benchmark)
    result = {"inputs": {"data": common.rel(data), "val": common.rel(val), "bench": common.rel(bench)},
              "hashes": hashes, "protocol": {"epochs": EPOCHS, "batch": BATCH, "episodes": EPISODES,
                                             "conditions": list(CONDITIONS), "sampling": "full_epochs"},
              "train_unique_scenes": len(training), "train_duplicates": EPISODES-len(training),
              "train_validation_overlap": 0, "train_bench_overlap": 0, "validation_bench_overlap": 0,
              "conditions": {}}
    for c in CONDITIONS:
        with np.load(data / f"train_{c}.npz") as z:
            if z["fg"].shape[:2] != (EPISODES, 64) or z["bg"].shape[:2] != (EPISODES, 64):
                raise ValueError(f"{c}: unequal episode or frame budget")
            phase = z["phase"]
            if c in ("state", "partial"):
                if not np.all(phase == 1):
                    raise ValueError(f"{c}: incorrect phase labels")
            elif not np.all(phase[:, :32] == 0) or not np.all(phase[:, 32:] == 1):
                raise ValueError(f"{c}: incorrect formation/operation frame layout")
        pairs = common.make_pairs(data, c, "video", EPISODES)
        if len(pairs) != budget(c)["pairs"] or np.any(pairs[:, 0] >= EPISODES):
            raise ValueError(f"{c}: incorrect valid video transition budget")
        result["conditions"][c] = budget(c)
    verify_video_controls(data, EPISODES)
    return result


def verify_video_controls(data, episodes):
    data = Path(data)
    with np.load(data / "train_state.npz") as s, np.load(data / "train_process.npz") as p:
        for key in ("fg", "bg"):
            if not np.array_equal(s[key][:episodes, :32], p[key][:episodes, 32:]):
                raise ValueError("state and process do not share identical operation clips")
    process_actions = common.load_json(data / "train_process_actions.json")["actions"]
    state_actions = common.load_json(data / "train_state_actions.json")["actions"]
    if any(state_actions[i][:32] != process_actions[i][32:] for i in range(episodes)):
        raise ValueError("state/process shared operation actions differ")
    shuffle_actions = common.load_json(data / "train_shuffle_actions.json")
    with np.load(data / "train_process.npz") as p, np.load(data / "train_shuffle.npz") as s:
        for key in ("fg", "bg"):
            original, shuffled = p[key], s[key]
            if not np.array_equal(original[:episodes, 32:], shuffled[:episodes, 32:]):
                raise ValueError("shuffle changed process operation frames")
            for i in range(episodes):
                order = shuffle_actions["extra"][i]["order"]
                if sorted(order) != list(range(32)) or not np.array_equal(original[i, order], shuffled[i, :32]):
                    raise ValueError("shuffle must contain exactly the process formation frames, permuted")
    for i in range(episodes):
        order = shuffle_actions["extra"][i]["order"]
        if (shuffle_actions["actions"][i][:32] != [process_actions[i][j] for j in order]
                or shuffle_actions["actions"][i][32:] != process_actions[i][32:]):
            raise ValueError("shuffle formation or operation actions differ from process")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    result = audit(common.resolve(a.data))
    out = common.RESULTS_ROOT / EXP_NAME / a.run / "audit.json"
    if out.exists() and common.load_json(out) != result:
        raise ValueError("existing main 03 audit differs; use a new run")
    common.dump_json(result, out)
    print(common.rel(out))


if __name__ == "__main__":
    main()
