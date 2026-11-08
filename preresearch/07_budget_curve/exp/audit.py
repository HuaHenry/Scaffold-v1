"""Preflight: equal frame budgets, prefix pairs, and train/validation/bench deduplication."""
import argparse
from pathlib import Path

import numpy as np

from .shared import CONDITIONS, EXP_NAME, LEVELS, common
from .generate_data import check_prefix, input_hashes
from datasets_v3 import state_hash


def check_scene_separation(training, validation, benchmark, include_validation_bench=False):
    if training & (validation | benchmark):
        raise ValueError("training final scenes overlap validation or bench contexts")
    if include_validation_bench and validation & benchmark:
        raise ValueError("validation scenes overlap the hard benchmark")


def audit(data, val, bench, levels=LEVELS):
    summary = common.load_summary(data)
    largest = max(levels)
    if summary["seed"] != 0 or summary["layout"] != {"formation": 32, "clip": 8, "state_clips": 8, "mixed_clips": 4}:
        raise ValueError("experiment 07 requires the existing T=64/F=32/seed=0 v3 data")
    scenes = common.load_json(Path(data) / "train_scenes.json")
    if len(scenes) < largest:
        raise ValueError(f"dataset needs {largest} episodes, only {len(scenes)} available")
    if largest > 256:
        source = common.load_json(Path(data) / "curve_source.json")
        if source["data"] != common.rel(data):
            raise ValueError("dataset does not match its provenance")
        if source["val"] != common.rel(val) or source["bench"] != common.rel(bench):
            raise ValueError("validation or bench differs from expanded dataset provenance")
        if EXP_NAME in ("07_03_simple_to_complex", "07_03_simple_to_complex_81920"):
            if (source.get("profile") != "simple_to_complex" or source["prefix_identical_episodes"] != 0
                    or summary.get("profile") != "simple_to_complex"
                    or summary.get("builder") != {"p_mistake": 0.0, "scaffold": False}):
                raise ValueError("new experiment needs a fully regenerated simple training dataset")
            if EXP_NAME == "07_03_simple_to_complex_81920":
                prefix_source = common.resolve(source.get("prefix_source", ""))
                if (source.get("extended_prefix_identical_episodes") != 40960
                        or common.load_json(prefix_source / "curve_source.json").get("data_sha1")
                        != input_hashes(prefix_source)
                        or check_prefix(prefix_source, Path(data)) != 40960):
                    raise ValueError("81920 dataset must preserve the frozen 40960 simple training prefix")
        elif source["prefix_identical_episodes"] != 256 or check_prefix(common.resolve(source["source"]), Path(data)) != 256:
            raise ValueError("original prefix has changed")
        if EXP_NAME == "07_03_budget_epochs":
            prefix_source = source.get("prefix_source")
            if (not prefix_source or source.get("extended_prefix_identical_episodes") != 4096
                    or check_prefix(common.resolve(prefix_source), Path(data)) != 4096):
                raise ValueError("20480 dataset must preserve the frozen 4096-episode prefix")
        if (source.get("data_sha1") != input_hashes(Path(data))
                or source.get("val_sha1") != input_hashes(Path(val))
                or source.get("bench_sha1") != {p.name: common.sha1_file(p)
                      for p in (Path(bench), Path(bench).with_suffix(".json"))}):
            raise ValueError("training, validation or bench input hash differs from provenance")
    hashes = [row["final_hash"] for row in scenes[:largest]]
    val_scenes = {row["final_hash"] for row in common.load_json(Path(val) / "train_scenes.json")}
    with np.load(bench) as b:
        bench_hashes = {state_hash(fg, bg) for fg, bg in zip(b["ctx_fg"], b["ctx_bg"])}
        n_bench = len(b["ctx_fg"])
    check_scene_separation(set(hashes), val_scenes, bench_hashes,
                           include_validation_bench=EXP_NAME == "07_03_simple_to_complex_81920")
    if EXP_NAME in ("07_03_simple_to_complex", "07_03_simple_to_complex_81920"):
        import datasets as base
        bm = common.load_json(Path(bench).with_suffix(".json"))
        with np.load(bench) as b:
            cascade, depth, ranks, split_ids = (b[k] for k in ("cascade", "depth", "rank_valid", "split_ids"))
        if (bm.get("profile") != "simple_to_complex" or n_bench != 512 or len(bench_hashes) != n_bench
                or not np.all(ranks) or not np.all(np.max(cascade, axis=1) >= 6)
                or not np.all(np.min(cascade, axis=1) >= 2)
                or not np.all(np.max(depth, axis=1) >= 3)
                or not np.array_equal(np.bincount(split_ids, minlength=4), np.full(4, 128))):
            raise ValueError("hard bench must contain unique, valid, nontrivial paired cascades")
        if (any(s["n_struct"] > 2 or not set(s["kinds"]).issubset(base.TRAIN_KINDS)
                or set(s["event_kinds"]) - {"single", "noop"} for s in scenes)
                or not any(k not in base.TRAIN_KINDS for s in bm["scenes"] for k in s["kinds"])):
            raise ValueError("training scenes must be simple and hard bench must contain unseen kinds")
    out = {"inputs": {"data": common.rel(data), "val": common.rel(val), "bench": common.rel(bench)},
           "hashes": {"data": input_hashes(Path(data)), "val": input_hashes(Path(val)),
                      "bench": {p.name: common.sha1_file(p) for p in
                                (Path(bench), Path(bench).with_suffix(".json"))}},
           "train_final_hashes_unique": len(set(hashes)), "train_internal_duplicates": largest - len(set(hashes)),
           "validation_overlap": 0, "bench_overlap": 0,
           "n_bench": n_bench, "levels": {}}
    for cond in CONDITIONS:
        with np.load(Path(data) / f"train_{cond}.npz") as d:
            if d["fg"].shape[:2] != (len(scenes), 64) or d["bg"].shape[:2] != (len(scenes), 64):
                raise ValueError(f"{cond} has unexpected video layout")
            if cond == "process" and not np.all((d["phase"][:, :32] == 0) & (d["phase"][:, 32:] == 1)):
                raise ValueError("process formation/operation frames are not 32/32")
            if cond == "state" and not np.all(d["phase"] == 1):
                raise ValueError("state has non-operation frames")
    for n in levels:
        entry = {"frames_per_condition": n * 64, "episodes": n, "conditions": {}}
        for cond in CONDITIONS:
            pair = common.make_pairs(data, cond, "video", n)
            if not np.array_equal(pair[:len(pair)//n, 1], pair[-len(pair)//n:, 1]):
                raise ValueError("episode pair pattern is not stable")
            expect = (56 if cond == "state" else 59) * n
            if len(pair) != expect or np.any(pair[:, 0] >= n):
                raise ValueError(f"{cond}@{n}: expected {expect} valid pairs, got {len(pair)}")
            entry["conditions"][cond] = {"video_pairs": len(pair),
                                            "formation_pairs": 31*n if cond == "process" else 0,
                                            "operation_pairs": 28*n if cond == "process" else 56*n,
                                            "batch_formation_fraction": 0.5 if cond == "process" else 0.0}
        out["levels"][str(n)] = entry
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=None)
    ap.add_argument("--data", type=Path, default=common.GAME_CODE / "output/data_v3_curve_4096")
    ap.add_argument("--val", type=Path, default=None)
    ap.add_argument("--bench", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="record full-epoch training protocol")
    a = ap.parse_args(argv)
    if a.epochs is not None and a.epochs <= 0:
        ap.error("--epochs must be positive")
    if EXP_NAME in ("07_02_budget_epochs", "07_03_budget_epochs", "07_03_simple_to_complex",
                    "07_03_simple_to_complex_81920") and a.epochs is None:
        ap.error("full-epoch budget curves require --epochs")
    r = a.run or common.current_run(EXP_NAME)
    data = common.resolve(a.data)
    provenance = common.load_json(data / "curve_source.json")
    val = common.resolve(a.val or provenance["val"])
    bench = common.resolve(a.bench or provenance["bench"])
    result = audit(data, val, bench)
    if a.epochs is not None:
        result["training_protocol"] = {"mode": "full_epochs", "epochs": a.epochs, "batch": 256}
        if EXP_NAME in ("07_03_budget_epochs", "07_03_simple_to_complex", "07_03_simple_to_complex_81920"):
            for level in result["levels"].values():
                level["conditions"]["process"]["batch_formation_fraction"] = 31 / 59
    target = common.RESULTS_ROOT / EXP_NAME / r / "audit.json"
    if target.exists() and common.load_json(target) != result:
        raise ValueError("existing run audit differs; use a new run for a different training protocol")
    common.dump_json(result, target)
    print(common.rel(target))


if __name__ == "__main__":
    main()
