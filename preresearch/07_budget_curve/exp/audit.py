"""Preflight: equal frame budgets, prefix pairs, and train/validation/bench deduplication."""
import argparse
from pathlib import Path

import numpy as np

from .shared import CONDITIONS, EXP_NAME, LEVELS, common
from .generate_data import check_prefix, input_hashes
from datasets_v3 import state_hash


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
        if source["prefix_identical_episodes"] != 256 or source["data"] != common.rel(data):
            raise ValueError("expanded dataset did not verify the original 256-episode prefix")
        if source["val"] != common.rel(val) or source["bench"] != common.rel(bench):
            raise ValueError("validation or bench differs from expanded dataset provenance")
        if check_prefix(common.resolve(source["source"]), Path(data)) != 256:
            raise ValueError("original prefix has changed")
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
    if set(hashes) & (val_scenes | bench_hashes):
        raise ValueError("training final scenes overlap validation or bench contexts")
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
    if EXP_NAME == "07_02_budget_epochs" and a.epochs is None:
        ap.error("07_02 requires --epochs")
    r = a.run or common.current_run(EXP_NAME)
    data = common.resolve(a.data)
    provenance = common.load_json(data / "curve_source.json")
    val = common.resolve(a.val or provenance["val"])
    bench = common.resolve(a.bench or provenance["bench"])
    result = audit(data, val, bench)
    if a.epochs is not None:
        result["training_protocol"] = {"mode": "full_epochs", "epochs": a.epochs, "batch": 256}
    target = common.RESULTS_ROOT / EXP_NAME / r / "audit.json"
    if target.exists() and common.load_json(target) != result:
        raise ValueError("existing run audit differs; use a new run for a different training protocol")
    common.dump_json(result, target)
    print(common.rel(target))


if __name__ == "__main__":
    main()
