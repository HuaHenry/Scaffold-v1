"""Extend deterministic v3 episodes without touching the original 256-episode dataset."""
from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path

import numpy as np

from .shared import PACKAGE, common
from datasets_v3 import generate_train, generate_bench, state_hash


def check_prefix(original, extended):
    """Assert that every old video, action script, and scene occupies the identical new prefix."""
    old = common.load_json(original / "train_scenes.json")
    new = common.load_json(extended / "train_scenes.json")
    n = len(old)
    if new[:n] != old:
        raise ValueError("extended dataset changed old scene descriptions or hashes")
    for cond in common.CONDITIONS:
        with np.load(original / f"train_{cond}.npz") as a, np.load(extended / f"train_{cond}.npz") as b:
            for key in ("fg", "bg", "cont", "phase"):
                if not np.array_equal(a[key], b[key][:n]):
                    raise ValueError(f"{cond} {key}: regenerated prefix differs from original")
        a = common.load_json(original / f"train_{cond}_actions.json")
        b = common.load_json(extended / f"train_{cond}_actions.json")
        if a["actions"] != b["actions"][:n] or a["extra"] != b["extra"][:n]:
            raise ValueError(f"{cond}: regenerated actions or extra differ from original")
    return n


def input_hashes(directory):
    names = ["train_summary.json", "train_scenes.json", "train_finals.npz"]
    names += [f"train_{c}{suffix}" for c in common.CONDITIONS
              for suffix in (".npz", "_actions.json")]
    return {name: common.sha1_file(directory / name) for name in names}


def prepare_validation(data, val, workers):
    """Generate only into an empty directory; never alter an existing validation set."""
    if not (val / "train_summary.json").is_file():
        if val.exists() and any(val.iterdir()):
            raise ValueError(f"incomplete validation directory: {val}; select an empty --val directory")
        import_module(f"{PACKAGE}.gen_val").main([
            "--data", str(data), "--out", str(val), "--episodes", "32", "--extra", "64",
            "--seed", "1000", "--workers", str(workers)])
    return input_hashes(val)


def disjoint_scenes(data, val):
    train = {row["final_hash"] for row in common.load_json(data / "train_scenes.json")}
    validation = {row["final_hash"] for row in common.load_json(val / "train_scenes.json")}
    if train & validation:
        raise ValueError(f"validation {val} overlaps expanded training; use an empty separate --val directory")
    return train, validation


def build(source, out, val, episodes=4096, workers=0):
    source, out, val = map(Path, (source, out, val))
    if source.resolve() == out.resolve():
        raise ValueError("source and output must be different folders")
    old = common.load_summary(source)
    if old["seed"] != 0 or old["layout"] != {"formation": 32, "clip": 8, "state_clips": 8, "mixed_clips": 4}:
        raise ValueError("source must be the original v3 T=64/F=32 seed-0 dataset")
    if episodes <= old["conditions"]["state"]["episodes"]:
        raise ValueError("requested new data must extend beyond the original dataset")
    existing = out / "train_summary.json"
    # A complete training set may precede provenance when validation/bench generation failed.
    # Recheck every required file and the original prefix before freezing it below.
    if not existing.is_file():
        if out.exists() and any(out.glob("train_*")):
            raise ValueError("partial dataset exists; use a fresh output directory")
        generate_train(out, episodes, 64, 0, workers, F=32)
    summary = common.load_summary(out)
    if (summary["seed"] != 0 or summary["layout"] != old["layout"]
            or any(summary["conditions"][c]["episodes"] != episodes for c in common.CONDITIONS)):
        raise ValueError("extended folder was generated with another layout, seed or episode count")
    data_hashes = input_hashes(out)
    provenance = out / "curve_source.json"
    if provenance.exists():
        previous = common.load_json(provenance)
        if (previous.get("data_sha1") != data_hashes or previous.get("val") != common.rel(val)
                or previous.get("val_sha1") != input_hashes(val)):
            raise ValueError("existing expanded data or validation differs from frozen provenance")
    prefix = check_prefix(source, out)
    val_hashes = prepare_validation(out, val, workers)
    train, validation = disjoint_scenes(out, val)
    old_bench = source / "bench.npz"
    with np.load(old_bench) as z:
        old_bench_hash = {state_hash(f, b) for f, b in zip(z["ctx_fg"], z["ctx_bg"])}
    if train & old_bench_hash:
        if not (out / "bench.npz").is_file():
            generate_bench(out, 128, 0, train | validation, workers)
        bench = out / "bench.npz"
        bench_origin = "regenerated: old bench intersects expanded training scenes"
    else:
        bench = old_bench
        bench_origin = "original v3 bench: still disjoint from expanded training"
    with np.load(bench) as z:
        overlap = sum(state_hash(f, b) in (train | validation) for f, b in zip(z["ctx_fg"], z["ctx_bg"]))
    if overlap:
        raise ValueError(f"selected bench overlaps {overlap} training/validation scenes")
    bench_hashes = {p.name: common.sha1_file(p) for p in (bench, bench.with_suffix(".json"))}
    if provenance.exists() and previous.get("bench_sha1") != bench_hashes:
        raise ValueError("existing bench differs from frozen provenance")
    report = {"source": common.rel(source), "data": common.rel(out), "val": common.rel(val),
              "episodes": episodes, "prefix_identical_episodes": prefix,
              "distinct_training_scenes": len(train), "bench": common.rel(bench),
              "bench_origin": bench_origin, "bench_overlap": overlap,
              "data_sha1": data_hashes, "val_sha1": val_hashes, "bench_sha1": bench_hashes}
    common.dump_json(report, out / "curve_source.json")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=common.GAME_CODE / "output/data_v3")
    ap.add_argument("--out", type=Path, default=common.GAME_CODE / "output/data_v3_curve_4096")
    ap.add_argument("--val", type=Path, default=None,
                    help="independent validation directory (default: OUT/validation; never overwrite old val)")
    ap.add_argument("--episodes", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args(argv)
    out = common.resolve(a.out)
    result = build(common.resolve(a.source), out, common.resolve(a.val) if a.val else out / "validation",
                   a.episodes, a.workers)
    print("data:", result["data"], "val:", result["val"], "bench:", result["bench"], result["bench_origin"])


if __name__ == "__main__":
    main()
