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


def prepare_validation(data, val, workers, extra=64):
    """Generate only into an empty directory; never alter an existing validation set."""
    if not (val / "train_summary.json").is_file():
        if val.exists() and any(val.iterdir()):
            raise ValueError(f"incomplete validation directory: {val}; select an empty --val directory")
        import_module(f"{PACKAGE}.gen_val").main([
            "--data", str(data), "--out", str(val), "--episodes", "32", "--extra", str(extra),
            "--seed", "1000", "--workers", str(workers)])
    return input_hashes(val)


def disjoint_scenes(data, val):
    train = {row["final_hash"] for row in common.load_json(data / "train_scenes.json")}
    validation = {row["final_hash"] for row in common.load_json(val / "train_scenes.json")}
    if train & validation:
        raise ValueError(f"validation {val} overlaps expanded training; use an empty separate --val directory")
    return train, validation


def frozen_hard_bench_disjoint(path, train, validation, profile):
    if not path.is_file():
        raise FileNotFoundError(f"frozen prefix benchmark missing: {path}")
    if common.load_json(path.with_suffix(".json")).get("profile") != profile:
        raise ValueError("prefix benchmark has a different difficulty profile")
    with np.load(path) as z:
        hashes = {state_hash(f, b) for f, b in zip(z["ctx_fg"], z["ctx_bg"])}
    return not (train | validation) & hashes


def build(source, out, val, episodes=4096, workers=0, prefix_source=None, val_extra=64,
          profile="v3"):
    if profile not in ("v3", "simple_to_complex"):
        raise ValueError(f"unknown data profile: {profile}")
    source, out, val = map(Path, (source, out, val))
    prefix_source = Path(prefix_source) if prefix_source is not None else None
    if prefix_source is not None and prefix_source.resolve() == out.resolve():
        raise ValueError("prefix source must not be the output directory")
    if profile == "simple_to_complex" and prefix_source is not None:
        frozen = common.load_json(prefix_source / "curve_source.json")
        if (frozen.get("profile") != profile or frozen.get("episodes") != 40960
                or frozen.get("data_sha1") != input_hashes(prefix_source)):
            raise ValueError("40960-episode prefix no longer matches its frozen provenance")
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
        if profile == "simple_to_complex":
            generate_train(out, episodes, 64, 0, workers, F=32,
                           profile=profile, p_mistake=0.0, scaffold=False)
        else:
            generate_train(out, episodes, 64, 0, workers, F=32)
    summary = common.load_summary(out)
    if (summary["seed"] != 0 or summary["layout"] != old["layout"]
            or (summary.get("profile", "v3") != profile)
            or (profile == "simple_to_complex" and summary["builder"] != {"p_mistake": 0.0, "scaffold": False})
            or any(summary["conditions"][c]["episodes"] != episodes for c in common.CONDITIONS)):
        raise ValueError("extended folder was generated with another layout, seed or episode count")
    data_hashes = input_hashes(out)
    provenance = out / "curve_source.json"
    if provenance.exists():
        previous = common.load_json(provenance)
        if (previous.get("data_sha1") != data_hashes or previous.get("val") != common.rel(val)
                or previous.get("val_sha1") != input_hashes(val)
                or previous.get("prefix_source") != (common.rel(prefix_source) if prefix_source else None)
                or previous.get("profile", "v3") != profile):
            raise ValueError("existing expanded data, prefix, profile or validation differs from frozen provenance")
    prefix = check_prefix(source, out) if profile == "v3" else 0
    extended_prefix = check_prefix(prefix_source, out) if prefix_source else None
    val_hashes = prepare_validation(out, val, workers, val_extra)
    train, validation = disjoint_scenes(out, val)
    old_bench = source / "bench.npz"
    with np.load(old_bench) as z:
        old_bench_hash = {state_hash(f, b) for f, b in zip(z["ctx_fg"], z["ctx_bg"])}
    if profile == "simple_to_complex":
        earlier_bench = prefix_source / "bench.npz" if prefix_source else None
        if earlier_bench and (frozen.get("bench") != common.rel(earlier_bench)
                              or frozen.get("bench_sha1") != {p.name: common.sha1_file(p)
                                    for p in (earlier_bench, earlier_bench.with_suffix(".json"))}):
            raise ValueError("40960-episode prefix benchmark differs from its frozen provenance")
        if earlier_bench and frozen_hard_bench_disjoint(earlier_bench, train, validation, profile):
            bench = earlier_bench
            bench_origin = "frozen previous hard bench: still disjoint from the full training/validation data"
        else:
            bench = out / "bench.npz"
            if not bench.is_file():
                generate_bench(out, 128, 2027, train | validation, workers, profile=profile)
            if common.load_json(bench.with_suffix(".json")).get("profile") != profile:
                raise ValueError("existing bench has a different difficulty profile")
            bench_origin = "hard transfer: regenerated separately for the full simple training dataset"
    elif train & old_bench_hash:
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
    if prefix_source:
        report["prefix_source"] = common.rel(prefix_source)
        report["extended_prefix_identical_episodes"] = extended_prefix
    if profile != "v3":
        report["profile"] = profile
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
    ap.add_argument("--prefix-source", type=Path, default=None,
                    help="also verify all old episodes/conditions against this frozen larger prefix")
    ap.add_argument("--val-extra", type=int, default=64,
                    help="additional candidate validation scenes to filter against the full training dataset")
    ap.add_argument("--profile", choices=("v3", "simple_to_complex"), default="v3")
    a = ap.parse_args(argv)
    out = common.resolve(a.out)
    result = build(common.resolve(a.source), out, common.resolve(a.val) if a.val else out / "validation",
                   a.episodes, a.workers,
                   common.resolve(a.prefix_source) if a.prefix_source else None, a.val_extra, a.profile)
    print("data:", result["data"], "val:", result["val"], "bench:", result["bench"], result["bench_origin"])


if __name__ == "__main__":
    main()
