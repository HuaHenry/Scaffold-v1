"""Validation set for monitoring (plan 00 section 2): 32 episodes of all four conditions, seed 1000, same scenes
across conditions, no scene identical to a training final state.

python -m exp.gen_val [--data DIR] [--out DIR] [--episodes 32] [--seed 1000] [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from . import common
from .common import resolve


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=common.DEFAULT_DATA, help="training data dir (for the overlap check)")
    ap.add_argument("--out", type=Path, default=common.DEFAULT_VAL)
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--extra", type=int, default=16, help="extra episodes generated to replace overlapping ones")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args(argv)
    data, out = resolve(a.data), resolve(a.out)

    import datasets_v3 as v3
    raw = out / "_raw"
    layout = common.load_summary(data)["layout"]
    v3.generate_train(raw, a.episodes + a.extra, a.T, a.seed, a.workers, F=layout["formation"])
    train_hash = {m["final_hash"] for m in common.load_json(data / "train_scenes.json")}
    scenes = common.load_json(raw / "train_scenes.json")
    keep = [i for i, m in enumerate(scenes) if m["final_hash"] not in train_hash][:a.episodes]
    if len(keep) < a.episodes:
        raise SystemExit(f"only {len(keep)} non-overlapping validation episodes; raise --extra")
    for c in common.CONDITIONS:
        z = np.load(raw / f"train_{c}.npz")
        np.savez_compressed(out / f"train_{c}.npz", **{k: z[k][keep] for k in z.files})
        j = common.load_json(raw / f"train_{c}_actions.json")
        for key in ("scenes", "extra", "actions"):
            j[key] = [j[key][i] for i in keep]
        (out / f"train_{c}_actions.json").write_text(json.dumps(j), encoding="utf-8")
    z = np.load(raw / "train_finals.npz")
    np.savez_compressed(out / "train_finals.npz", **{k: z[k][keep] for k in z.files})
    (out / "train_scenes.json").write_text(json.dumps([scenes[i] for i in keep]), encoding="utf-8")
    summ = common.load_json(raw / "train_summary.json")
    for c in summ["conditions"].values():
        c["episodes"] = len(keep)
    common.dump_json(summ, out / "train_summary.json")
    common.dump_json({"generated": a.episodes + a.extra, "kept": len(keep),
                      "overlap_removed": len([m for m in scenes if m["final_hash"] in train_hash]),
                      "kept_indices": keep}, out / "overlap.json")
    shutil.rmtree(raw)
    print(f"validation set written to {common.rel(out)} ({len(keep)} episodes)")


if __name__ == "__main__":
    main()
