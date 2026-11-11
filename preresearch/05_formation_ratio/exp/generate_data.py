"""Replay only formation segments of frozen scenes, with resumable episode shards."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import shutil
import zipfile

import numpy as np

from .shared import (EPISODES, VERSION, budget, check_output_tree, common, new_levels, run_lock, run_path, variant_hashes,
                     verify_manifest, verify_variant, write_json, write_npz)
import datasets as D
import datasets_v3 as v3
from core import check_violations


def partition_actions(actions, slots):
    if slots <= 0:
        raise ValueError("formation needs an initial frame and four settling frames")
    if not actions:
        return [[] for _ in range(slots)]
    if slots > len(actions):
        edges = np.linspace(0, len(actions), slots + 1).astype(int)
        return [actions[start:end] for start, end in zip(edges[:-1], edges[1:])]
    cuts, seen = [], set()
    for i, action in enumerate(actions):
        key = (action["x"], action["y"])
        if key in seen:
            cuts.append(i)
            seen.clear()
        seen.add(key)
    if len(cuts) + 1 > slots:
        raise ValueError(f"too many repeated-cell actions: need at least {len(cuts) + 1} construction slots, have {slots}")
    boundaries = [0, *cuts, len(actions)]
    lengths = np.diff(boundaries)
    allocations = np.ones(len(lengths), dtype=int)
    used = min(slots, len(actions))
    while allocations.sum() < used:
        score = np.where(allocations < lengths, lengths / (allocations + 1), -1)
        allocations[int(np.argmax(score))] += 1
    frames = []
    for start, end, count in zip(boundaries[:-1], boundaries[1:], allocations):
        for indices in np.array_split(np.arange(start, end), count):
            frames.append(actions[int(indices[0]):int(indices[-1]) + 1])
    frames += [[] for _ in range(slots - len(frames))]
    if [a for frame in frames for a in frame] != actions:
        raise ValueError("formation partition lost or reordered actions")
    return frames


def replay_formation(frame_actions, F, index, expected_fg, expected_bg, seed=0):
    if F not in (8, 16, 48):
        raise ValueError("only new F=8/16/48 need formation replay")
    if len(frame_actions) != 32:
        raise ValueError("source must contain exactly 32 formation action frames")
    if any(frame_actions[t] for t in range(28, 32)) or frame_actions[0]:
        raise ValueError("source has actions outside the original construction slots")
    actions = [a for frame in frame_actions[1:28] for a in frame]
    frames = partition_actions(actions, F - 5)
    script = [[(v3.to_action(action), "build") for action in frame] for frame in frames]
    snapshots, world, ok = v3.run_formation(script, D.int_seed(seed, "world", index), F)
    if not ok or len(snapshots) != F:
        raise ValueError("compressed construction contains an unapplied action")
    if not np.array_equal(world.grid, expected_fg) or not np.array_equal(world.background, expected_bg):
        raise ValueError("changing construction timing changed the frozen final structure")
    settled_fg, settled_bg = D.settle_copy(world.grid, world.background, D.int_seed(seed, "world", index))
    if (not np.array_equal(settled_fg, world.grid) or not np.array_equal(settled_bg, world.background)
            or sum(check_violations(world.grid, world.background).values())):
        raise ValueError("compressed final structure is unstable or violates rules")
    formed = D.pack(snapshots, [0] * F)
    encoded = [a for frame in formed["actions"] for a in frame]
    if encoded != [D.act_dict(v3.to_action(a)) for a in actions]:
        raise ValueError("applied formation actions differ from the frozen complete action order")
    return formed, {"n_actions": len(actions), "max_actions_per_frame": max(map(len, frames), default=0)}


def build_episode(F, index, formation_actions, state, seed=0):
    final_fg, final_bg = state["fg"][0], state["bg"][0]
    formed, stats = replay_formation(formation_actions, F, index, final_fg, final_bg, seed)
    n_ops = 64 - F
    result = {key: np.concatenate((formed[key], state[key][:n_ops]), axis=0) for key in ("fg", "bg", "cont", "phase")}
    result["actions"] = formed["actions"] + state["actions"][:n_ops]
    if result["fg"].shape != (64, D.DH, D.DW):
        raise ValueError("incorrect episode frame budget")
    result["phase"][:F] = 0
    result["phase"][F:] = 1
    return result, stats


def _worker(args):
    F, index, formation_actions, state = args
    try:
        result, stats = build_episode(F, index, formation_actions, state)
        return index, result, stats, None
    except ValueError as exc:
        return index, None, None, str(exc)


def cache_arrays(bundle, out):
    check_output_tree(out)
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    path = bundle["inputs"]["data"] / "train_state.npz"
    expected = bundle["manifest"]["hashes"]["data"][path.name]
    marker = cache / "state_complete.json"
    if marker.exists():
        done = common.load_json(marker)
        if (done["source_sha1"] != expected or set(done["hashes"]) != {f"{k}.npy" for k in ("fg", "bg", "cont", "phase")}
                or any(common.sha1_file(cache / name) != sha for name, sha in done["hashes"].items())):
            raise ValueError("uncompressed source cache changed")
    else:
        with zipfile.ZipFile(path) as archive:
            for key in ("fg", "bg", "cont", "phase"):
                print(f"[05] caching frozen state.{key} for bounded-memory replay", flush=True)
                target = cache / f"{key}.npy"
                temporary = target.with_suffix(".tmp")
                with archive.open(f"{key}.npy") as reader, temporary.open("wb") as writer:
                    shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
                temporary.replace(target)
        write_json(marker, {"source_sha1": expected, "hashes": {f"{k}.npy": common.sha1_file(cache / f"{k}.npy") for k in ("fg", "bg", "cont", "phase")}})
    arrays = {k: np.load(cache / f"{k}.npy", mmap_mode="r", allow_pickle=False) for k in ("fg", "bg", "cont", "phase")}
    if any(arrays[k].shape != ((EPISODES, 64, D.DH, D.DW) if k in ("fg", "bg") else (EPISODES, 64)) for k in arrays):
        raise ValueError("source state array layout differs from the full matched experiment")
    actions = common.load_json(bundle["inputs"]["data"] / "train_state_actions.json")["actions"]
    process = common.load_json(bundle["inputs"]["data"] / "train_process_actions.json")["actions"]
    if len(actions) != EPISODES or len(process) != EPISODES:
        raise ValueError("source action episode count differs")
    return arrays, actions, process


def preflight(bundle, out, arrays, actions, process, n=32):
    if n <= 0:
        raise ValueError("preflight count must be positive")
    n = min(n, EPISODES)
    head = min(8, n)
    candidates = np.unique(np.concatenate((np.arange(head), np.linspace(head, EPISODES - 1, n - head, dtype=int))))
    failures = []
    for F in new_levels(bundle["manifest"]):
        for i in candidates:
            try:
                replay_formation(process[int(i)][:32], F, int(i), arrays["fg"][i, 0], arrays["bg"][i, 0])
            except ValueError as exc:
                failures.append({"F": F, "episode": int(i), "error": str(exc)})
    report = {"checked_per_setting": len(candidates), "failures": failures,
              "source": bundle["manifest"], "status": "blocked" if failures else "sample_preflight_pass",
              "note": "sample check is not proof that all episodes are compatible; no scenes may be replaced or filtered"}
    write_json(out / "formation_preflight.json", report)
    if failures:
        first = failures[0]
        raise ValueError(f"matched compress is infeasible on this source: F={first['F']}, episode={first['episode']}: {first['error']}; see formation_preflight.json. No training started; do not filter scenes or replace endpoints.")
    return report


def shard_identity(bundle, F, start, end):
    return {"version": VERSION, "source": bundle["manifest"], "F": F, "start": start, "end": end, "mode": "compress"}


def read_shard(directory, expected):
    done = common.load_json(directory / "complete.json")
    if (done["identity"] != expected or set(done["hashes"]) != {"episodes.npz", "actions.json"}
            or any(common.sha1_file(directory / name) != sha for name, sha in done["hashes"].items())):
        raise ValueError("saved episode shard changed; refusing replay")
    return done


def create_variant(bundle, out, F, arrays, actions, process, workers=4, chunk=128):
    check_output_tree(out)
    target = out / "data" / f"f{F}"
    if (target / "complete.json").exists():
        print(f"[05] reuse complete F={F} data; no replay", flush=True)
        return verify_variant(target, bundle["manifest"], F)
    target.mkdir(parents=True, exist_ok=True)
    generation = {"version": VERSION, "source": bundle["manifest"], "F": F, "shard_size": chunk}
    marker = target / "generation.json"
    if marker.exists() and common.load_json(marker) != generation:
        raise ValueError("generation settings or shard size changed; use the original shard size or a new run")
    if not marker.exists():
        write_json(marker, generation)
    names = common.load_json(bundle["inputs"]["data"] / "train_scenes.json")
    chunks = []
    pool = ProcessPoolExecutor(workers) if workers > 1 else None
    try:
        with v3.GenerationProgress(f"05 replay F={F}", EPISODES, "episodes") as progress:
            for start in range(0, EPISODES, chunk):
                end = min(EPISODES, start + chunk)
                directory = target / "shards" / f"{start:06d}_{end:06d}"
                expected = shard_identity(bundle, F, start, end)
                if (directory / "complete.json").exists():
                    read_shard(directory, expected)
                else:
                    args = [(F, i, process[i][:32], {**{k: arrays[k][i] for k in arrays}, "actions": actions[i]}) for i in range(start, end)]
                    records = list(pool.map(_worker, args, chunksize=2)) if pool else list(map(_worker, args))
                    failures = [{"F": F, "episode": i, "error": error} for i, _, _, error in records if error]
                    if failures:
                        write_json(target / "blocked.json", {"source": bundle["manifest"], "failures": failures})
                        raise ValueError(f"F={F} complete formation cannot preserve episode {failures[0]['episode']}; see blocked.json; completed shards retained")
                    directory.mkdir(parents=True, exist_ok=True)
                    write_npz(directory / "episodes.npz", {k: np.stack([r[1][k] for r in records]) for k in arrays})
                    write_json(directory / "actions.json", {"actions": [r[1]["actions"] for r in records], "stats": [r[2] for r in records]})
                    write_json(directory / "complete.json", {"identity": expected, "hashes": {name: common.sha1_file(directory / name) for name in ("episodes.npz", "actions.json")}})
                chunks.append(directory)
                progress.advance(end - start)
    finally:
        if pool:
            pool.shutdown()
    gathered_actions, stats = [], []
    work = target / "assemble"
    work.mkdir(exist_ok=True)
    maps = {k: np.lib.format.open_memmap(work / f"{k}.npy", mode="w+", dtype=arrays[k].dtype, shape=arrays[k].shape) for k in arrays}
    for directory in chunks:
        marker = common.load_json(directory / "complete.json")
        start, end = marker["identity"]["start"], marker["identity"]["end"]
        with np.load(directory / "episodes.npz") as z:
            for k in arrays:
                maps[k][start:end] = z[k]
        content = common.load_json(directory / "actions.json")
        gathered_actions.extend(content["actions"])
        stats.extend(content["stats"])
    with v3.GenerationProgress(f"05 compress/write F={F} full training arrays"):
        for a in maps.values():
            a.flush()
        write_npz(target / "train_process.npz", maps)
    write_json(target / "train_process_actions.json", {"schema": v3.SCHEMA, "condition": "process",
               "scenes": [row["kind"] for row in names], "extra": [{} for _ in names], "actions": gathered_actions})
    summary = {"schema": v3.SCHEMA, "seed": 0, "grid": [D.DH, D.DW],
               "profile": "simple_to_complex", "builder": {"p_mistake": 0.0, "scaffold": False},
               "layout": {"formation": F, "clip": 8, "state_clips": 8, "mixed_clips": (64 - F) // 8},
               "formation_mode": "compress", "conditions": {"process": {"episodes": EPISODES, "frames_per_episode": 64,
               "total_frames": EPISODES * 64, "fg_shape": list(maps["fg"].shape),
               "continuous_transitions": budget(F)["pairs"], "formation_frames": EPISODES * F}}}
    write_json(target / "train_summary.json", summary)
    done = {"version": VERSION, "source": bundle["manifest"], "F": F, "episodes": EPISODES,
            "mode": "compress", "hashes": variant_hashes(target), "budget": budget(F),
            "stats": {"formation_actions": sum(r["n_actions"] for r in stats), "max_actions_per_frame": max(r["max_actions_per_frame"] for r in stats)}}
    write_json(target / "complete.json", done)
    (target / "blocked.json").unlink(missing_ok=True)
    print(f"[05] F={F}: complete; saved data reused on subsequent runs", flush=True)
    return done


def prepare_all(bundle, name, workers=4, chunk=128, preflight_count=32):
    if min(workers, chunk, preflight_count) <= 0:
        raise ValueError("workers, shard size and preflight count must be positive")
    out = run_path(name)
    levels = new_levels(bundle["manifest"])
    complete = all((out / "data" / f"f{F}" / "complete.json").exists() for F in levels)
    if complete:
        return {str(F): verify_variant(out / "data" / f"f{F}", bundle["manifest"], F) for F in levels}
    arrays, actions, process = cache_arrays(bundle, out)
    preflight(bundle, out, arrays, actions, process, preflight_count)
    return {str(F): create_variant(bundle, out, F, arrays, actions, process, workers, chunk) for F in levels}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shard-size", type=int, default=128)
    ap.add_argument("--preflight-count", type=int, default=32)
    a = ap.parse_args(argv)
    with run_lock(run_path(a.run)):
        prepare_all(verify_manifest(a.run), a.run, a.workers, a.shard_size, a.preflight_count)


if __name__ == "__main__":
    main()
