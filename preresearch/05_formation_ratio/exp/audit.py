"""Audit dose variants without resampling scenes, repairing data or changing endpoints."""
import argparse
from collections import Counter
from contextlib import ExitStack
import zipfile

import numpy as np

from .generate_data import cache_arrays, read_shard, shard_identity
from .shared import (EPISODES, VERSION, budget, common, new_levels, resolve_file, run_lock, run_path, verify_manifest,
                     verify_variant, write_json)
import datasets_v3 as v3


def validate_episode(arrays, actions, source_state, source_actions, source_formation_actions, F):
    n_ops = 64 - F
    if F not in (8, 16, 48) or len(actions) != 64 or len(source_formation_actions) < 32:
        raise ValueError("invalid dose or action frame layout")
    if arrays["fg"].shape != source_state["fg"].shape or arrays["bg"].shape != arrays["fg"].shape:
        raise ValueError("variant must preserve the 64-frame grid layout")
    for key in ("fg", "bg", "cont"):
        if not np.array_equal(arrays[key][F:], source_state[key][:n_ops]):
            raise ValueError("operation frames must be the exact frozen state prefix")
    if actions[F:] != source_actions[:n_ops]:
        raise ValueError("operation actions differ from the same frozen event prefix")
    expected_phase = np.array([0] * F + [1] * n_ops, dtype=arrays["phase"].dtype)
    expected_cont = np.ones(64, bool)
    expected_cont[0] = False
    expected_cont[F::8] = False
    if not np.array_equal(arrays["phase"], expected_phase) or not np.array_equal(arrays["cont"], expected_cont):
        raise ValueError("incorrect formation labels or reset boundaries")
    if actions[0] or any(actions[t] for t in range(F - 4, F)):
        raise ValueError("initial and final four settling frames must have empty actions")
    canonical = lambda a: tuple(sorted(v3.to_action(a).as_dict().items()))
    if [canonical(a) for frame in actions[:F] for a in frame] != [canonical(a) for frame in source_formation_actions[:32] for a in frame]:
        raise ValueError("compressed formation must preserve all original actions in order")
    if (not np.array_equal(arrays["fg"][F - 1], source_state["fg"][0])
            or not np.array_equal(arrays["bg"][F - 1], source_state["bg"][0])):
        raise ValueError("variant final structure differs from the frozen scene")
    for frame in actions[:F]:
        cells = [(a["x"], a["y"]) for a in frame]
        if len(cells) != len(set(cells)):
            raise ValueError("a training action map cannot encode repeated same-cell actions")


def open_array_stream(archive, stack, key, reference):
    reader = stack.enter_context(archive.open(f"{key}.npy"))
    version = np.lib.format.read_magic(reader)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(reader)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(reader)
    else:
        raise ValueError("unsupported NPY array version")
    if fortran or dtype.hasobject or shape != reference.shape or dtype != reference.dtype:
        raise ValueError("assembled variant has incorrect array layout or dtype")
    return reader


def compare_array_batch(reader, saved):
    payload = reader.read(saved.nbytes)
    if len(payload) != saved.nbytes:
        raise ValueError("assembled variant is truncated")
    actual = np.frombuffer(payload, dtype=saved.dtype).reshape(saved.shape)
    if not np.array_equal(actual, saved):
        raise ValueError("assembled variant arrays differ from saved episode shards")


def audit_variants(bundle, name):
    out = run_path(name)
    markers = {str(F): verify_variant(out / "data" / f"f{F}", bundle["manifest"], F) for F in new_levels(bundle["manifest"])}
    arrays, source_actions, formation_actions = cache_arrays(bundle, out)
    scenes = common.load_json(bundle["inputs"]["data"] / "train_scenes.json")
    if len(scenes) != EPISODES:
        raise ValueError("source scene metadata is incomplete")
    scene_distribution = {"scene_types": dict(Counter(row["kind"] for row in scenes)),
                          "component_counts": dict(Counter(str(row["n_struct"]) for row in scenes))}
    report = {"version": VERSION, "source": bundle["manifest"], "inputs": bundle["audited"]["inputs"],
              "hashes": bundle["audited"]["hashes"], "protocol": bundle["manifest"]["protocol"],
              "endpoints": bundle["manifest"]["endpoints"], "variants": {},
              "levels": {str(F): budget(F) for F in bundle["manifest"]["protocol"]["levels"]},
              "scene_distribution": scene_distribution,
              "note": "same complete frozen structures and action order; formation timing and operation budget differ"}
    for F in new_levels(bundle["manifest"]):
        target = out / "data" / f"f{F}"
        summary = common.load_json(target / "train_summary.json")
        if summary["layout"] != {"formation": F, "clip": 8, "state_clips": 8, "mixed_clips": (64 - F) // 8}:
            raise ValueError("variant summary has incorrect frame boundaries")
        condition = summary.get("conditions", {}).get("process", {})
        expected = {"episodes": EPISODES, "frames_per_episode": 64, "total_frames": EPISODES * 64,
                    "continuous_transitions": budget(F)["pairs"], "formation_frames": EPISODES * F,
                    "fg_shape": list(arrays["fg"].shape)}
        if any(condition.get(k) != value for k, value in expected.items()):
            raise ValueError("variant summary has incorrect episode or transition budget")
        counts = Counter()
        final_content = common.load_json(target / "train_process_actions.json")
        if final_content.get("scenes") != [row["kind"] for row in scenes]:
            raise ValueError("assembled scene metadata differs from the frozen scene order")
        final_actions = final_content["actions"]
        if len(final_actions) != EPISODES:
            raise ValueError("assembled variant has incorrect action episode count")
        offset = 0
        print(f"[05] auditing F={F}: complete script, matched final and nested operations", flush=True)
        with ExitStack() as stack:
            archive = stack.enter_context(zipfile.ZipFile(target / "train_process.npz"))
            readers = {k: open_array_stream(archive, stack, k, arrays[k]) for k in arrays}
            progress = stack.enter_context(v3.GenerationProgress(f"05 audit F={F}", EPISODES, "episodes"))
            for directory in sorted((target / "shards").iterdir()):
                saved = common.load_json(directory / "complete.json")
                start, end = saved["identity"]["start"], saved["identity"]["end"]
                read_shard(directory, shard_identity(bundle, F, start, end))
                if start != offset or not start < end <= EPISODES:
                    raise ValueError("shards must cover every source episode once in order")
                with np.load(directory / "episodes.npz", allow_pickle=False) as shard:
                    batch = {k: shard[k] for k in arrays}
                actions = common.load_json(directory / "actions.json")["actions"]
                if len(actions) != end - start:
                    raise ValueError("shard action count differs from its episode range")
                for k in arrays:
                    if batch[k].shape != (end - start,) + arrays[k].shape[1:] or batch[k].dtype != arrays[k].dtype:
                        raise ValueError("shard array has incorrect layout")
                    compare_array_batch(readers[k], batch[k])
                for j, i in enumerate(range(start, end)):
                    validate_episode({k: batch[k][j] for k in arrays}, actions[j],
                                     {k: arrays[k][i] for k in arrays}, source_actions[i], formation_actions[i], F)
                    if final_actions[i] != actions[j]:
                        raise ValueError("assembled actions differ from saved episode shards")
                    counts.update(a["type"] for frame in actions[j] for a in frame)
                offset = end
                progress.advance(end - start)
            if offset != EPISODES or any(reader.read(1) for reader in readers.values()):
                raise ValueError("some frozen episodes were excluded or extra array data was added")
        report["variants"][str(F)] = {"data": common.rel(target), "hashes": markers[str(F)]["hashes"],
                                       "operation_prefix_clips": (64 - F) // 8,
                                       "episodes": EPISODES, "scene_distribution": scene_distribution,
                                       "action_types": dict(counts), "stats": markers[str(F)]["stats"]}
    path = out / "audit.json"
    if path.exists() and common.load_json(path) != report:
        raise ValueError("existing audit differs; use a new run")
    if not path.exists():
        write_json(path, report)
    return report


def verify_audit(name):
    bundle = verify_manifest(name)
    audited = common.load_json(run_path(name) / "audit.json")
    expected = {"source": bundle["manifest"], "protocol": bundle["manifest"]["protocol"],
                "inputs": bundle["audited"]["inputs"], "hashes": bundle["audited"]["hashes"],
                "endpoints": bundle["manifest"]["endpoints"], "version": VERSION,
                "levels": {str(F): budget(F) for F in bundle["manifest"]["protocol"]["levels"]}}
    if any(audited.get(k) != value for k, value in expected.items()):
        raise ValueError("audited source or protocol changed")
    for F in new_levels(bundle["manifest"]):
        data = resolve_file(audited["variants"][str(F)]["data"])
        if data.resolve() != (run_path(name) / "data" / f"f{F}").resolve():
            raise ValueError("variant path is outside its bound run")
        marker = verify_variant(data, bundle["manifest"], F)
        if marker["hashes"] != audited["variants"][str(F)]["hashes"]:
            raise ValueError("variant data differs from audit")
    return audited


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    with run_lock(run_path(a.run)):
        audit_variants(verify_manifest(a.run), a.run)


if __name__ == "__main__":
    main()
