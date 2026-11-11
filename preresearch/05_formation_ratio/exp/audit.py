"""Read-only population audit; no trajectory generation or frame cache."""
import argparse
from collections import Counter
import zipfile

import numpy as np

from .shared import (VERSION, array_sha1, budget, common, dose_key, resolve_file,
                     run_lock, run_path, verify_manifest, write_json)


def array_header(archive, key):
    with archive.open(key + ".npy") as reader:
        version = np.lib.format.read_magic(reader)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(reader)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(reader)
        else:
            raise ValueError("unsupported source NPY version")
    if dtype.hasobject or fortran:
        raise ValueError("source arrays must be non-object C-order arrays")
    return shape, dtype


def population(data, episodes):
    data = resolve_file(data)
    summary = common.load_summary(data)
    if summary["layout"]["formation"] != 32 or summary["layout"]["clip"] != 8:
        raise ValueError("supervision sampling requires the unchanged F32 source layout")
    if summary["conditions"]["process"]["episodes"] != episodes:
        raise ValueError("source must contain the full frozen episode population")
    with zipfile.ZipFile(data / "train_process.npz") as archive:
        for key in ("fg", "bg", "cont", "phase"):
            shape, dtype = array_header(archive, key)
            expected = (episodes, 64, common.H, common.W) if key in ("fg", "bg") else (episodes, 64)
            allowed = {"fg": ("uint8",), "bg": ("uint8",), "phase": ("int8", "uint8"), "cont": ("bool",)}
            if shape != expected or dtype.name not in allowed[key]:
                raise ValueError(f"source {key} array layout changed: {shape}, {dtype}")
    with np.load(data / "train_process.npz", allow_pickle=False) as z:
        phase, cont = z["phase"], z["cont"]
    expected_phase = np.array([0] * 32 + [1] * 32, dtype=np.uint8)
    expected_cont = np.ones(64, bool)
    expected_cont[[0, 32, 40, 48, 56]] = False
    if not np.all(phase == expected_phase) or not np.all(cont == expected_cont):
        raise ValueError("source phase labels or reset boundaries changed")
    pairs = common.make_pairs(data, "process", "video", episodes)
    mask = phase[pairs[:, 0], pairs[:, 1]] == 0
    if mask.sum() != episodes * 31 or (~mask).sum() != episodes * 28:
        raise ValueError("incorrect unchanged formation/operation pool sizes")
    pools = {"total": len(pairs), "formation": int(mask.sum()), "operation": int((~mask).sum()),
             "pairs_sha1": array_sha1(pairs), "formation_indices_sha1": array_sha1(np.flatnonzero(mask)),
             "operation_indices_sha1": array_sha1(np.flatnonzero(~mask)),
             "natural_formation_share": 31 / 59, "formation_pairs_per_episode": 31,
             "operation_pairs_per_episode": 28, "operation_clips": 4}
    scenes = common.load_json(data / "train_scenes.json")
    payload = common.load_json(data / "train_process_actions.json")
    actions = payload["actions"]
    if len(scenes) != episodes or len(actions) != episodes:
        raise ValueError("incomplete scene/action population")
    if payload.get("scenes") != [row["kind"] for row in scenes]:
        raise ValueError("source scene/action order differs")
    counts = {"formation": Counter(), "operation": Counter()}
    collisions, examples = 0, []
    for i, episode in enumerate(actions):
        if len(episode) != 64:
            raise ValueError("source action episode has incorrect frame count")
        for t, frame in enumerate(episode):
            cells = []
            for a in frame:
                if (a["type"] not in common.TYPE_IDX or not 0 <= int(a["x"]) < common.W
                        or not 0 <= int(a["y"]) < common.H or a.get("layer", "front") not in ("front", "back")):
                    raise ValueError("source contains actions not representable by the frozen model")
                cells.append((int(a["x"]), int(a["y"])))
            counts["formation" if t < 32 else "operation"].update(a["type"] for a in frame)
            if len(cells) != len(set(cells)):
                collisions += 1
                if len(examples) < 8:
                    examples.append({"episode": i, "frame": t})
    return {"pools": pools,
            "scene_distribution": {"scene_types": dict(Counter(row["kind"] for row in scenes)),
                                   "component_counts": dict(Counter(str(row["n_struct"]) for row in scenes))},
            "action_types": {k: dict(v) for k, v in counts.items()},
            "inherited_action_encoding": {"collision_frames": collisions, "examples": examples,
                "note": "existing single-token encoding is unchanged; duplicate same-cell actions use the last token; no new merging, filtering or replay"}}


def audit_sampling(bundle, name):
    out = run_path(name)
    details = population(bundle["inputs"]["data"], bundle["manifest"]["protocol"]["episodes"])
    report = {"version": VERSION, "source": bundle["manifest"], "protocol": bundle["manifest"]["protocol"],
              "inputs": bundle["manifest"]["inputs"], "hashes": bundle["manifest"]["hashes"], **details,
              "budgets": {dose_key(r): budget(r, bundle["manifest"]["protocol"]["steps"])
                          for r in bundle["manifest"]["protocol"]["ratios"]},
              "note": "identical original process pairs for every dose; only replacement sampling mixture changes; all doses newly trained"}
    path = out / "audit.json"
    if path.exists() and common.load_json(path) != report:
        raise ValueError("existing supervision audit changed; use a new run")
    if not path.exists():
        write_json(path, report)
    print(f"[05] audited {details['pools']['total']} original pairs; no frame cache or variants generated", flush=True)
    return report


def verify_audit(name):
    bundle = verify_manifest(name)
    audited = common.load_json(run_path(name) / "audit.json")
    expected = {"version": VERSION, "source": bundle["manifest"], "protocol": bundle["manifest"]["protocol"],
                "inputs": bundle["manifest"]["inputs"], "hashes": bundle["manifest"]["hashes"]}
    if any(audited.get(k) != v for k, v in expected.items()):
        raise ValueError("audited source or supervision protocol changed")
    expected_pools = {"total": 59 * audited["protocol"]["episodes"],
                      "formation": 31 * audited["protocol"]["episodes"],
                      "operation": 28 * audited["protocol"]["episodes"]}
    if any(audited.get("pools", {}).get(k) != v for k, v in expected_pools.items()):
        raise ValueError("audited pair population is incomplete")
    if audited.get("budgets") != {dose_key(r): budget(r, audited["protocol"]["steps"]) for r in audited["protocol"]["ratios"]}:
        raise ValueError("audited supervision exposure budget changed")
    return audited


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    with run_lock(run_path(a.run)):
        audit_sampling(verify_manifest(a.run), a.run)


if __name__ == "__main__":
    main()
