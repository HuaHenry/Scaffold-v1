"""Derive noisy formation only; unchanged scenes, final states and downstream operations."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from .noise import D, derive, v3
from .shared import DEFAULT_SOURCE, NOISE, code_identity, common, file_set, integer, qualifies, scale, source_bundle, verify_data, write_json


def derive_one(args):
    return derive(*args)


def derive_results(tasks, workers):
    if workers == 1:
        yield from map(derive_one, tasks)
        return
    from itertools import islice
    tasks = iter(tasks)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        while batch := list(islice(tasks, workers * 4)):
            yield from pool.map(derive_one, batch)


def check_pair(source_fg, source_bg, source_actions, fg, bg, actions):
    if fg.shape != source_fg[:32].shape or bg.shape != source_bg[:32].shape or len(actions) != 32:
        raise ValueError("formation frame budget changed")
    if any(actions[t] for t in (0, 28, 29, 30, 31)):
        raise ValueError("initial terrain/settling actions changed")
    for t in (0, 31):
        if not np.array_equal(fg[t], source_fg[t]) or not np.array_equal(bg[t], source_bg[t]):
            raise ValueError("paired terrain/final structure changed")
    for frame in actions:
        if len({(a["x"], a["y"]) for a in frame}) != len(frame):
            raise ValueError("conflicting encoded action tokens")


def extract_arrays(path, workspace, n):
    arrays = {}
    with zipfile.ZipFile(path) as archive:
        for key in ("fg", "bg", "cont", "phase"):
            with archive.open(f"{key}.npy") as stream:
                version = np.lib.format.read_magic(stream)
                shape, fortran, dtype = (np.lib.format.read_array_header_1_0(stream) if version == (1, 0)
                                         else np.lib.format.read_array_header_2_0(stream))
                if fortran or dtype.hasobject or n > shape[0]:
                    raise ValueError("invalid source array or prefix")
                dest = np.lib.format.open_memmap(Path(workspace) / f"source_{key}.npy", mode="w+", dtype=dtype, shape=(n, *shape[1:]))
                per = int(np.prod(shape[1:]))
                for start in range(0, n, 64):
                    count = min(64, n - start)
                    raw = stream.read(count * per * dtype.itemsize)
                    if len(raw) != count * per * dtype.itemsize:
                        raise ValueError("truncated source array")
                    dest[start:start + count] = np.frombuffer(raw, dtype=dtype).reshape(count, *shape[1:])
                    if start == 0 or start + count == n or start % 4096 == 0:
                        print(f"[08 data] source {key} {start+count}/{n}", flush=True)
                dest.flush()
                arrays[key] = dest
    return arrays


def report_audit(records, n):
    eligible = [r for r in records if r["eligible"]]
    accepted = [r for r in eligible if not r["fallback"]]
    actions = sum(r["actions"] for r in records)
    reasons = {}
    for r in records:
        if r["fallback_reason"]:
            reasons[r["fallback_reason"]] = reasons.get(r["fallback_reason"], 0) + 1
    chunks = [size for r in records for size in r["chunk_sizes"]]
    return {"episodes": n, "eligible_episodes": len(eligible), "excluded_fluid_episodes": n - len(eligible),
            "effective_episodes": sum(r["effective_episode"] for r in accepted),
            "effective_episode_rate": sum(r["effective_episode"] for r in accepted) / max(1, len(eligible)),
            "successful_offset_actions": sum(r["successful_offset_actions"] for r in accepted),
            "skipped_invalid_actions": sum(r["skipped_invalid_actions"] for r in records),
            "chunk_size_histogram": {str(s): chunks.count(s) for s in sorted(set(chunks))},
            "chunk_size_mean": float(np.mean(chunks)), "chunk_size_max": max(chunks),
            "falling_metric": "material-cell changes before/after tick; proxy, not distinct collapse count",
            "fallback_episodes": sum(r["fallback"] for r in eligible),
            "fallback_rate": sum(r["fallback"] for r in eligible) / max(1, len(eligible)),
            "episodes_with_accepted_mistakes": sum(r["triggered"] > 0 for r in accepted),
            "triggered_attempts": {k: sum(r[k] for r in records) for k in ("delay", "offset", "omit")},
            "candidate_actions": sum(r["candidate_actions"] for r in records),
            "accepted_trigger_rate": sum(r["triggered"] for r in accepted) / max(1, sum(r["candidate_actions"] for r in eligible)),
            "physical_transition_fraction": sum(r["physical_transitions"] for r in records) / (n * 31),
            "remove_action_fraction": sum(r["remove_actions"] for r in records) / max(1, actions),
            "falling_cell_events": sum(r["falling_cell_events"] for r in records),
            "fallback_reasons": reasons, "same_final_structure": True, "same_operation_frames_actions": True}


def generate(out, source=DEFAULT_SOURCE, episodes=81920, noise_seed=0, max_fallback=.1, workers=8):
    n, workers = integer(episodes, "episodes"), integer(workers, "workers")
    integer(noise_seed, "noise seed", 0)
    if not np.isfinite(max_fallback) or not 0 <= max_fallback <= 1:
        raise ValueError("fallback threshold must be finite in [0,1]")
    with v3.GenerationProgress("08 frozen source preflight (hashing)"):
        bundle = source_bundle(source, n)
    out = scale.check_output_tree(out)
    scale.helpers.forbid_source_output(out, bundle["source"], bundle["audited"])
    allowed = common.RESULTS_ROOT / "08_noisy_formation"
    if allowed not in out.parents:
        raise ValueError(f"08 data must be a versioned directory under {allowed}")
    with scale.run_lock(out):
        return generate_locked(out, bundle, n, noise_seed, max_fallback, workers)


def generate_locked(out, bundle, n, noise_seed, max_fallback, workers):
    request = {"version": 1, "source": common.rel(bundle["source"]), "source_identity": bundle["manifest"],
               "episodes": n, "noise_seed": noise_seed, "max_fallback": max_fallback, "code": code_identity()}
    if (out / "complete.json").exists():
        manifest, _ = verify_data(out)
        if manifest["request"] != request:
            raise ValueError("generation request changed; use a new data version")
        return out
    if out.exists() and any(p.name != "pipeline.lock" for p in out.iterdir()):
        raise ValueError("uncommitted data directory is not empty; keep it and use a new version")
    out.mkdir(parents=True, exist_ok=True)
    source_data = bundle["inputs"]["data"]
    with v3.GenerationProgress("08 reading frozen actions/scenes"):
        source_actions = common.load_json(source_data / "train_process_actions.json")["actions"][:n]
        scenes = common.load_json(source_data / "train_scenes.json")[:n]
        summary = common.load_summary(source_data)
    with tempfile.TemporaryDirectory(prefix="work-", dir=out) as work:
        with v3.GenerationProgress("08 extracting frozen prefix"):
            arrays = extract_arrays(source_data / "train_process.npz", work, n)
        audits, baseline_records = {}, None
        for variant, p in NOISE.items():
            records, output_actions = [], []
            target = out / variant
            if variant != "process":
                target.mkdir()
                fg = np.lib.format.open_memmap(Path(work) / f"{variant}_fg.npy", mode="w+", dtype=np.uint8, shape=arrays["fg"].shape)
                bg = np.lib.format.open_memmap(Path(work) / f"{variant}_bg.npy", mode="w+", dtype=np.uint8, shape=arrays["bg"].shape)
            tasks = ((arrays["fg"][e], arrays["bg"][e], source_actions[e], D.int_seed(summary["seed"], "world", e),
                      p, noise_seed, not any(k in ("dune", "pond") for k in scenes[e]["kinds"]),
                      None if baseline_records is None else baseline_records[e]) for e in range(n))
            with v3.GenerationProgress(f"08 {variant} formation", n, "episodes") as progress:
                for e, (f, b, acts, meta) in enumerate(derive_results(tasks, workers)):
                    check_pair(arrays["fg"][e], arrays["bg"][e], source_actions[e], f, b, acts)
                    records.append(meta)
                    if variant != "process":
                        fg[e, :32], bg[e, :32] = f, b
                        fg[e, 32:], bg[e, 32:] = arrays["fg"][e, 32:], arrays["bg"][e, 32:]
                        output_actions.append(acts + source_actions[e][32:])
                        if (output_actions[-1][32:] != source_actions[e][32:]
                                or not np.array_equal(fg[e, 32:], arrays["fg"][e, 32:])
                                or not np.array_equal(bg[e, 32:], arrays["bg"][e, 32:])):
                            raise ValueError("paired operation frames/actions changed")
                    progress.advance()
            audits[variant] = report_audit(records, n)
            if variant == "process":
                baseline_records = records
            if variant != "process":
                fg.flush()
                bg.flush()
                with v3.GenerationProgress(f"08 {variant} compress/write (sample100% is not completion)"):
                    np.savez_compressed(target / "train_process.npz", fg=fg, bg=bg, cont=arrays["cont"], phase=arrays["phase"])
                    write_json(target / "train_process_actions.json", {"actions": output_actions})
                derived_summary = {**summary, "conditions": {"process": {**summary["conditions"]["process"], "episodes": n}},
                                   "experiment08": {"variant": variant, "p": p, "same_final": True}}
                write_json(target / "train_summary.json", derived_summary)
                write_json(target / "episode_audit.json", records)
                del fg, bg
        manifest = {**request, "request": request, "audits": audits,
                    "qualified": qualifies(audits, max_fallback),
                    "noise_sampling": "per examined eligible original place action in frames1..18; delays capped at20",
                    "repair": "state-aware legal remove/place at21..27; late original construction completed here; real ticks; no actions28..31",
                    "pair_audit": {"episodes_checked": n, "terrain_and_final_identical": True,
                                   "operation_frames_actions_identical": True, "cont_phase_copied": True,
                                   "source_strict_replay": True, "real_ticks_per_episode": 31},
                    "not_a_multitask_dataset": True}
        write_json(out / "manifest.json", manifest)
    with v3.GenerationProgress("08 final hashing/audit"):
        write_json(out / "complete.json", {"manifest_sha1": common.sha1_file(out / "manifest.json"), "hashes": file_set(out)})
        verify_data(out, require_qualified=False)
    if not manifest["qualified"]:
        raise ValueError("noise acceptance failed (fallback/monotonicity/high>=30%); retained for diagnosis, not training")
    print(f"[08 data] complete: {out}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--source-run", default=DEFAULT_SOURCE)
    ap.add_argument("--episodes", type=int, default=81920)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--max-fallback", type=float, default=.1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.dry_run:
        source_bundle(a.source_run, a.episodes)
        print(f"[08 data] source checks passed; {a.episodes} original scenes; no writes/training")
        return
    generate(a.out, a.source_run, a.episodes, a.noise_seed, a.max_fallback, a.workers)


if __name__ == "__main__":
    main()
