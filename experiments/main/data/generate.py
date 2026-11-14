"""Generate or import an immutable task dataset. This entry never starts training."""
import argparse
from importlib import import_module
from pathlib import Path

import numpy as np

from ..shared import (CONDITIONS, DEFAULT_SOURCE, ROOT, VERSION, code_identity, common,
                      disjoint, file_hash, files_identity, frozen_write, integer, load_json,
                      lock, name, safe_path, write_json, D)
from ..tasks.registry import TASKS, select
from ..tasks.physics_tasks import bench_item, episode, sample_spec
from .storage import allocate, flush, import_legacy, legacy_count
from .audit import verify


def bench_write(path, rows, groups=None):
    arrays = {k: np.stack([r[k] for r in rows]).astype(np.uint8) for k in rows[0]}
    arrays["scene_ids"] = np.arange(len(rows), dtype=np.int64)
    arrays["group"] = np.zeros(len(rows), np.int16) if groups is None else np.asarray(groups, np.int16)
    with Path(path).open("wb") as stream:
        np.savez_compressed(stream, **arrays)


def synthetic_task(root, task, config):
    used, metadata = set(), {}
    for split, count in (("train", config["episodes"]), ("validation", config["val_episodes"]), ("test", config["test_scenes"])):
        arrays = allocate(root / split, count, CONDITIONS if split == "train" else ("state",)) if split != "test" else None
        scene_meta, bench = [], []
        local_hashes = set()
        for index in range(count):
            for attempt in range(config["max_attempts"]):
                spec = sample_spec(task, config["seed"], split, index, attempt)
                try:
                    result, meta, _, final = episode(spec)
                    if split != "train" and (meta["final_hash"] in used or meta["final_hash"] in local_hashes):
                        continue
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"{task}/{split}/{index}: exhausted generation retries; reduce count or expand generator diversity")
            local_hashes.add(meta["final_hash"])
            meta["attempt"] = attempt
            scene_meta.append(meta)
            if split == "test":
                bench.append(bench_item(spec, *final))
            else:
                for c, values in arrays.items():
                    for k, dest in values.items():
                        dest[index] = result[c][k]
            if (index + 1) % 100 == 0:
                print(f"[data] {task}/{split} {index+1}/{count}", flush=True)
        if arrays is not None:
            flush(arrays)
            del arrays
        else:
            bench_write(root / "bench.npz", bench)
        write_json(root / f"{split}_scenes.json", scene_meta)
        metadata[split + ("_scenes" if split == "test" else "_episodes")] = count
        metadata[split + "_unique_final_states"] = len(local_hashes)
        metadata[split + "_duplicates"] = count - len(local_hashes)
        used |= local_hashes
    metadata["train_validation_overlap"] = metadata["train_test_overlap"] = metadata["validation_test_overlap"] = 0
    return metadata


def source_identity(source, audited):
    return {"source_run": str(Path(source).resolve()), "audit_sha256": file_hash(Path(source) / "audit.json"),
            "inputs": audited["inputs"], "hashes": audited["hashes"]}


def legacy_task(root, source, config):
    audited = common.load_json(source / "audit.json")
    paths = import_module("preresearch.03_main.exp.audit").verify_inputs(audited)
    expected_source = load_json(Path(root).parent / "request.json")["source_identity"]
    if source_identity(source, audited) != expected_source:
        raise ValueError("T01 source identity changed since the generation request")
    for key in ("train_validation_overlap", "train_bench_overlap", "validation_bench_overlap"):
        if audited.get(key) != 0:
            raise ValueError("legacy source separation audit is invalid")
    n = config["episodes"]
    if n > audited["protocol"]["episodes"]:
        raise ValueError("T01 cannot exceed current03 source episode count")
    import_legacy(root / "train", paths["data"], n)
    val_total = legacy_count(paths["val"])
    if config["val_episodes"] > val_total:
        raise ValueError("requested validation prefix exceeds legacy validation data")
    import_legacy(root / "validation", paths["val"], config["val_episodes"], ("state",))
    b = common.load_bench(paths["bench"])
    rollout_api = import_module("preresearch.01_pilot_validity.exp.rollout")
    rows = []
    for i in range(len(b["ctx_fg"])):
        maps = np.stack([rollout_api.bench_action_maps([b["ivs"][i][k]], 32)[0] for k in range(2)])
        rows.append({"ctx_fg": b["ctx_fg"][i], "ctx_bg": b["ctx_bg"][i], "act": maps,
                     "target_fg": b["fut_fg"][i, :, -1], "target_bg": b["fut_bg"][i, :, -1]})
    bench_write(root / "bench.npz", rows, b["split_ids"])
    train_scenes = common.load_json(paths["data"] / "train_scenes.json")[:n]
    val_scenes = common.load_json(paths["val"] / "train_scenes.json")[:config["val_episodes"]]
    write_json(root / "train_scenes.json", train_scenes)
    write_json(root / "validation_scenes.json", val_scenes)
    write_json(root / "test_scenes.json", b["meta"]["scenes"])
    if source_identity(source, common.load_json(source / "audit.json")) != expected_source:
        raise ValueError("T01 source audit changed during import")
    return {"train_episodes": n, "validation_episodes": config["val_episodes"], "test_scenes": len(rows),
            "source": str(source), "source_audit_sha256": file_hash(source / "audit.json"),
            "source_hashes": audited["hashes"], "legacy_test_is_reused": True,
            "groups": list(b["splits"]), "train_unique_final_states": len({s['final_hash'] for s in train_scenes}),
            "train_validation_overlap": 0, "train_test_overlap": 0, "validation_test_overlap": 0}


def build_task(payload):
    root, task, config, source, request_hash = payload
    target = Path(root) / task
    target.mkdir()
    meta = legacy_task(target, Path(source), config) if task == "T01" else synthetic_task(target, task, config)
    write_json(target / "task_complete.json", {"request_sha256": request_hash, "metadata": meta,
                                               "hashes": files_identity(target, ("task_complete.json",))})
    return task, meta


def generate(out, config, source=DEFAULT_SOURCE, dry_run=False, workers=1):
    source = Path(source).expanduser().resolve()
    out = disjoint(out, [ROOT / "runs", source, *[ROOT / p for p in ("tasks", "data", "training", "evaluation", "tests", "extensions")]])
    if out == ROOT or out in ROOT.parents:
        raise ValueError("dataset output cannot contain main source tree")
    tasks = select(config["tasks"])
    config = {**config, "tasks": tasks}
    for k in ("episodes", "val_episodes", "test_scenes", "max_attempts"):
        integer(config[k], k)
    integer(config["seed"], "seed", 0)
    workers = integer(workers, "workers")
    if "T01" in tasks:
        audited = common.load_json(source / "audit.json")
        paths = import_module("preresearch.03_main.exp.audit").verify_inputs(audited)
        disjoint(out, [*paths.values()])
        if audited.get("protocol", {}).get("episodes") != 81920 or common.load_summary(paths["data"]).get("profile") != "simple_to_complex":
            raise ValueError("T01 requires the current frozen81920 simple_to_complex experiment03 source")
        if config["episodes"] > legacy_count(paths["data"]) or config["val_episodes"] > legacy_count(paths["val"]):
            raise ValueError("T01 requested prefix exceeds frozen training/validation data")
        if any(audited.get(k) != 0 for k in ("train_validation_overlap", "train_bench_overlap", "validation_bench_overlap")):
            raise ValueError("T01 source separation audit is invalid")
    request = {"version": VERSION, "config": config, "source": str(source) if "T01" in tasks else None,
               "source_identity": source_identity(source, audited) if "T01" in tasks else None,
               "code": code_identity()}
    estimated = (config["episodes"] * 4 + config["val_episodes"]) * 64 * 28 * 56 * 3 * len(tasks)
    print(f"[data] {len(tasks)} tasks; approximate uncompressed array storage {estimated/1024**3:.2f} GiB", flush=True)
    if out.exists() and (out / "complete.json").exists():
        m = verify(out)
        if m["request"] != request:
            raise ValueError("dataset request changed; select a new output directory")
        return out
    if out.exists():
        bound = (out / "request.json").exists()
        if bound and load_json(out / "request.json") != request:
            raise ValueError("dataset request changed; select a new output directory")
        allowed = {"generation.lock"}
        if bound:
            allowed |= {*tasks, "request.json", "manifest.json", "manifest.json.tmp", "complete.json.tmp", "request.json.tmp"}
        if set(p.name for p in out.iterdir()) - allowed:
            raise ValueError("dataset directory contains unrelated or uncommitted files without a matching request")
    if dry_run:
        print("[data] dry-run: no files or training created", flush=True)
        return out
    out.mkdir(parents=True, exist_ok=True)
    with lock(out / "generation.lock"):
        frozen_write(out / "request.json", request)
        task_data, pending = {}, []
        request_hash = file_hash(out / "request.json")
        for task in tasks:
            target = out / task
            completion = target / "task_complete.json"
            if completion.exists():
                record = load_json(completion)
                if record["request_sha256"] != request_hash or record["hashes"] != files_identity(target, ("task_complete.json",)):
                    raise ValueError("committed task data changed")
                task_data[task] = record["metadata"]
                continue
            if target.exists():
                raise ValueError(f"uncommitted {task} data remains after interruption; keep it and choose a new dataset output")
            pending.append((str(out), task, config, str(source), request_hash))
        if workers == 1:
            for payload in pending:
                task, meta = build_task(payload)
                task_data[task] = meta
        else:
            from concurrent.futures import ProcessPoolExecutor
            import multiprocessing
            with ProcessPoolExecutor(max_workers=min(workers, max(1, len(pending))),
                                     mp_context=multiprocessing.get_context("spawn")) as pool:
                for task, meta in pool.map(build_task, pending):
                    task_data[task] = meta
        task_data = {task: task_data[task] for task in tasks}
        manifest = {"version": VERSION, "request": request, "config": config, "code": request["code"],
                    "tasks": {t: TASKS[t] for t in tasks}, "task_data": task_data,
                    "layout": {"frames": 64, "formation": 32, "clip": 8},
                    "metrics": ["change_f1", "cell_acc_mask"], "horizon": 32,
                    "target": "exact frame32, not first fixed point", "no_task_composition": True}
        frozen_write(out / "manifest.json", manifest)
        for filename in ("request.json.tmp", "manifest.json.tmp", "complete.json.tmp"):
            (out / filename).unlink(missing_ok=True)
        write_json(out / "complete.json", {"version": VERSION, "manifest_sha256": file_hash(out / "manifest.json"),
                                            "hashes": files_identity(out, ("complete.json",))})
        verify(out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--episodes", required=True, type=int, help="maximum frozen episodes per task, chosen after experiment06")
    ap.add_argument("--val-episodes", type=int, default=32, help="default32 matches the actual frozen experiment03 validation arrays")
    ap.add_argument("--test-scenes", type=int, default=128, help="new tasks only; T01 retains all512 original scenes")
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--max-attempts", type=int, default=100)
    ap.add_argument("--workers", type=int, default=1, help="CPU processes across independent tasks; does not change dataset identity")
    ap.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    config = {"tasks": select(a.tasks), "episodes": a.episodes, "val_episodes": a.val_episodes,
              "test_scenes": a.test_scenes, "seed": a.seed, "max_attempts": a.max_attempts}
    generate(a.out, config, a.source_run, a.dry_run, a.workers)


if __name__ == "__main__":
    main()
