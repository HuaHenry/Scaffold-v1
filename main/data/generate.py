"""Generate or import an immutable task dataset. This entry never starts training."""
import argparse
from importlib import import_module
from pathlib import Path

import numpy as np

from ..shared import (CONDITIONS, DEFAULT_SOURCE, ROOT, RESULTS_ROOT, VERSION, code_identity, common,
                      disjoint, file_hash, files_identity, frozen_write, integer, load_json,
                      lock, name, safe_path, write_json, D)
from ..tasks.registry import TASKS, select
from ..tasks.physics_tasks import bench_item, episode, sample_spec
from .storage import allocate, flush, import_legacy, legacy_count
from .audit import verify
from .progress import Monitor, Reporter, initialize_worker


def bench_write(path, rows, groups=None):
    arrays = {k: np.stack([r[k] for r in rows]).astype(np.uint8) for k in rows[0]}
    arrays["scene_ids"] = np.arange(len(rows), dtype=np.int64)
    arrays["group"] = np.zeros(len(rows), np.int16) if groups is None else np.asarray(groups, np.int16)
    with Path(path).open("wb") as stream:
        np.savez_compressed(stream, **arrays)


def synthetic_task(root, task, config, reporter=None):
    reporter = reporter or Reporter(task)
    used, metadata = set(), {}
    for split, count in (("train", config["episodes"]), ("validation", config["val_episodes"]), ("test", config["test_scenes"])):
        with reporter.stage(f"{split} 分配数据"):
            arrays = allocate(root / split, count, CONDITIONS if split == "train" else ("state",)) if split != "test" else None
        reporter.start(f"{split} 样本生成", count, "集", "samples")
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
            reporter.update(index + 1)
        reporter.finish()
        with reporter.stage(f"{split} 写盘与保存元数据"):
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


def legacy_task(root, source, config, reporter=None):
    reporter = reporter or Reporter("T01")
    with reporter.stage("核验03冻结来源"):
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
    import_legacy(root / "train", paths["data"], n, reporter=reporter)
    val_total = legacy_count(paths["val"])
    if config["val_episodes"] > val_total:
        raise ValueError("requested validation prefix exceeds legacy validation data")
    import_legacy(root / "validation", paths["val"], config["val_episodes"], ("state",), reporter=reporter)
    with reporter.stage("读取03测试集"):
        b = common.load_bench(paths["bench"])
    rollout_api = import_module("preresearch.01_pilot_validity.exp.rollout")
    rows = []
    with reporter.stage("转换03测试场景", len(b["ctx_fg"]), "场景"):
        for i in range(len(b["ctx_fg"])):
            maps = np.stack([rollout_api.bench_action_maps([b["ivs"][i][k]], 32)[0] for k in range(2)])
            rows.append({"ctx_fg": b["ctx_fg"][i], "ctx_bg": b["ctx_bg"][i], "act": maps,
                         "target_fg": b["fut_fg"][i, :, -1], "target_bg": b["fut_bg"][i, :, -1]})
            reporter.update(i + 1)
    with reporter.stage("压缩测试集与保存场景清单"):
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


def build_task(payload, reporter=None):
    root, task, config, source, request_hash = payload
    reporter = reporter or Reporter(task)
    target = Path(root) / task
    target.mkdir()
    meta = legacy_task(target, Path(source), config, reporter) if task == "T01" else synthetic_task(target, task, config, reporter)
    with reporter.stage("任务文件哈希与提交", unit="字节"):
        hashes = files_identity(target, ("task_complete.json",), progress=reporter.update)
        write_json(target / "task_complete.json", {"request_sha256": request_hash, "metadata": meta, "hashes": hashes})
    reporter.committed()
    return task, meta


def generate(out, config, source=DEFAULT_SOURCE, dry_run=False, workers=1, progress_every=10):
    workers = integer(workers, "workers")
    with Monitor(select(config["tasks"]), config, workers, progress_every) as monitor:
        reporter = Reporter("总流程", monitor.events, progress_every)
        result = _generate(out, config, source, dry_run, workers, monitor, reporter)
    if not dry_run:
        print(f"[总进度] 整体完成，冻结校验通过；数据目录：{result}", flush=True)
    return result


def _generate(out, config, source, dry_run, workers, monitor, reporter):
    source = Path(source).expanduser().resolve()
    out = disjoint(out, [ROOT, RESULTS_ROOT / "runs", source])
    tasks = select(config["tasks"])
    config = {**config, "tasks": tasks}
    for k in ("episodes", "val_episodes", "test_scenes", "max_attempts"):
        integer(config[k], k)
    integer(config["seed"], "seed", 0)
    workers = integer(workers, "workers")
    if "T01" in tasks:
        with reporter.stage("预检03来源（完整哈希读取）"):
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
        with reporter.stage("核验已完成数据", unit="字节"):
            m = verify(out, progress=reporter.update)
        if m["request"] != request:
            raise ValueError("dataset request changed; select a new output directory")
        for task in tasks:
            Reporter(task, monitor.events, monitor.interval).committed(reused=True)
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
                task_reporter = Reporter(task, monitor.events, monitor.interval)
                with task_reporter.stage("核验已提交任务", unit="字节"):
                    hashes = files_identity(target, ("task_complete.json",), progress=task_reporter.update)
                    if record["request_sha256"] != request_hash or record["hashes"] != hashes:
                        raise ValueError("committed task data changed")
                task_data[task] = record["metadata"]
                task_reporter.committed(reused=True)
                continue
            if target.exists():
                raise ValueError(f"uncommitted {task} data remains after interruption; keep it and choose a new dataset output")
            pending.append((str(out), task, config, str(source), request_hash))
        if workers == 1:
            for payload in pending:
                task, meta = build_task(payload, Reporter(payload[1], monitor.events, monitor.interval))
                task_data[task] = meta
        else:
            from concurrent.futures import ProcessPoolExecutor
            import multiprocessing
            with ProcessPoolExecutor(max_workers=min(workers, max(1, len(pending))),
                                     mp_context=multiprocessing.get_context("spawn"),
                                     initializer=initialize_worker, initargs=(monitor.events, monitor.interval)) as pool:
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
        with reporter.stage("整体文件哈希与提交", unit="字节"):
            hashes = files_identity(out, ("complete.json",), progress=reporter.update)
            write_json(out / "complete.json", {"version": VERSION, "manifest_sha256": file_hash(out / "manifest.json"), "hashes": hashes})
        with reporter.stage("最终冻结审计", unit="字节"):
            verify(out, progress=reporter.update)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--episodes", required=True, type=int, help="generate M episodes per task; training later chooses any prefix N<=M")
    ap.add_argument("--val-episodes", type=int, default=32, help="default32 matches the actual frozen experiment03 validation arrays")
    ap.add_argument("--test-scenes", type=int, default=128, help="new tasks only; T01 retains all512 original scenes")
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--max-attempts", type=int, default=100)
    ap.add_argument("--workers", type=int, default=1, help="CPU processes across independent tasks; does not change dataset identity")
    ap.add_argument("--progress-every", type=float, default=10, help="progress/throughput/ETA refresh interval in seconds; does not change data identity")
    ap.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    config = {"tasks": select(a.tasks), "episodes": a.episodes, "val_episodes": a.val_episodes,
              "test_scenes": a.test_scenes, "seed": a.seed, "max_attempts": a.max_attempts}
    generate(a.out, config, a.source_run, a.dry_run, a.workers, a.progress_every)


if __name__ == "__main__":
    main()
