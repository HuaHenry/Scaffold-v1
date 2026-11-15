"""A frozen execution subset of a main run, preserving original model identities."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, ExitStack
import fcntl
from pathlib import Path
from queue import Empty, Queue
import shutil
import signal
from threading import Event

from main import shared
from main.training import protocol, queue, train
from main.evaluation import analyze, evaluate

DEFAULT_TASKS = ("T01", "W01", "W02", "G03", "G07")
SNAPSHOTS = ("protocol.json", "jobs.json", "run.json")
TRAIN_FILES = ("config.json", "done.json", "ckpt.pt", "train_complete.json")


def launcher_identity():
    return shared.file_hash(Path(__file__))


def selected_jobs(p, tasks):
    from main.tasks.registry import select
    tasks = select(tasks)
    if not set(tasks).issubset(p["tasks"]):
        raise ValueError("selected tasks must exist in the source run")
    if "single" not in p["modes"]:
        raise ValueError("a mixed model cannot be reused as a single-task model")
    rows = protocol.jobs(p)
    return [r for task in tasks for r in rows if r["mode"] == "single" and r["tasks"] == [task]]


@contextmanager
def source_guard(source, rows):
    paths = [source / "pipeline.lock"]
    for row in rows:
        base = source / "train" / row["id"]
        paths += [base / "training.lock", base / f"evaluation_{row['tasks'][0]}.lock"]
    with ExitStack() as stack:
        for path in paths:
            if not path.exists():
                continue
            stream = stack.enter_context(path.open("rb"))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError(f"source is still running: {path}; stop it before importing") from exc
        yield


def copy_plan(source, p, row):
    base = source / "train" / row["id"]
    if not (base / "train_complete.json").exists():
        return None
    train.validate_done(base, p, row)
    protocol.task_inputs(p, row)
    paths = list(TRAIN_FILES)
    paths += [n for n in ("train.log", "train_log.jsonl") if (base / n).is_file()]
    task = row["tasks"][0]
    if (base / "eval" / task / "complete.json").exists():
        evaluate.validate_eval(base, p, row, task)
        paths += [f"eval/{task}/{n}" for n in shared.files_identity(base / "eval" / task)]
        if (base / "eval.log").is_file():
            paths.append("eval.log")
    return {n: shared.file_hash(base / n) for n in paths}


def copied_hashes(out, record):
    return {n: shared.file_hash(out / n) for n in record}


def verify_subset(run, full_data=True):
    run = protocol.run_root(run)
    p, manifest = protocol.verify_run(run, full_data=full_data)
    subset = shared.load_json(run / "subset.json")
    marker = shared.load_json(run / "subset_complete.json")
    expected = {"subset_sha256": shared.file_hash(run / "subset.json"),
                "snapshots": {n: shared.file_hash(run / n) for n in SNAPSHOTS}}
    if marker != expected or subset["source_files"] != expected["snapshots"]:
        raise ValueError("subset/source snapshot identity changed")
    if subset["launcher_sha256"] != launcher_identity():
        raise ValueError("subset launcher changed; keep the frozen launcher or create a new run")
    rows = selected_jobs(p, subset["tasks"])
    if rows != subset["jobs"] or subset["modes"] != ["single"]:
        raise ValueError("subset task list changed")
    ids = {r["id"] for r in rows}
    if not set(subset["copied"]).issubset(ids):
        raise ValueError("copied jobs are outside the selected subset")
    for key, record in subset["copied"].items():
        if copied_hashes(run / "train" / key, record) != record:
            raise ValueError(f"copied model or evaluation changed: {key}")
    return p, subset, rows, manifest


def prepare_subset(source, run, tasks=DEFAULT_TASKS, full_data=True):
    source, run = protocol.run_root(source), protocol.run_root(run)
    shared.disjoint(run, [source])
    tasks = list(tasks)
    if (run / "subset_complete.json").exists():
        with shared.lock(run / "pipeline.lock"):
            _, subset, _, _ = verify_subset(run, full_data)
            if subset["tasks"] != tasks or subset["source_run"] != str(source):
                raise ValueError("subset selection/source changed; use a new run")
            print_status(run, full_data=False)
            return run
    p = shared.load_json(source / "protocol.json")
    shared.disjoint(run, [p["data"]])
    rows = selected_jobs(p, tasks)
    with source_guard(source, rows), shared.lock(run / "pipeline.lock"):
        p, _ = protocol.verify_run(source, full_data=full_data)
        source_files = {n: shared.file_hash(source / n) for n in SNAPSHOTS}
        copied = {}
        for row in rows:
            print(f"[subset prepare] checking {row['id']}", flush=True)
            record = copy_plan(source, p, row)
            if record is not None:
                copied[row["id"]] = record
        subset = {"version": 1, "kind": "single_task_execution_subset", "source_run": str(source),
                  "source_files": source_files, "tasks": tasks, "modes": ["single"], "jobs": rows,
                  "copied": copied, "launcher_sha256": launcher_identity(),
                  "selection_note": "按场景类型与计算预算保留代表任务；不是按测试成绩筛选；历史完整训练协议逐字节保留"}
        if not (run / "subset.json").exists() and any(n.name != "pipeline.lock" for n in run.iterdir()):
            raise ValueError("target is not empty and has no frozen subset request")
        shared.frozen_write(run / "subset.json", subset)
        for name in SNAPSHOTS:
            target = run / name
            if target.exists():
                if shared.file_hash(target) != source_files[name]:
                    raise ValueError(f"target source snapshot changed: {name}")
            else:
                part = target.with_name(name + ".copying")
                shutil.copyfile(source / name, part)
                if shared.file_hash(part) != source_files[name]:
                    raise ValueError(f"source snapshot changed during copy: {name}")
                part.replace(target)
        for row in rows:
            key = row["id"]
            if key not in copied:
                print(f"[subset prepare] {key}: no committed model; train from step0 later", flush=True)
                continue
            target = run / "train" / key
            temporary = target.with_name(key + ".importing")
            record = copied[key]
            if target.exists():
                if copied_hashes(target, record) != record:
                    raise ValueError(f"target copied job changed: {key}")
            else:
                temporary.mkdir(parents=True, exist_ok=True)
                for name, expected in record.items():
                    dest = temporary / name
                    if dest.exists():
                        if shared.file_hash(dest) != expected:
                            raise ValueError(f"partial import changed: {key}/{name}")
                        continue
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    part = dest.with_name(dest.name + ".copying")
                    shutil.copyfile(source / "train" / key / name, part)
                    if shared.file_hash(part) != expected:
                        raise ValueError(f"source changed during copy: {key}/{name}")
                    part.replace(dest)
                train.validate_done(temporary, p, row)
                if (temporary / "eval" / row["tasks"][0] / "complete.json").exists():
                    evaluate.validate_eval(temporary, p, row, row["tasks"][0])
                temporary.rename(target)
            print(f"[subset prepare] copied completed {key}; original records unchanged", flush=True)
        if {n: shared.file_hash(source / n) for n in SNAPSHOTS} != source_files:
            raise ValueError("source protocol changed during import")
        marker = {"subset_sha256": shared.file_hash(run / "subset.json"), "snapshots": source_files}
        shared.frozen_write(run / "subset_complete.json", marker)
        verify_subset(run, full_data=False)
    print_status(run, full_data=False)
    print("[subset] prepared; NO training started. Use --stage train to start GPUs.", flush=True)
    return run


def print_status(run, full_data=False):
    _, subset, rows, _ = verify_subset(run, full_data)
    complete = [r for r in rows if (Path(run) / "train" / r["id"] / "train_complete.json").exists()]
    evaluated = [r for r in complete if (Path(run) / "train" / r["id"] / "eval" / r["tasks"][0] / "complete.json").exists()]
    print(f"[subset] tasks={','.join(subset['tasks'])}; models={len(rows)}; "
          f"copied={len(subset['copied'])}; training_complete={len(complete)}; "
          f"evaluation_complete={len(evaluated)}; remaining_training={len(rows)-len(complete)}", flush=True)


def execute_subset(run, cards="0,1,2,3", interval=30, evaluate_only=False):
    cards = queue.devices(cards)
    if not 0 < interval < float("inf"):
        raise ValueError("progress interval must be positive and finite")
    run = protocol.run_root(run)
    with shared.lock(run / "pipeline.lock"):
        p, _, rows, _ = verify_subset(run)
        if evaluate_only:
            queue.preflight(run, p, rows)
        pending, stop = Queue(), Event()
        for row in rows:
            pending.put(row)

        def worker(card):
            ok = True
            while not stop.is_set():
                try:
                    row = pending.get_nowait()
                except Empty:
                    return ok
                ok = queue.run_job(run, p, row, card, evaluate_only, stop, interval) and ok
            return False

        def terminate(signum, frame):
            stop.set()
            raise KeyboardInterrupt("subset queue terminated")

        old_int, old_term = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, terminate)
        pool = ThreadPoolExecutor(max_workers=min(len(cards), len(rows)))
        try:
            if not all(list(pool.map(worker, cards[:len(rows)]))):
                raise RuntimeError("some selected jobs failed; see their failed.json/train.log/eval.log")
        except BaseException:
            stop.set()
            raise
        finally:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            try:
                pool.shutdown(wait=True, cancel_futures=True)
            finally:
                signal.signal(signal.SIGINT, old_int)
                signal.signal(signal.SIGTERM, old_term)
    print_status(run)


def analyze_subset(run, boot=10000, seed=12345):
    run = protocol.run_root(run)
    with shared.lock(run / "pipeline.lock"):
        p, subset, rows, _ = verify_subset(run)
        samples, identities = {}, {}
        for row in rows:
            out = run / "train" / row["id"]
            task = row["tasks"][0]
            values = evaluate.validate_eval(out, p, row, task)
            samples[("single", task, row["condition"], row["seed"])] = values
            identities[row["id"]] = {"training": shared.file_hash(out / "train_complete.json"),
                                      "evaluation": shared.file_hash(out / "eval" / task / "complete.json")}
        view = {**p, "tasks": subset["tasks"], "modes": ["single"],
                "training_prefixes": {t: p["training_prefixes"][t] for t in subset["tasks"]},
                "selection_note": p["selection_note"] + "; " + subset["selection_note"]}
        report = analyze.summarize(samples, view, boot, seed)
        report.update(identities=identities, source_run=subset["source_run"],
                      copied_jobs=list(subset["copied"]), subset_sha256=shared.file_hash(run / "subset.json"),
                      training_protocol_sha256=shared.file_hash(run / "protocol.json"),
                      protocol_role="analysis subset view; actual training protocol preserved in protocol.json")
        base = run / "analysis" / f"subset_boot{boot}_seed{seed}"
        out, suffix = base, 1
        while out.exists():
            suffix += 1
            out = base.with_name(f"{base.name}-{suffix}")
        out.mkdir(parents=True)
        shared.write_json(out / "results.json", report)
        text = analyze.markdown(report)
        text = text.replace("- 17项是17个互斥生成模板，不是17种物理机制；水/沙对应模板平行，不进行跨机制组合。",
                            "- 本报告只纳入冻结的代表任务子集，不宣称覆盖全部17种场景模板。")
        (out / "results.md").write_text(text, encoding="utf-8")
        analyze.plot(report, out / "figs")
        print(f"[subset] analysis: {out}", flush=True)
        return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("prepare", "train", "evaluate", "analyze", "status"), required=True)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--source-run", type=Path, help="prepare only; completed original main run")
    ap.add_argument("--tasks", help="prepare only; default T01,W01,W02,G03,G07")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--progress-every", type=float, default=30)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--bootstrap-seed", type=int, default=12345)
    a = ap.parse_args(argv)
    if a.stage == "prepare":
        if a.source_run is None:
            ap.error("prepare requires --source-run")
        from main.tasks.registry import select
        prepare_subset(a.source_run, a.run, select(a.tasks) if a.tasks else DEFAULT_TASKS)
    else:
        if a.source_run is not None or a.tasks is not None:
            ap.error("execution inherits the frozen subset; do not pass prepare parameters")
        if a.stage in ("train", "evaluate"):
            execute_subset(a.run, a.gpus, a.progress_every, a.stage == "evaluate")
        elif a.stage == "status":
            print_status(a.run)
        else:
            analyze_subset(a.run, a.boot, a.bootstrap_seed)


if __name__ == "__main__":
    main()
