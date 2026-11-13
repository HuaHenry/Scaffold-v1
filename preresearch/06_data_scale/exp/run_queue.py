"""Independent single-GPU jobs; final-only evaluation never triggers training."""
from concurrent.futures import ThreadPoolExecutor
import os
from queue import Empty, Queue
import subprocess
import sys
from threading import Event
import zipfile

from .make_jobs import jobs
from .shared import (PILOT_PACKAGE, PROJECT, check_output_tree, common, record_eval,
                     resolve_file, run_path, validate_config, validate_done, validate_eval,
                     verify_audit, write_json)


def devices(value):
    cards = [v.strip() for v in value.split(",")]
    if (not cards or any(not v or (v != "cpu" and not v.isdigit()) for v in cards)
            or len(set(cards)) != len(cards)):
        raise ValueError("GPU IDs must be distinct nonnegative integers or cpu")
    return cards


def checked_jobs(name, audited):
    rows = common.load_json(run_path(name) / "jobs/data_scale.json")
    if rows != jobs(name, audited):
        raise ValueError("job list differs from audited design")
    return rows


def preflight_final(rows, audited):
    for row in rows:
        done = validate_done(resolve_file(row["out"]), row, audited)
        if (resolve_file(row["out"]) / "eval_complete.json").exists():
            validate_eval(resolve_file(row["out"]), audited, done)


def run_job(row, card, audited, interval=30, stop=None, evaluate_only=False):
    stop = stop if stop is not None else Event()
    out = check_output_tree(resolve_file(row["out"]))
    if evaluate_only:
        validate_done(out, row, audited)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if card == "cpu" else card,
               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    label = row["id"]
    try:
        done, stages = None, []
        if (out / "train_complete.json").exists():
            done = validate_done(out, row, audited)
            print(f"[06] {label}: reuse completed training", flush=True)
        else:
            if evaluate_only:
                raise ValueError("final-only evaluation cannot start training")
            if any((out / n).exists() for n in ("eval.json", "eval_samples.npz", "eval_complete.json")):
                raise ValueError("evaluation without committed training")
            if (out / "config.json").exists():
                validate_config(out, row, audited)
                print(f"[06] {label}: interrupted training restarts at step0; no optimizer resume", flush=True)
            stages.append(("train", [sys.executable, *row["cmd"][1:]]))
        if (out / "eval_complete.json").exists():
            validate_eval(out, audited, done)
        else:
            stages.append(("eval", [sys.executable, "-m", f"{PILOT_PACKAGE}.evaluate", "--run", str(out),
                "--bench", str(resolve_file(audited["inputs"]["bench"])), "--splits", "all"]))
        for stage, command in stages:
            if stop.is_set():
                raise RuntimeError("queue interrupted before starting task")
            print(f"[06] {label}: {stage} on {card}; log={common.rel(out / (stage + '.log'))}", flush=True)
            with (out / f"{stage}.log").open("w") as log:
                process = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    last = ""
                    while process.poll() is None:
                        if stop.wait(interval):
                            raise RuntimeError("queue interrupted; terminating child")
                        with (out / f"{stage}.log").open("rb") as reader:
                            reader.seek(max(0, (out / f"{stage}.log").stat().st_size - 4096))
                            lines = reader.read().decode("utf-8", errors="replace").splitlines()
                        latest = next((v for v in reversed(lines) if any(s in v for s in ("[train]", "[val]", "[eval]"))), "loading/checking data")
                        if latest != last:
                            print(f"[06] {label} GPU {card}: {latest}", flush=True)
                            last = latest
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
            if process.returncode:
                raise RuntimeError(f"{stage} failed with exit {process.returncode}; see {stage}.log")
            if stage == "train":
                done = validate_done(out, row, audited)
            else:
                record_eval(out, audited, done)
        (out / "failed.json").unlink(missing_ok=True)
        print(f"[06] {label}: complete", flush=True)
        return True
    except (ValueError, KeyError, OSError, RuntimeError, EOFError, zipfile.BadZipFile) as exc:
        write_json(out / "failed.json", {"id": label, "error": str(exc)})
        print(f"[06] {label}: FAILED: {exc}", flush=True)
        return False


def execute_queue(name, cards="0,1,2,3", interval=30, evaluate_only=False):
    cards = devices(cards)
    if not 0 < interval < float("inf"):
        raise ValueError("progress interval must be finite and positive")
    audited = verify_audit(name)
    rows = checked_jobs(name, audited)
    if evaluate_only:
        preflight_final(rows, audited)
    queue, stop = Queue(), Event()
    for row in rows:
        queue.put(row)

    def worker(card):
        ok = True
        while not stop.is_set():
            try:
                row = queue.get_nowait()
            except Empty:
                return ok
            ok = run_job(row, card, audited, interval, stop, evaluate_only) and ok
        return False

    pool = ThreadPoolExecutor(max_workers=min(len(cards), len(rows)))
    try:
        if not all(list(pool.map(worker, cards[:len(rows)]))):
            raise RuntimeError("one or more data-scale tasks failed; analysis was not started")
    except BaseException:
        stop.set()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
