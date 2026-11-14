"""One job per device, checked reuse and evaluation-only preflight."""
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
import os
import signal
from queue import Empty, Queue
import subprocess
import sys
from threading import Event

from .shared import (PROJECT, common, evaluation_path, jobs, record_eval, resolve_file,
                     run_path, scale, validate_config, validate_done, validate_eval,
                     verify_run, write_json)

devices = import_module("preresearch.06_data_scale.exp.run_queue").devices


def preflight_final(rows, p):
    for row in rows:
        done = validate_done(resolve_file(row["out"]), row, p)
        if (evaluation_path(row) / "eval_complete.json").exists():
            validate_eval(row, p, done)


def run_job(row, card, p, interval=30, stop=None, evaluate_only=False):
    stop = stop or Event()
    status = run_path(row["run"]) / "queue" / row["id"]
    status.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if card == "cpu" else card,
               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    label = row["id"]
    try:
        out, stages = resolve_file(row["out"]), []
        if row.get("reuse") or (out / "train_complete.json").exists():
            done = validate_done(out, row, p)
            print(f"[08] {label}: reuse verified final EMA", flush=True)
        else:
            if evaluate_only:
                raise ValueError("evaluation-only cannot start training")
            if (out / "config.json").exists():
                validate_config(out, row, p)
                print(f"[08] {label}: interrupted training restarts at step0", flush=True)
            stages.append("train")
            done = None
        if (evaluation_path(row) / "eval_complete.json").exists():
            if done is None:
                raise ValueError("evaluation without committed training")
            validate_eval(row, p, done)
        else:
            stages.append("eval")
        for stage in stages:
            if stop.is_set():
                raise RuntimeError("queue interrupted")
            module = "evaluate" if stage == "eval" else "train"
            cmd = [sys.executable, "-m", f"preresearch.08_noisy_formation.exp.{module}",
                   "--run", row["run"], "--variant", row["variant"]]
            path = status / f"{stage}.log"
            print(f"[08] {label}: {stage} device={card}; log={path}", flush=True)
            with path.open("w") as log:
                child = subprocess.Popen(cmd, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    while child.poll() is None:
                        if stop.wait(interval):
                            raise RuntimeError("queue interrupted; terminating child")
                        with path.open("rb") as reader:
                            reader.seek(max(0, path.stat().st_size - 4096))
                            lines = reader.read().decode("utf8", errors="replace").splitlines()
                        latest = next((s for s in reversed(lines) if any(k in s for k in ("[train]", "[val]", "[eval]"))), "checking/loading frozen inputs")
                        print(f"[08] {label} {card}: {latest}", flush=True)
                finally:
                    if child.poll() is None:
                        child.terminate()
                        try:
                            child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
            if child.returncode:
                raise RuntimeError(f"{stage} failed ({child.returncode}); see {path}")
            if stage == "train":
                done = validate_done(out, row, p)
            else:
                record_eval(row, p, done)
        (status / "failed.json").unlink(missing_ok=True)
        print(f"[08] {label}: complete", flush=True)
        return True
    except Exception as exc:
        write_json(status / "failed.json", {"job": row, "error": str(exc)})
        print(f"[08] {label}: FAILED: {exc}", flush=True)
        return False


def execute_queue(name, cards="0,1,2,3", interval=30, evaluate_only=False):
    cards = devices(cards)
    if not 0 < interval < float("inf"):
        raise ValueError("progress interval must be finite and positive")
    with scale.run_lock(run_path(name)):
        p, _ = verify_run(name)
        rows = jobs(name, p)
        if evaluate_only:
            preflight_final(rows, p)
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
                ok = run_job(row, card, p, interval, stop, evaluate_only) and ok
            return False

        def terminate(signum, frame):
            stop.set()
            raise KeyboardInterrupt("experiment08 queue terminated")

        previous_int = signal.getsignal(signal.SIGINT)
        previous = signal.signal(signal.SIGTERM, terminate)
        pool = ThreadPoolExecutor(max_workers=min(len(cards), len(rows)))
        try:
            if not all(list(pool.map(worker, cards[:len(rows)]))):
                raise RuntimeError("one or more experiment08 jobs failed; analysis not started")
        except BaseException:
            stop.set()
            raise
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                pool.shutdown(wait=True, cancel_futures=True)
            finally:
                signal.signal(signal.SIGTERM, previous)
                signal.signal(signal.SIGINT, previous_int)
