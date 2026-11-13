"""Independent dose tasks on individual GPUs, with periodic readable progress."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from queue import Queue, Empty
from threading import Event
import subprocess
import sys

from .audit import verify_audit
from .make_jobs import jobs
from .shared import PROJECT, check_output_tree, common, record_eval, resolve_file, run_lock, run_path, validate_config, validate_done, validate_eval, write_json
import zipfile


def run_job(row, card, audited, interval=30, stop=None, evaluate_only=False):
    stop = stop if stop is not None else Event()
    out = check_output_tree(resolve_file(row["out"]))
    if evaluate_only:
        validate_done(out, row, audited)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if card == "cpu" else card,
               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    try:
        stages, done = [], None
        if (out / "done.json").exists():
            done = validate_done(out, row, audited)
            print(f"[05] ratio={row['formation_ratio']:.1%}: reuse completed training", flush=True)
        else:
            if (out / "eval.json").exists() or (out / "eval_samples.npz").exists():
                raise ValueError("evaluation without completed training")
            if (out / "config.json").exists():
                validate_config(out, row, audited)
                print(f"[05] ratio={row['formation_ratio']:.1%}: interrupted training restarts from step 0; no optimizer resume", flush=True)
            stages.append(("train", [sys.executable, *row["cmd"][1:]]))
        if (out / "eval_complete.json").exists():
            validate_eval(out, audited, done)
        else:
            if (out / "eval.json").exists() or (out / "eval_samples.npz").exists():
                print(f"[05] ratio={row['formation_ratio']:.1%}: uncommitted evaluation restarts; completed training is retained", flush=True)
            stages.append(("eval", [sys.executable, "-m", "preresearch.01_pilot_validity.exp.evaluate", "--run", str(out),
                                    "--bench", str(resolve_file(audited["inputs"]["bench"])), "--splits", "all"]))
        for stage, command in stages:
            if stop.is_set():
                raise RuntimeError("queue interrupted before starting task")
            print(f"[05] ratio={row['formation_ratio']:.1%} {stage} on GPU {card}; log={common.rel(out / (stage + '.log'))}", flush=True)
            with (out / f"{stage}.log").open("w") as log:
                process = subprocess.Popen(command, cwd=PROJECT, env=env,
                                           stdout=log, stderr=subprocess.STDOUT)
                try:
                    last = ""
                    while process.poll() is None:
                        if stop.wait(interval):
                            raise RuntimeError("queue interrupted; terminating child process")
                        with (out / f"{stage}.log").open("rb") as reader:
                            reader.seek(max(0, (out / f"{stage}.log").stat().st_size - 4096))
                            records = reader.read().decode("utf-8", errors="replace").splitlines()
                        latest = next((line for line in reversed(records) if "[train]" in line or "[val]" in line or "[eval]" in line), "loading/checking data")
                        if latest != last:
                            print(f"[05] ratio={row['formation_ratio']:.1%} GPU {card}: {latest}", flush=True)
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
        print(f"[05] ratio={row['formation_ratio']:.1%}: training and evaluation complete", flush=True)
        return True
    except (ValueError, KeyError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
        write_json(out / "failed.json", {"formation_ratio": row["formation_ratio"], "error": str(exc)})
        print(f"[05] ratio={row['formation_ratio']:.1%}: FAILED: {exc}", flush=True)
        return False


def execute_queue(name, cards="0,1,2,3", interval=30, evaluate_only=False):
    devices = [c.strip() for c in cards.split(",")]
    if not devices or any(not c or (c != "cpu" and not c.isdigit()) for c in devices) or len(set(devices)) != len(devices) or interval <= 0:
        raise ValueError("devices must be distinct and progress interval positive")
    audited = verify_audit(name)
    rows = common.load_json(run_path(name) / "jobs/formation_ratio.json")
    if rows != jobs(name, audited):
        raise ValueError("job list differs from audited formation experiment")
    if evaluate_only:
        for row in rows:
            validate_done(resolve_file(row["out"]), row, audited)
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
            ok = run_job(row, card, audited, interval, stop, evaluate_only=evaluate_only) and ok
        return False

    pool = ThreadPoolExecutor(max_workers=min(len(devices), len(rows)))
    try:
        if not all(list(pool.map(worker, devices[:len(rows)]))):
            raise RuntimeError("one or more formation-ratio tasks failed")
    except BaseException:
        stop.set()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--progress-every", type=float, default=30)
    a = ap.parse_args(argv)
    with run_lock(run_path(a.run)):
        execute_queue(a.run, a.gpus, a.progress_every)


if __name__ == "__main__":
    main()
