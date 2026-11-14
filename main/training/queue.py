"""Four-condition single/mixed jobs on independent GPUs; evaluate-only has full preflight."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Event

from ..shared import PROJECT, load_json, write_json
from .protocol import verify_run
from .train import validate_done


def devices(value):
    result = value.split(",")
    if not result or len(set(result)) != len(result) or any(v != "cpu" and not v.isdigit() for v in result):
        raise ValueError("distinct GPU IDs or cpu required")
    return result


def preflight(run, p, rows):
    from ..evaluation.evaluate import validate_eval
    for row in rows:
        out = Path(run) / "train" / row["id"]
        validate_done(out, p, row)
        for task in row["tasks"]:
            if (out / "eval" / task / "complete.json").exists():
                validate_eval(out, p, row, task)


def run_job(run, p, row, card, evaluate_only, stop, interval):
    from ..evaluation.evaluate import validate_eval
    out = Path(run) / "train" / row["id"]
    stages = []
    if (out / "train_complete.json").exists():
        validate_done(out, p, row)
    elif evaluate_only:
        raise ValueError("evaluate-only cannot train a missing checkpoint")
    else:
        stages.append(("train", "main.training.train"))
    complete = True
    for task in row["tasks"]:
        if (out / "eval" / task / "complete.json").exists():
            validate_eval(out, p, row, task)
        else:
            complete = False
    if not complete:
        stages.append(("eval", "main.evaluation.evaluate"))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if card == "cpu" else card,
               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    out.mkdir(parents=True, exist_ok=True)
    try:
        for stage, module in stages:
            if stop.is_set():
                raise RuntimeError("queue interrupted")
            command = [sys.executable, "-m", module, "--run", str(run), "--job", row["id"]]
            print(f"[queue] {row['id']} {stage} on {card}", flush=True)
            with (out / f"{stage}.log").open("w") as log:
                process = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    last = ""
                    while process.poll() is None:
                        if stop.wait(interval):
                            raise RuntimeError("queue interrupted; terminating child")
                        with (out / f"{stage}.log").open("rb") as reader:
                            reader.seek(max(0, (out / f"{stage}.log").stat().st_size - 2048))
                            lines = reader.read().decode(errors="replace").splitlines()
                        latest = lines[-1] if lines else "loading"
                        if latest != last:
                            print(f"[queue] {row['id']}: {latest}", flush=True)
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
                raise RuntimeError(f"{stage} exit {process.returncode}; see {out / (stage + '.log')}")
            if stage == "train":
                validate_done(out, p, row)
            else:
                for task in row["tasks"]:
                    validate_eval(out, p, row, task)
        (out / "failed.json").unlink(missing_ok=True)
        return True
    except Exception as exc:
        write_json(out / "failed.json", {"job": row["id"], "error": str(exc)})
        print(f"[queue] FAILED {row['id']}: {exc}", flush=True)
        return False


def execute(run, cards="0,1,2,3", evaluate_only=False, interval=30):
    run = Path(run).resolve()
    cards = devices(cards)
    if not 0 < interval < float("inf"):
        raise ValueError("progress interval must be finite and positive")
    p, _ = verify_run(run)
    rows = load_json(run / "jobs.json")
    if evaluate_only:
        preflight(run, p, rows)
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
            ok = run_job(run, p, row, card, evaluate_only, stop, interval) and ok
        return False

    pool = ThreadPoolExecutor(max_workers=min(len(cards), len(rows)))
    try:
        if not all(list(pool.map(worker, cards[:len(rows)]))):
            raise RuntimeError("some jobs failed; analysis not started")
    except BaseException:
        stop.set()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
