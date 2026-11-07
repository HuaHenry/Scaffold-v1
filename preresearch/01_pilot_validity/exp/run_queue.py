"""Multi-GPU job queue (plan 00 section 11).

python -m exp.run_queue --jobs <run>/jobs/pilot_validity_steps6000.json --gpus 0,1,2,3 --eval-after
`out` in a job file is a result folder relative to the results root (experiments/preresearch/), inside the run folder;
re-running the SAME job file resumes (finished folders are skipped); a new run (new FW_RUN) has its own job files.
Each GPU runs one job at a time (CUDA_VISIBLE_DEVICES). `--gpus cpu` (or cpu,cpu) runs on CPU for smoke tests.
A job with out/done.json is skipped; a failure writes out/failed.json and the queue continues.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import common
from .common import ROOT, resolve


def _cmd(cmd):
    return [sys.executable if c == "python" else c for c in cmd]


def validate_curve_resume(job, out, bench_args):
    """Reject stale 07 completion/evaluation artifacts rather than silently skipping them."""
    cmd = job["cmd"]
    if "--select-best-val" not in cmd:
        return
    if not (out / "done.json").exists():
        if (out / "eval.json").exists():
            raise ValueError("evaluation exists without completed training")
        return
    cfg = common.load_json(out / "config.json")
    args = cfg["args"]
    done = common.load_json(out / "done.json")
    for flag in ("condition", "seed", "protocol", "steps", "batch", "episodes", "lr",
                 "change-weight", "depth", "data", "val", "val-every"):
        key = flag.replace("-", "_")
        actual = cmd[cmd.index("--" + flag) + 1]
        if str(args[key]) != actual and not (key == "lr" and float(args[key]) == float(actual)):
            raise ValueError(f"completed job changed {flag}")
    epochs = int(cmd[cmd.index("--epochs") + 1]) if "--epochs" in cmd else None
    if args.get("epochs") != epochs or (epochs is not None and
            (args["balanced_phases"] or done.get("epochs") != epochs
             or done.get("n_pairs") != (56 if args["condition"] == "state" else 59) * args["episodes"]
             or done.get("samples_seen") != epochs * done["n_pairs"]
             or done.get("steps_per_epoch") != (done["n_pairs"] + args["batch"] - 1) // args["batch"]
             or done.get("steps") != epochs * done["steps_per_epoch"]
             or done.get("best_step") not in [k * done["steps_per_epoch"] for k in range(1, epochs + 1)])):
        raise ValueError("completed job differs from the full-epoch sampling protocol")
    if (not args["select_best_val"] or args["balanced_phases"] != (epochs is None and args["condition"] == "process")
            or done.get("selection") != "val_cascade_acc" or done.get("best_step") is None
            or not (out / "ckpt.pt").is_file()
            or done.get("ckpt_sha1") != common.sha1_file(out / "ckpt.pt")
            or cfg.get("data_sha1") != common.data_hashes(common.resolve(args["data"]), args["condition"])
            or cfg.get("val_sha1") != common.data_hashes(common.resolve(args["val"]), "state")):
        raise ValueError("completed job checkpoint or input identity changed")
    if (out / "eval.json").exists():
        ev = common.load_json(out / "eval.json")
        bench = resolve(bench_args[1])
        if (ev.get("checkpoint_sha1") != done["ckpt_sha1"] or ev.get("bench") != common.rel(bench)
                or ev.get("bench_sha1") != common.sha1_file(bench)
                or ev.get("bench_json_sha1") != common.sha1_file(bench.with_suffix(".json"))):
            raise ValueError("completed evaluation has different checkpoint or bench")


def run_job(job: dict, gpu: str, eval_after: bool, bench_args: list[str], log) -> bool:
    out = common.resolve_out(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if gpu == "cpu" else gpu, PYTHONUNBUFFERED="1")
    try:
        if "--select-best-val" in job["cmd"] and eval_after and not bench_args:
            raise ValueError("experiment 07 requires an explicit audited --bench")
        validate_curve_resume(job, out, bench_args)
    except (ValueError, KeyError, IndexError, FileNotFoundError) as exc:
        log(f"FAILED  {job['name']} [resume] {exc}")
        return False
    stages = []
    if not (out / "done.json").exists():
        stages.append(("train", job["cmd"]))
    if eval_after and (not (out / "eval.json").exists() or not (out / "eval_samples.npz").exists()):
        stages.append(("eval", ["python", "-m", "exp.evaluate", "--run", str(out), "--splits", "all", *bench_args]))
    for stage, cmd in stages:
        (out / "failed.json").unlink(missing_ok=True)
        t0 = time.time()
        with open(out / f"{stage}.log", "w") as lf:
            rc = subprocess.call(_cmd(cmd), cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
        if rc != 0:
            common.dump_json({"stage": stage, "returncode": rc, "cmd": [common.rel(c) if c.startswith("/") else c for c in cmd], "gpu": gpu}, out / "failed.json")
            log(f"FAILED  {job['name']} [{stage}] rc={rc} (see {common.rel(out / (stage + '.log'))})")
            return False
        log(f"{stage:5s} ok {job['name']} on gpu {gpu} ({time.time() - t0:.0f}s)")
    if not stages:
        log(f"skip    {job['name']} (done)")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--gpus", default="0,1,2,3", help="comma list of GPU ids, or cpu / cpu,cpu")
    ap.add_argument("--eval-after", action="store_true", help="evaluate on FormBench after each training")
    ap.add_argument("--bench", type=Path, default=None, help="bench.npz for --eval-after")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    jobs = json.loads(resolve(a.jobs).read_text())
    if a.dry_run:
        for j in jobs:
            print(j["name"], "->", j["out"], "\n   ", " ".join(j["cmd"]))
        return 0
    bench_args = ["--bench", str(resolve(a.bench))] if a.bench else []
    q: queue.Queue = queue.Queue()
    for j in jobs:
        q.put(j)
    lock, failed = threading.Lock(), []

    def log(msg):
        with lock:
            print(time.strftime("%H:%M:%S"), msg, flush=True)

    def worker(gpu):
        while True:
            try:
                job = q.get_nowait()
            except queue.Empty:
                return
            if not run_job(job, gpu, a.eval_after, bench_args, log):
                failed.append(job["name"])

    ths = [threading.Thread(target=worker, args=(g.strip(),)) for g in a.gpus.split(",")]
    [t.start() for t in ths]
    [t.join() for t in ths]
    if failed:
        log(f"{len(failed)} job(s) failed: {failed}")
        return 1
    log("all jobs finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
