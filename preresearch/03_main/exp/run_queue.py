"""Four-card main-experiment queue with input and checkpoint identity checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from queue import Empty, Queue
import subprocess
import sys

from .audit import verify_inputs
from .make_jobs import jobs
from .shared import CONDITIONS, EXP_NAME, PROJECT, common, validate_completed, validate_evaluation


def run_job(row, gpu, paths, audited):
    out = common.resolve_out(row["out"])
    out.mkdir(parents=True, exist_ok=True)
    condition = row["name"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if gpu == "cpu" else gpu, PYTHONUNBUFFERED="1")
    root = PROJECT / "preresearch/01_pilot_validity"
    try:
        stages = []
        if (out / "done.json").exists():
            done = validate_completed(out, condition, audited)
        else:
            if (out / "eval.json").exists():
                raise ValueError("evaluation exists without completed main training")
            stages.append(("train", [sys.executable, *row["cmd"][1:]]))
            done = None
        if (out / "eval.json").exists() and (out / "eval_samples.npz").exists():
            validate_evaluation(out, audited, done)
        else:
            stages.append(("eval", [sys.executable, "-m", "exp.evaluate", "--run", str(out),
                                    "--splits", "all", "--bench", str(paths["bench"])]))
        for stage, command in stages:
            with (out / f"{stage}.log").open("w") as log:
                rc = subprocess.call(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            if rc:
                raise RuntimeError(f"{stage} exited {rc}; see {stage}.log")
            if stage == "train":
                done = validate_completed(out, condition, audited)
            else:
                validate_evaluation(out, audited, done)
            print(f"{condition}: {stage} complete on GPU {gpu}", flush=True)
        (out / "failed.json").unlink(missing_ok=True)
        return True
    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
        common.dump_json({"condition": condition, "error": str(exc)}, out / "failed.json")
        print(f"{condition}: FAILED: {exc}", flush=True)
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--gpus", default="0,1,2,3")
    a = ap.parse_args(argv)
    directory = common.RESULTS_ROOT / EXP_NAME / a.run
    audited = common.load_json(directory / "audit.json")
    paths = verify_inputs(audited)
    rows = common.load_json(directory / "jobs/main03.json")
    if rows != jobs(a.run, paths["data"], paths["val"]) or len(rows) != len(CONDITIONS):
        raise ValueError("main 03 requires the exact four audited training jobs")
    gpus = a.gpus.split(",")
    if not gpus or any(not g.strip() for g in gpus):
        ap.error("--gpus must not be empty")
    tasks = Queue()
    for row in rows:
        tasks.put(row)

    def worker(gpu):
        ok = True
        while True:
            try:
                row = tasks.get_nowait()
            except Empty:
                return ok
            ok = run_job(row, gpu, paths, audited) and ok

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        succeeded = list(pool.map(worker, gpus))
    return 0 if all(succeeded) else 1


if __name__ == "__main__":
    raise SystemExit(main())
