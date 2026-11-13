"""Evaluate all recorded final EMA checkpoints, with all-task preflight and no training."""
import argparse

from .analysis import analyze
from .run_queue import checked_jobs, devices, execute_queue, preflight_final
from .shared import run_lock, run_path, verify_audit


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--progress-every", type=float, default=30)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=12345, help="bootstrap seed")
    ap.add_argument("--no-analysis", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="read-only preflight of every final checkpoint")
    a = ap.parse_args(argv)
    devices(a.gpus)
    if not 0 < a.progress_every < float("inf") or a.boot <= 0 or a.seed < 0:
        ap.error("positive interval/count and nonnegative seed required")
    if a.dry_run:
        audited = verify_audit(a.run)
        preflight_final(checked_jobs(a.run, audited), audited)
        print("[06] all final checkpoints passed read-only preflight; no training or evaluation started", flush=True)
        return
    with run_lock(run_path(a.run)):
        execute_queue(a.run, a.gpus, a.progress_every, evaluate_only=True)
        if not a.no_analysis:
            analyze(a.run, a.boot, a.seed)


if __name__ == "__main__":
    main()
