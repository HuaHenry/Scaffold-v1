"""Evaluate completed final-step EMA models only; never start training."""
import argparse
import os
import sys

from .analysis import analyze
from .audit import verify_audit
from .make_jobs import jobs
from .run_queue import execute_queue
from .shared import resolve_file, run_lock, run_path, validate_done, validate_eval


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="new final-model run name; old best-model runs are not accepted")
    ap.add_argument("--gpus", default=os.environ.get("GPUS", "2,3"))
    ap.add_argument("--progress-every", type=float, default=30)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--no-analysis", action="store_true", help="evaluate only, without generating an analysis report")
    ap.add_argument("--dry-run", action="store_true", help="validate every final checkpoint; do not write or launch inference")
    a = ap.parse_args(argv)
    cards = [c.strip() for c in a.gpus.split(",")]
    if (a.progress_every <= 0 or a.boot <= 0 or a.seed < 0 or len(set(cards)) != len(cards)
            or any(not c or (c != "cpu" and not c.isdigit()) for c in cards)):
        ap.error("distinct GPU IDs or cpu, positive counts/interval, and nonnegative seed required")
    audited = verify_audit(a.run)
    rows = jobs(a.run, audited)
    for row in rows:
        out = resolve_file(row["out"])
        done = validate_done(out, row, audited)
        if (out / "eval_complete.json").exists():
            validate_eval(out, audited, done)
        print(f"[05 eval] ratio={row['formation_ratio']:.1%}: final step={done['checkpoint_step']}, weights=ema", flush=True)
    if a.dry_run:
        print("[05 eval] all final checkpoints verified; no writes, inference or training", flush=True)
        return
    with run_lock(run_path(a.run)):
        execute_queue(a.run, a.gpus, a.progress_every, evaluate_only=True)
        if not a.no_analysis:
            analyze(a.run, a.boot, a.seed)
    print("[05 eval] final-model evaluation complete; no training was started", flush=True)


def cli(argv=None):
    try:
        main(argv)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f"[05 eval] ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from None


if __name__ == "__main__":
    cli()
