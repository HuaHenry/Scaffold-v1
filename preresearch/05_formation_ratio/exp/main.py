"""Experiment 05 v2: formation supervision fractions, equal updates, no frame replay."""
import argparse
import os
import sys

from .analysis import analyze
from .audit import audit_sampling, verify_audit
from .make_jobs import save_jobs
from .run_queue import execute_queue
from .shared import (DEFAULT_SOURCE, RATIOS, STEPS, bind_run, budget, common, require_current,
                     run_lock, run_path, validate_ratios, validate_source)


def fresh_name(tag):
    base = "_".join(p for p in (common.now_ts(), common.slug(tag)) if p)
    name, suffix = base, 1
    while run_path(name).exists():
        suffix += 1
        name = f"{base}-{suffix}"
    return name


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Using the existing zhouqi environment, from this experiment folder:
  env -u FW_RUN bash run_all.sh --dry-run
  env -u FW_RUN bash run_all.sh --tag supervision_ratio --stage prepare
  bash run_all.sh --run <prepared-run-name> --stage train --gpus 0,1,2,3
  bash run_all.sh --run <completed-run-name> --stage analyze

Ratios are supervision fractions, NOT numbers of frames. Default ratios:
0, 0.125, 0.25, 0.5, 0.75; batch256; 94400 updates per dose; training seed0.
All five doses are trained from scratch, including 0 and 0.5. No experiment03
endpoint enters the equal-budget dose curve. One source process dataset is used
for all doses: formation31 and operation28 legal one-step pairs per episode.
Preparation only audits original inputs and writes jobs; no new trajectories,
no 16GB frame cache, no scene filtering, no changed physics or final structures.
Old --levels and frame compression are retired. Use a NEW supervision_ratio run;
the failed compression run and its cache remain untouched. Interrupted training
restarts at step0; completed models and committed evaluations are reused.
--steps changes the common budget for a NEW run. --seed is bootstrap seed only.
""")
    ap.add_argument("--source-run", help="completed experiment03 source; resume inherits recorded source")
    ap.add_argument("--run", default=os.environ.get("FW_RUN"))
    ap.add_argument("--tag", default="supervision_ratio")
    ap.add_argument("--ratios", help="comma-separated fractions in [0,1]; default 0,0.125,0.25,0.5,0.75")
    ap.add_argument("--steps", type=int, help="common updates per dose; default 94400; resume inherits saved budget")
    ap.add_argument("--stage", choices=("all", "prepare", "train", "analyze"), default="all")
    ap.add_argument("--dry-run", action="store_true", help="read-only source checks; no output directory or training")
    ap.add_argument("--gpus", default=os.environ.get("GPUS", "0,1,2,3"))
    ap.add_argument("--progress-every", type=float, default=30)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345, help="bootstrap seed, not training seed")
    a = ap.parse_args(argv)
    if a.progress_every <= 0 or a.boot <= 0 or a.seed < 0:
        ap.error("counts/interval must be positive, bootstrap seed nonnegative")
    cards = [c.strip() for c in a.gpus.split(",")]
    if any(not c or (c != "cpu" and not c.isdigit()) for c in cards) or len(set(cards)) != len(cards):
        ap.error("--gpus requires distinct nonnegative GPU IDs or cpu")
    recorded = None
    if a.run and (run_path(a.run) / "source.json").exists():
        recorded = common.load_json(run_path(a.run) / "source.json")
        require_current(recorded)
    if a.stage in ("train", "analyze") and recorded is None:
        ap.error("train/analyze requires a prepared v2 --run or FW_RUN")
    try:
        ratios = validate_ratios(a.ratios.split(",")) if a.ratios else (
            validate_ratios(recorded["protocol"]["ratios"]) if recorded else RATIOS)
        steps = a.steps if a.steps is not None else (recorded["protocol"]["steps"] if recorded else STEPS)
        budget(ratios[0], steps)
    except ValueError as exc:
        ap.error(str(exc))
    source = a.source_run or (recorded["source_run"] if recorded else DEFAULT_SOURCE)
    bundle = validate_source(source, ratios, steps)
    if recorded and bundle["manifest"] != recorded:
        raise ValueError("source, ratios or common updates changed; use a new run")
    print("[05] supervision sampling v2; unchanged process trajectories; ALL doses new, seed0", flush=True)
    for ratio in ratios:
        b = budget(ratio, steps)
        print(f"[05] ratio={ratio:.1%}: F/O batch={b['formation_per_batch']}/{b['operation_per_batch']}, "
              f"updates={steps}, samples={b['samples']}", flush=True)
    if a.dry_run:
        print("[05] read-only checks complete; no writes, generation, evaluation or training", flush=True)
        return
    name = a.run or fresh_name(a.tag)
    out = run_path(name)
    with run_lock(out):
        bind_run(name, bundle)
        print(f"[05] run={name}; output={common.rel(out)}", flush=True)
        if a.stage in ("all", "prepare"):
            audited = audit_sampling(bundle, name)
            save_jobs(name, audited)
        if a.stage in ("all", "train"):
            if a.stage == "train":
                save_jobs(name, verify_audit(name))
            execute_queue(name, a.gpus, a.progress_every)
        if a.stage in ("all", "train", "analyze"):
            result = analyze(name, a.boot, a.seed)
            print(f"[05] complete: {common.rel(result)}", flush=True)
        else:
            print("[05] preparation complete; no frame variants/cache generated, no training started", flush=True)


def cli(argv=None):
    try:
        main(argv)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f"[05] ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from None


if __name__ == "__main__":
    cli()
