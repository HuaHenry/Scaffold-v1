"""Run the matched five-dose experiment; source checking never starts training."""
import argparse
import os

from .analysis import analyze
from .audit import audit_variants, verify_audit
from .generate_data import cache_arrays, preflight, prepare_all
from .make_jobs import save_jobs
from .run_queue import execute_queue
from .shared import DEFAULT_SOURCE, LEVELS, bind_run, budget, common, new_levels, run_lock, run_path, validate_levels, validate_source


def fresh_name(tag):
    base = "_".join(p for p in (common.now_ts(), common.slug(tag)) if p)
    name, suffix = base, 1
    while run_path(name).exists():
        suffix += 1
        name = f"{base}-{suffix}"
    return name


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Usage (from this experiment folder, using the existing zhouqi environment):
  bash run_all.sh --dry-run
  bash run_all.sh --stage preflight --tag formation_check
  bash run_all.sh --levels 0,16,32,48 --tag ratio_f16_f48 --stage prepare
  bash run_all.sh --run <prepared-run-name> --stage train --gpus 0,1
  bash run_all.sh --run <completed-run-name> --stage analyze

New runs default to all five doses. F=8 has known infeasible frozen scenes;
there is no automatic scene filtering or dose fallback. Explicit --levels
0,16,32,48 uses the planned two-new-dose fallback, not a five-dose result.
All selected scenes must still pass complete replay and audit before training.
This implementation matches experiment 03: training seed 0 and five full epochs,
not the original three-seed/6000-update protocol. Prefix mode is not implemented.
More formation means fewer operation clips and unequal effective update counts.

Results: experiments/preresearch/05_formation_ratio/<run>/, including source.json,
formation_preflight.json, data/f*/, audit.json, jobs/, train/, and analysis/.
Sources, validation, benchmark and F=0/32 checkpoints are never rewritten.
Replaying uses a bounded-memory cache (about 16.4 GB), episode shards, and full
assembled variants; allow ample disk space. Resuming reuses completed data and
models; retain the original shard size. Interrupted training starts at step 0;
interrupted evaluation reruns inference only. --seed controls bootstrap, not training.
""")
    ap.add_argument("--source-run", help="completed experiment 03 run; resume inherits recorded source")
    ap.add_argument("--run", default=os.environ.get("FW_RUN"))
    ap.add_argument("--tag", default="formation_ratio")
    ap.add_argument("--levels", help="comma-separated sorted doses; new runs default to 0,8,16,32,48; explicit fallback: 0,16,32,48")
    ap.add_argument("--stage", choices=("all", "preflight", "prepare", "train", "analyze"), default="all")
    ap.add_argument("--dry-run", action="store_true", help="read-only source/model identity check; no formation replay or files")
    ap.add_argument("--gpus", default=os.environ.get("GPUS", "0,1,2,3"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shard-size", type=int, default=128)
    ap.add_argument("--preflight-count", type=int, default=32)
    ap.add_argument("--progress-every", type=float, default=30)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345, help="bootstrap seed; training seed is fixed at 0")
    a = ap.parse_args(argv)
    if min(a.workers, a.shard_size, a.preflight_count, a.progress_every, a.boot) <= 0 or a.seed < 0:
        ap.error("counts and progress interval must be positive; bootstrap seed nonnegative")
    cards = [c.strip() for c in a.gpus.split(",")]
    if any(not c or (c != "cpu" and not c.isdigit()) for c in cards) or len(set(cards)) != len(cards):
        ap.error("--gpus requires distinct nonnegative GPU IDs or cpu")
    if a.stage in ("train", "analyze") and not a.run:
        ap.error("--stage train/analyze requires --run or FW_RUN")
    recorded = None
    if a.run:
        out = run_path(a.run)
        if (out / "source.json").exists():
            recorded = common.load_json(out / "source.json")
    try:
        levels = validate_levels(tuple(int(v.strip()) for v in a.levels.split(","))) if a.levels else (
            validate_levels(recorded["protocol"]["levels"]) if recorded else LEVELS)
    except ValueError as exc:
        ap.error(str(exc))
    source = a.source_run or (recorded["source_run"] if recorded else DEFAULT_SOURCE)
    bundle = validate_source(source, levels)
    if recorded and recorded != bundle["manifest"]:
        raise ValueError("existing run has different source files or protocol; changing levels requires a new run")
    print(f"[05] seed 0, 81920 episodes, 64 frames, 5 full epochs; F=0/32 reused, new={new_levels(bundle['manifest'])}", flush=True)
    if levels != LEVELS:
        print("[05] EXPLICIT REDUCED DESIGN: not a complete five-dose experiment", flush=True)
    for F in levels:
        b = budget(F)
        print(f"[05] F={F}: operation clips={b['operation_clips']}, transitions/episode={b['pairs_per_episode']}, updates={b['steps']}", flush=True)
    if a.dry_run:
        print("[05] dry-run complete: no writes, replay, generation, evaluation or training; compression feasibility not checked", flush=True)
        return
    name = a.run or fresh_name(a.tag)
    out = run_path(name)
    with run_lock(out):
        bind_run(name, bundle)
        print(f"[05] run={name}; output={common.rel(out)}", flush=True)
        if a.stage == "preflight":
            arrays, actions, process = cache_arrays(bundle, out)
            preflight(bundle, out, arrays, actions, process, a.preflight_count)
            print("[05] sample preflight only; no full variant generation or training", flush=True)
            return
        if a.stage in ("all", "prepare"):
            prepare_all(bundle, name, a.workers, a.shard_size, a.preflight_count)
            audited = audit_variants(bundle, name)
            save_jobs(name, audited)
        if a.stage in ("all", "train"):
            if a.stage == "train":
                save_jobs(name, verify_audit(name))
            execute_queue(name, a.gpus, a.progress_every)
        if a.stage in ("all", "train", "analyze"):
            result = analyze(name, a.boot, a.seed)
            print(f"[05] complete: {common.rel(result)}", flush=True)
        else:
            print("[05] preparation complete; no training started", flush=True)


if __name__ == "__main__":
    main()
