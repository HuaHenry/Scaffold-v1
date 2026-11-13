"""Experiment06: four-condition data curves from experiment03's frozen prefixes."""
import argparse
import os
import sys

from .analysis import analyze
from .make_jobs import save_jobs
from .run_queue import devices, execute_queue
from .shared import (CONDITIONS, DEFAULT_SOURCE, bind_run,
                     budget, common, prefix_audit, protocol, run_lock, run_path, validate_source)


def fresh_name(tag):
    base = "_".join(v for v in (common.now_ts(), common.slug(tag)) if v)
    name, suffix = base, 1
    while run_path(name).exists():
        suffix += 1
        name = f"{base}-{suffix}"
    return name


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""From preresearch/06_data_scale using the zhouqi environment:
  env -u FW_RUN bash run_all.sh --dry-run
  env -u FW_RUN bash run_all.sh --stage prepare
  bash run_all.sh --run <prepared-run> --stage train --gpus 0,1,2,3
  bash run_all.sh --run <completed-run> --stage analyze
  bash evaluate_final.sh --run <completed-run> --gpus 0,1,2,3

Default: N=5120,10240,20480,40960,81920; four conditions; seeds0,1,2;
94400 updates per cell, batch256. All 60 cells train from scratch.
No generation, scene filtering, physical changes, or experiment03 model reuse.
Only final EMA is evaluated; validation is monitoring only.
Resume inherits source, sizes, seeds and steps; changes require a NEW run.
Interrupted training restarts at step0, not optimizer resume.
--seed is bootstrap seed; --seeds are training seeds.
""")
    ap.add_argument("--source-run", help="existing experiment03 run with frozen data audit; no source model reuse")
    ap.add_argument("--run", default=os.environ.get("FW_RUN"))
    ap.add_argument("--tag", default="data_scale_final")
    ap.add_argument("--sizes", help="comma-separated increasing episode counts; default 5120,10240,20480,40960,81920")
    ap.add_argument("--seeds", help="comma-separated increasing training seeds; default 0,1,2")
    ap.add_argument("--steps", type=int, help="equal updates per cell; default 94400")
    ap.add_argument("--stage", choices=("all", "prepare", "train", "analyze"), default="all")
    ap.add_argument("--dry-run", action="store_true", help="read-only source/prefix checks; no directories or jobs written")
    ap.add_argument("--gpus", default=os.environ.get("GPUS", "0,1,2,3"))
    ap.add_argument("--progress-every", type=float, default=30)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=12345, help="bootstrap seed, not training seed")
    ap.add_argument("--allow-incomplete", action="store_true", help="analyze only: use full curves from complete common seeds and list missing cells")
    a = ap.parse_args(argv)
    devices(a.gpus)
    if not 0 < a.progress_every < float("inf") or a.boot <= 0 or a.seed < 0:
        ap.error("finite positive progress interval, positive boot count and nonnegative bootstrap seed required")
    if a.allow_incomplete and a.stage != "analyze":
        ap.error("--allow-incomplete is only valid with --stage analyze")
    recorded = common.load_json(run_path(a.run) / "source.json") if a.run and (run_path(a.run) / "source.json").exists() else None
    if a.stage in ("train", "analyze") and recorded is None:
        ap.error("train/analyze requires a prepared --run or FW_RUN")
    p = recorded["protocol"] if recorded else protocol()
    sizes = a.sizes.split(",") if a.sizes is not None else p["sizes"]
    seeds = a.seeds.split(",") if a.seeds is not None else p["training_seeds"]
    steps = a.steps if a.steps is not None else p["steps"]
    source = a.source_run or (recorded["source_run"] if recorded else DEFAULT_SOURCE)
    bundle = validate_source(source, sizes, seeds, steps)
    if recorded and bundle["manifest"] != recorded:
        raise ValueError("source, sizes, seeds or updates changed; use a new run")
    audited = prefix_audit(bundle)
    design = audited["protocol"]
    print(f"[06] sizes={design['sizes']}; seeds={design['training_seeds']}; "
          f"{len(design['sizes']) * len(CONDITIONS) * len(design['training_seeds'])} NEW tasks", flush=True)
    for n in design["sizes"]:
        a_prefix = audited["prefixes"][str(n)]
        print(f"[06] N={n}: unique structures={a_prefix['unique_final_structures']}; "
              f"state/partial pairs={budget('state', n, steps)['pairs']}; "
              f"process/shuffle pairs={budget('process', n, steps)['pairs']}; "
              f"updates={steps}; samples={steps * 256}", flush=True)
    if a.dry_run:
        if recorded and common.load_json(run_path(a.run) / "audit.json") != audited:
            raise ValueError("recorded prefix audit changed")
        print("[06] read-only source and prefix checks complete; no writes or training", flush=True)
        return
    name = a.run or fresh_name(a.tag)
    with run_lock(run_path(name)):
        out = bind_run(name, bundle, audited)
        save_jobs(name, audited)
        print(f"[06] run={name}; output={common.rel(out)}", flush=True)
        if a.stage in ("all", "train"):
            execute_queue(name, a.gpus, a.progress_every)
        if a.stage in ("all", "train", "analyze"):
            analyze(name, a.boot, a.seed, a.allow_incomplete)
        else:
            print("[06] preparation complete; no generation or training started", flush=True)


def cli(argv=None):
    try:
        main(argv)
    except (ValueError, KeyError, OSError, RuntimeError, EOFError) as exc:
        print(f"[06] ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from None


if __name__ == "__main__":
    cli()
