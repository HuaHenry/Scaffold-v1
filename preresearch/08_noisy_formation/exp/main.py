"""Prepare, run or evaluate experiment08's original single structural task."""
import argparse
from pathlib import Path

from .shared import prepare, verify_run


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    build = sub.add_parser("prepare", help="bind qualified paired data; no training")
    build.add_argument("--run", required=True)
    build.add_argument("--data", required=True, type=Path)
    build.add_argument("--episodes", type=int, default=81920)
    build.add_argument("--steps", type=int, default=94400)
    build.add_argument("--reuse-scale-run", help="reuse all four completed matching seed0 experiment06 baselines read-only")
    for cmd in ("run", "evaluate"):
        parser = sub.add_parser(cmd)
        parser.add_argument("--run", required=True)
        parser.add_argument("--gpus", default="0,1,2,3")
        parser.add_argument("--progress-every", type=float, default=30)
    audit = sub.add_parser("audit")
    audit.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    if a.command == "prepare":
        out = prepare(a.run, a.data, a.episodes, a.steps, a.reuse_scale_run)
        print(f"[08] prepared: {out}; seed0 only; training not started", flush=True)
    elif a.command == "audit":
        verify_run(a.run)
        print("[08] frozen run/data checks passed; no training", flush=True)
    else:
        from .run_queue import execute_queue
        execute_queue(a.run, a.gpus, a.progress_every, a.command == "evaluate")


if __name__ == "__main__":
    main()
