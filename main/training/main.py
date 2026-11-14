"""Training is separate from data generation; prepare explicitly freezes the chosen settings."""
import argparse
from pathlib import Path

from ..shared import lock, load_json
from ..data.audit import verify
from .protocol import make_protocol, prepare, run_root, single_training_seed, verify_run
from .queue import devices, execute, preflight


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--stage", choices=("prepare", "train", "evaluate", "analyze"), required=True)
    ap.add_argument("--data", type=Path)
    ap.add_argument("--tasks", help="prepare only; default all tasks in the frozen dataset")
    ap.add_argument("--episodes", type=int, help="prepare only; use the first N episodes per task from a larger frozen dataset")
    ap.add_argument("--formation-ratio")
    ap.add_argument("--selection-note", default="")
    ap.add_argument("--steps", type=int, help="prepare only; default94400")
    ap.add_argument("--seeds", help="prepare only; exactly one training seed, default0")
    ap.add_argument("--modes", choices=("single", "mixed", "both"), help="prepare only; default both")
    ap.add_argument("--mixed-budget", choices=("per_task", "total"), help="prepare only; default per_task")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--progress-every", type=float, default=30)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--bootstrap-seed", type=int, default=12345)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    devices(a.gpus)
    out = run_root(a.run)
    if a.stage == "prepare":
        if a.data is None or a.episodes is None or a.formation_ratio is None:
            ap.error("prepare requires --data --episodes --formation-ratio and --selection-note")
        seeds = single_training_seed(("0" if a.seeds is None else a.seeds).split(","))
        manifest = verify(a.data)
        modes = a.modes or "both"
        p = make_protocol(manifest, a.data, a.tasks or "all", a.episodes, a.formation_ratio,
                          a.steps if a.steps is not None else 94400,
                          seeds, ("single", "mixed") if modes == "both" else (modes,),
                          a.mixed_budget or "per_task", selection_note=a.selection_note)
        for task, prefix in p["training_prefixes"].items():
            print(f"[训练数据] {task}: 已生成{prefix['available_episodes']:,}集，使用前{prefix['used_episodes']:,}集；验证/测试不截断", flush=True)
        from ..shared import disjoint
        disjoint(out, [a.data])
        if a.dry_run:
            from .protocol import jobs
            print(f"read-only prepare checks complete: {len(jobs(p))} jobs; no writes")
            return
        if out.exists() and not (out / "protocol.json").exists() and any(p.name != "pipeline.lock" for p in out.iterdir()):
            raise ValueError("unbound run output is not empty")
        if (out / "protocol.json").exists() and load_json(out / "protocol.json") != p:
            raise ValueError("settings changed; create a new run")
        with lock(out / "pipeline.lock"):
            prepare(out, p, manifest)
        print(f"prepared: {out}; no training started")
        return
    if any(v is not None for v in (a.data, a.episodes, a.formation_ratio, a.tasks, a.steps, a.seeds, a.modes, a.mixed_budget)) or a.selection_note:
        ap.error("resume inherits all data/training settings; do not pass prepare parameters")
    p, _ = verify_run(out)
    if a.dry_run:
        if a.stage in ("evaluate", "analyze"):
            preflight(out, p, load_json(out / "jobs.json"))
        print("read-only run checks complete; no jobs launched")
        return
    with lock(out / "pipeline.lock"):
        if a.stage in ("train", "evaluate"):
            execute(out, a.gpus, a.stage == "evaluate", a.progress_every)
        else:
            from ..evaluation.analyze import analyze
            analyze(out, a.boot, a.bootstrap_seed)


if __name__ == "__main__":
    main()
