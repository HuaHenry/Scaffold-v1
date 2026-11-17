"""Experiment 09 CLI: frozen inference, truth/oracle audits, paired statistics and plots."""
import argparse
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

from .analysis import analyze
from .evaluation import (audit_targets, cached, evaluate_task, oracle_checks, save_task)
from .measurement import load_formation
from .shared import (DRIFT_HORIZON, FORMATION_HORIZON, HORIZON, PILOT, PROJECT, VERSION,
                     common, output_path, protect_output, read_npz, resolve, write_json, write_npz)
from .sources import load_sources, verify_evaluation_inputs

TASKS = ("intervention", "drift", "formation")


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="Experiment 09 output run name")
    ap.add_argument("--source-run", action="append", help="Audited experiment 03 run path/name; repeat for independent repetitions")
    ap.add_argument("--models", type=Path, help="Explicit JSON: models=[{condition, seed, path, source_run}]")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Required paired training seeds, defaults to all three")
    ap.add_argument("--val", type=Path, help="Ordered Process validation directory, defaults to the source audit")
    ap.add_argument("--mode", choices=("all", "evaluate", "analysis"), default="all")
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit-per-split", type=int, help="Explicit diagnostic subset; never labeled a full experiment")
    ap.add_argument("--formation-limit", type=int, help="Explicit diagnostic prefix of formation validation")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=2027)
    ap.add_argument("--oracle-seed", type=int, default=12345)
    ap.add_argument("--oracle-exact-threshold", type=float, default=.95)
    ap.add_argument("--strict-audit", action="store_true", help="Abort before learned inference if truth/oracle acceptance fails")
    ap.add_argument("--save-trajectories", action="store_true", help="Save all grids in addition to per-tick scores; uses more disk")
    ap.add_argument("--no-plots", action="store_true")
    return ap


def device_for(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu for local diagnostics")
    return device


def prepare_bench(path, limit):
    bench = common.load_bench(path)
    n = len(bench["ctx_fg"])
    if (n != 512 or bench["fut_fg"].shape != (n, 2, HORIZON, common.H, common.W)
            or bench["fut_bg"].shape != bench["fut_fg"].shape
            or not np.array_equal(np.bincount(bench["split_ids"], minlength=4), np.full(4, 128))):
        raise ValueError("Experiment 09 requires the frozen 512-scene, four-split FormBench")
    if any(any(t < 0 or t >= HORIZON for t in common.iv_by_t(iv)) for pair in bench["ivs"] for iv in pair):
        raise ValueError("Intervention scripts extend outside the 32-tick benchmark")
    bench["case_ids"] = np.arange(n)
    return common.select_bench(bench, limit=limit)


def code_identity():
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += [PROJECT / "preresearch/01_pilot_validity/exp" / name for name in
              ("common.py", "metrics.py", "rollout.py", "evaluate.py", "model.py")]
    paths += [PROJECT / "game_code" / name for name in ("core.py", "datasets.py", "datasets_v3.py")]
    return {str(path.relative_to(PROJECT)): common.sha1_file(path) for path in sorted(paths)}


def protocol(args, audit, models, bench, val, val_hashes, formation):
    return {"version": VERSION, "source_inputs": audit["inputs"], "source_hashes": audit["hashes"],
            "models": models, "seeds": args.seeds, "splits": bench["splits"],
            "benchmark_case_ids": bench["case_ids"].tolist(), "formation_episode_ids": formation["episode_ids"].tolist(),
            "validation": str(val), "validation_hashes": val_hashes,
            "horizons": {"intervention": HORIZON, "drift": DRIFT_HORIZON, "formation": FORMATION_HORIZON},
            "stop_on_fixed_point": False, "teacher_forcing": False, "weights": "validation-selected EMA",
            "oracle_seed": args.oracle_seed, "oracle_exact_threshold": args.oracle_exact_threshold,
            "save_trajectories": args.save_trajectories,
            "diagnostic_only": len(args.seeds) < 3 or args.limit_per_split is not None or args.formation_limit is not None,
            "code_sha1": code_identity()}


def evaluate(args, out, source, bench, formation):
    truths, truth_report = audit_targets(bench, formation)
    for task, values in truths.items():
        if task == "intervention":
            values.update({"case_ids": bench["case_ids"], "split_ids": bench["split_ids"]})
        write_npz(out / f"audit_{task}.npz", values)
    oracles = {}
    for task in TASKS:
        path = out / "raw/oracle_reseed" / f"{task}.npz"
        identity = {"protocol": source, "task": task, "model": "oracle_reseed"}
        if cached(path, identity):
            oracles[task] = read_npz(path)
        else:
            values, trajectories = evaluate_task(task, bench, formation, save_trajectories=args.save_trajectories,
                                                  oracle_seed=args.oracle_seed)
            save_task(path, identity, values, trajectories)
            oracles[task] = values
    checks = oracle_checks(oracles, args.oracle_exact_threshold)
    truth_ok = truth_report["intervention"]["all_ticks_zero"]
    checks["true_intervention_all_ticks_zero"] = truth_ok
    acceptance = (truth_ok and truth_report["contexts"]["all_ticks_zero"]
                  and checks["formation_probe_passed"] and checks["noop_engine_static"]
                  and checks["noop_engine_violation_free"])
    audit = {"truth": truth_report, "checks": checks, "acceptance_passed": bool(acceptance),
             "qualified": bool(acceptance and not source["diagnostic_only"]),
             "diagnostic_only": source["diagnostic_only"]}
    write_json(out / "audit.json", audit)
    if not acceptance:
        print("[09] WARNING: truth/oracle acceptance failed; inspect audit.json. Rest-rule flags on valid falling frames are not proof of illegal dynamics.", flush=True)
        if args.strict_audit:
            raise ValueError("Truth/oracle audit failed; learned-model evaluation was not started")
    device = device_for(args.device)
    loader = import_module(f"{PILOT}.evaluate").load_run
    for row in source["models"]:
        directory = out / "raw" / f"{row['condition']}_seed{row['seed']}"
        pending = [task for task in TASKS if not cached(directory / f"{task}.npz",
                   {"protocol": source, "task": task, "model": row})]
        if not pending:
            continue
        print(f"[09] {row['condition']} seed={row['seed']} on {device}", flush=True)
        model, checkpoint = loader(Path(row["path"]), device)
        if checkpoint["cfg"].get("depth") != 8:
            raise ValueError("Loaded predictor architecture differs from experiment 03")
        del checkpoint
        model.requires_grad_(False)
        for task in pending:
            values, trajectories = evaluate_task(task, bench, formation, model, device, args.batch, args.save_trajectories)
            save_task(directory / f"{task}.npz", {"protocol": source, "task": task, "model": row}, values, trajectories)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main(argv=None):
    args = parser().parse_args(argv)
    if (args.batch <= 0 or args.boot <= 0 or any(seed < 0 for seed in args.seeds)
            or any(value is not None and value <= 0 for value in (args.limit_per_split, args.formation_limit))
            or not 0 <= args.oracle_exact_threshold <= 1):
        raise ValueError("Batch, bootstrap and limits must be positive; oracle threshold is in [0,1]")
    if args.models is not None and args.source_run:
        raise ValueError("Choose --models or --source-run, not both")
    out = output_path(args.run)
    if args.mode == "analysis":
        source = common.load_json(out / "source.json")
        analyze(out, source, TASKS, args.boot, args.bootstrap_seed, not args.no_plots)
        print(out / "analysis/results.json")
        return
    audit, models, roots = load_sources(args.source_run, args.seeds, args.models)
    val = resolve(args.val) if args.val else resolve(audit["inputs"]["val"])
    bench_path, val_hashes = verify_evaluation_inputs(audit, val)
    protect_output(out, [*roots, val, *[resolve(value) for value in audit["inputs"].values()]])
    bench = prepare_bench(bench_path, args.limit_per_split)
    formation = load_formation(val, args.formation_limit)
    source = protocol(args, audit, models, bench, val, val_hashes, formation)
    if (out / "source.json").exists() and common.load_json(out / "source.json") != source:
        raise ValueError("Existing experiment 09 run has different sources/protocol; choose a new --run")
    write_json(out / "source.json", source)
    evaluate(args, out, source, bench, formation)
    if args.mode == "all":
        analyze(out, source, TASKS, args.boot, args.bootstrap_seed, not args.no_plots)
    print(out)


if __name__ == "__main__":
    main()
