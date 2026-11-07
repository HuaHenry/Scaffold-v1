"""Create paired video-budget jobs for the fixed-step or full-epoch protocol."""
import argparse
import math
from importlib import import_module
from pathlib import Path

from .shared import CONDITIONS, EXP_NAME, LEVELS, PACKAGE, common
from .generate_data import input_hashes


def epoch_steps(episodes, condition, batch, epochs):
    return epochs * math.ceil((56 if condition == "state" else 59) * episodes / batch)


def jobs(run, data, seeds, steps=12000, batch=256, lr=3e-4, change_weight=8, depth=8,
         val=common.DEFAULT_VAL, epochs=None):
    if batch <= 0 or (epochs is not None and epochs <= 0):
        raise ValueError("batch and epochs must be positive")
    if epochs is None and batch % 2:
        raise ValueError("batch must be even for 50:50 process batches")
    Ctx = import_module(f"{PACKAGE}.make_jobs").Ctx
    ctx = Ctx(steps, data, run, batch)
    ctx.fz = {"lr": lr, "change_weight": change_weight, "depth": depth, "unroll": 1}
    result = []
    for n in LEVELS:
        for s in seeds:
            for cond in CONDITIONS:
                rid = f"{cond}_n{n}_s{s}"
                job_steps = epoch_steps(n, cond, batch, epochs) if epochs is not None else steps
                j = ctx.job(EXP_NAME, rid, cond, s, steps=job_steps, episodes=n, tag=f"budget{n}")
                j["cmd"] += ["--log-batch-composition", "--select-best-val", "--val", common.rel(val),
                             "--val-every", "500"]
                if epochs is not None:
                    j["cmd"] += ["--epochs", str(epochs)]
                elif cond == "process":
                    j["cmd"].append("--balanced-phases")
                result.append(j)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=None)
    ap.add_argument("--data", type=Path, default=common.GAME_CODE / "output/data_v3_curve_4096")
    ap.add_argument("--val", type=Path, default=None)
    ap.add_argument("--seeds", default="0", help="default: one seed for the full curve; more seeds are optional")
    ap.add_argument("--steps", type=int, default=None, help="fixed-step protocol (default 12000 without --epochs)")
    ap.add_argument("--epochs", type=int, default=None, help="full, no-replacement passes per budget level")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--change-weight", type=float, default=8)
    ap.add_argument("--depth", type=int, default=8)
    a = ap.parse_args(argv)
    if a.epochs is not None and a.steps is not None:
        ap.error("--epochs and --steps are mutually exclusive")
    if a.epochs is not None and a.batch != 256:
        ap.error("07_02 audit fixes batch at 256")
    if a.epochs is None:
        a.steps = 12000 if a.steps is None else a.steps
    seeds = tuple(int(s) for s in a.seeds.split(","))
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    run = a.run or common.current_run(EXP_NAME)
    d = common.RESULTS_ROOT / EXP_NAME / run
    if not (d / "audit.json").is_file():
        raise FileNotFoundError("run exp.audit first; it verifies equal frames and evaluation deduplication")
    audited = common.load_json(d / "audit.json")
    protocol = audited.get("training_protocol", {})
    if (a.epochs is None and protocol or a.epochs is not None
            and protocol != {"mode": "full_epochs", "epochs": a.epochs, "batch": a.batch}):
        raise ValueError("job training protocol differs from the audited run")
    data = common.resolve(a.data)
    val = common.resolve(a.val or audited["inputs"]["val"])
    if (audited["inputs"]["data"] != common.rel(data)
            or audited["inputs"]["val"] != common.rel(val)
            or audited["hashes"]["data"] != input_hashes(data)
            or audited["hashes"]["val"] != input_hashes(val)):
        raise ValueError("training or validation input differs from audit")
    protocol = f"epochs{a.epochs}" if a.epochs is not None else f"steps{a.steps}"
    out = d / "jobs" / f"curve_{protocol}_seeds{'-'.join(map(str, seeds))}.json"
    rows = jobs(run, data, seeds, a.steps, a.batch, a.lr, a.change_weight, a.depth, val, a.epochs)
    if out.exists():
        if common.load_json(out) != rows:
            raise ValueError("existing job list differs from requested training configuration")
        print(f"reusing {common.rel(out)}")
        return
    common.dump_json(rows, out)
    print(f"{len(rows)} jobs -> {common.rel(out)}")


if __name__ == "__main__":
    main()
