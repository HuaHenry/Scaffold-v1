"""Register four seed-0 6000-step video jobs in one experiment-02 run.

python -m exp.make_jobs --pilot01 /path/to/01/pilot_validity.json --frozen /path/to/frozen.json
"""
from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path

from .shared import EXP_NAME, PACKAGE, common


def make_jobs(run, steps, batch, data, frozen, pilot01):
    pilot = common.load_json(pilot01)
    a = pilot["A_ordering"]["id"]["cascade_f1"]
    gain = a["state"]["mean"] - a["apply_edit"]["mean"]
    if pilot["checks"]["V2"]["pass"] is not True or not (
        gain >= .12 and a["state"]["mean"] < a["oracle_reseed"]["mean"] and
        a["random_physics"]["mean"] < a["apply_edit"]["mean"] + .05
    ):
        raise ValueError("experiment 01 V1/V2 prerequisite not met under the current plan")
    fz = common.load_json(frozen)
    ctx_class = import_module(f"{PACKAGE}.make_jobs").Ctx
    ctx = ctx_class(steps, data, run, batch)
    ctx.fz = {key: fz[key] for key in ("lr", "change_weight", "depth", "unroll")}
    return [ctx.job(EXP_NAME, f"{condition}_s0", condition, 0) for condition in common.CONDITIONS]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot01", type=Path, required=True, help="specific experiment 01 pilot_validity.json")
    ap.add_argument("--frozen", type=Path, default=common.FROZEN)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--data", type=Path, default=common.DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    run = common.current_run(EXP_NAME)
    path = common.resolve_out(a.out) if a.out else common.RESULTS_ROOT / EXP_NAME / run / "jobs" / f"pilot_signal_steps{a.steps}.json"
    if path.exists():
        print(f"{common.rel(path)} exists; reusing this job list")
        return
    jobs = make_jobs(run, a.steps, a.batch, common.resolve(a.data), common.resolve(a.frozen), common.resolve(a.pilot01))
    common.dump_json(jobs, path)
    print(f"{len(jobs)} jobs -> {common.rel(path)}")


if __name__ == "__main__":
    main()
