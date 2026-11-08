"""Create exactly four isolated main-experiment training jobs."""
import argparse
from importlib import import_module

from .audit import verify_inputs
from .shared import BATCH, CONDITIONS, EPOCHS, EPISODES, EXP_NAME, PACKAGE, budget, common


def jobs(run, data, val):
    ctx = import_module(f"{PACKAGE}.make_jobs").Ctx(None, data, run, BATCH)
    ctx.fz = {"lr": 3e-4, "change_weight": 8, "depth": 8, "unroll": 1}
    rows = []
    for condition in CONDITIONS:
        row = ctx.job(EXP_NAME, condition, condition, 0, steps=budget(condition)["steps"],
                      episodes=EPISODES, tag="main03")
        row["cmd"] += ["--epochs", str(EPOCHS), "--select-best-val", "--val", common.rel(val),
                       "--val-every", "500", "--log-batch-composition"]
        rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    directory = common.RESULTS_ROOT / EXP_NAME / a.run
    audited = common.load_json(directory / "audit.json")
    paths = verify_inputs(audited)
    expected = {"epochs": EPOCHS, "batch": BATCH, "episodes": EPISODES,
                "conditions": list(CONDITIONS), "sampling": "full_epochs"}
    if audited.get("protocol") != expected:
        raise ValueError("main experiment must use the frozen four-condition protocol")
    rows = jobs(a.run, paths["data"], paths["val"])
    out = directory / "jobs" / "main03.json"
    if out.exists() and common.load_json(out) != rows:
        raise ValueError("main 03 job list changed; use a new run")
    common.dump_json(rows, out)
    print(common.rel(out))


if __name__ == "__main__":
    main()
