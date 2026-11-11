"""Three new formation doses; F=0 and F=32 refer to completed main models."""
import argparse
from importlib import import_module

from .audit import verify_audit
from .shared import BATCH, EPOCHS, EPISODES, EXP_NAME, PACKAGE, TRAIN_DEFAULTS, budget, common, new_levels, resolve_file, run_lock, run_path, write_json


def jobs(name, audited):
    rows = []
    for F in new_levels({"protocol": audited["protocol"]}):
        data = resolve_file(audited["variants"][str(F)]["data"])
        ctx = import_module(f"{PACKAGE}.make_jobs").Ctx(None, data, name, BATCH)
        ctx.fz = {"lr": 3e-4, "change_weight": 8, "depth": 8, "unroll": 1}
        row = ctx.job(EXP_NAME, f"process_f{F}_compress", "process", 0,
                      steps=budget(F)["steps"], episodes=EPISODES, tag="formation_ratio")
        row["cmd"][row["cmd"].index("--data") + 1] = str(data)
        row["cmd"] += ["--epochs", str(EPOCHS), "--select-best-val", "--val", str(resolve_file(audited["inputs"]["val"])),
                       "--val-every", "500", "--log-batch-composition"]
        for key, value in TRAIN_DEFAULTS.items():
            row["cmd"] += ["--" + key.replace("_", "-"), str(value)]
        row.update(F=F, condition="process", data=common.rel(data))
        rows.append(row)
    return rows


def save_jobs(name, audited):
    rows = jobs(name, audited)
    path = run_path(name) / "jobs" / "formation_ratio.json"
    if path.exists() and common.load_json(path) != rows:
        raise ValueError("job settings changed; use a new run")
    if not path.exists():
        write_json(path, rows)
    print(f"[05] {len(rows)} new training tasks; completed F=0/32 source models reused: {common.rel(path)}", flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    with run_lock(run_path(a.run)):
        save_jobs(a.run, verify_audit(a.run))


if __name__ == "__main__":
    main()
