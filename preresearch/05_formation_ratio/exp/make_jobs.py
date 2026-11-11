"""All supervision doses are new equal-budget training tasks, including 0 and 0.5."""
import argparse

from .audit import verify_audit
from .shared import PROJECT, common, dose_key, run_lock, run_path, training_args, write_json


def jobs(name, audited):
    rows = []
    for ratio in audited["protocol"]["ratios"]:
        args = training_args(name, ratio, audited)
        rows.append({"run": name, "id": args["id"], "dose": dose_key(ratio), "formation_ratio": ratio,
                     "out": args["out"], "steps": args["steps"], "seed": 0,
                     "cwd": str(PROJECT), "cmd": ["python", "-m", "preresearch.05_formation_ratio.exp.train_ratio",
                                                   "--run", name, "--formation-ratio", str(ratio)]})
    return rows


def save_jobs(name, audited):
    rows = jobs(name, audited)
    path = run_path(name) / "jobs/formation_ratio.json"
    if path.exists() and common.load_json(path) != rows:
        raise ValueError("training job list changed; use a new run")
    if not path.exists():
        write_json(path, rows)
    print(f"[05] {len(rows)} NEW training tasks, same update budget; no experiment 03 models reused", flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    with run_lock(run_path(a.run)):
        save_jobs(a.run, verify_audit(a.run))


if __name__ == "__main__":
    main()
