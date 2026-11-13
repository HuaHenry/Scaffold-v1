"""Build all (prefix size, condition, training seed) jobs without reusing models."""
from .shared import PROJECT, common, job_id, run_path, train_cli, write_json


def jobs(name, audited):
    p = audited["protocol"]
    rows = []
    for n in p["sizes"]:
        for condition in p["conditions"]:
            for seed in p["training_seeds"]:
                key = job_id(condition, n, seed)
                train_cli(name, condition, n, seed, audited)
                rows.append({"run": name, "id": key, "condition": condition, "episodes": n,
                    "seed": seed, "steps": p["steps"], "out": common.rel(run_path(name) / "train" / key),
                    "cwd": str(PROJECT), "cmd": ["python", "-m", "preresearch.06_data_scale.exp.train_scale",
                        "--run", name, "--condition", condition, "--episodes", str(n), "--seed", str(seed)]})
    return rows


def save_jobs(name, audited):
    rows = jobs(name, audited)
    path = run_path(name) / "jobs/data_scale.json"
    if path.exists() and common.load_json(path) != rows:
        raise ValueError("job list changed; use a new run")
    if not path.exists():
        write_json(path, rows)
    print(f"[06] {len(rows)} NEW training tasks; no experiment03 models reused", flush=True)
    return rows
