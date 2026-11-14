"""Explicitly freeze settings selected on the anchor, then build single/mixed jobs."""
from pathlib import Path

from ..shared import (CONDITIONS, ROOT, RESULTS_ROOT, TRAIN_DEFAULTS, VERSION, code_identity, digest, disjoint,
                      file_hash, frozen_write, integer, load_json, name, ratio, safe_path, write_json)
from ..data.audit import verify
from ..tasks.registry import select


def single_training_seed(seeds):
    values = [integer(s, "training seed", 0) for s in seeds]
    if len(values) != 1:
        raise ValueError("main experiment requires exactly one training seed; default is 0")
    return values


def make_protocol(manifest, data, tasks, episodes, formation_ratio, steps=94400, seeds=(0,),
                  modes=("single", "mixed"), mixed_budget="per_task", overrides=None, selection_note=""):
    tasks = select(list(manifest["tasks"]) if tasks == "all" else tasks)
    if not set(tasks).issubset(manifest["tasks"]):
        raise ValueError("selected tasks are not present in frozen data")
    episodes, steps = integer(episodes, "episodes"), integer(steps, "steps")
    r = ratio(formation_ratio)
    if r in (0, 1):
        raise ValueError("four-condition main study requires 0 < formation ratio < 1; endpoints belong to ablations")
    seeds = single_training_seed(seeds)
    modes = list(modes)
    if not modes or len(set(modes)) != len(modes) or set(modes) - {"single", "mixed"}:
        raise ValueError("modes must select single and/or mixed")
    if mixed_budget not in ("per_task", "total"):
        raise ValueError("mixed budget must be per_task or total")
    if not selection_note.strip():
        raise ValueError("record how experiment05/06 or development data selected the frozen settings")
    for task in tasks:
        if episodes > manifest["task_data"][task]["train_episodes"]:
            raise ValueError("requested prefix exceeds generated/imported task data")
    settings = {**TRAIN_DEFAULTS, **(overrides or {})}
    if set(settings) != set(TRAIN_DEFAULTS):
        raise ValueError("unknown training overrides")
    for k in ("dim", "depth", "heads", "batch", "val_every", "log_every"):
        integer(settings[k], k)
    if settings["dim"] % settings["heads"]:
        raise ValueError("dim must be divisible by heads")
    if not 0 <= settings["ema"] < 1 or settings["lr"] <= 0 or settings["lr_min"] < 0:
        raise ValueError("invalid EMA or learning rate")
    return {"version": VERSION, "tasks": tasks, "episodes_per_task": episodes,
            "formation_ratio": str(r), "steps_single": steps, "training_seeds": seeds,
            "modes": modes, "mixed_budget": mixed_budget, "settings": settings,
            "conditions": list(CONDITIONS), "selection_note": selection_note,
            "data": str(Path(data).resolve()), "data_manifest_sha256": file_hash(Path(data) / "manifest.json"),
            "data_complete_sha256": file_hash(Path(data) / "complete.json"), "code": code_identity(),
            "sampling": "task-balanced cumulative quota; replacement within each phase",
            "selection": "last", "weights": "ema", "metrics": ["change_f1", "cell_acc_mask"],
            "horizon": 32, "validation_selection": False}


def jobs(protocol):
    seeds = single_training_seed(protocol["training_seeds"])
    rows = []
    tasks = protocol["tasks"]
    groups = [("single", [t]) for t in tasks] if "single" in protocol["modes"] else []
    if "mixed" in protocol["modes"]:
        groups.append(("mixed", tasks))
    for mode, group in groups:
        steps = protocol["steps_single"] * (len(group) if mode == "mixed" and protocol["mixed_budget"] == "per_task" else 1)
        for c in CONDITIONS:
            for seed in seeds:
                key = f"{mode}_{group[0] if mode == 'single' else 'all'}_{c}_seed{seed}"
                rows.append({"id": key, "mode": mode, "tasks": group, "condition": c,
                             "seed": seed, "steps": steps})
    return rows


def run_root(value):
    return disjoint(value, [ROOT, RESULTS_ROOT / "datasets"])


def prepare(out, protocol, manifest):
    single_training_seed(protocol["training_seeds"])
    out = run_root(out)
    disjoint(out, [protocol["data"]])
    if out.exists() and not (out / "protocol.json").exists():
        if any(p.name != "pipeline.lock" for p in out.iterdir()):
            raise ValueError("unbound run directory is not empty")
    frozen_write(out / "protocol.json", protocol)
    frozen_write(out / "jobs.json", jobs(protocol))
    write_json(out / "run.json", {"kind": "multitask_main", "protocol_sha256": file_hash(out / "protocol.json"),
                                  "task_count": len(protocol["tasks"]), "job_count": len(jobs(protocol))})
    return out


def verify_run(out, full_data=True):
    out = run_root(out)
    p = load_json(out / "protocol.json")
    single_training_seed(p["training_seeds"])
    disjoint(out, [p["data"]])
    manifest = verify(p["data"], full=full_data)
    if (file_hash(Path(p["data"]) / "manifest.json") != p["data_manifest_sha256"]
            or file_hash(Path(p["data"]) / "complete.json") != p["data_complete_sha256"]
            or code_identity() != p["code"] or load_json(out / "jobs.json") != jobs(p)):
        raise ValueError("run code, dataset or jobs changed; use a new run")
    if load_json(out / "run.json")["protocol_sha256"] != file_hash(out / "protocol.json"):
        raise ValueError("run protocol was modified")
    return p, manifest


def task_inputs(p, row):
    marker = load_json(Path(p["data"]) / "complete.json")["hashes"]
    names = []
    for task in row["tasks"]:
        names += [f"{task}/train/{row['condition']}_{k}.npy" for k in ("fg", "bg", "act")]
        names += [f"{task}/validation/state_{k}.npy" for k in ("fg", "bg", "act")]
        names += [f"{task}/bench.npz"]
    expected = {n: marker[n] for n in names}
    if {n: file_hash(Path(p["data"]) / n) for n in names} != expected:
        raise ValueError("task input files changed")
    return expected
