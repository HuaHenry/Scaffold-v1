"""Resolve paired completed experiment 03 checkpoints without substituting other studies."""
from pathlib import Path

from .shared import CONDITIONS, common, main, resolve


def validate_checkpoint(path, condition, seed, audit, verify_hashes=True):
    path = Path(path).resolve()
    config = common.load_json(path / "config.json")
    args = config["args"]
    done = common.load_json(path / "done.json")
    budget = main.budget(condition)
    expected = {"condition": condition, "seed": seed, "episodes": main.EPISODES,
                "epochs": 5, "steps": budget["steps"], "batch": 256, "protocol": "video",
                "lr": 3e-4, "change_weight": 8, "depth": 8, "unroll": 1,
                "data": audit["inputs"]["data"], "val": audit["inputs"]["val"],
                "val_every": 500, "select_best_val": True, "balanced_phases": False,
                "log_batch_composition": True}
    if any(args.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{path}: checkpoint does not follow the frozen experiment 03 protocol")
    expected_done = {"steps": budget["steps"], "epochs": 5, "n_pairs": budget["pairs"],
                     "samples_seen": budget["samples"], "steps_per_epoch": budget["steps_per_epoch"],
                     "selection": "val_cascade_acc"}
    if (any(done.get(key) != value for key, value in expected_done.items())
            or done.get("best_step") not in [i * budget["steps_per_epoch"] for i in range(1, 6)]
            or abs(done.get("form_frac_mean", -1) - budget["formation_fraction"]) > 1e-6):
        raise ValueError(f"{path}: incomplete epochs or wrong checkpoint selection")
    data_hashes = {key: audit["hashes"]["data"][key] for key in
                   (f"train_{condition}.npz", f"train_{condition}_actions.json", "train_summary.json")}
    val_hashes = {key: audit["hashes"]["val"][key] for key in
                  ("train_state.npz", "train_state_actions.json", "train_summary.json")}
    if config.get("data_sha1") != data_hashes or config.get("val_sha1") != val_hashes:
        raise ValueError(f"{path}: training/validation identity differs from the audit")
    if not (path / "ckpt.pt").is_file():
        raise FileNotFoundError(path / "ckpt.pt")
    if verify_hashes and done.get("ckpt_sha1") != common.sha1_file(path / "ckpt.pt"):
        raise ValueError(f"{path}: selected checkpoint identity changed")
    return {"condition": condition, "seed": seed, "path": str(path), "checkpoint_sha1": done["ckpt_sha1"],
            "config_sha1": common.sha1_file(path / "config.json"),
            "done_sha1": common.sha1_file(path / "done.json"), "best_step": done["best_step"]}


def require_pairs(models, seeds):
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Training seeds must be a nonempty list without duplicates")
    keys = [(row["condition"], row["seed"]) for row in models]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate condition/seed checkpoints; select sources explicitly")
    missing = [(condition, seed) for seed in seeds for condition in CONDITIONS if (condition, seed) not in keys]
    extra = set(keys) - {(condition, seed) for seed in seeds for condition in CONDITIONS}
    if missing or extra:
        raise ValueError(f"Unpaired source models. Missing={missing}; extra={sorted(extra)}. "
                         "No training is launched and missing repetitions are never fabricated. "
                         "Use --seeds 0 explicitly for a one-repetition diagnostic.")


def load_sources(source_runs, seeds, manifest=None):
    if manifest is not None:
        manifest_path = resolve(manifest)
        supplied = common.load_json(manifest_path)
        entries = supplied["models"]
        source_runs = sorted({row["source_run"] for row in entries})
    else:
        entries = None
    if not source_runs:
        raise ValueError("Provide --source-run (repeatable) or --models JSON with a models list")
    audits, roots = {}, {}
    for value in source_runs:
        direct = resolve(value)
        source = direct if (direct / "audit.json").is_file() else common.RESULTS_ROOT / "03_main" / str(value)
        source = source.resolve()
        if common.load_json(source / "run.json").get("exp") != "03_main":
            raise ValueError(f"Not an experiment 03 source: {source}")
        audit = common.load_json(source / "audit.json")
        if audits and any(audit[key] != next(iter(audits.values()))[key] for key in ("inputs", "hashes")):
            raise ValueError("All repetitions must share identical frozen training, validation and benchmark data")
        audits[value], roots[value] = audit, source
    audit = next(iter(audits.values()))
    if entries is None:
        entries = []
        for value, source in roots.items():
            for config in sorted((source / "train").rglob("config.json")):
                args = common.load_json(config).get("args", {})
                if args.get("condition") in CONDITIONS and args.get("seed") in seeds:
                    entries.append({"condition": args["condition"], "seed": args["seed"],
                                    "path": str(config.parent), "source_run": value})
    require_pairs(entries, seeds)
    seen_checkpoints, models = set(), []
    for row in entries:
        source = roots[row["source_run"]]
        path = resolve(row["path"])
        if source not in path.parents:
            raise ValueError("Explicit model path must be inside its audited experiment 03 source")
        identity = validate_checkpoint(path, row["condition"], row["seed"], audit)
        if identity["checkpoint_sha1"] in seen_checkpoints:
            raise ValueError("The same checkpoint cannot be counted as independent models")
        seen_checkpoints.add(identity["checkpoint_sha1"])
        models.append(identity)
    models.sort(key=lambda row: (row["seed"], CONDITIONS.index(row["condition"])))
    return audit, models, list(roots.values())


def verify_evaluation_inputs(audit, val):
    bench = resolve(audit["inputs"]["bench"])
    for path in (bench, bench.with_suffix(".json")):
        if common.sha1_file(path) != audit["hashes"]["bench"][path.name]:
            raise ValueError(f"Benchmark identity changed: {path}")
    identities = {}
    for name in ("train_process.npz", "train_process_actions.json", "train_summary.json"):
        path = val / name
        identity = common.sha1_file(path)
        if val == resolve(audit["inputs"]["val"]) and identity != audit["hashes"]["val"][name]:
            raise ValueError(f"Validation identity changed: {path}")
        identities[name] = identity
    return bench, identities
