"""Frozen experiment03 prefixes, equal updates, and final-checkpoint identities."""
from contextlib import contextmanager
import fcntl
from importlib import import_module
import os
from pathlib import Path
import sys
import zipfile

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
PILOT_PACKAGE = "preresearch.01_pilot_validity.exp"
MAIN_PACKAGE = "preresearch.03_main.exp"
common = import_module(f"{PILOT_PACKAGE}.common")
metrics = import_module(f"{PILOT_PACKAGE}.metrics")
helpers = import_module("preresearch.04_ood_generalization.exp.shared")
write_json, resolve_file = helpers.write_json, helpers.resolve_file
EXP_NAME, VERSION = "06_data_scale", 1
DEFAULT_SOURCE = "20261111-044057_four_conditions"
EPISODES, BATCH, STEPS = 81920, 256, 94400
SIZES = (5120, 10240, 20480, 40960, 81920)
SEEDS = (0, 1, 2)
CONDITIONS = common.CONDITIONS
TRAIN_DEFAULTS = {"dim": 384, "depth": 8, "heads": 6, "wd": 0.05, "ema": 0.999,
                  "warmup": 300, "lr": 3e-4, "lr_min": 3e-5, "change_weight": 8,
                  "unroll": 1, "val_every": 500}
METRICS = ("cascade_f1", "change_f1", "cell_acc_mask", "rank_acc", "any_violation")


def integers(values, label, minimum, maximum=None):
    result = []
    for value in values:
        try:
            n = int(value)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"{label} must contain finite integers") from exc
        if isinstance(value, bool) or str(value) != str(n):
            raise ValueError(f"{label} must contain integers")
        if n < minimum or (maximum is not None and n > maximum):
            raise ValueError(f"{label} outside allowed range")
        result.append(n)
    if not result or result != sorted(set(result)):
        raise ValueError(f"{label} must be nonempty, sorted and distinct")
    return tuple(result)


def protocol(sizes=SIZES, seeds=SEEDS, steps=STEPS):
    sizes = integers(sizes, "sizes", 1, EPISODES)
    seeds = integers(seeds, "training seeds", 0)
    if len(sizes) < 2:
        raise ValueError("a data curve requires at least two sizes")
    steps = integers([steps], "updates", 1)[0]
    return {"mode": "nested_prefix_equal_updates", "source_episodes": EPISODES,
            "sizes": list(sizes), "training_seeds": list(seeds), "conditions": list(CONDITIONS),
            "steps": steps, "batch": BATCH, "epochs": None, "sampling": "uniform_with_replacement",
            "pair_protocol": "video", "train_defaults": dict(TRAIN_DEFAULTS),
            "checkpoint_selection": "last", "checkpoint_weights": "ema",
            "all_cells_retrained": True, "reuse_main_endpoints": False}


def budget(condition, n, steps=STEPS):
    if condition not in CONDITIONS:
        raise ValueError("unknown condition")
    n = integers([n], "episodes", 1, EPISODES)[0]
    steps = integers([steps], "updates", 1)[0]
    per_episode = 56 if condition in ("state", "partial") else 59
    return {"episodes": n, "pairs": n * per_episode, "pairs_per_episode": per_episode,
            "steps": steps, "batch": BATCH, "samples": steps * BATCH, "epochs": None,
            "exposures_per_pair": steps * BATCH / (n * per_episode)}


def job_id(condition, n, seed):
    budget(condition, n)
    integers([seed], "training seeds", 0)
    return f"{condition}_n{n}_seed{seed}"


def check_output_tree(out):
    out = Path(out).absolute()
    if out.resolve() != out:
        raise ValueError("experiment06 output must not pass through symbolic links")
    if out.exists():
        for directory, dirs, files in os.walk(out, followlinks=False):
            if any((Path(directory) / n).is_symlink() for n in (*dirs, *files)):
                raise ValueError("experiment06 output contains symbolic links; refusing writes")
    return out


def run_path(name):
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("run must be a simple run name")
    return check_output_tree(common.RESULTS_ROOT / EXP_NAME / name)


@contextmanager
def run_lock(out):
    out = check_output_tree(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "pipeline.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another experiment06 pipeline is using this run") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def array_header(path, key):
    with zipfile.ZipFile(path) as z, z.open(f"{key}.npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"unsupported npy header {version}")
    if dtype.hasobject:
        raise ValueError("object arrays are not allowed")
    return shape, dtype


def validate_source(value, sizes=SIZES, seeds=SEEDS, steps=STEPS):
    p = protocol(sizes, seeds, steps)
    source = helpers.source_path(value)
    audited = common.load_json(source / "audit.json")
    inputs = import_module(f"{MAIN_PACKAGE}.audit").verify_inputs(audited)
    summary = common.load_summary(inputs["data"])
    if (audited.get("protocol", {}).get("episodes") != EPISODES
            or summary.get("profile") != "simple_to_complex"
            or summary.get("layout") != {"formation": 32, "clip": 8, "state_clips": 8, "mixed_clips": 4}
            or any(summary["conditions"][c]["episodes"] != EPISODES for c in CONDITIONS)
            or any(audited.get(k) != 0 for k in ("train_validation_overlap", "train_bench_overlap", "validation_bench_overlap"))):
        raise ValueError("source must be experiment03's frozen 81920-episode simple training data")
    for c in CONDITIONS:
        for key in ("fg", "bg", "cont", "phase"):
            shape, dtype = array_header(inputs["data"] / f"train_{c}.npz", key)
            expected = (EPISODES, 64, common.H, common.W) if key in ("fg", "bg") else (EPISODES, 64)
            if shape != expected or (key in ("fg", "bg") and dtype != np.dtype("uint8")):
                raise ValueError(f"{c}/{key}: frozen array shape or dtype differs")
    manifest = {"version": VERSION, "source_run": common.rel(source),
                "source_audit_sha1": common.sha1_file(source / "audit.json"),
                "inputs": audited["inputs"], "hashes": audited["hashes"], "protocol": p}
    return {"source": source, "inputs": inputs, "manifest": manifest, "audited": audited}


def prefix_audit(bundle):
    m = bundle["manifest"]
    scenes = common.load_json(bundle["inputs"]["data"] / "train_scenes.json")
    if len(scenes) != EPISODES:
        raise ValueError("scene count differs from frozen source")
    prefixes = {}
    import hashlib
    import json
    for n in m["protocol"]["sizes"]:
        selected = scenes[:n]
        unique = len({s["final_hash"] for s in selected})
        kinds = {}
        for scene in selected:
            for kind in set(scene["kinds"]):
                kinds[kind] = kinds.get(kind, 0) + 1
        prefixes[str(n)] = {"episodes": n, "selection": {"start": 0, "stop": n, "step": 1},
            "scene_prefix_sha1": hashlib.sha1(json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "unique_final_structures": unique, "duplicate_episodes": n - unique, "kind_scene_counts": kinds,
            "conditions": {c: budget(c, n, m["protocol"]["steps"]) for c in CONDITIONS}}
    return {"version": VERSION, "source": m, "inputs": m["inputs"], "hashes": m["hashes"],
            "protocol": m["protocol"], "prefixes": prefixes,
            "prefix_identity": "same frozen file SHA1 plus identical episode index range for all conditions"}


def bind_run(name, bundle, audited):
    out = run_path(name)
    helpers.forbid_source_output(out, bundle["source"], bundle["audited"])
    expected = {"source.json": bundle["manifest"], "audit.json": audited}
    if not (out / "source.json").exists() and out.exists():
        if set(x.name for x in out.iterdir()) - {"pipeline.lock", "run.json"}:
            raise ValueError("unbound output directory is not empty; use a new run")
    if (out / "run.json").exists() and common.load_json(out / "run.json").get("exp") != EXP_NAME:
        raise ValueError("output belongs to another experiment")
    for filename, value in expected.items():
        if (out / filename).exists() and common.load_json(out / filename) != value:
            raise ValueError("source, prefixes or training budget changed; use a new run")
    common.start_run(EXP_NAME, name=name)
    for filename, value in expected.items():
        if not (out / filename).exists():
            write_json(out / filename, value)
    return out


def verify_audit(name):
    out = run_path(name)
    recorded = common.load_json(out / "source.json")
    p = recorded["protocol"]
    bundle = validate_source(recorded["source_run"], p["sizes"], p["training_seeds"], p["steps"])
    audited = prefix_audit(bundle)
    if bundle["manifest"] != recorded or common.load_json(out / "audit.json") != audited:
        raise ValueError("frozen source or prefix audit changed")
    return audited


def train_cli(name, condition, n, seed, audited):
    p = audited["protocol"]
    if n not in p["sizes"] or seed not in p["training_seeds"] or condition not in p["conditions"]:
        raise ValueError("task is not in this run's recorded design")
    out = run_path(name) / "train" / job_id(condition, n, seed)
    args = {**p["train_defaults"], "condition": condition, "episodes": n, "steps": p["steps"],
            "batch": p["batch"], "seed": seed, "protocol": "video", "exp": EXP_NAME,
            "id": job_id(condition, n, seed), "out": str(out),
            "data": str(resolve_file(audited["inputs"]["data"])),
            "val": str(resolve_file(audited["inputs"]["val"])), "log_every": 50}
    return [v for k, value in args.items() for v in ("--" + k.replace("_", "-"), str(value))] + ["--log-batch-composition"]


def expected_args(row, audited):
    trainer = import_module(f"{PILOT_PACKAGE}.train")
    a = trainer.get_args(train_cli(row["run"], row["condition"], row["episodes"], row["seed"], audited))
    return {k: common.rel(v) if isinstance(v, Path) else v for k, v in vars(a).items()}


def expected_data_hashes(audited, condition):
    return {k: audited["hashes"]["data"][k] for k in
            (f"train_{condition}.npz", f"train_{condition}_actions.json", "train_summary.json")}


def validate_config(out, row, audited):
    cfg = common.load_json(Path(out) / "config.json")
    if (cfg.get("args") != expected_args(row, audited)
            or cfg.get("data_sha1") != expected_data_hashes(audited, row["condition"])
            or cfg.get("episodes_used") != row["episodes"]
            or cfg.get("n_pairs") != budget(row["condition"], row["episodes"], row["steps"])["pairs"]):
        raise ValueError("training configuration, source or prefix differs")
    return cfg


def training_identity(out, row, audited):
    out = Path(out)
    return {"version": VERSION, "id": row["id"], "checkpoint_step": row["steps"],
            "selection": "last", "weights": "ema", "samples_seen": row["steps"] * BATCH,
            "prefix": audited["prefixes"][str(row["episodes"])],
            "validation_hashes": audited["hashes"]["val"],
            "hashes": {n: common.sha1_file(out / n) for n in ("config.json", "done.json", "ckpt.pt")}}


def validate_done(out, row, audited):
    import torch
    out = Path(out)
    cfg = validate_config(out, row, audited)
    done = common.load_json(out / "done.json")
    marker = common.load_json(out / "train_complete.json")
    if marker != training_identity(out, row, audited):
        raise ValueError("completed training identity changed")
    if (done.get("selection") != "last" or done.get("steps") != row["steps"]
            or done.get("samples_seen") != row["steps"] * BATCH or done.get("epochs") is not None
            or done.get("n_pairs") != cfg["n_pairs"] or done.get("checkpoint_step") != row["steps"]
            or done.get("weights") != "ema" or done.get("ckpt_sha1") != marker["hashes"]["ckpt.pt"]):
        raise ValueError("incomplete training budget or non-final checkpoint")
    ck = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=True)
    model = import_module(f"{PILOT_PACKAGE}.model").build_model(
        {k: cfg["args"][k] for k in ("dim", "depth", "heads")})
    if (ck.get("step") != row["steps"] or ck.get("args") != cfg["args"] or ck.get("cfg") != model.cfg
            or ck.get("selection") != "last" or ck.get("weights") != "ema"):
        raise ValueError("checkpoint is not this task's final EMA")
    for key in ("model", "ema"):
        if not all(torch.is_tensor(v) and torch.isfinite(v).all() for v in ck[key].values()):
            raise ValueError("invalid or nonfinite checkpoint weights")
        model.load_state_dict(ck[key], strict=True)
    for key in ("val", "train_raw", "train_ema"):
        score = done.get(key, {}).get("cascade_acc", float("nan"))
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"missing or invalid {key} accuracy")
    return done


def evaluation_identity(out, done):
    out = Path(out)
    return {"version": VERSION, "checkpoint_sha1": done["ckpt_sha1"],
            "training_complete_sha1": common.sha1_file(out / "train_complete.json"),
            "checkpoint_step": done["checkpoint_step"], "selection": "last", "weights": "ema",
            "hashes": {n: common.sha1_file(out / n) for n in ("eval.json", "eval_samples.npz")}}


def validate_eval(out, audited, done):
    out = Path(out)
    ev = common.load_json(out / "eval.json")
    bp = resolve_file(audited["inputs"]["bench"])
    if (ev.get("bench") != common.rel(bp) or ev.get("n_samples") != 512
            or ev.get("bench_sha1") != audited["hashes"]["bench"][bp.name]
            or ev.get("bench_json_sha1") != audited["hashes"]["bench"][bp.with_suffix(".json").name]
            or ev.get("checkpoint_sha1") != done["ckpt_sha1"]):
        raise ValueError("evaluation differs from frozen bench or final checkpoint")
    marker = out / "eval_complete.json"
    if marker.exists() and common.load_json(marker) != evaluation_identity(out, done):
        raise ValueError("committed evaluation identity changed; refusing overwrite")
    bench = common.load_bench(bp)
    with np.load(out / "eval_samples.npz", allow_pickle=False) as z:
        samples = {k: z[k] for k in (*METRICS, "split_ids", "rank_valid", "n_cascade_true", "pred_fg", "pred_bg")}
    import_module("preresearch.05_formation_ratio.exp.shared").validate_metrics(samples, bench)
    return ev


def record_eval(out, audited, done):
    validate_eval(out, audited, done)
    path = Path(out) / "eval_complete.json"
    if not path.exists():
        write_json(path, evaluation_identity(out, done))
