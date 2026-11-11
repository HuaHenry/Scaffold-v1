"""Version 2: equal-update supervision mixtures on unchanged frozen trajectories."""
from contextlib import contextmanager
import fcntl
import hashlib
from importlib import import_module
import os
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
PACKAGE = "preresearch.01_pilot_validity.exp"
MAIN_PACKAGE = "preresearch.03_main.exp"
common = import_module(f"{PACKAGE}.common")
metrics = import_module(f"{PACKAGE}.metrics")
helpers = import_module("preresearch.04_ood_generalization.exp.shared")
write_json, write_npz, resolve_file = helpers.write_json, helpers.write_npz, helpers.resolve_file
EXP_NAME, VERSION = "05_formation_ratio", 2
MODE = "supervision_sampling"
RATIOS = (0.0, 0.125, 0.25, 0.5, 0.75)
EPISODES, BATCH, STEPS = 81920, 256, 94400
DEFAULT_SOURCE = "20261111-044057_four_conditions"
TRAIN_DEFAULTS = {"dim": 384, "depth": 8, "heads": 6, "wd": 0.05, "ema": 0.999,
                  "warmup": 300, "lr": 3e-4, "lr_min": 3e-5, "change_weight": 8,
                  "unroll": 1, "val_every": 500}


def formation_count(ratio, batch=BATCH):
    if batch <= 0 or not np.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError("ratio must be finite in [0,1] and batch positive")
    n = ratio * batch
    if not np.isclose(n, round(n), atol=1e-10, rtol=0):
        raise ValueError("ratio * batch must be an integer; no silent rounding")
    return int(round(n))


def validate_ratios(ratios):
    ratios = tuple(float(v) for v in ratios)
    for ratio in ratios:
        formation_count(ratio)
    if ratios != tuple(sorted(set(ratios))) or not {0.0, 0.5}.issubset(ratios):
        raise ValueError("ratios must be sorted, distinct, and include 0 and 0.5")
    return ratios


def dose_key(ratio):
    return f"r{formation_count(ratio):03d}"


def budget(ratio, steps=STEPS):
    if steps <= 0 or not isinstance(steps, int):
        raise ValueError("updates must be a positive integer")
    n = formation_count(ratio)
    return {"formation_ratio": ratio, "steps": steps, "batch": BATCH,
            "samples": steps * BATCH, "formation_per_batch": n, "operation_per_batch": BATCH - n,
            "formation_samples": steps * n, "operation_samples": steps * (BATCH - n),
            "sampling": "stratified_with_replacement", "epochs": None}


def check_output_tree(out):
    out = Path(out).absolute()
    if out.resolve() != out:
        raise ValueError("experiment 05 output must not pass through a symbolic link")
    if out.exists():
        for directory, dirs, files in os.walk(out, followlinks=False):
            if any((Path(directory) / name).is_symlink() for name in (*dirs, *files)):
                raise ValueError("experiment 05 output contains a symbolic link; refusing writes")
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
            raise ValueError("another experiment 05 pipeline is using this run") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def array_sha1(array):
    return hashlib.sha1(np.ascontiguousarray(array).tobytes()).hexdigest()


def protocol(ratios=RATIOS, steps=STEPS):
    ratios = validate_ratios(ratios)
    budget(ratios[0], steps)
    return {"mode": MODE, "episodes": EPISODES, "condition": "process", "trajectory_frames": 64,
            "formation_frames": 32, "operation_clips": 4, "ratios": list(ratios), "steps": steps,
            "batch": BATCH, "epochs": None, "training_seeds": [0], "train_defaults": TRAIN_DEFAULTS,
            "sampling": "stratified_with_replacement", "pair_protocol": "video",
            "all_doses_retrained": True, "reuse_main_endpoints": False}


def validate_source(value, ratios=RATIOS, steps=STEPS):
    p = protocol(ratios, steps)
    source = helpers.source_path(value)
    print(f"[05] checking frozen experiment 03 source: {common.rel(source)}", flush=True)
    _, _, _, audited = import_module(f"{MAIN_PACKAGE}.analysis").load_results(source)
    inputs = {k: resolve_file(v) for k, v in audited["inputs"].items()}
    manifest = {"version": VERSION, "source_run": common.rel(source),
                "source_audit_sha1": common.sha1_file(source / "audit.json"),
                "inputs": audited["inputs"], "hashes": audited["hashes"], "protocol": p}
    return {"source": source, "inputs": inputs, "audited": audited, "manifest": manifest}


def require_current(recorded):
    if recorded.get("version") != VERSION or recorded.get("protocol", {}).get("mode") != MODE:
        raise ValueError("this run uses the retired frame-compression protocol; keep it unchanged and create a new supervision_ratio run")


def bind_run(name, bundle):
    out = run_path(name)
    helpers.forbid_source_output(out, bundle["source"], bundle["audited"])
    path = out / "source.json"
    if path.exists():
        require_current(common.load_json(path))
        if common.load_json(path) != bundle["manifest"]:
            raise ValueError("source or sampling budget changed; use a new experiment 05 run")
    elif out.exists() and any(p.name != "pipeline.lock" for p in out.iterdir()):
        metadata_only = set(p.name for p in out.iterdir()) <= {"pipeline.lock", "run.json"}
        if not metadata_only:
            raise ValueError("unbound output directory is not empty; use a new run")
    common.start_run(EXP_NAME, name=name)
    if not path.exists():
        write_json(path, bundle["manifest"])
    return out


def verify_manifest(name):
    recorded = common.load_json(run_path(name) / "source.json")
    require_current(recorded)
    p = recorded["protocol"]
    bundle = validate_source(recorded["source_run"], p["ratios"], p["steps"])
    if bundle["manifest"] != recorded:
        raise ValueError("frozen source identity or sampling protocol changed")
    return bundle


def training_args(name, ratio, audited):
    p = audited["protocol"]
    b = budget(ratio, p["steps"])
    key = dose_key(ratio)
    return {**TRAIN_DEFAULTS, "condition": "process", "episodes": p["episodes"], "steps": b["steps"],
            "batch": BATCH, "formation_ratio": ratio, "protocol": "video", "seed": 0, "epochs": None,
            "sampling_mode": MODE, "select_best_val": True, "log_batch_composition": True,
            "data": audited["inputs"]["data"], "val": audited["inputs"]["val"], "run": name,
            "id": f"{key}_seed0", "out": common.rel(run_path(name) / "train" / f"{key}_seed0"),
            "device": "auto", "log_every": 50}


def expected_data_hashes(audited, kind):
    condition = "process" if kind == "data" else "state"
    return {k: audited["hashes"][kind][k] for k in
            (f"train_{condition}.npz", f"train_{condition}_actions.json", "train_summary.json")}


def validate_config(out, row, audited):
    cfg = common.load_json(Path(out) / "config.json")
    expected = training_args(row["run"], row["formation_ratio"], audited)
    if any(cfg.get("args", {}).get(k) != v for k, v in expected.items()):
        raise ValueError("training configuration differs from the supervision sampling protocol")
    if (cfg.get("version") != VERSION or cfg.get("data_sha1") != expected_data_hashes(audited, "data")
            or cfg.get("val_sha1") != expected_data_hashes(audited, "val")
            or cfg.get("pools") != audited["pools"]):
        raise ValueError("training inputs or pair pools differ from the audited source")
    return cfg


def validate_done(out, row, audited):
    validate_config(out, row, audited)
    done = common.load_json(Path(out) / "done.json")
    b = budget(row["formation_ratio"], audited["protocol"]["steps"])
    expected = {"version": VERSION, "sampling_mode": MODE, "steps": b["steps"], "epochs": None,
                "steps_per_epoch": None, "samples_seen": b["samples"], "formation_samples_seen": b["formation_samples"],
                "operation_samples_seen": b["operation_samples"], "formation_ratio": row["formation_ratio"],
                "form_frac_mean": row["formation_ratio"], "n_pairs": audited["pools"]["total"],
                "selection": "val_cascade_acc"}
    if any(done.get(k) != v for k, v in expected.items()):
        raise ValueError("incomplete updates or incorrect formation/operation exposure")
    best = done.get("best_step", 0)
    if (not isinstance(best, int) or best <= 0 or best > b["steps"]
            or (best != b["steps"] and best % TRAIN_DEFAULTS["val_every"] != 0)
            or not isinstance(done.get("best_val"), dict)
            or done.get("ckpt_sha1") != common.sha1_file(Path(out) / "ckpt.pt")):
        raise ValueError("invalid selected validation checkpoint or checkpoint identity")
    validate_checkpoint(out, training_args(row["run"], row["formation_ratio"], audited), done)
    return done


def validate_checkpoint(out, args, done):
    import torch
    try:
        ckpt = torch.load(Path(out) / "ckpt.pt", map_location="cpu", weights_only=True)
        model = import_module(f"{PACKAGE}.model").build_model(
            {k: args[k] for k in ("dim", "depth", "heads")})
        if (ckpt.get("args") != args or ckpt.get("cfg") != model.cfg
                or ckpt.get("step") != done["best_step"] or ckpt.get("selection") != "val_cascade_acc"):
            raise ValueError("checkpoint internal identity differs from this ratio's selected model")
        ema = ckpt["ema"]
        if not all(torch.is_tensor(v) and torch.isfinite(v).all() for v in ema.values()):
            raise ValueError("checkpoint EMA contains invalid or nonfinite weights")
        model.load_state_dict(ema, strict=True)
        score = done["best_val"].get("cascade_acc", float("nan"))
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("selected validation score is invalid")
    except Exception as exc:
        raise ValueError(f"cannot validate the selected supervision checkpoint: {exc}") from exc


def validate_metrics(samples, bench):
    import_module("preresearch.04_ood_generalization.exp.load").validate_samples(samples, bench)
    n = len(bench["split_ids"])
    for key in ("change_f1", "any_violation"):
        values = samples[key]
        if values.shape != (n, 2) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError(f"invalid dose metric {key}")
    for i in range(n):
        initial = (bench["ctx_fg"][i], bench["ctx_bg"][i])
        for k in range(2):
            score = metrics.score_sample((samples["pred_fg"][i, k], samples["pred_bg"][i, k]),
                (bench["fut_fg"][i, k, -1], bench["fut_bg"][i, k, -1]), initial, bench["A"][i, k])
            for key in ("change_f1", "any_violation"):
                if not np.isclose(score[key], samples[key][i, k], atol=1e-10, rtol=0):
                    raise ValueError(f"cached {key} differs from predicted states")


def evaluation_identity(out, done):
    return {"version": VERSION, "checkpoint_sha1": done["ckpt_sha1"],
            "hashes": {n: common.sha1_file(Path(out) / n) for n in ("eval.json", "eval_samples.npz")}}


def validate_eval(out, audited, done):
    out = Path(out)
    ev = common.load_json(out / "eval.json")
    bench_path = resolve_file(audited["inputs"]["bench"])
    if (ev.get("bench") != common.rel(bench_path) or ev.get("n_samples") != 512
            or ev.get("bench_sha1") != audited["hashes"]["bench"][bench_path.name]
            or ev.get("bench_json_sha1") != audited["hashes"]["bench"][bench_path.with_suffix(".json").name]
            or ev.get("checkpoint_sha1") != done["ckpt_sha1"]):
        raise ValueError("evaluation differs from the frozen bench or selected checkpoint")
    marker = out / "eval_complete.json"
    if marker.exists() and common.load_json(marker) != evaluation_identity(out, done):
        raise ValueError("completed evaluation identity changed; refusing overwrite")
    bench = common.load_bench(bench_path)
    with np.load(out / "eval_samples.npz", allow_pickle=False) as z:
        samples = {k: z[k] for k in ("cascade_f1", "change_f1", "cell_acc_mask", "rank_acc", "any_violation",
                                    "split_ids", "rank_valid", "n_cascade_true", "pred_fg", "pred_bg")}
    if len(samples["split_ids"]) != 512:
        raise ValueError("incomplete evaluation samples")
    validate_metrics(samples, bench)
    return ev


def record_eval(out, audited, done):
    validate_eval(out, audited, done)
    path = Path(out) / "eval_complete.json"
    if not path.exists():
        write_json(path, evaluation_identity(out, done))
