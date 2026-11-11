"""Formation-dose protocol and source identity checks, independent of older experiments."""
from contextlib import contextmanager
import fcntl
from importlib import import_module
import math
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
main_shared = import_module(f"{MAIN_PACKAGE}.shared")
helpers = import_module("preresearch.04_ood_generalization.exp.shared")
write_json, write_npz, resolve_file = helpers.write_json, helpers.write_npz, helpers.resolve_file
EXP_NAME = "05_formation_ratio"
VERSION = 1
LEVELS = (0, 8, 16, 32, 48)
EPISODES, EPOCHS, BATCH = 81920, 5, 256
DEFAULT_SOURCE = "20261111-044057_four_conditions"
TRAIN_DEFAULTS = {"dim": 384, "heads": 6, "wd": 0.05, "ema": 0.999,
                  "warmup": 300, "lr_min": 3e-5}


def budget(F, episodes=EPISODES):
    if F not in LEVELS or episodes <= 0:
        raise ValueError("formation frames must be 0/8/16/32/48 and episodes positive")
    clips = (64 - F) // 8
    form = max(0, F - 1)
    pairs = form + clips * 7
    steps_per_epoch = math.ceil(episodes * pairs / BATCH)
    return {"F": F, "formation_frames_fraction": F / 64, "operation_clips": clips,
            "episodes": episodes, "frames": episodes * 64, "pairs_per_episode": pairs,
            "pairs": episodes * pairs, "formation_pairs": form * episodes,
            "formation_fraction": form / pairs, "epochs": EPOCHS,
            "steps_per_epoch": steps_per_epoch, "steps": steps_per_epoch * EPOCHS,
            "samples": episodes * pairs * EPOCHS}


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


def validate_levels(levels):
    levels = tuple(levels)
    if (levels != tuple(sorted(set(levels))) or not {0, 32}.issubset(levels)
            or not set(levels).issubset(LEVELS) or len(levels) < 3):
        raise ValueError("levels must be sorted distinct doses, including F=0/32 and at least one new dose")
    return levels


def new_levels(manifest):
    return tuple(F for F in validate_levels(manifest["protocol"]["levels"]) if F not in (0, 32))


def validate_source(value, levels=LEVELS):
    levels = validate_levels(levels)
    source = helpers.source_path(value)
    print(f"[05] validating completed source: {common.rel(source)}", flush=True)
    _, _, models, audited = import_module(f"{MAIN_PACKAGE}.analysis").load_results(source)
    inputs = {k: resolve_file(v) for k, v in audited["inputs"].items()}
    endpoints = {"0": models["state"], "32": models["process"]}
    endpoint_hashes = {}
    for F, path in endpoints.items():
        out = resolve_file(path)
        args = common.load_json(out / "config.json")["args"]
        if any(args.get(k) != value for k, value in TRAIN_DEFAULTS.items()):
            raise ValueError("source model hyperparameters differ from the matched dose protocol")
        endpoint_hashes[F] = {n: common.sha1_file(out / n) for n in
                              ("config.json", "done.json", "ckpt.pt", "eval.json", "eval_samples.npz")}
    manifest = {"version": VERSION, "source_run": common.rel(source), "source_audit_sha1": common.sha1_file(source / "audit.json"),
                "inputs": audited["inputs"], "hashes": audited["hashes"],
                "endpoints": endpoints, "endpoint_hashes": endpoint_hashes,
                "protocol": {"episodes": EPISODES, "epochs": EPOCHS, "batch": BATCH,
                             "training_seeds": [0], "levels": list(levels), "sampling": "full_epochs",
                             "full_five_dose_design": levels == LEVELS, "train_defaults": TRAIN_DEFAULTS,
                             "formation_mode": "compress", "settle_frames": 4}}
    return {"source": source, "audited": audited, "inputs": inputs, "manifest": manifest}


def bind_run(name, bundle):
    out = run_path(name)
    helpers.forbid_source_output(out, bundle["source"], bundle["audited"])
    path = out / "source.json"
    if path.exists() and common.load_json(path) != bundle["manifest"]:
        raise ValueError("source identity or protocol changed; use a new experiment 05 run")
    common.start_run(EXP_NAME, name=name)
    if not path.exists():
        write_json(path, bundle["manifest"])
    return out


def verify_manifest(name):
    out = run_path(name)
    recorded = common.load_json(out / "source.json")
    bundle = validate_source(recorded["source_run"], recorded["protocol"]["levels"])
    if recorded != bundle["manifest"]:
        raise ValueError("frozen source or endpoint models changed")
    return bundle


def variant_hashes(path):
    return {name: common.sha1_file(Path(path) / name) for name in
            ("train_process.npz", "train_process_actions.json", "train_summary.json")}


def verify_variant(path, source, F):
    path = Path(path)
    marker = common.load_json(path / "complete.json")
    if (F not in (8, 16, 48) or marker.get("version") != VERSION
            or marker.get("source") != source or marker.get("F") != F
            or marker.get("episodes") != EPISODES or marker.get("mode") != "compress"
            or marker.get("budget") != budget(F)
            or marker.get("hashes") != variant_hashes(path)):
        raise ValueError(f"F={F} data identity changed; refusing regeneration or training")
    return marker


def validate_config(out, row, audited):
    cfg = common.load_json(Path(out) / "config.json")
    b = budget(row["F"])
    expected = {**TRAIN_DEFAULTS, "condition": "process", "episodes": EPISODES, "epochs": EPOCHS, "steps": b["steps"],
                "batch": BATCH, "protocol": "video", "seed": 0, "lr": 3e-4, "depth": 8,
                "change_weight": 8, "unroll": 1, "balanced_phases": False,
                "select_best_val": True, "val_every": 500, "log_batch_composition": True,
                "data": row["data"], "val": audited["inputs"]["val"]}
    if any(cfg["args"].get(k) != v for k, v in expected.items()):
        raise ValueError("formation-ratio training has different settings")
    if (cfg.get("data_sha1") != audited["variants"][str(row["F"])]["hashes"]
            or cfg.get("val_sha1") != {k: audited["hashes"]["val"][k] for k in
                                       ("train_state.npz", "train_state_actions.json", "train_summary.json")}):
        raise ValueError("training data or validation identity changed")
    return cfg


def validate_done(out, row, audited):
    validate_config(out, row, audited)
    done = common.load_json(Path(out) / "done.json")
    b = budget(row["F"])
    if (any(done.get(k) != b[k] for k in ("steps", "epochs", "steps_per_epoch"))
            or done.get("samples_seen") != b["samples"] or done.get("n_pairs") != b["pairs"]
            or done.get("selection") != "val_cascade_acc"
            or done.get("best_step") not in [k * b["steps_per_epoch"] for k in range(1, EPOCHS + 1)]
            or not np.isclose(done.get("form_frac_mean", -1), b["formation_fraction"], atol=1e-6, rtol=0)):
        raise ValueError("incomplete epoch coverage or incorrect formation dose")
    if done.get("ckpt_sha1") != common.sha1_file(Path(out) / "ckpt.pt"):
        raise ValueError("checkpoint identity changed")
    return done


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
            predicted = (samples["pred_fg"][i, k], samples["pred_bg"][i, k])
            target = (bench["fut_fg"][i, k, -1], bench["fut_bg"][i, k, -1])
            score = metrics.score_sample(predicted, target, initial, bench["A"][i, k])
            for key in ("change_f1", "any_violation"):
                if not np.isclose(score[key], samples[key][i, k], atol=1e-10, rtol=0):
                    raise ValueError(f"cached {key} differs from predicted states")


def evaluation_identity(out, done):
    return {"version": VERSION, "checkpoint_sha1": done["ckpt_sha1"],
            "hashes": {name: common.sha1_file(Path(out) / name) for name in ("eval.json", "eval_samples.npz")}}


def validate_eval(out, audited, done):
    out = Path(out)
    result = main_shared.validate_evaluation(out, audited, done)
    marker = out / "eval_complete.json"
    if marker.exists() and common.load_json(marker) != evaluation_identity(out, done):
        raise ValueError("completed evaluation identity changed; refusing overwrite")
    bench = common.load_bench(resolve_file(audited["inputs"]["bench"]))
    with np.load(out / "eval_samples.npz", allow_pickle=False) as z:
        samples = {k: z[k] for k in ("cascade_f1", "change_f1", "cell_acc_mask", "rank_acc", "any_violation",
                                    "split_ids", "rank_valid", "n_cascade_true", "pred_fg", "pred_bg")}
    if len(samples["split_ids"]) != 512:
        raise ValueError("incomplete evaluation samples")
    validate_metrics(samples, bench)
    return result


def record_eval(out, audited, done):
    validate_eval(out, audited, done)
    marker = Path(out) / "eval_complete.json"
    if not marker.exists():
        write_json(marker, evaluation_identity(out, done))
