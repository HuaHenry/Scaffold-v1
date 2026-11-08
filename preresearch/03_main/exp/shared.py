"""Independent main-experiment protocol; reuse only the unchanged pilot training interfaces."""
from importlib import import_module
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
PACKAGE = "preresearch.01_pilot_validity.exp"
common = import_module(f"{PACKAGE}.common")
metrics = import_module(f"{PACKAGE}.metrics")
EXP_NAME = "03_main"
CONDITIONS = ("state", "process", "partial", "shuffle")
EPISODES = 81920
EPOCHS = 5
BATCH = 256
DEFAULT_DATA = common.GAME_CODE / "output/data_v3_simple_train_hard_eval_81920"
METRICS = ("cascade_f1", "change_f1", "structure_iou", "cell_acc_mask", "rank_acc",
           "any_violation", "cascade_f1__s1", "cascade_f1__s4")


def budget(condition, episodes=EPISODES):
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    mixed = condition in ("process", "shuffle")
    pairs = (59 if mixed else 56) * episodes
    steps_per_epoch = math.ceil(pairs / BATCH)
    return {"episodes": episodes, "frames": episodes * 64, "pairs": pairs,
            "epochs": EPOCHS, "samples": pairs * EPOCHS, "steps_per_epoch": steps_per_epoch,
            "steps": EPOCHS * steps_per_epoch, "formation_fraction": 31 / 59 if mixed else 0.0}


def data_hashes(directory):
    directory = Path(directory)
    names = ["train_summary.json", "train_scenes.json", "train_finals.npz"]
    names += [f"train_{c}{suffix}" for c in CONDITIONS for suffix in (".npz", "_actions.json")]
    return {name: common.sha1_file(directory / name) for name in names}


def bench_hashes(bench):
    bench = Path(bench)
    return {p.name: common.sha1_file(p) for p in (bench, bench.with_suffix(".json"))}


def require_data(data):
    data = Path(data)
    if not (data / "curve_source.json").is_file():
        raise FileNotFoundError(f"main 03 reuses existing data only; enlarged dataset is not complete: {data}")
    return common.load_json(data / "curve_source.json")


def check_separation(train, val, bench):
    if train & val or train & bench or val & bench:
        raise ValueError("training, validation and benchmark scenes must be pairwise disjoint")


def validate_completed(out, condition, audited):
    out = Path(out)
    cfg = common.load_json(out / "config.json")
    args = cfg["args"]
    done = common.load_json(out / "done.json")
    b = budget(condition)
    expected = {"condition": condition, "episodes": EPISODES, "epochs": EPOCHS,
                "steps": b["steps"], "batch": BATCH, "protocol": "video", "lr": 3e-4,
                "change_weight": 8, "depth": 8, "unroll": 1, "seed": 0,
                "data": audited["inputs"]["data"], "val": audited["inputs"]["val"],
                "val_every": 500, "select_best_val": True, "balanced_phases": False,
                "log_batch_composition": True}
    if any(args.get(k) != v for k, v in expected.items()):
        raise ValueError(f"{condition}: completed training differs from the frozen main protocol")
    if (done.get("steps") != b["steps"] or done.get("epochs") != EPOCHS
            or done.get("n_pairs") != b["pairs"] or done.get("samples_seen") != b["samples"]
            or done.get("steps_per_epoch") != b["steps_per_epoch"]
            or done.get("selection") != "val_cascade_acc"
            or done.get("best_step") not in [k*b["steps_per_epoch"] for k in range(1, EPOCHS+1)]
            or abs(done.get("form_frac_mean", -1) - b["formation_fraction"]) > 1e-6):
        raise ValueError(f"{condition}: incomplete full-epoch coverage or incorrect checkpoint selection")
    expected_data = {k: audited["hashes"]["data"][k] for k in
                     (f"train_{condition}.npz", f"train_{condition}_actions.json", "train_summary.json")}
    expected_val = {k: audited["hashes"]["val"][k] for k in
                    ("train_state.npz", "train_state_actions.json", "train_summary.json")}
    if (cfg.get("data_sha1") != expected_data or cfg.get("val_sha1") != expected_val
            or done.get("ckpt_sha1") != common.sha1_file(out / "ckpt.pt")):
        raise ValueError(f"{condition}: checkpoint or training/validation identity changed")
    return done


def validate_evaluation(out, audited, done):
    out = Path(out)
    ev = common.load_json(out / "eval.json")
    bench = common.resolve(audited["inputs"]["bench"])
    if (ev.get("bench") != common.rel(bench) or ev.get("n_samples") != 512
            or ev.get("bench_sha1") != audited["hashes"]["bench"][bench.name]
            or ev.get("bench_json_sha1") != audited["hashes"]["bench"][bench.with_suffix(".json").name]
            or ev.get("checkpoint_sha1") != done["ckpt_sha1"]):
        raise ValueError("evaluation differs from its frozen benchmark or selected checkpoint")
    return ev
