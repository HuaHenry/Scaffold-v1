"""Frozen interfaces, atomic outputs, and experiment 09 protocol constants."""
from importlib import import_module
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
PILOT = "preresearch.01_pilot_validity.exp"
MAIN = "preresearch.03_main.exp"
common = import_module(f"{PILOT}.common")
metrics = import_module(f"{PILOT}.metrics")
main = import_module(f"{MAIN}.shared")
CONDITIONS = main.CONDITIONS
EXP_NAME = "09_physics_rollout"
VERSION = 1
HORIZON = 32
DRIFT_HORIZON = 128
FORMATION_HORIZON = 31
VIOLATIONS = ("floating_struct", "struct_on_loose", "floating_leaf", "floating_sand", "floating_back")
METRICS = (*VIOLATIONS, "violations", "any_violation", "mass_error", "cell_acc_all",
           "cell_acc_mask", "cell_error", "cascade_f1", "change_f1", "structure_iou",
           "exact", "changed_fraction", "state_changed", "ever_changed", "ever_violation")
CONTRASTS = (("process", "state"), ("process", "partial"), ("process", "shuffle"))


def resolve(value):
    value = str(value)
    if value.startswith("<results>/"):
        return (common.RESULTS_ROOT / value[len("<results>/"):]).resolve()
    return common.resolve(value).resolve()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def write_npz(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def read_npz(path):
    with np.load(path, allow_pickle=False) as stream:
        return {key: stream[key].copy() for key in stream.files}


def output_path(name):
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("--run must be a new run name, not a path")
    return (common.RESULTS_ROOT / EXP_NAME / name).resolve()


def protect_output(out, paths):
    out = Path(out).resolve()
    for path in paths:
        path = Path(path).resolve()
        if out == path or path in out.parents or out in path.parents:
            raise ValueError(f"Output overlaps frozen source: {path}")
