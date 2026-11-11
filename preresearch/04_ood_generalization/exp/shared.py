"""Experiment 04 imports existing interfaces without changing training or physics."""
from importlib import import_module
from pathlib import Path
import json
import os
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
MAIN_PACKAGE = "preresearch.03_main.exp"
PILOT_PACKAGE = "preresearch.01_pilot_validity.exp"
main_shared = import_module(f"{MAIN_PACKAGE}.shared")
common = main_shared.common
metrics = main_shared.metrics
CONDITIONS = main_shared.CONDITIONS
METRICS = ("cascade_f1", "cell_acc_mask", "rank_acc")
CONTRASTS = (("process", "state"), ("process", "partial"), ("process", "shuffle"))
EXP_NAME = "04_ood_generalization"
VERSION = 1


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
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
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def write_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def resolve_file(value):
    value = str(value)
    if value.startswith("<results>/"):
        return common.RESULTS_ROOT / value[len("<results>/"):]
    return common.resolve(value)


def source_path(value):
    direct = resolve_file(value)
    path = direct if (direct / "run.json").is_file() else common.RESULTS_ROOT / "03_main" / str(value)
    path = path.resolve()
    if not (path / "run.json").is_file() or common.load_json(path / "run.json").get("exp") != "03_main":
        raise ValueError("--source-run must identify an existing completed experiment 03 run")
    return path


def output_path(run):
    if not run or Path(run).name != run or run in (".", ".."):
        raise ValueError("--run must be a run name, not a path")
    return (common.RESULTS_ROOT / EXP_NAME / run).resolve()


def forbid_source_output(out, source, audited):
    out = Path(out).resolve()
    protected = [source, *[common.resolve(p) for p in audited["inputs"].values()]]
    for path in protected:
        path = Path(path).resolve()
        if out == path or path in out.parents or out in path.parents:
            raise ValueError("experiment 04 output must be separate from source models and frozen data")


def bind_source(out, manifest):
    path = Path(out) / "source.json"
    if path.exists():
        if common.load_json(path) != clean(manifest):
            raise ValueError("source files or settings changed; use a new experiment 04 run")
    else:
        write_json(path, manifest)


def start_run(label=""):
    name = os.environ.get("FW_RUN")
    if name:
        output_path(name)
        return common.start_run(EXP_NAME, label, name).name
    base = "_".join(p for p in (common.now_ts(), common.slug(label)) if p)
    name, suffix = base, 1
    while output_path(name).exists():
        suffix += 1
        name = f"{base}-{suffix}"
    return common.start_run(EXP_NAME, label, name).name
