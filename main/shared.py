"""Shared paths, atomic records and immutable identities for the multi-task main study."""
from contextlib import contextmanager
from fractions import Fraction
import fcntl
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
RESULTS_ROOT = PROJECT / "experiments/main"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
PILOT = "preresearch.01_pilot_validity.exp"


def validate_environment():
    for variable, expected in (("FW_PROJECT", PROJECT), ("FW_GAME_CODE", PROJECT / "game_code")):
        if variable in os.environ and Path(os.environ[variable]).expanduser().resolve() != expected.resolve():
            raise ValueError(f"{variable} must point to this repository for the frozen main experiment")


validate_environment()
common = import_module(f"{PILOT}.common")
model_api = import_module(f"{PILOT}.model")
train_api = import_module(f"{PILOT}.train")
metrics_api = import_module(f"{PILOT}.metrics")
D = import_module("datasets")
core = import_module("core")
CONDITIONS = ("state", "process", "partial", "shuffle")
METRICS = ("change_f1", "cell_acc_mask")
H, W, HORIZON, VERSION = 28, 56, 32, 1
DEFAULT_SOURCE = PROJECT / "experiments/preresearch/03_main/20261111-044057_four_conditions"
TRAIN_DEFAULTS = {"dim": 384, "depth": 8, "heads": 6, "batch": 256, "lr": .0003,
                  "lr_min": .00003, "warmup": 300, "wd": .05, "ema": .999,
                  "change_weight": 8, "val_every": 500, "log_every": 50}


def integer(value, label, minimum=1):
    try:
        n = int(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if isinstance(value, bool) or str(n) != str(value) or n < minimum:
        raise ValueError(f"invalid {label}")
    return n


def ratio(value):
    try:
        r = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("formation ratio must be finite in [0,1]") from exc
    if not 0 <= r <= 1:
        raise ValueError("formation ratio must be in [0,1]")
    return r


def quota(value, seen):
    r = ratio(value)
    return seen * r.numerator // r.denominator


def name(value):
    if not value or Path(value).name != value or value in (".", ".."):
        raise ValueError("name must be a simple directory name")
    return value


def safe_path(path):
    p = Path(path).expanduser().absolute()
    if p.resolve() != p:
        raise ValueError("output paths must not pass through symbolic links")
    if p.exists():
        for directory, dirs, files in os.walk(p, followlinks=False):
            if any((Path(directory) / n).is_symlink() for n in (*dirs, *files)):
                raise ValueError("output tree contains symbolic links")
    return p


def disjoint(out, protected):
    out = safe_path(out)
    for value in protected:
        p = Path(value).resolve()
        if out == p or out in p.parents or p in out.parents:
            raise ValueError("output overlaps a protected input or source directory")
    return out


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(p)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def files_identity(root, excludes=()):
    root = Path(root)
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.name not in excludes and not p.name.endswith(".lock")}


def code_identity():
    validate_environment()
    if Path(common.PROJECT).resolve() != PROJECT or Path(common.GAME_CODE).resolve() != PROJECT / "game_code":
        raise ValueError("cached common module uses a different project or game source")
    modules = {"core": "game_code/core.py", "datasets": "game_code/datasets.py", "datasets_v3": "game_code/datasets_v3.py"}
    modules.update({f"{PILOT}.{n}": f"preresearch/01_pilot_validity/exp/{n}.py"
                    for n in ("model", "train", "common", "metrics", "rollout")})
    modules.update({"preresearch.03_main.exp.audit": "preresearch/03_main/exp/audit.py",
                    "preresearch.03_main.exp.shared": "preresearch/03_main/exp/shared.py",
                    "preresearch.04_ood_generalization.exp.statistics": "preresearch/04_ood_generalization/exp/statistics.py"})
    external = []
    for key, relative in modules.items():
        module = import_module(key)
        expected = (PROJECT / relative).resolve()
        if Path(module.__file__).resolve() != expected:
            raise ValueError(f"actual module source differs from this repository: {key}")
        external.append(expected)
    own = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in ROOT.rglob("*.py")
           if not any(x in p.relative_to(ROOT).parts for x in ("datasets", "runs", "tests"))}
    return {"main": own, "dependencies": {p.relative_to(PROJECT).as_posix(): file_hash(p) for p in external}}


@contextmanager
def lock(path):
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"another process holds {path.name}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def frozen_write(path, record):
    path = Path(path)
    if path.exists():
        if load_json(path) != record:
            raise ValueError(f"immutable record changed: {path.name}; use a new run")
    else:
        write_json(path, record)


def pack(frames):
    return {"fg": np.stack([f[0] for f in frames]).astype(np.uint8),
            "bg": np.stack([f[1] for f in frames]).astype(np.uint8),
            "act": np.stack([common.encode_actions(f[2]) for f in frames]).astype(np.uint8)}


def score(pred, target, initial):
    return {k: float(v) for k, v in D.score_outcome(pred[0], pred[1], target[0], target[1], initial[0], initial[1]).items() if k in METRICS}
