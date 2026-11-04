"""Shared helpers: paths, data loading, protocol slicing, action encoding, seeds (plan 00 sections 2-5)."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]                       # this experiment's code folder (…/preresearch/01_pilot_validity)


def _find_project() -> Path:
    """Project root = FW_PROJECT, else the nearest parent of this folder that contains `game_code/`.
    Nothing below hard-codes an absolute path, so the whole tree can be moved or copied to another machine."""
    if os.environ.get("FW_PROJECT"):
        return Path(os.environ["FW_PROJECT"]).expanduser().resolve()
    for p in ROOT.parents:
        if (p / "game_code").is_dir():
            return p
    raise SystemExit("cannot locate the project root (a folder containing game_code/); set FW_PROJECT=/path/to/project")


PROJECT = _find_project()
GAME_CODE = Path(os.environ.get("FW_GAME_CODE", PROJECT / "game_code")).expanduser().resolve()
if str(GAME_CODE) not in sys.path:
    sys.path.insert(0, str(GAME_CODE))

DEFAULT_DATA = Path(os.environ.get("FW_DATA", GAME_CODE / "output" / "data_v3")).expanduser().resolve()
DEFAULT_BENCH = DEFAULT_DATA / "bench.npz"
DEFAULT_VAL = Path(os.environ.get("FW_VAL", DEFAULT_DATA.parent / "data_val_v3")).expanduser().resolve()
FROZEN = ROOT / "exp" / "configs" / "frozen.json"

# All pre-experiment RESULTS live under RESULTS_ROOT/<experiment folder>/<run>/<category>/<item>/ (plan 00 s.11.1).
RESULTS_ROOT = Path(os.environ.get("FW_RESULTS", PROJECT / "experiments" / "preresearch")).expanduser().resolve()
EXP_NAME = "01_pilot_validity"          # this experiment's folder under RESULTS_ROOT


def rel(p) -> str:
    """Portable form of a path for anything written to disk: relative to the project root (or the results root),
    never absolute, whenever the path lives inside the project."""
    p = Path(p)
    if not p.is_absolute():
        return p.as_posix()
    for base in (RESULTS_ROOT, PROJECT):
        try:
            r = p.resolve().relative_to(base).as_posix()
            return r if base == PROJECT else f"<results>/{r}"
        except ValueError:
            continue
    return p.as_posix()


H, W = 28, 56
N_MAT = 21
N_ACT = 127                                                       # 1 + 3 types * 21 materials * 2 layers
CONDITIONS = ("state", "process", "partial", "shuffle")
SPLITS = ("id", "combo_ood", "scale_ood", "multi_ood")
TYPE_IDX = {"place": 0, "remove": 1, "interact": 2}
HORIZON = 32


def resolve(p) -> Path:
    """Input paths: absolute as is; relative ones are tried against this code folder, then the project root
    (never the shell cwd)."""
    p = Path(p)
    if p.is_absolute():
        return p
    return ROOT / p if (ROOT / p).exists() or not (PROJECT / p).exists() else PROJECT / p


def resolve_out(p) -> Path:
    """Result paths: absolute as is; relative ones are relative to RESULTS_ROOT (not to the shell cwd)."""
    p = Path(p)
    return p if p.is_absolute() else RESULTS_ROOT / p


# --------------------------------------------------------------------------- result folders (plan 00 s.11)
def now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def slug(x) -> str:
    return re.sub(r"[^A-Za-z0-9.+=_-]+", "-", str(x)).strip("-")


#   RESULTS_ROOT/<exp>/<run>/<category>/<item>/        one RUN = one big folder holding everything of one execution
#     run.json  jobs/  precheck/  tune/  train/  baseline/  analysis/
# A new execution (run_all.sh) = a new <run> folder (`<YYYYMMDD-HHMMSS>[_label]`); old runs are never touched.
_RUN_NOTE = {"shown": False}


def category(kind: str, rid: str = "") -> str:
    """Sub-folder of a run that holds this kind of result."""
    rid = str(rid)
    if kind == "freeze" or (kind == "train" and rid.startswith("tune_")):
        return "tune"
    if kind == "train" and rid.startswith("compose_"):
        return "compose"
    return kind


def item_name(kind: str, rid: str, settings: str = "") -> str:
    """`<id>[_<settings>]` - the run folder already carries the timestamp and the category folder the kind."""
    return "_".join(p for p in (slug(rid), slug(settings)) if p)


def list_runs(exp: str = EXP_NAME) -> list[Path]:
    """Run folders of an experiment, oldest first (names start with the timestamp)."""
    base = RESULTS_ROOT / exp
    return sorted(d for d in base.iterdir() if (d / "run.json").is_file()) if base.is_dir() else []


def start_run(exp: str = EXP_NAME, label: str = "", name: str | None = None) -> Path:
    """Create the big folder of a new run (or, with an explicit `name`, make sure it exists)."""
    name = name or "_".join(p for p in (now_ts(), slug(label)) if p)
    d = RESULTS_ROOT / exp / name
    if not (d / "run.json").exists():
        d.mkdir(parents=True, exist_ok=True)
        dump_json({"name": name, "exp": exp, "label": label, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "git": git_rev()}, d / "run.json")
    return d


def current_run(exp: str = EXP_NAME, create: bool = True) -> str | None:
    """Name of the run that new results go to: `FW_RUN` if set (created when missing), else the newest existing run
    of this experiment, else (only with create=True) a brand-new run. run_all.sh always starts a new run."""
    if os.environ.get("FW_RUN"):
        name = os.environ["FW_RUN"]
        if create:
            start_run(exp, name=name)
        return name
    runs = list_runs(exp)
    if runs:
        if not _RUN_NOTE["shown"]:
            _RUN_NOTE["shown"] = True
            print(f"[run] FW_RUN not set - using the newest run {rel(runs[-1])} "
                  f"(start a fresh one with: export FW_RUN=$(python -m exp.new_run))", file=sys.stderr, flush=True)
        return runs[-1].name
    return start_run(exp).name if create else None


def run_dir(exp: str = EXP_NAME, create: bool = True) -> Path | None:
    name = current_run(exp, create)
    return RESULTS_ROOT / exp / name if name else None


def new_result_dir(exp: str, kind: str, rid: str, settings: str = "", create: bool = True,
                   run: str | None = None) -> Path:
    base = RESULTS_ROOT / exp / (run or current_run(exp)) / category(kind, rid) / item_name(kind, rid, settings)
    d, k = base, 1
    while d.exists():
        k += 1
        d = base.with_name(f"{base.name}-{k}")
    if create:
        d.mkdir(parents=True)
    return d


def write_meta(d: Path, kind: str, rid: str, settings: dict, **extra) -> dict:
    """meta.json = the registry entry of a result folder (logical id, kind, settings). Lookups use it, not the name."""
    d = Path(d)
    parts = ()
    for cand in (d, d.resolve()):
        try:
            parts = cand.relative_to(RESULTS_ROOT).parts
            break
        except ValueError:
            continue
    exp, run = (parts[0], parts[1]) if len(parts) == 4 else (parts[0] if parts else d.parent.name, None)
    meta = {"id": rid, "kind": kind, "exp": exp, "run": run, "name": d.name,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "settings": settings, **extra}
    dump_json(meta, d / "meta.json")
    return meta


def iter_results(exp: str | None = None, run: str | None = None):
    """Yield (dir, meta) for registered result folders, oldest first. Layout: `<exp>/<run>/<category>/<item>`;
    folders of the older flat layout (`<exp>/<item>`) are still found (run = None)."""
    if not RESULTS_ROOT.exists():
        return
    exps = [RESULTS_ROOT / exp] if exp else sorted(d for d in RESULTS_ROOT.iterdir() if d.is_dir())
    found = []

    def add(d):
        if (d / "meta.json").is_file():
            found.append((d, json.loads((d / "meta.json").read_text(encoding="utf-8"))))

    for e in exps:
        if not e.is_dir():
            continue
        for r in e.iterdir():
            if not r.is_dir():
                continue
            if (r / "run.json").is_file():
                if run and r.name != run:
                    continue
                for c in r.iterdir():
                    if c.is_dir():
                        for d in c.iterdir():
                            if d.is_dir():
                                add(d)
            elif not run:
                add(r)
    yield from sorted(found, key=lambda t: (t[1].get("run") or "", t[1].get("created", ""), t[0].name))


def find_results(rid: str, kind: str | None = None, exp: str | None = None, require: str | None = None,
                 match: dict | None = None, run: str | None = None) -> list[Path]:
    """Result folders with this logical id (and kind / settings), keeping only those that contain `require`.
    `run` restricts the search to one run folder (default: all runs)."""
    out = []
    for d, m in iter_results(exp, run):
        if m.get("id") != rid or (kind and m.get("kind") != kind):
            continue
        if require and not (d / require).exists():
            continue
        if match and any(m.get("settings", {}).get(k) != v for k, v in match.items()):
            continue
        out.append(d)
    return out


def find_latest(rid: str, kind: str | None = None, exp: str | None = None, require: str | None = None,
                match: dict | None = None, run: str | None = None) -> Path | None:
    r = find_results(rid, kind, exp, require, match, run)
    return r[-1] if r else None


def resolve_run(arg, kind: str = "train", require: str | None = "ckpt.pt", steps: int | None = None) -> Path:
    """A run given as a folder path (absolute, relative to RESULTS_ROOT or to the cwd) or as its logical id
    (e.g. `state_s0` -> the newest finished result with that id, trained for `steps` steps if given; the current
    run folder is searched first, then all runs)."""
    p = Path(arg)
    for cand in ([p] if p.is_absolute() else [RESULTS_ROOT / p, Path.cwd() / p]):
        if cand.is_dir():
            return cand
    match = {"steps": steps} if steps else None
    cur = current_run(EXP_NAME, create=False)
    d = (find_latest(str(arg), kind, EXP_NAME, require, match, cur) if cur else None) \
        or find_latest(str(arg), kind, None, require, match)
    if d is None:
        raise SystemExit(f"no {kind} result found for {arg!r} (steps={steps}) under {rel(RESULTS_ROOT)}")
    print(f"[run] {arg} -> {rel(d)}", flush=True)
    return d


def bench_label(path) -> str:
    p = Path(path)
    return p.parent.name if p.name == "bench.npz" else p.stem


def frozen_defaults() -> dict:
    d = {"lr": 3e-4, "change_weight": 8.0, "depth": 8, "unroll": 1}
    if FROZEN.exists():
        d.update({k: v for k, v in json.loads(FROZEN.read_text()).items() if k in d})
    return d


def train_settings(episodes: int, protocol: str, steps: int, batch: int, lr: float, change_weight: float,
                   depth: int, unroll: int = 1) -> str:
    """Setting part of a training folder name; make_jobs and train.py must produce the same string."""
    s = f"ep{episodes}_{protocol}_steps{steps}_bs{batch}_lr{lr:g}_cw{change_weight:g}_d{depth}"
    return s + (f"_unr{unroll}" if unroll > 1 else "")


# --------------------------------------------------------------------------- seeds / provenance
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return "unknown"


def load_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_jsonable), encoding="utf-8")


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return rel(o)
    raise TypeError(type(o))


# --------------------------------------------------------------------------- action encoding (00 section 4)
def encode_actions(actions, h: int = H, w: int = W) -> np.ndarray:
    """list[dict] -> int64 [h,w]. remove / interact never carry a material; noop is ignored."""
    m = np.zeros((h, w), dtype=np.int64)
    for a in actions or ():
        t = TYPE_IDX.get(a["type"])
        if t is None:
            continue
        x, y = int(a["x"]), int(a["y"])
        if not (0 <= x < w and 0 <= y < h):
            continue
        mat = int(a.get("material", 0)) if a["type"] == "place" else 0
        layer = 1 if a.get("layer", "front") == "back" else 0
        m[y, x] = 1 + ((t * N_MAT + mat) * 2 + layer)
    return m


def iv_by_t(iv: dict) -> dict:
    """Compound intervention {kind, steps:[{t, actions}]} -> {t: [actions]} (actions of step t precede tick t)."""
    out: dict = {}
    for st in iv["steps"]:
        out.setdefault(int(st["t"]), []).extend(st["actions"])
    return out


def iv_action_maps(iv: dict, length: int = 1, h: int = H, w: int = W) -> np.ndarray:
    by_t = iv_by_t(iv)
    length = max(length, max(by_t) + 1 if by_t else 1)
    return np.stack([encode_actions(by_t.get(t, []), h, w) for t in range(length)])


# --------------------------------------------------------------------------- training data
def load_summary(data_dir) -> dict:
    return load_json(Path(data_dir) / "train_summary.json")


def load_condition(data_dir, condition: str, episodes: int | None = None, with_actions: bool = True) -> dict:
    """fg,bg [E,T,H,W] uint8; cont,phase [E,T]; actions[e][t] -> list[dict]. `episodes` keeps the first N (prefix)."""
    d = np.load(Path(data_dir) / f"train_{condition}.npz")
    out = {k: d[k] for k in ("fg", "bg", "cont", "phase")}
    if episodes is not None:
        out = {k: v[:episodes] for k, v in out.items()}
    if with_actions:
        acts = load_json(Path(data_dir) / f"train_{condition}_actions.json")["actions"]
        out["actions"] = acts[:episodes] if episodes is not None else acts
    return out


def segments(condition: str, layout: dict, T: int) -> list[tuple[int, int]]:
    """Video-clip boundaries [a,b) of one episode (00 section 5)."""
    clip, F = int(layout["clip"]), int(layout["formation"])
    if condition in ("state", "partial"):
        return [(i * clip, (i + 1) * clip) for i in range(T // clip)]
    segs = [(0, F)] if F > 0 else []
    return segs + [(F + j * clip, F + (j + 1) * clip) for j in range((T - F) // clip)]


def make_pairs(data_dir, condition: str, protocol: str = "video", episodes: int | None = None) -> np.ndarray:
    """int32 [N,2]: (episode, t) meaning the transition t-1 -> t."""
    d = np.load(Path(data_dir) / f"train_{condition}.npz")
    cont = d["cont"]
    E, T = cont.shape
    if episodes is not None:
        E = min(E, episodes)
    if protocol == "causal":
        e, t = np.nonzero(cont[:E])
        keep = t > 0
        return np.stack([e[keep], t[keep]], 1).astype(np.int32)
    if protocol != "video":
        raise ValueError(f"unknown protocol {protocol!r}")
    segs = segments(condition, load_summary(data_dir)["layout"], T)
    ts = np.array([t for a, b in segs for t in range(a + 1, b)], dtype=np.int32)
    e = np.repeat(np.arange(E, dtype=np.int32), len(ts))
    return np.stack([e, np.tile(ts, E)], 1)


def build_act_maps(actions, pairs: np.ndarray) -> np.ndarray:
    """uint8 [N,H,W]; token of the actions applied before the tick that produced frame t."""
    out = np.zeros((len(pairs), H, W), dtype=np.uint8)
    for i, (e, t) in enumerate(pairs):
        acts = actions[int(e)][int(t)]
        if acts:
            out[i] = encode_actions(acts)
    return out


def data_hashes(data_dir, condition: str) -> dict:
    data_dir = Path(data_dir)
    files = [data_dir / f"train_{condition}.npz", data_dir / f"train_{condition}_actions.json",
             data_dir / "train_summary.json"]
    return {f.name: sha1_file(f) for f in files if f.exists()}


# --------------------------------------------------------------------------- bench
def load_bench(path=None) -> dict:
    """bench.npz + bench.json (same stem). Adds `ivs[i][k]`, `t_last[i,k]`, `A[i,k,H,W]` (intervention cells)."""
    path = Path(path) if path else DEFAULT_BENCH
    z = np.load(path)
    b = {k: z[k] for k in z.files}
    meta = load_json(path.with_suffix(".json"))
    b["meta"] = meta
    b["ivs"] = meta["interventions"]
    b["scene_kinds"] = meta["scene_kinds"]
    b["splits"] = meta["splits"]
    N = b["ctx_fg"].shape[0]
    b["t_last"] = np.array([[max(iv_by_t(iv)) if iv_by_t(iv) else 0 for iv in pair] for pair in b["ivs"]],
                           dtype=np.int64)
    import datasets_v3 as v3
    A = np.zeros((N, 2, H, W), dtype=bool)
    for i in range(N):
        for k in range(2):
            for x, y in v3.action_cells(b["ivs"][i][k], b["ctx_fg"][i]):
                A[i, k, y, x] = True
    b["A"] = A
    return b


def select_bench(b: dict, splits=None, limit: int | None = None) -> dict:
    """Keep the given splits (names; None/'all' = all) and at most `limit` samples per split (quick local checks)."""
    names = b["splits"]
    want = list(range(len(names))) if splits in (None, "all") else [
        names.index(s) for s in (splits.split(",") if isinstance(splits, str) else splits)]
    idx = []
    for s in want:
        ii = np.nonzero(b["split_ids"] == s)[0]
        idx.extend(ii[:limit] if limit else ii)
    idx = np.array(sorted(idx))
    n = len(b["split_ids"])
    out = {}
    for k, v in b.items():
        if isinstance(v, np.ndarray) and v.shape[:1] == (n,):
            out[k] = v[idx]
        elif k in ("ivs", "scene_kinds"):
            out[k] = [v[i] for i in idx]
        else:
            out[k] = v
    out["index"] = idx
    return out
