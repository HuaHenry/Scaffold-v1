"""Reference predictors (plan 00 section 7), adapted to the v3 compound interventions.

Interface: predict_traj(name, ctx_fg, ctx_bg, iv, seed=0, key=0) -> (fg[32,H,W], bg[32,H,W]).
`iv` is the v3 intervention dict {kind, steps:[{t, actions}]}; `key` seeds random_physics (sample*2 + k).
"""
from __future__ import annotations

import random

import numpy as np

from . import common
from .common import HORIZON
import datasets as D
import datasets_v3 as v3
from core import FormWorld

NAMES = ("copy_last", "apply_edit", "random_physics", "oracle_reseed")
ORACLE_SEED = 12345


def _repeat(fg, bg):
    return np.repeat(fg[None], HORIZON, 0), np.repeat(bg[None], HORIZON, 0)


def copy_last(fg, bg, iv, seed=0, key=0):
    return _repeat(fg, bg)


def apply_edit(fg, bg, iv, seed=0, key=0):
    """Literal effect of the actions: no tick, no background clean-up."""
    w = FormWorld(grid=fg.copy(), background=bg.copy())
    for st in sorted(iv["steps"], key=lambda s: s["t"]):
        for a in st["actions"]:
            w.apply_action(v3.to_action(a))
    return _repeat(w.grid, w.background)


def oracle_reseed(fg, bg, iv, seed=0, key=0):
    r = v3.rollout(fg, bg, iv, ORACLE_SEED)
    return r["fg"], r["bg"]


def random_physics(fg, bg, iv, seed=0, key=0):
    """A random other load-bearing block is removed instead; the real engine plays it out."""
    rng = random.Random(key)
    true_cells = v3.action_cells(iv, fg)
    cands = D.candidate_interventions(fg, rng, 12)
    pick = next((a for a in cands if (a.x, a.y) not in true_cells), cands[0] if cands else None)
    if pick is None:
        return _repeat(fg, bg)
    alt = v3.mk("single", (0, [v3._a("remove", pick.x, pick.y)]))
    r = v3.rollout(fg, bg, alt, ORACLE_SEED)
    return r["fg"], r["bg"]


_FN = {"copy_last": copy_last, "apply_edit": apply_edit, "random_physics": random_physics,
       "oracle_reseed": oracle_reseed}


def predict_traj(name, fg, bg, iv, seed=0, key=0):
    return _FN[name](fg, bg, iv, seed, key)


def predict(name, fg, bg, iv, seed=0, key=0):
    f, b = predict_traj(name, fg, bg, iv, seed, key)
    return f[-1], b[-1]


def _job(args):
    name, i, k, fg, bg, iv = args
    f, b = predict_traj(name, fg, bg, iv, 0, 2 * i + k)
    return i, k, f, b


def predict_bench(name: str, bench: dict, workers: int = 0):
    """Trajectories for every (sample, intervention): uint8 [N,2,32,H,W] x2."""
    N = bench["ctx_fg"].shape[0]
    tf = np.zeros((N, 2, HORIZON, common.H, common.W), dtype=np.uint8)
    tb = np.zeros_like(tf)
    jobs = [(name, i, k, bench["ctx_fg"][i], bench["ctx_bg"][i], bench["ivs"][i][k])
            for i in range(N) for k in range(2)]
    if workers > 1 and name in ("oracle_reseed", "random_physics"):
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as ex:
            results = list(ex.map(_job, jobs, chunksize=8))
    else:
        results = map(_job, jobs)
    for i, k, f, b in results:
        tf[i, k], tb[i, k] = f, b
    return tf, tb
