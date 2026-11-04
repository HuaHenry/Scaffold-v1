"""Statistics (plan 00 section 12): paired bootstrap, seed-level bootstrap, Holm correction.

All inputs are per-sample metric arrays (already averaged over seeds where the plan says so)."""
from __future__ import annotations

import numpy as np

from . import common
from .metrics import sample_level

N_BOOT = 10_000
BOOT_SEED = 12345


def load_samples(run_dir) -> dict:
    z = np.load(common.resolve_run(run_dir, require=None) / "eval_samples.npz")
    return {k: z[k] for k in z.files}


def seed_mean(samples_list: list[dict], metric: str) -> np.ndarray:
    """Mean over seeds of the per-sample metric -> [N] (nan stays nan only if all seeds are nan)."""
    with np.errstate(all="ignore"):
        return np.nanmean(np.stack([sample_level(s, metric) for s in samples_list]), axis=0)


def _valid(a, b):
    ok = ~(np.isnan(a) | np.isnan(b))
    return a[ok], b[ok]


def boot_mean(x: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return {"mean": float(x.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5)),
            "n": int(len(x))}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """a - b over samples, paired resampling. Returns mean diff, 95% CI, SE (std of bootstrap diffs), p (two-sided)."""
    a, b = _valid(np.asarray(a, float), np.asarray(b, float))
    d = a - b
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    p = 2 * min((m <= 0).mean(), (m >= 0).mean())
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi), "se": float(m.std(ddof=1)),
            "excludes_0": bool(lo > 0 or hi < 0), "p": float(min(1.0, max(p, 1.0 / n_boot))), "n": int(len(d))}


def seed_level_bootstrap(A: np.ndarray, B: np.ndarray, n_boot: int = 2000, seed: int = BOOT_SEED) -> dict:
    """Two-level bootstrap of mean(A) - mean(B). A [sa,N], B [sb,N]: per-seed per-sample metrics.
    Resamples the seeds of each group (with replacement), then the samples; includes training noise."""
    rng = np.random.default_rng(seed)
    ok = ~(np.isnan(A).any(0) | np.isnan(B).any(0))
    A, B = A[:, ok], B[:, ok]
    N = A.shape[1]
    nidx = rng.integers(0, N, (n_boot, N))

    def grp(M):
        sidx = rng.integers(0, M.shape[0], (n_boot, M.shape[0]))
        return np.mean([M[sidx[:, j][:, None], nidx].mean(1) for j in range(M.shape[0])], axis=0)

    m = grp(A) - grp(B)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return {"diff": float(A.mean() - B.mean()), "lo": float(lo), "hi": float(hi), "se": float(m.std(ddof=1)),
            "excludes_0": bool(lo > 0 or hi < 0), "n": int(N)}


def holm(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni: True = rejected (significant) in the input order."""
    order = np.argsort(pvals)
    rej = [False] * len(pvals)
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (len(pvals) - rank):
            rej[i] = True
        else:
            break
    return rej


def aa_partitions(seeds: list[int], k: int = 3) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """All distinct splits of `seeds` into two groups of size k (unordered), in a fixed order: 10 for 6 seeds."""
    from itertools import combinations
    out, first = [], seeds[0]
    for g in combinations(seeds, k):
        if first in g:
            out.append((g, tuple(s for s in seeds if s not in g)))
    return out
