"""Paired scene bootstrap; interventions from a scene never become independent cases."""
import hashlib

import numpy as np


def rng_for(seed, key):
    digest = hashlib.sha256(str(key).encode()).digest()
    return np.random.default_rng(np.random.SeedSequence([seed, int.from_bytes(digest[:4], "little")]))


def bootstrap_means(values, n_boot, rng):
    values = np.asarray(values, dtype=float)
    if n_boot <= 0 or values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("bootstrap requires nonempty finite scene values and positive count")
    draws = np.empty(n_boot)
    for start in range(0, n_boot, 128):
        count = min(128, n_boot - start)
        indices = rng.integers(len(values), size=(count, len(values)))
        draws[start:start + count] = values[indices].mean(axis=1)
    return draws


def estimate(values, n_boot=2000, seed=12345, key="mean"):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if not n:
        return {"n_cases": 0, "mean": None, "ci": None, "status": "empty"}
    ci = np.percentile(bootstrap_means(values, n_boot, rng_for(seed, key)), [2.5, 97.5]).tolist() if n >= 2 else None
    return {"n_cases": n, "mean": float(values.mean()), "ci": ci,
            "status": "descriptive_small_group" if n < 8 else "scene_bootstrap"}


def paired(left, right, mask, n_boot, seed, key):
    left, right = np.asarray(left), np.asarray(right)
    selected = np.asarray(mask, dtype=bool) & np.isfinite(left) & np.isfinite(right)
    return estimate((left - right)[selected], n_boot, seed, key)


def difference_of_differences(left, right, reference, other, n_boot, seed, key):
    left, right = np.asarray(left, float), np.asarray(right, float)
    finite = np.isfinite(left) & np.isfinite(right)
    reference, other = np.asarray(reference, bool) & finite, np.asarray(other, bool) & finite
    if np.any(reference & other):
        raise ValueError("reference and comparison strata must contain different scenes")
    a, b = (left - right)[reference], (left - right)[other]
    if not len(a) or not len(b):
        return {"delta_reference": None, "delta_other": None, "difference": None, "ci": None,
                "n_reference": len(a), "n_other": len(b), "status": "empty"}
    rng = rng_for(seed, key)
    ci = None
    if min(len(a), len(b)) >= 2:
        draws = bootstrap_means(b, n_boot, rng) - bootstrap_means(a, n_boot, rng)
        ci = np.percentile(draws, [2.5, 97.5]).tolist()
    return {"delta_reference": float(a.mean()), "delta_other": float(b.mean()),
            "difference": float(b.mean() - a.mean()), "ci": ci,
            "n_reference": len(a), "n_other": len(b), "status": "scene_bootstrap" if min(len(a), len(b)) >= 8 else "descriptive_small_group"}


def cluster_values(values, membership):
    values, membership = np.asarray(values, float), np.asarray(membership, bool)
    if values.ndim != 2 or values.shape != membership.shape:
        raise ValueError("cluster membership must match scene-by-intervention values")
    finite = membership & np.isfinite(values)
    count = finite.sum(axis=1)
    total = np.where(finite, values, 0).sum(axis=1)
    return np.divide(total, count, out=np.full(len(values), np.nan), where=count > 0)
