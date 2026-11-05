"""Metrics (plan 00 section 6). Per-sample scoring reuses datasets.score_outcome and core.check_violations."""
from __future__ import annotations

import numpy as np

from . import common  # noqa: F401  (puts game_code on sys.path)
import datasets as D
from core import check_violations

N_MAT = 21
PER_INTERVENTION = ("cell_acc_all", "cell_acc_mask", "structure_iou", "change_f1", "cascade_f1",
                    "cascade_recall", "exact", "violations", "any_violation", "mass_error",
                    "has_cascade", "n_cascade_true", "n_cascade_pred")
HEADLINE = ("cascade_f1", "change_f1", "structure_iou", "cell_acc_mask", "rank_acc", "any_violation")


def changed(a, b) -> np.ndarray:
    """a, b = (fg, bg). Cells where either layer differs."""
    return (a[0] != b[0]) | (a[1] != b[1])


def _f1(pc: np.ndarray, tc: np.ndarray) -> float:
    if not (pc.any() or tc.any()):
        return 1.0
    tp, fp, fn = (pc & tc).sum(), (pc & ~tc).sum(), (~pc & tc).sum()
    return float(2 * tp / max(1, 2 * tp + fp + fn))


def mass_error(P, T) -> int:
    """Sum over materials of |count_P - count_T| (both layers, EMPTY excluded)."""
    cp = np.bincount(P[0].ravel(), minlength=N_MAT) + np.bincount(P[1].ravel(), minlength=N_MAT)
    ct = np.bincount(T[0].ravel(), minlength=N_MAT) + np.bincount(T[1].ravel(), minlength=N_MAT)
    return int(np.abs(cp[1:] - ct[1:]).sum())


def score_sample(P, T, S, A, violations: bool = True) -> dict:
    """P predicted, T true, S initial state (each (fg, bg)); A bool[H,W] cells covered by the intervention."""
    out = D.score_outcome(P[0], P[1], T[0], T[1], S[0], S[1])
    pc, tc = changed(P, S), changed(T, S)
    pcc, tcc = pc & ~A, tc & ~A
    out["cascade_f1"] = _f1(pcc, tcc)
    out["cascade_recall"] = float((pcc & tcc).sum() / max(1, tcc.sum()))
    out["exact"] = float(np.array_equal(P[0], T[0]) and np.array_equal(P[1], T[1]))
    out["mass_error"] = float(mass_error(P, T))
    out["has_cascade"] = float(tcc.any())
    out["n_cascade_true"] = float(tcc.sum())
    out["n_cascade_pred"] = float(pcc.sum())
    if violations:
        v = sum(check_violations(P[0].copy(), P[1].copy()).values())
        out["violations"], out["any_violation"] = float(v), float(v > 0)
    else:
        out["violations"], out["any_violation"] = np.nan, np.nan
    return out


def short_scores(P, T, S, A) -> dict:
    """Cheap metrics for the 1-step / 4-step horizons."""
    pc, tc = changed(P, S), changed(T, S)
    return {"change_f1": _f1(pc, tc), "cascade_f1": _f1(pc & ~A, tc & ~A)}


def rank_acc(p0, p1, t0, t1):
    """Rank two predicted change rates against the ground-truth intervention order."""
    if p0 == p1 or t0 == t1:
        return 0.5
    return float((p0 > p1) == (t0 > t1))


def change_rate(S, P) -> float:
    return D.change_rate(S[0], S[1], P[0], P[1])


def sample_level(samples: dict, metric: str) -> np.ndarray:
    """[N] per-sample value: mean over the two interventions (nan-aware); rank_acc is already per sample.
    Samples where the metric is undefined (e.g. rank_acc of rank-invalid samples) are nan."""
    v = samples[metric]
    if v.ndim == 1:
        return v.astype(np.float64)
    with np.errstate(all="ignore"):
        return np.nanmean(v.astype(np.float64), axis=1)


def masked_mean(x: np.ndarray) -> tuple[float, int]:
    x = np.asarray(x, dtype=np.float64)
    ok = ~np.isnan(x)
    return (float(x[ok].mean()) if ok.any() else float("nan")), int(ok.sum())
