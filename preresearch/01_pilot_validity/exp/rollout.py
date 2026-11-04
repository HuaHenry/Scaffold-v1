"""Auto-regressive roll-outs of a learned model (plan 00 section 10), batched, compound interventions supported."""
from __future__ import annotations

import numpy as np
import torch

from .common import H, W, HORIZON, iv_by_t, encode_actions, N_ACT


def bench_action_maps(ivs, length: int | None = None) -> np.ndarray:
    """ivs: list of intervention dicts -> uint8 [B,L,H,W]; L = largest step index + 1 (>= 1)."""
    by = [iv_by_t(iv) for iv in ivs]
    L = max([max(b) + 1 if b else 1 for b in by] + [length or 1])
    out = np.zeros((len(ivs), L, H, W), dtype=np.uint8)
    for i, b in enumerate(by):
        for t, acts in b.items():
            out[i, t] = encode_actions(acts)
    return out


@torch.no_grad()
def rollout_maps(model, fg0, bg0, act_maps, horizon: int = HORIZON, stop_on_fixed_point: bool = True):
    """fg0,bg0: long [B,H,W]; act_maps: [B,L,H,W]. Tick t uses act_maps[:,t] if t < L, otherwise noop.
    Returns fg [B,horizon,H,W], bg (long). With stop_on_fixed_point the remaining frames are padded once the
    whole batch is unchanged after the last scripted action (identical to continuing for a deterministic model)."""
    dev = fg0.device
    B, L = act_maps.shape[:2]
    fg, bg = fg0, bg0
    out_f, out_b = [], []
    zero = torch.zeros_like(fg0)
    for t in range(horizon):
        act = act_maps[:, t].long() if t < L else zero
        nf, nb = model.step(fg, bg, act)
        out_f.append(nf)
        out_b.append(nb)
        fixed = bool((nf == fg).all() and (nb == bg).all())
        fg, bg = nf, nb
        if stop_on_fixed_point and fixed and t >= L:
            break
    while len(out_f) < horizon:
        out_f.append(out_f[-1])
        out_b.append(out_b[-1])
    return torch.stack(out_f, 1), torch.stack(out_b, 1)


def rollout(model, fg0, bg0, first_action, horizon: int = HORIZON, stop_on_fixed_point: bool = True):
    """Plan interface: first_action = int64 action map [B,H,W] used at tick 0, noop afterwards."""
    f, b = rollout_maps(model, fg0, bg0, first_action[:, None], horizon, stop_on_fixed_point)
    return {"fg": f, "bg": b}


def rollout_script(model, fg0, bg0, actions_per_frame, horizon: int | None = None):
    """Follow a per-frame action script (one episode, B=1): list (length T) of list[dict], or an encoded
    tensor [B,T,H,W]. Never stops early."""
    maps = actions_per_frame
    if not torch.is_tensor(maps):
        maps = torch.from_numpy(np.stack([encode_actions(a) for a in actions_per_frame])[None]).to(fg0.device)
    f, b = rollout_maps(model, fg0, bg0, maps, horizon or maps.shape[1], stop_on_fixed_point=False)
    return {"fg": f, "bg": b}


def fixed_point_step(traj_fg: np.ndarray, traj_bg: np.ndarray, ctx_fg: np.ndarray, ctx_bg: np.ndarray,
                     t_last: np.ndarray) -> np.ndarray:
    """First tick s > t_last whose state equals the previous one (deterministic => fixed point); else 31.
    traj [N,2,32,H,W], ctx [N,H,W], t_last [N,2]."""
    N = traj_fg.shape[0]
    fp = np.full((N, 2), traj_fg.shape[2] - 1, dtype=np.int64)
    for k in range(2):
        pf = np.concatenate([ctx_fg[:, None], traj_fg[:, k, :-1]], 1)
        pb = np.concatenate([ctx_bg[:, None], traj_bg[:, k, :-1]], 1)
        same = (traj_fg[:, k] == pf).all((2, 3)) & (traj_bg[:, k] == pb).all((2, 3))      # [N,32]
        same &= np.arange(traj_fg.shape[2])[None] > t_last[:, k:k + 1]
        has = same.any(1)
        fp[has, k] = same[has].argmax(1)
    return fp
