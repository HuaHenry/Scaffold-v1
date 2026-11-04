"""Training entry point (plan 00 section 9).

python -m exp.train --condition state --seed 0 --protocol video --steps 6000 --batch 256

Without --out the result goes to experiments/preresearch/<exp>/<timestamp>_train_<id>_<settings>/ (plan 00 s.11).
A relative --out is relative to that results root. `--id` is the logical name used to find the run later
(default `<condition>[_<tag>]_s<seed>`).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import common
from .common import H, W, resolve
from .model import build_model, n_params

frozen_defaults = common.frozen_defaults


def get_args(argv=None):
    fz = frozen_defaults()
    ap = argparse.ArgumentParser(description="Train one GridFormer on one data condition (fixed number of steps).")
    ap.add_argument("--condition", required=True, choices=common.CONDITIONS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--protocol", default="video", choices=("video", "causal"))
    ap.add_argument("--data", type=Path, default=common.DEFAULT_DATA, help="v3 data dir (train_*.npz)")
    ap.add_argument("--val", type=Path, default=common.DEFAULT_VAL, help="validation dir (monitoring only)")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", type=Path, default=None, help="result folder (default: new timestamped folder)")
    ap.add_argument("--id", default=None, help="logical run id, default <condition>[_<tag>]_s<seed>")
    ap.add_argument("--exp", default=common.EXP_NAME, help="experiment folder under the results root")
    ap.add_argument("--episodes", type=int, default=None, help="use only the first N episodes (prefix)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--unroll", type=int, default=fz["unroll"])
    ap.add_argument("--lr", type=float, default=fz["lr"])
    ap.add_argument("--lr-min", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--change-weight", type=float, default=fz["change_weight"])
    ap.add_argument("--depth", type=int, default=fz["depth"])
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--log-batch-composition", action="store_true",
                    help="log the share of formation (phase==0) transitions per batch")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu | mps")
    return ap.parse_args(argv)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- data on device
class Data:
    """One condition's frames (uint8, on device), pre-encoded action maps and the transition pairs."""

    def __init__(self, data_dir, condition, protocol, episodes, dev):
        d = common.load_condition(data_dir, condition, episodes)
        E, T = d["fg"].shape[:2]
        self.E, self.T = E, T
        self.fg = torch.from_numpy(d["fg"]).to(dev)
        self.bg = torch.from_numpy(d["bg"]).to(dev)
        self.phase = torch.from_numpy(d["phase"]).to(dev)
        act = np.zeros((E, T, H, W), dtype=np.uint8)
        for e in range(E):
            for t, acts in enumerate(d["actions"][e]):
                if acts:
                    act[e, t] = common.encode_actions(acts)
        self.act = torch.from_numpy(act).to(dev)
        self.pairs_np = common.make_pairs(data_dir, condition, protocol, episodes)
        self.pairs = torch.from_numpy(self.pairs_np.astype(np.int64)).to(dev)
        self.valid = np.zeros((E, T), dtype=bool)
        self.valid[self.pairs_np[:, 0], self.pairs_np[:, 1]] = True

    def chain_pairs(self, k: int) -> torch.Tensor:
        """Pairs whose next k-1 transitions are also valid (needed for K-step unrolling)."""
        if k <= 1:
            return self.pairs
        p = self.pairs_np
        ok = np.ones(len(p), dtype=bool)
        for j in range(1, k):
            tt = p[:, 1] + j
            ok &= (tt < self.T) & self.valid[p[:, 0], np.minimum(tt, self.T - 1)]
        return self.pairs[torch.from_numpy(ok).to(self.pairs.device)]

    def batch(self, e, t):
        return (self.fg[e, t - 1].long(), self.bg[e, t - 1].long(), self.fg[e, t].long(), self.bg[e, t].long(),
                self.act[e, t].long())


def weighted_ce(lf, lb, tf, tb, pf, pb, w_change):
    chg = (tf != pf) | (tb != pb)
    w = 1.0 + (w_change - 1.0) * chg.float()
    ce = F.cross_entropy(lf.float(), tf, reduction="none") + F.cross_entropy(lb.float(), tb, reduction="none")
    return (ce * w).sum() / w.sum(), chg


@torch.no_grad()
def transition_accuracy(model, data: Data, pairs=None, max_n: int | None = None, batch: int = 512) -> dict:
    """Accuracy on cells that changed (changed-cell acc) and on changed cells not touched by an action."""
    pairs = data.pairs if pairs is None else pairs
    if max_n is not None and len(pairs) > max_n:
        pairs = pairs[torch.randperm(len(pairs), generator=torch.Generator().manual_seed(0))[:max_n].to(pairs.device)]
    model.eval()
    c_ok = c_n = k_ok = k_n = 0
    for i in range(0, len(pairs), batch):
        e, t = pairs[i:i + batch, 0], pairs[i:i + batch, 1]
        pf, pb, tf, tb, act = data.batch(e, t)
        nf, nb = model.step(pf, pb, act)
        chg = (tf != pf) | (tb != pb)
        ok = (nf == tf) & (nb == tb)
        casc = chg & (act == 0)
        c_ok += int((ok & chg).sum()); c_n += int(chg.sum())
        k_ok += int((ok & casc).sum()); k_n += int(casc.sum())
    return {"chg_acc": c_ok / max(1, c_n), "cascade_acc": k_ok / max(1, k_n), "n_changed": c_n, "n_cascade": k_n}


def lr_at(step, a):
    if step <= a.warmup:
        return a.lr * step / max(1, a.warmup)
    p = (step - a.warmup) / max(1, a.steps - a.warmup)
    return a.lr_min + 0.5 * (a.lr - a.lr_min) * (1 + math.cos(math.pi * min(1.0, p)))


def main(argv=None) -> dict:
    a = get_args(argv)
    a.data, a.val = resolve(a.data), resolve(a.val)
    a.id = a.id or f"{a.condition}{'_' + a.tag if a.tag else ''}_s{a.seed}"
    n_ep = a.episodes or common.load_summary(a.data)["conditions"][a.condition]["episodes"]
    sett = common.train_settings(n_ep, a.protocol, a.steps, a.batch, a.lr, a.change_weight, a.depth, a.unroll)
    a.out = common.resolve_out(a.out) if a.out else common.new_result_dir(a.exp, "train", a.id, sett)
    a.out.mkdir(parents=True, exist_ok=True)
    common.write_meta(a.out, "train", a.id, {
        "condition": a.condition, "seed": a.seed, "protocol": a.protocol, "episodes": n_ep, "steps": a.steps,
        "batch": a.batch, "lr": a.lr, "change_weight": a.change_weight, "depth": a.depth, "unroll": a.unroll,
        "tag": a.tag, "data": common.bench_label(a.data / "bench.npz")})
    dev = pick_device(a.device)
    use_amp = dev.type == "cuda"
    common.seed_everything(a.seed)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    data = Data(a.data, a.condition, a.protocol, a.episodes, dev)
    pairs = data.chain_pairs(a.unroll)
    if len(pairs) == 0:
        sys.exit("no training transitions")
    val = None
    if a.val_every > 0 and (a.val / "train_state.npz").exists():
        val = Data(a.val, "state", "video", None, dev)
    elif a.val_every > 0:
        print(f"[warn] validation set {a.val} not found, skipping validation (run: python -m exp.gen_val)",
              flush=True)

    model = build_model({"dim": a.dim, "depth": a.depth, "heads": a.heads}).to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    decay = [p for p in model.parameters() if p.ndim >= 2]          # weight decay only on matrices / embeddings
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=(dev.type == "cuda"))

    cfg = {"args": {k: common.rel(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
           "n_pairs": int(len(pairs)), "n_params": n_params(model), "episodes_used": data.E,
           "data_sha1": common.data_hashes(a.data, a.condition), "git_rev": common.git_rev(),
           "torch": torch.__version__, "device": str(dev),
           "device_name": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
           "formation_share_of_pairs": float((data.phase[pairs[:, 0], pairs[:, 1]] == 0).float().mean())}
    common.dump_json(cfg, a.out / "config.json")
    print(f"[train] {a.condition} seed={a.seed} protocol={a.protocol} pairs={len(pairs)} params={n_params(model) / 1e6:.1f}M "
          f"device={dev}", flush=True)

    g = torch.Generator(device=dev).manual_seed(a.seed)
    log = open(a.out / "train_log.jsonl", "w")
    acc_loss = acc_cacc = acc_form = torch.zeros((), device=dev)
    n_acc = 0
    form_total, form_n = 0.0, 0
    t0 = time.time()
    last_val: dict = {}
    for step in range(1, a.steps + 1):
        model.train()
        lr = lr_at(step, a)
        for grp in opt.param_groups:
            grp["lr"] = lr
        idx = torch.randint(len(pairs), (a.batch,), device=dev, generator=g)
        e, t = pairs[idx, 0], pairs[idx, 1]
        cur_f, cur_b = data.fg[e, t - 1].long(), data.bg[e, t - 1].long()
        loss = 0.0
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
            for j in range(a.unroll):
                tt = t + j
                gp_f, gp_b = data.fg[e, tt - 1].long(), data.bg[e, tt - 1].long()
                tf, tb, act = data.fg[e, tt].long(), data.bg[e, tt].long(), data.act[e, tt].long()
                lf, lb = model(cur_f, cur_b, act)
                lj, chg = weighted_ce(lf, lb, tf, tb, gp_f, gp_b, a.change_weight)
                loss = loss + lj / a.unroll
                if j == 0:
                    ok = (lf.argmax(1) == tf) & (lb.argmax(1) == tb)
                    cacc = (ok & chg).sum() / chg.sum().clamp(min=1)
                cur_f, cur_b = lf.argmax(1).detach(), lb.argmax(1).detach()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        d = min(a.ema, (1 + step) / (10 + step))
        with torch.no_grad():
            ep, mp = list(ema.parameters()), list(model.parameters())
            torch._foreach_mul_(ep, d)
            torch._foreach_add_(ep, mp, alpha=1 - d)
        acc_loss = acc_loss + loss.detach()
        acc_cacc = acc_cacc + cacc.detach()
        if a.log_batch_composition:
            acc_form = acc_form + (data.phase[e, t] == 0).float().mean()
        n_acc += 1
        if step % a.log_every == 0 or step == a.steps:
            rec = {"step": step, "loss": float(acc_loss) / n_acc, "chg_acc": float(acc_cacc) / n_acc, "lr": lr}
            if a.log_batch_composition:
                rec["form_frac"] = float(acc_form) / n_acc
                form_total += rec["form_frac"] * n_acc
                form_n += n_acc
            log.write(json.dumps(rec) + "\n")
            log.flush()
            acc_loss = acc_cacc = acc_form = torch.zeros((), device=dev)
            n_acc = 0
            if step % (a.log_every * 10) == 0:
                print(f"[train] step {step}/{a.steps} loss {rec['loss']:.4f} chg_acc {rec['chg_acc']:.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        if val is not None and (step % a.val_every == 0 or step == a.steps):
            last_val = transition_accuracy(ema, val)
            log.write(json.dumps({"step": step, "type": "val", **{f"val_{k}": v for k, v in last_val.items()}}) + "\n")
            log.flush()
            print(f"[val] step {step} chg_acc {last_val['chg_acc']:.3f} cascade_acc {last_val['cascade_acc']:.3f}",
                  flush=True)
    log.close()

    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "cfg": model.cfg, "step": a.steps,
                "args": cfg["args"]}, a.out / "ckpt.pt")
    done = {"steps": a.steps, "seconds": time.time() - t0, "n_pairs": int(len(pairs)),
            "val": last_val, "train_raw": transition_accuracy(model, data, max_n=4096),
            "train_ema": transition_accuracy(ema, data, max_n=4096)}
    if form_n:
        done["form_frac_mean"] = form_total / form_n
    common.dump_json(done, a.out / "done.json")
    print(f"[train] done in {done['seconds']:.0f}s -> {common.rel(a.out)}", flush=True)
    return done


if __name__ == "__main__":
    main()
