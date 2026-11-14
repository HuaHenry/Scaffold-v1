"""Train one frozen single-task or mixed-task job; save last raw and EMA weights."""
import argparse
import copy
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from ..shared import file_hash, frozen_write, load_json, lock, model_api, quota, train_api, write_json, common
from .protocol import single_training_seed, task_inputs, verify_run
from .sampler import Sampler, STATE_OPERATION


@torch.no_grad()
def validation(model, sampler, device, samples=256):
    model.eval()
    result = {}
    for task, data in sampler.data.items():
        rng = np.random.default_rng(0)
        n = min(samples, data.count * len(STATE_OPERATION))
        es = rng.integers(data.count, size=n)
        ts = rng.choice(STATE_OPERATION, n)
        values = [torch.from_numpy(v).long().to(device) for v in data.batch(es, ts)]
        pf, pb, tf, tb, act = values
        nf, nb = model.step(pf, pb, act)
        changed = (tf != pf) | (tb != pb)
        correct = (nf == tf) & (nb == tb)
        result[task] = {"changed_cell_accuracy": float((correct & changed).sum() / changed.sum().clamp(min=1)),
                        "n_changed": int(changed.sum())}
    return result


def identity(out, p, row):
    return {"job": row, "protocol_sha256": file_hash(Path(out).parents[1] / "protocol.json"),
            "hashes": {n: file_hash(Path(out) / n) for n in ("config.json", "done.json", "ckpt.pt")}}


def validate_done(out, p, row):
    out = Path(out)
    cfg = load_json(out / "config.json")
    if cfg != {"protocol": p, "job": row}:
        raise ValueError("completed training configuration changed")
    marker = load_json(out / "train_complete.json")
    if marker != identity(out, p, row):
        raise ValueError("completed training hash identity changed")
    done = load_json(out / "done.json")
    batch = p["settings"]["batch"]
    if (done.get("step") != row["steps"] or done.get("selection") != "last" or done.get("weights") != "ema"
            or done.get("samples_seen") != row["steps"] * batch):
        raise ValueError("completed training budget differs")
    total = row["steps"] * batch
    for i, task in enumerate(row["tasks"]):
        n = total // len(row["tasks"]) + int(i < total % len(row["tasks"]))
        formed = quota(p["formation_ratio"], n) if row["condition"] in ("process", "shuffle") else 0
        if done["exposure"][task] != {"samples": n, "formation": formed, "operation": n - formed}:
            raise ValueError("task or phase exposure differs from exact quota")
    ck = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=True)
    expected_cfg = {k: p["settings"][k] for k in ("dim", "depth", "heads")}
    model = model_api.build_model(expected_cfg)
    if (ck.get("job") != row or ck.get("step") != row["steps"] or ck.get("cfg") != model.cfg
            or ck.get("selection") != "last" or ck.get("weights") != "ema"):
        raise ValueError("checkpoint is not this job's final EMA")
    for kind in ("model", "ema"):
        if not all(torch.is_tensor(v) and torch.isfinite(v).all() for v in ck[kind].values()):
            raise ValueError("checkpoint contains invalid weights")
        model.load_state_dict(ck[kind], strict=True)
    return done


def train_job(run, p, row, device="auto"):
    seeds = single_training_seed(p["training_seeds"])
    if row["seed"] != seeds[0]:
        raise ValueError("job seed differs from the single recorded training seed")
    out = Path(run) / "train" / row["id"]
    task_inputs(p, row)
    if (out / "train_complete.json").exists():
        return validate_done(out, p, row)
    if any(out.glob("eval*")):
        raise ValueError("evaluation exists without completed training")
    with lock(out / "training.lock"):
        frozen_write(out / "config.json", {"protocol": p, "job": row})
        dev = train_api.pick_device(device)
        common.seed_everything(row["seed"])
        settings = p["settings"]
        sampler = Sampler(p, row)
        val_sampler = Sampler(p, row, "validation")
        model = model_api.build_model({k: settings[k] for k in ("dim", "depth", "heads")}).to(dev)
        ema = copy.deepcopy(model).eval()
        for param in ema.parameters():
            param.requires_grad_(False)
        groups = [{"params": [v for v in model.parameters() if v.ndim >= 2], "weight_decay": settings["wd"]},
                  {"params": [v for v in model.parameters() if v.ndim < 2], "weight_decay": 0.0}]
        opt = torch.optim.AdamW(groups, lr=settings["lr"], betas=(.9, .95), fused=dev.type == "cuda")
        args = SimpleNamespace(**settings, steps=row["steps"])
        started, last_val = time.time(), {}
        with (out / "train_log.jsonl").open("w") as log:
            for step in range(1, row["steps"] + 1):
                model.train()
                for group in opt.param_groups:
                    group["lr"] = train_api.lr_at(step, args)
                batch = [torch.from_numpy(v).long().to(dev) for v in sampler.batch(settings["batch"])]
                pf, pb, tf, tb, act = batch
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    lf, lb = model(pf, pb, act)
                    loss, _ = train_api.weighted_ce(lf, lb, tf, tb, pf, pb, settings["change_weight"])
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite training loss")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                decay = min(settings["ema"], (1 + step) / (10 + step))
                with torch.no_grad():
                    ep, mp = list(ema.parameters()), list(model.parameters())
                    torch._foreach_mul_(ep, decay)
                    torch._foreach_add_(ep, mp, alpha=1 - decay)
                if step % settings["log_every"] == 0 or step == row["steps"]:
                    record = {"step": step, "loss": float(loss.detach()), "seconds": time.time() - started}
                    log.write(__import__("json").dumps(record) + "\n")
                    log.flush()
                    print(f"[train] {row['id']} step {step}/{row['steps']} loss={record['loss']:.4f}", flush=True)
                if step % settings["val_every"] == 0 or step == row["steps"]:
                    last_val = validation(ema, val_sampler, dev)
                    log.write(__import__("json").dumps({"step": step, "validation": last_val}) + "\n")
                    log.flush()
        ck = {"model": model.state_dict(), "ema": ema.state_dict(), "cfg": model.cfg,
              "step": row["steps"], "job": row, "selection": "last", "weights": "ema"}
        torch.save(ck, out / "ckpt.tmp")
        (out / "ckpt.tmp").replace(out / "ckpt.pt")
        done = {"step": row["steps"], "selection": "last", "weights": "ema", "seconds": time.time() - started,
                "samples_seen": sampler.total, "exposure": sampler.exposure(), "validation": last_val}
        write_json(out / "done.json", done)
        write_json(out / "train_complete.json", identity(out, p, row))
        try:
            return validate_done(out, p, row)
        except Exception:
            (out / "train_complete.json").unlink(missing_ok=True)
            raise


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--job", required=True)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    p, _ = verify_run(a.run, full_data=False)
    row = next((r for r in load_json(a.run / "jobs.json") if r["id"] == a.job), None)
    if row is None:
        raise ValueError("job is not in frozen run")
    train_job(a.run, p, row, a.device)


if __name__ == "__main__":
    main()
