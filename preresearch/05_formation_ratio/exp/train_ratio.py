"""Exact supervision-mixture training on original one-step transitions."""
import argparse
import copy
import fcntl
from importlib import import_module
import json
import time
from types import SimpleNamespace

import numpy as np
import torch

from .audit import verify_audit
from .shared import (MODE, PACKAGE, VERSION, array_sha1, check_output_tree,
                     common, expected_data_hashes, formation_count, resolve_file,
                     training_args, validate_config, validate_done, write_json)

pilot = import_module(f"{PACKAGE}.train")
models = import_module(f"{PACKAGE}.model")


def phase_pools(pairs, phase):
    labels = phase[pairs[:, 0], pairs[:, 1]]
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("transition phases must be formation=0 or operation=1")
    return tuple(torch.nonzero(labels == k, as_tuple=True)[0] for k in (0, 1))


def sample_ratio_batch(pools, batch, ratio, generator):
    n_form = formation_count(ratio, batch)
    device = pools[0].device
    selected = []
    for pool, count in zip(pools, (n_form, batch - n_form)):
        if count:
            if len(pool) == 0:
                raise ValueError("requested supervision pool is empty")
            selected.append(pool[torch.randint(len(pool), (count,), device=device, generator=generator)])
    indices = torch.cat(selected)
    return indices[torch.randperm(batch, device=device, generator=generator)]


def verify_pools(data, pools, expected):
    actual = {"total": len(data.pairs), "formation": len(pools[0]), "operation": len(pools[1]),
              "pairs_sha1": array_sha1(data.pairs_np),
              "formation_indices_sha1": array_sha1(pools[0].cpu().numpy()),
              "operation_indices_sha1": array_sha1(pools[1].cpu().numpy())}
    if any(expected.get(k) != v for k, v in actual.items()):
        raise ValueError("loaded training transition pools differ from the frozen audit")


def fit(data, val, args, out, config):
    dev = data.pairs.device
    pairs, pools = data.chain_pairs(1), phase_pools(data.chain_pairs(1), data.phase)
    verify_pools(data, pools, config["pools"])
    n_form = formation_count(args.formation_ratio, args.batch)
    model = models.build_model({"dim": args.dim, "depth": args.depth, "heads": args.heads}).to(dev)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    opt = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": args.wd},
        {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.95), fused=dev.type == "cuda")
    config.update(n_params=models.n_params(model), torch=torch.__version__, device=str(dev),
                  device_name=torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu")
    write_json(out / "config.json", config)
    generator = torch.Generator(device=dev).manual_seed(args.seed)
    seen = form_seen = 0
    best_val, last_val, best_step = None, {}, None
    accumulated_loss = torch.zeros((), device=dev)
    accumulated_accuracy = torch.zeros((), device=dev)
    accumulated_steps = 0
    start = time.monotonic()
    print(f"[train] ratio={args.formation_ratio:g} F/O per batch={n_form}/{args.batch - n_form}, "
          f"pairs={len(pairs)} updates={args.steps} device={dev}; original physical transitions", flush=True)
    with (out / "train_log.jsonl").open("w") as log:
        for step in range(1, args.steps + 1):
            model.train()
            lr = pilot.lr_at(step, args)
            for group in opt.param_groups:
                group["lr"] = lr
            indices = sample_ratio_batch(pools, args.batch, args.formation_ratio, generator)
            e, t = pairs[indices, 0], pairs[indices, 1]
            pf, pb, tf, tb, act = data.batch(e, t)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                lf, lb = model(pf, pb, act)
                loss, changed = pilot.weighted_ce(lf, lb, tf, tb, pf, pb, args.change_weight)
                correct = (lf.argmax(1) == tf) & (lb.argmax(1) == tb)
                accuracy = (correct & changed).sum() / changed.sum().clamp(min=1)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss; refusing to commit a model")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            decay = min(args.ema, (1 + step) / (10 + step))
            with torch.no_grad():
                torch._foreach_mul_(list(ema.parameters()), decay)
                torch._foreach_add_(list(ema.parameters()), list(model.parameters()), alpha=1 - decay)
            actual_form = int((data.phase[e, t] == 0).sum())
            if actual_form != n_form:
                raise ValueError("sampled supervision batch has the wrong formation ratio")
            seen += len(indices)
            form_seen += actual_form
            accumulated_loss += loss.detach()
            accumulated_accuracy += accuracy.detach()
            accumulated_steps += 1
            if step % args.log_every == 0 or step == args.steps:
                record = {"step": step, "loss": float(accumulated_loss) / accumulated_steps,
                          "chg_acc": float(accumulated_accuracy) / accumulated_steps, "lr": lr,
                          "form_frac": form_seen / seen, "samples_seen": seen,
                          "formation_samples_seen": form_seen, "operation_samples_seen": seen - form_seen}
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(f"[train] step {step}/{args.steps} loss={record['loss']:.4f} "
                      f"chg_acc={record['chg_acc']:.3f} form_frac={record['form_frac']:.3f} "
                      f"elapsed={time.monotonic() - start:.0f}s", flush=True)
                accumulated_loss.zero_()
                accumulated_accuracy.zero_()
                accumulated_steps = 0
            if pilot.validation_due(step, args.steps, args.val_every, None):
                last_val = pilot.transition_accuracy(ema, val)
                if not np.isfinite(last_val["cascade_acc"]) or last_val["n_cascade"] <= 0:
                    raise ValueError("validation requires nonempty finite cascade accuracy")
                log.write(json.dumps({"type": "val", "step": step, **last_val}) + "\n")
                log.flush()
                print(f"[val] step {step} cascade_acc={last_val['cascade_acc']:.4f}", flush=True)
                if pilot.better_validation(last_val, best_val):
                    best_val, best_step = last_val.copy(), step
                    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "cfg": model.cfg,
                                "step": step, "args": config["args"], "selection": "val_cascade_acc"}, out / "ckpt.tmp")
                    (out / "ckpt.tmp").replace(out / "ckpt.pt")
    if best_step is None:
        raise ValueError("training finished without a valid selected checkpoint")
    done = {"version": VERSION, "sampling_mode": MODE, "steps": args.steps, "epochs": None,
            "steps_per_epoch": None, "samples_seen": seen, "formation_samples_seen": form_seen,
            "operation_samples_seen": seen - form_seen, "formation_ratio": args.formation_ratio,
            "form_frac_mean": form_seen / seen, "n_pairs": len(pairs), "seconds": time.monotonic() - start,
            "val": last_val, "best_val": best_val, "best_step": best_step, "selection": "val_cascade_acc",
            "ckpt_sha1": common.sha1_file(out / "ckpt.pt")}
    write_json(out / "done.json", done)
    print(f"[train] complete ratio={args.formation_ratio:g}, best_step={best_step}: {common.rel(out)}", flush=True)
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--formation-ratio", type=float, required=True)
    a = ap.parse_args(argv)
    formation_count(a.formation_ratio)
    audited = verify_audit(a.run)
    if a.formation_ratio not in audited["protocol"]["ratios"]:
        raise ValueError("ratio is not in the frozen job protocol")
    expected = training_args(a.run, a.formation_ratio, audited)
    row = {"run": a.run, "formation_ratio": a.formation_ratio}
    out = check_output_tree(resolve_file(expected["out"]))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("this supervision ratio is already training") from exc
        if (out / "done.json").exists():
            validate_done(out, row, audited)
            print("[train] completed model reused", flush=True)
            return
        if (out / "eval.json").exists() or (out / "eval_samples.npz").exists():
            raise ValueError("evaluation exists without completed training; refusing overwrite")
        if (out / "config.json").exists():
            validate_config(out, row, audited)
            print("[train] restarting incomplete training from step 0; no optimizer resume", flush=True)
        args = SimpleNamespace(**expected)
        dev = pilot.pick_device(args.device)
        common.seed_everything(args.seed)
        if dev.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        print("[train] loading unchanged process frames and action maps; no data variants", flush=True)
        data = pilot.Data(resolve_file(args.data), "process", "video", args.episodes, dev)
        val = pilot.Data(resolve_file(args.val), "state", "video", None, dev)
        config = {"version": VERSION, "args": expected, "pools": audited["pools"],
                  "data_sha1": expected_data_hashes(audited, "data"), "val_sha1": expected_data_hashes(audited, "val"),
                  "episodes_used": data.E, "n_pairs": len(data.pairs), "git_rev": common.git_rev()}
        fit(data, val, args, out, config)


if __name__ == "__main__":
    main()
