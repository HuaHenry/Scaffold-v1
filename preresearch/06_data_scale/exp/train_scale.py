"""Internal queue entry: reuse the shared trainer, commit only final-step EMA weights."""
import argparse
from importlib import import_module
from pathlib import Path

from .make_jobs import jobs
from .shared import (PILOT_PACKAGE, common, resolve_file, train_cli, training_identity,
                     validate_config, validate_done, verify_audit, write_json)


def train_task(row, audited):
    import torch
    out = resolve_file(row["out"])
    if (out / "train_complete.json").exists():
        return validate_done(out, row, audited)
    if any((out / n).exists() for n in ("eval.json", "eval_samples.npz", "eval_complete.json")):
        raise ValueError("evaluation exists without committed training")
    if (out / "config.json").exists():
        validate_config(out, row, audited)
    val = resolve_file(audited["inputs"]["val"])
    expected_val = {k: audited["hashes"]["val"][k] for k in
                    ("train_state.npz", "train_state_actions.json", "train_summary.json")}
    if common.data_hashes(val, "state") != expected_val:
        raise ValueError("frozen validation inputs changed")
    trainer = import_module(f"{PILOT_PACKAGE}.train")
    trainer.main(train_cli(row["run"], row["condition"], row["episodes"], row["seed"], audited))
    cfg = validate_config(out, row, audited)
    ck = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=True)
    done = common.load_json(out / "done.json")
    if (ck.get("step") != row["steps"] or ck.get("args") != cfg["args"]
            or done.get("selection") != "last" or done.get("samples_seen") != row["steps"] * cfg["args"]["batch"]):
        raise ValueError("trainer did not finish the requested final-step budget")
    ck.update(selection="last", weights="ema")
    torch.save(ck, out / "ckpt.tmp")
    (out / "ckpt.tmp").replace(out / "ckpt.pt")
    done.update(checkpoint_step=row["steps"], weights="ema", ckpt_sha1=common.sha1_file(out / "ckpt.pt"))
    write_json(out / "done.json", done)
    write_json(out / "train_complete.json", training_identity(out, row, audited))
    try:
        return validate_done(out, row, audited)
    except Exception:
        (out / "train_complete.json").unlink(missing_ok=True)
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--condition", required=True, choices=common.CONDITIONS)
    ap.add_argument("--episodes", required=True, type=int)
    ap.add_argument("--seed", required=True, type=int)
    a = ap.parse_args(argv)
    audited = verify_audit(a.run)
    row = next((r for r in jobs(a.run, audited) if
                (r["condition"], r["episodes"], r["seed"]) == (a.condition, a.episodes, a.seed)), None)
    if row is None:
        ap.error("task not present in the recorded run design")
    out = resolve_file(row["out"])
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (Path(out) / "training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another trainer is using this task") from exc
        train_task(row, audited)


if __name__ == "__main__":
    main()
