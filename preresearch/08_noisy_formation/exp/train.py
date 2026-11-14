"""Queue-only single-seed training; commit the final EMA, never validation-best."""
import argparse
from importlib import import_module

from .shared import (PILOT, common, input_preflight, jobs, resolve_file, scale,
                     train_cli, train_identity, validate_config, validate_done,
                     verify_run, write_json)


def train_task(row, p):
    if p.get("training_seeds") != [0] or row.get("seed") != 0 or row not in jobs(row["run"], p):
        raise ValueError("experiment08 requires the recorded single-seed job")
    import torch
    out = resolve_file(row["out"])
    if row.get("reuse"):
        return validate_done(out, row, p)
    if (out / "train_complete.json").exists():
        return validate_done(out, row, p)
    if any((out / n).exists() for n in ("eval.json", "eval_samples.npz", "eval_complete.json")):
        raise ValueError("evaluation exists without committed training")
    if (out / "config.json").exists():
        validate_config(out, row, p)
    input_preflight(row, p)
    import_module(f"{PILOT}.train").main(train_cli(row, p))
    cfg = validate_config(out, row, p)
    ck = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=True)
    done = common.load_json(out / "done.json")
    if (ck.get("step") != row["steps"] or ck.get("args") != cfg["args"]
            or done.get("selection") != "last" or done.get("samples_seen") != row["steps"] * p["batch"]):
        raise ValueError("trainer did not finish the final-step budget")
    ck.update(selection="last", weights="ema")
    torch.save(ck, out / "ckpt.tmp")
    (out / "ckpt.tmp").replace(out / "ckpt.pt")
    done.update(checkpoint_step=row["steps"], weights="ema", ckpt_sha1=common.sha1_file(out / "ckpt.pt"))
    write_json(out / "done.json", done)
    write_json(out / "train_complete.json", train_identity(out, row, p))
    try:
        return validate_done(out, row, p)
    except Exception:
        (out / "train_complete.json").unlink(missing_ok=True)
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--variant", required=True)
    a = ap.parse_args(argv)
    p, _ = verify_run(a.run)
    row = next((r for r in jobs(a.run, p) if r["variant"] == a.variant), None)
    if row is None:
        ap.error("variant is not in this recorded run")
    if row.get("reuse"):
        validate_done(resolve_file(row["out"]), row, p)
        return
    with scale.run_lock(resolve_file(row["out"])):
        train_task(row, p)


if __name__ == "__main__":
    main()
