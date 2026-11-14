"""Evaluate final EMA on the frozen full bench; all writes stay in experiment08."""
import argparse
from importlib import import_module
import time

import numpy as np

from .shared import (PILOT, common, evaluation_path, jobs, record_eval, resolve_file,
                     scale, validate_done, validate_eval, verify_run, write_json)


def evaluate_task(row, p, batch=512, device="auto"):
    done = validate_done(resolve_file(row["out"]), row, p)
    out = evaluation_path(row)
    if (out / "eval_complete.json").exists():
        return validate_eval(row, p, done)
    bench_path = resolve_file(p["inputs"]["bench"])
    hashes = {n: common.sha1_file(bench_path.parent / n) for n in p["hashes"]["bench"]}
    if hashes != p["hashes"]["bench"]:
        raise ValueError("frozen bench changed")
    evaluator = import_module(f"{PILOT}.evaluate")
    dev = import_module(f"{PILOT}.train").pick_device(device)
    model, _ = evaluator.load_run(resolve_file(row["out"]), dev)
    bench = common.load_bench(bench_path)
    start = time.monotonic()
    fg, bg = evaluator.predict_bench_model(model, bench, dev, batch)
    samples = evaluator.score_bench(bench, fg, bg)
    summary = evaluator.summarize(samples, bench)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "eval.json", {"run": common.rel(resolve_file(row["out"])), "variant": row["variant"],
               "bench": common.rel(bench_path), "bench_sha1": hashes[bench_path.name],
               "bench_json_sha1": hashes[bench_path.with_suffix(".json").name],
               "checkpoint_sha1": done["ckpt_sha1"], "seconds": time.monotonic() - start, **summary})
    np.savez_compressed(out / "eval_samples.npz", **samples)
    record_eval(row, p, done)
    print(f"[eval] {row['variant']}: full frozen bench complete", flush=True)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--variant", required=True)
    a = ap.parse_args(argv)
    p, _ = verify_run(a.run)
    row = next((r for r in jobs(a.run, p) if r["variant"] == a.variant), None)
    if row is None:
        ap.error("variant is not in the recorded run")
    with scale.run_lock(evaluation_path(row)):
        evaluate_task(row, p)


if __name__ == "__main__":
    main()
