"""Evaluate existing checkpoints only; all new files stay in experiment 04."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from pathlib import Path
from queue import Empty, Queue
import os
import subprocess
import sys
import time

import numpy as np

from .shared import (CONDITIONS, PILOT_PACKAGE, PROJECT, VERSION, common, main_shared, output_path,
                     resolve_file, write_json, write_npz)
from .probes import verify_probe


def probe_identity(run_dir, name):
    directory = Path(run_dir) / "data" / name
    done = common.load_json(directory / "complete.json")
    verify_probe(directory, done["settings"])
    if done["status"] != "ready":
        raise ValueError("probe has no usable cases")
    return directory, done


def evaluation_identity(run_dir, name, condition, batch):
    directory, done = probe_identity(run_dir, name)
    source = common.load_json(Path(run_dir) / "source.json")
    return {"version": VERSION, "probe": name, "probe_hashes": done["hashes"], "condition": condition,
            "source": source, "batch": batch}, directory, source


def load_cached(directory, expected):
    done = common.load_json(Path(directory) / "complete.json")
    if done["identity"] != expected:
        raise ValueError("cached evaluation identity changed; use a new run")
    if any(common.sha1_file(Path(directory) / name) != sha1 for name, sha1 in done["hashes"].items()):
        raise ValueError("cached evaluation was modified; refusing to overwrite")
    return done


def evaluate_probe(run, name, condition, batch=128, device="cuda"):
    if condition not in (*CONDITIONS, "oracle_reseed") or batch <= 0:
        raise ValueError("unknown condition or invalid batch")
    run_dir = output_path(run)
    expected, data_dir, source = evaluation_identity(run_dir, name, condition, batch)
    out = run_dir / "evaluate" / name / condition
    if (out / "complete.json").exists():
        print(f"[04] reuse {name}/{condition}", flush=True)
        return load_cached(out, expected)
    if out.exists() and any(p.name not in ("task.log", "failed.json") for p in out.iterdir()):
        raise ValueError("partial evaluation files exist; use a new run")
    out.mkdir(parents=True, exist_ok=True)
    evaluate = import_module(f"{PILOT_PACKAGE}.evaluate")
    bench = common.load_bench(data_dir / "bench.npz")
    n = len(bench["ctx_fg"])
    model, dev = None, None
    if condition in CONDITIONS:
        import torch
        from importlib import import_module as load_module
        path = resolve_file(source["sources"][condition])
        audited = common.load_json(resolve_file(source["source_run"]) / "audit.json")
        main_shared.validate_completed(path, condition, audited)
        if common.sha1_file(path / "ckpt.pt") != source["models"][condition]["ckpt.pt"]:
            raise ValueError("source checkpoint changed")
        dev = load_module(f"{PILOT_PACKAGE}.train").pick_device(device)
        checkpoint = torch.load(path / "ckpt.pt", map_location=dev, weights_only=False)
        model = load_module(f"{PILOT_PACKAGE}.model").build_model(checkpoint["cfg"]).to(dev)
        model.load_state_dict(checkpoint["ema"])
        model.eval()
    parts, t0 = [], time.monotonic()
    print(f"[04] {name}/{condition}: start {n} scenes", flush=True)
    for start in range(0, n, batch):
        selected = np.arange(start, min(n, start + batch))
        chunk = {}
        for key, value in bench.items():
            if isinstance(value, np.ndarray) and key not in ("act_vocab", "iv_vocab") and value.shape[:1] == (n,):
                chunk[key] = value[selected]
            elif key in ("ivs", "scene_kinds"):
                chunk[key] = [value[i] for i in selected]
            else:
                chunk[key] = value
        if condition == "oracle_reseed":
            tf, tb = import_module(f"{PILOT_PACKAGE}.baselines").predict_bench(condition, chunk, workers=1)
        else:
            tf, tb = evaluate.predict_bench_model(model, chunk, dev, batch)
        parts.append(evaluate.score_bench(chunk, tf, tb))
        print(f"[04] {name}/{condition}: {selected[-1] + 1}/{n} scenes, {time.monotonic() - t0:.0f}s", flush=True)
    samples = {key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]}
    summary = evaluate.summarize(samples, bench)
    write_npz(out / "eval_samples.npz", samples)
    write_json(out / "eval.json", {"identity": expected, "seconds": time.monotonic() - t0, **summary})
    done = {"identity": expected, "hashes": {key: common.sha1_file(out / key) for key in ("eval_samples.npz", "eval.json")}}
    write_json(out / "complete.json", done)
    (out / "failed.json").unlink(missing_ok=True)
    return done


def check_oracle(run, name):
    run_dir = output_path(run)
    data_dir, _ = probe_identity(run_dir, name)
    meta = common.load_json(data_dir / "bench.json")
    with np.load(run_dir / "evaluate" / name / "oracle_reseed" / "eval_samples.npz") as z:
        exact = z["exact"].mean(axis=1)
    rows = []
    if name == "scale":
        for kind in ("tower", "bridge"):
            mask = np.array([r["kind"] == kind and r["within_training_parameter_support"] for r in meta["groups"]])
            n = int(mask.sum())
            value = float(exact[mask].mean()) if n else None
            rows.append({"kind": kind, "n_cases": n, "oracle_exact": value,
                         "status": "not_available" if not n else ("pass" if value >= .95 else "fail")})
    report = {"probe": name, "criterion": "oracle exact >= 0.95 within training parameter support",
              "groups": rows, "note": "missing groups cannot count as a validity pass"}
    write_json(run_dir / "evaluate" / name / "oracle_check.json", report)
    if any(r["status"] == "fail" for r in rows):
        raise ValueError("scale oracle validation failed; models were not evaluated on this invalid probe")
    return report


def run_evaluations(run, names, gpus="0,1,2,3", batch=128):
    run_dir = output_path(run)
    available = []
    for name in names:
        if common.load_json(run_dir / "data" / name / "complete.json")["status"] != "ready":
            print(f"[04] {name}: no unique valid cases, recorded as unavailable", flush=True)
            continue
        evaluate_probe(run, name, "oracle_reseed", batch, "cpu")
        check_oracle(run, name)
        available.append(name)
    cards = [card.strip() for card in gpus.split(",")]
    if not cards or any(not card for card in cards) or len(set(cards)) != len(cards):
        raise ValueError("--gpus must list distinct nonempty devices")
    tasks = Queue()
    for name in available:
        for c in CONDITIONS:
            expected, _, _ = evaluation_identity(run_dir, name, c, batch)
            out = run_dir / "evaluate" / name / c
            if (out / "complete.json").exists():
                load_cached(out, expected)
            else:
                tasks.put((name, c))

    def worker(gpu):
        ok = True
        while True:
            try:
                name, c = tasks.get_nowait()
            except Empty:
                return ok
            out = run_dir / "evaluate" / name / c
            out.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-m", "exp.evaluate", "--run", run, "--probe", name,
                       "--condition", c, "--batch", str(batch), "--device", "cpu" if gpu == "cpu" else "cuda"]
            print(f"[04] launch {name}/{c} on {gpu}; log={common.rel(out / 'task.log')}", flush=True)
            with (out / "task.log").open("a") as log:
                result = subprocess.run(command, cwd=PROJECT / "preresearch" / EXP_NAME_DIRECTORY,
                                        env=dict(os.environ, CUDA_VISIBLE_DEVICES="" if gpu == "cpu" else gpu,
                                                 PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1"),
                                        stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                ok = False
                write_json(out / "failed.json", {"returncode": result.returncode, "probe": name, "condition": c})
                print(f"[04] FAILED {name}/{c}: see task.log", flush=True)
            else:
                load_cached(out, evaluation_identity(run_dir, name, c, batch)[0])
                print(f"[04] complete {name}/{c} on {gpu}", flush=True)
    with ThreadPoolExecutor(max_workers=len(cards)) as pool:
        if not all(pool.map(worker, cards)):
            raise RuntimeError("probe evaluation failed; existing checkpoints were not modified")


EXP_NAME_DIRECTORY = "04_ood_generalization"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--probe", choices=("scale", "components"), required=True)
    ap.add_argument("--condition", choices=(*CONDITIONS, "oracle_reseed"), required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    evaluate_probe(a.run, a.probe, a.condition, a.batch, a.device)


if __name__ == "__main__":
    main()
