"""Every task uses exact frame32, change F1 and the same foreground/background mask accuracy."""
import argparse
from pathlib import Path

import numpy as np
import torch

from ..shared import METRICS, file_hash, files_identity, load_json, lock, model_api, score, train_api, write_json
from ..training.protocol import task_inputs, verify_run
from ..training.train import validate_done


def evaluation_identity(out, task):
    return {"checkpoint_sha256": file_hash(Path(out) / "ckpt.pt"),
            "training_complete_sha256": file_hash(Path(out) / "train_complete.json"),
            "hashes": files_identity(Path(out) / "eval" / task, ("complete.json",))}


def validate_eval(out, p, row, task):
    validate_done(out, p, row)
    path = Path(out) / "eval" / task
    marker = load_json(path / "complete.json")
    if marker != evaluation_identity(out, task):
        raise ValueError("committed evaluation identity changed")
    bench_path = Path(p["data"]) / task / "bench.npz"
    report = load_json(path / "metrics.json")
    if report["bench_sha256"] != file_hash(bench_path) or report["horizon"] != 32 or report["metrics"] != list(METRICS):
        raise ValueError("evaluation benchmark or protocol changed")
    with np.load(bench_path, allow_pickle=False) as z:
        bench = {k: z[k] for k in z.files}
    with np.load(path / "samples.npz", allow_pickle=False) as z:
        samples = {k: z[k] for k in z.files}
    n = len(bench["ctx_fg"])
    for key in ("scene_ids", "group"):
        if not np.array_equal(samples[key], bench[key]):
            raise ValueError("evaluation scene pairing/order differs")
    for key in ("pred_fg", "pred_bg"):
        if samples[key].shape != (n, 2, 28, 56) or samples[key].dtype != np.uint8 or samples[key].max() > 20:
            raise ValueError("invalid predicted state")
    for metric in METRICS:
        if samples[metric].shape != (n, 2) or not np.isfinite(samples[metric]).all():
            raise ValueError("invalid metric values")
    for i in range(n):
        initial = bench["ctx_fg"][i], bench["ctx_bg"][i]
        for j in range(2):
            values = score((samples["pred_fg"][i, j], samples["pred_bg"][i, j]),
                (bench["target_fg"][i, j], bench["target_bg"][i, j]), initial)
            if any(not np.isclose(samples[k][i, j], values[k], atol=1e-12, rtol=0) for k in METRICS):
                raise ValueError("metrics cannot be reproduced from predicted states")
    return samples


@torch.no_grad()
def evaluate_job(run, p, row, device="auto", batch=32):
    out = Path(run) / "train" / row["id"]
    validate_done(out, p, row)
    task_inputs(p, row)
    dev = train_api.pick_device(device)
    ck = torch.load(out / "ckpt.pt", map_location=dev, weights_only=True)
    model = model_api.build_model({k: p["settings"][k] for k in ("dim", "depth", "heads")}).to(dev)
    model.load_state_dict(ck["ema"], strict=True)
    model.eval()
    for task in row["tasks"]:
        target = out / "eval" / task
        if (target / "complete.json").exists():
            validate_eval(out, p, row, task)
            continue
        with lock(out / f"evaluation_{task}.lock"):
            target.mkdir(parents=True, exist_ok=True)
            bp = Path(p["data"]) / task / "bench.npz"
            with np.load(bp, allow_pickle=False) as z:
                bench = {k: z[k] for k in z.files}
            n = len(bench["ctx_fg"])
            pred_f = np.empty((n, 2, 28, 56), np.uint8)
            pred_b = np.empty_like(pred_f)
            values = {k: np.empty((n, 2), float) for k in METRICS}
            for j in range(2):
                for start in range(0, n, batch):
                    sl = slice(start, min(n, start + batch))
                    fg = torch.from_numpy(bench["ctx_fg"][sl]).long().to(dev)
                    bg = torch.from_numpy(bench["ctx_bg"][sl]).long().to(dev)
                    maps = torch.from_numpy(bench["act"][sl, j]).long().to(dev)
                    for t in range(32):
                        fg, bg = model.step(fg, bg, maps[:, t])
                    pred_f[sl, j], pred_b[sl, j] = fg.cpu().numpy().astype(np.uint8), bg.cpu().numpy().astype(np.uint8)
                print(f"[eval] {row['id']} {task} intervention {j+1}/2", flush=True)
            for i in range(n):
                for j in range(2):
                    sc = score((pred_f[i, j], pred_b[i, j]), (bench["target_fg"][i, j], bench["target_bg"][i, j]),
                               (bench["ctx_fg"][i], bench["ctx_bg"][i]))
                    for k in METRICS:
                        values[k][i, j] = sc[k]
            with (target / "samples.tmp").open("wb") as stream:
                np.savez_compressed(stream, **values, pred_fg=pred_f, pred_bg=pred_b,
                                    scene_ids=bench["scene_ids"], group=bench["group"])
            (target / "samples.tmp").replace(target / "samples.npz")
            write_json(target / "metrics.json", {"task": task, "n_scenes": n, "horizon": 32, "metrics": list(METRICS),
                "bench_sha256": file_hash(bp), "means": {k: float(v.mean()) for k, v in values.items()}})
            write_json(target / "complete.json", evaluation_identity(out, task))
            try:
                validate_eval(out, p, row, task)
            except Exception:
                (target / "complete.json").unlink(missing_ok=True)
                raise


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--job", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args(argv)
    if a.batch <= 0:
        ap.error("batch must be positive")
    p, _ = verify_run(a.run, full_data=False)
    row = next((r for r in load_json(a.run / "jobs.json") if r["id"] == a.job), None)
    if row is None:
        raise ValueError("job not in frozen run")
    evaluate_job(a.run, p, row, a.device, a.batch)


if __name__ == "__main__":
    main()
