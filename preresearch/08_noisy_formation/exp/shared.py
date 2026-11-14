"""Single structural benchmark, frozen experiment03 inputs and final EMA identities."""
from importlib import import_module
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
PILOT = "preresearch.01_pilot_validity.exp"
common = import_module(f"{PILOT}.common")
helpers = import_module("preresearch.04_ood_generalization.exp.shared")
scale = import_module("preresearch.06_data_scale.exp.shared")
write_json, resolve_file = helpers.write_json, helpers.resolve_file
EXP_NAME, VERSION = "08_noisy_formation", 1
DEFAULT_SOURCE = "20261111-044057_four_conditions"
VARIANTS = {"state": "state", "process": "process", "partial": "partial", "shuffle": "shuffle",
            "process_noisy_low": "process", "process_noisy_high": "process"}
NOISE = {"process": 0., "process_noisy_low": .1, "process_noisy_high": .3}
DEFAULTS = dict(scale.TRAIN_DEFAULTS)
METRICS = scale.METRICS


def integer(value, label, minimum=1):
    if isinstance(value, bool):
        raise ValueError(f"invalid {label}")
    try:
        result = int(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"invalid {label}") from exc
    if str(result) != str(value) or result < minimum:
        raise ValueError(f"invalid {label}")
    return result


def run_path(name):
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("run must be a simple name")
    return scale.check_output_tree(common.RESULTS_ROOT / EXP_NAME / name)


def code_identity():
    expected = {"core": PROJECT / "game_code/core.py", "datasets": PROJECT / "game_code/datasets.py",
                "datasets_v3": PROJECT / "game_code/datasets_v3.py"}
    for variable, path in (("FW_PROJECT", PROJECT), ("FW_GAME_CODE", PROJECT / "game_code")):
        import os
        if variable in os.environ and Path(os.environ[variable]).resolve() != path:
            raise ValueError("experiment08 must use this repository's project and physics")
    for module, path in expected.items():
        if Path(import_module(module).__file__).resolve() != path:
            raise ValueError("actual physics module differs from frozen source")
    dependencies = list(expected.values())
    dependencies += [PROJECT / f"preresearch/01_pilot_validity/exp/{n}.py" for n in ("common", "model", "train", "evaluate", "rollout", "metrics")]
    dependencies += [PROJECT / f"preresearch/{sub}/exp/{n}.py" for sub, n in
                     (("03_main", "audit"), ("03_main", "shared"), ("04_ood_generalization", "shared"),
                      ("04_ood_generalization", "statistics"), ("04_ood_generalization", "load"),
                      ("05_formation_ratio", "shared"), ("06_data_scale", "shared"),
                      ("06_data_scale", "make_jobs"), ("06_data_scale", "run_queue"))]
    own = list(Path(__file__).parent.glob("*.py"))
    return {common.rel(p): common.sha1_file(p) for p in sorted(own + dependencies)}


def source_bundle(value, episodes):
    integer(episodes, "episodes")
    if episodes > 81920:
        raise ValueError("episodes exceeds frozen experiment03 source")
    return scale.validate_source(value, (40960, 81920), (0,), 94400)


def file_set(root):
    root = Path(root)
    return {p.relative_to(root).as_posix(): common.sha1_file(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.name not in ("complete.json", "generation.lock", "pipeline.lock")}


def verify_data(path, full=True, require_qualified=True):
    path = scale.check_output_tree(path)
    manifest = common.load_json(path / "manifest.json")
    done = common.load_json(path / "complete.json")
    if done.get("manifest_sha1") != common.sha1_file(path / "manifest.json") or manifest["code"] != code_identity():
        raise ValueError("derived dataset manifest or code changed")
    bundle = source_bundle(manifest["source"], manifest["episodes"])
    if manifest["source_identity"] != bundle["manifest"]:
        raise ValueError("frozen experiment03 source changed")
    if full and file_set(path) != done["hashes"]:
        raise ValueError("derived dataset changed")
    if require_qualified and (not manifest.get("qualified") or not qualifies(manifest["audits"], manifest["max_fallback"])):
        raise ValueError("derived data failed noise acceptance; retained for diagnosis, not training")
    return manifest, bundle


def qualifies(audits, max_fallback=.1):
    fractions = [audits[v]["physical_transition_fraction"] for v in NOISE]
    return (all(audits[v]["fallback_rate"] < min(.1, max_fallback) for v in NOISE if v != "process")
            and fractions[0] < fractions[1] < fractions[2] and fractions[2] >= .3)


def jobs(name, p):
    rows = []
    for v, c in VARIANTS.items():
        ref = p.get("baseline_reuse", {}).get(v)
        rows.append({"id": f"{v}_seed0", "run": name, "variant": v, "condition": c, "seed": 0,
                     "episodes": p["episodes"], "steps": p["steps"], "data": p["data_paths"][v],
                     "out": ref["row"]["out"] if ref else common.rel(run_path(name) / "train" / f"{v}_seed0"),
                     "reuse": bool(ref)})
    return rows


def train_cli(row, p):
    args = {**p["train_defaults"], "condition": row["condition"], "episodes": row["episodes"],
            "steps": row["steps"], "batch": p["batch"], "seed": 0, "protocol": "video",
            "exp": EXP_NAME, "id": row["id"], "out": str(resolve_file(row["out"])),
            "data": str(resolve_file(row["data"])), "val": str(resolve_file(p["inputs"]["val"])), "log_every": 50}
    return [v for k, value in args.items() for v in ("--" + k.replace("_", "-"), str(value))] + ["--log-batch-composition"]


def expected_args(row, p):
    a = import_module(f"{PILOT}.train").get_args(train_cli(row, p))
    return {k: common.rel(v) if isinstance(v, Path) else v for k, v in vars(a).items()}


def validate_config(out, row, p):
    cfg = common.load_json(Path(out) / "config.json")
    if (cfg["args"] != expected_args(row, p) or cfg["data_sha1"] != p["training_hashes"][row["variant"]]
            or cfg["episodes_used"] != row["episodes"] or cfg["n_pairs"] != row["episodes"] * (59 if row["condition"] in ("process", "shuffle") else 56)):
        raise ValueError("training configuration/source/prefix changed")
    return cfg


def input_preflight(row, p):
    if common.data_hashes(resolve_file(row["data"]), row["condition"]) != p["training_hashes"][row["variant"]]:
        raise ValueError("training data changed")
    val_keys = ("train_state.npz", "train_state_actions.json", "train_summary.json")
    if common.data_hashes(resolve_file(p["inputs"]["val"]), "state") != {k: p["hashes"]["val"][k] for k in val_keys}:
        raise ValueError("validation data changed")


def train_identity(out, row, p):
    return {"version": VERSION, "job": row, "protocol_sha1": common.sha1_file(run_path(row["run"]) / "protocol.json"),
            "selection": "last", "weights": "ema", "hashes": {n: common.sha1_file(Path(out) / n) for n in ("config.json", "done.json", "ckpt.pt")}}


def validate_done(out, row, p):
    import torch
    out = Path(out)
    if row.get("reuse"):
        ref = p["baseline_reuse"][row["variant"]]
        audited = common.load_json(scale.run_path(ref["run"]) / "audit.json")
        if common.sha1_file(out / "train_complete.json") != ref["train_complete_sha1"]:
            raise ValueError("reused baseline changed")
        if audited["inputs"] != p["inputs"] or audited["hashes"] != p["hashes"]:
            raise ValueError("reused baseline frozen inputs differ")
        return scale.validate_done(out, ref["row"], audited)
    cfg = validate_config(out, row, p)
    done = common.load_json(out / "done.json")
    if common.load_json(out / "train_complete.json") != train_identity(out, row, p):
        raise ValueError("training identity changed")
    if (done["selection"] != "last" or done["steps"] != row["steps"] or done["epochs"] is not None
            or done["checkpoint_step"] != row["steps"] or done["weights"] != "ema"
            or done["samples_seen"] != row["steps"] * p["batch"] or done["n_pairs"] != cfg["n_pairs"]
            or done["ckpt_sha1"] != common.sha1_file(out / "ckpt.pt")):
        raise ValueError("training budget or final checkpoint differs")
    ck = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=True)
    model = import_module(f"{PILOT}.model").build_model({k: p["train_defaults"][k] for k in ("dim", "depth", "heads")})
    if ck["args"] != cfg["args"] or ck["cfg"] != model.cfg or ck["step"] != row["steps"] or ck.get("selection") != "last" or ck.get("weights") != "ema":
        raise ValueError("checkpoint is not this job's final EMA")
    for key in ("model", "ema"):
        if not all(torch.is_tensor(v) and torch.isfinite(v).all() for v in ck[key].values()):
            raise ValueError("invalid model weights")
        model.load_state_dict(ck[key], strict=True)
    return done


def run_identity(out):
    return {n: common.sha1_file(Path(out) / n) for n in ("protocol.json", "jobs.json")}


def prepare(name, dataset, episodes=81920, steps=94400, reuse_scale_run=None):
    n, steps = integer(episodes, "episodes"), integer(steps, "updates")
    dataset = scale.check_output_tree(dataset)
    m, bundle = verify_data(dataset)
    if n > m["episodes"]:
        raise ValueError("training prefix exceeds paired dataset")
    paths = {v: common.rel(bundle["inputs"]["data"] if v in common.CONDITIONS else Path(dataset) / v) for v in VARIANTS}
    p = {"version": VERSION, "task": "original_single_structural_intervention", "training_seeds": [0],
         "episodes": n, "prefix": {"start": 0, "stop": n, "step": 1}, "steps": steps, "batch": 256,
         "epochs": None, "sampling": "uniform_with_replacement", "pair_protocol": "video",
         "selection": "last", "weights": "ema", "train_defaults": dict(DEFAULTS),
         "dataset": common.rel(Path(dataset).absolute()), "dataset_complete_sha1": common.sha1_file(Path(dataset) / "complete.json"),
         "inputs": bundle["manifest"]["inputs"], "hashes": bundle["manifest"]["hashes"],
         "data_paths": paths, "code": code_identity(), "baseline_reuse": {},
         "training_hashes": {v: common.data_hashes(resolve_file(paths[v]), c) for v, c in VARIANTS.items()}}
    if reuse_scale_run:
        audited = scale.verify_audit(reuse_scale_run)
        sp = audited["protocol"]
        if (audited["inputs"] != p["inputs"] or audited["hashes"] != p["hashes"] or n not in sp["sizes"]
                or 0 not in sp["training_seeds"] or steps != sp["steps"] or sp["batch"] != p["batch"]
                or sp["train_defaults"] != p["train_defaults"]):
            raise ValueError("experiment06 baselines do not match the requested inputs/budget")
        scale_jobs = import_module("preresearch.06_data_scale.exp.make_jobs").jobs(reuse_scale_run, audited)
        for c in common.CONDITIONS:
            row = next(r for r in scale_jobs if (r["condition"], r["episodes"], r["seed"]) == (c, n, 0))
            out = resolve_file(row["out"])
            scale.validate_done(out, row, audited)
            p["baseline_reuse"][c] = {"run": reuse_scale_run, "row": row,
                                      "train_complete_sha1": common.sha1_file(out / "train_complete.json")}
    out = run_path(name)
    if out == Path(dataset).absolute() or out in Path(dataset).absolute().parents or Path(dataset).absolute() in out.parents:
        raise ValueError("run and paired data directories must be separate")
    with scale.run_lock(out):
        if (out / "protocol.json").exists():
            if common.load_json(out / "protocol.json") != p:
                raise ValueError("run design changed; use a new run name")
            if (out / "run_complete.json").exists():
                verify_run(name)
                return out
        elif set(x.name for x in out.iterdir()) - {"pipeline.lock"}:
            raise ValueError("unbound run directory is not empty")
        if (out / "jobs.json").exists() and common.load_json(out / "jobs.json") != jobs(name, p):
            raise ValueError("job list changed")
        common.start_run(EXP_NAME, name=name)
        write_json(out / "protocol.json", p)
        write_json(out / "jobs.json", jobs(name, p))
        write_json(out / "run_complete.json", run_identity(out))
    return out


def verify_run(name):
    out = run_path(name)
    p = common.load_json(out / "protocol.json")
    if p["training_seeds"] != [0]:
        raise ValueError("experiment08 is single-seed only")
    if common.load_json(out / "run_complete.json") != run_identity(out):
        raise ValueError("run manifest changed or was not committed")
    m, bundle = verify_data(resolve_file(p["dataset"]))
    n = integer(p["episodes"], "episodes")
    integer(p["steps"], "updates")
    if (p["code"] != code_identity() or p["dataset_complete_sha1"] != common.sha1_file(resolve_file(p["dataset"]) / "complete.json")
            or n > m["episodes"] or p["batch"] != 256 or p["train_defaults"] != DEFAULTS
            or p["inputs"] != bundle["manifest"]["inputs"] or p["hashes"] != bundle["manifest"]["hashes"]
            or p["prefix"] != {"start": 0, "stop": n, "step": 1}):
        raise ValueError("run code/dataset/budget changed")
    if common.load_json(out / "jobs.json") != jobs(name, p):
        raise ValueError("job list changed")
    return p, bundle


def evaluation_path(row):
    return run_path(row["run"]) / "evaluate" / row["id"]


def evaluation_identity(row, done):
    out = evaluation_path(row)
    return {"version": VERSION, "job": row, "checkpoint_sha1": done["ckpt_sha1"],
            "training_complete_sha1": common.sha1_file(resolve_file(row["out"]) / "train_complete.json"),
            "selection": "last", "weights": "ema", "checkpoint_step": done["checkpoint_step"],
            "hashes": {n: common.sha1_file(out / n) for n in ("eval.json", "eval_samples.npz")}}


def validate_eval(row, p, done):
    import numpy as np
    out = evaluation_path(row)
    ev = common.load_json(out / "eval.json")
    bp = resolve_file(p["inputs"]["bench"])
    if (ev.get("bench") != common.rel(bp) or ev.get("n_samples") != 512 or ev.get("variant") != row["variant"]
            or ev.get("bench_sha1") != p["hashes"]["bench"][bp.name]
            or ev.get("bench_json_sha1") != p["hashes"]["bench"][bp.with_suffix(".json").name]
            or ev.get("checkpoint_sha1") != done["ckpt_sha1"]):
        raise ValueError("evaluation differs from frozen bench or final checkpoint")
    marker = out / "eval_complete.json"
    if marker.exists() and common.load_json(marker) != evaluation_identity(row, done):
        raise ValueError("completed evaluation changed; refusing overwrite")
    bench = common.load_bench(bp)
    with np.load(out / "eval_samples.npz", allow_pickle=False) as z:
        samples = {k: z[k] for k in (*METRICS, "split_ids", "rank_valid", "n_cascade_true", "pred_fg", "pred_bg")}
    import_module("preresearch.05_formation_ratio.exp.shared").validate_metrics(samples, bench)
    return ev


def record_eval(row, p, done):
    validate_eval(row, p, done)
    path = evaluation_path(row) / "eval_complete.json"
    if not path.exists():
        write_json(path, evaluation_identity(row, done))
