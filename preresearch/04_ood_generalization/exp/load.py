"""Read and validate paired source predictions and true scene properties."""
from importlib import import_module
from itertools import combinations

import numpy as np

from .shared import CONDITIONS, METRICS, MAIN_PACKAGE, VERSION, common, metrics, resolve_file, source_path
import datasets as D
import datasets_v3 as v3


def validate_samples(samples, bench):
    n = len(bench["split_ids"])
    for key in ("split_ids", "rank_valid"):
        if not np.array_equal(samples[key], bench[key]):
            raise ValueError(f"source evaluation {key} differs from benchmark row order")
    if not np.array_equal(samples["n_cascade_true"], bench["cascade"]):
        raise ValueError("source n_cascade_true is not aligned with benchmark interventions")
    for key in ("cascade_f1", "cell_acc_mask"):
        values = samples[key]
        if values.shape != (n, 2) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError(f"invalid source metric {key}")
    if samples["rank_acc"].shape != (n,):
        raise ValueError("invalid rank metric shape")
    if not np.isfinite(samples["rank_acc"][bench["rank_valid"]]).all():
        raise ValueError("valid ranking cases contain missing scores")
    if samples["pred_fg"].shape != (n, 2) + bench["ctx_fg"].shape[1:] or samples["pred_bg"].shape != samples["pred_fg"].shape:
        raise ValueError("prediction shape differs from benchmark")
    for i in range(n):
        initial = (bench["ctx_fg"][i], bench["ctx_bg"][i])
        rates = []
        for k in range(2):
            predicted = (samples["pred_fg"][i, k], samples["pred_bg"][i, k])
            target = (bench["fut_fg"][i, k, -1], bench["fut_bg"][i, k, -1])
            score = metrics.score_sample(predicted, target, initial, bench["A"][i, k], False)
            for key in ("cascade_f1", "cell_acc_mask"):
                if not np.isclose(score[key], samples[key][i, k], atol=1e-10):
                    raise ValueError(f"cached {key} cannot be reproduced on benchmark row {i}, intervention {k}")
            rates.append(metrics.change_rate(initial, predicted))
        if bench["rank_valid"][i]:
            expected = metrics.rank_acc(*rates, *bench["labels"][i])
            if not np.isclose(expected, samples["rank_acc"][i]):
                raise ValueError("cached ranking differs from predicted states")


def scene_features(bench, training):
    compositions = {tuple(sorted(row["kinds"])) for row in training}
    train_kinds = set().union(*(set(row["kinds"]) for row in training))
    ranges = {}
    for kind, field in (("tower", "h"), ("bridge", "span")):
        values = [int(spec[field]) for row in training for spec in row["specs"] if spec["kind"] == kind]
        ranges[kind] = {"parameter": field, "observed_min": min(values) if values else None,
                        "observed_max": max(values) if values else None,
                        "generator_base_range": [4, 7] if kind == "tower" else [4, 8]}
    rows = []
    for i, scene in enumerate(bench["meta"]["scenes"]):
        kinds, specs, starts = scene["kinds"], scene["specs"], scene["x0s"]
        if not (len(kinds) == len(specs) == len(starts) == len(scene["mirrored"])):
            raise ValueError("incomplete scene metadata")
        sizes = {}
        for kind, field in (("tower", "h"), ("bridge", "span")):
            values = [int(s[field]) for s in specs if s["kind"] == kind]
            if values:
                sizes[kind] = max(values)
        heights = []
        for spec, x in zip(specs, starts):
            occupied = (bench["ctx_fg"][i, :D.SURF, x:x + spec["width"]] != 0) | (bench["ctx_bg"][i, :D.SURF, x:x + spec["width"]] != 0)
            yy = np.nonzero(occupied)[0]
            heights.append(int(D.SURF - yy.min()) if len(yy) else 0)
        rows.append({"index": i, "split": bench["splits"][int(bench["split_ids"][i])],
                     "final_hash": v3.state_hash(bench["ctx_fg"][i], bench["ctx_bg"][i]),
                     "kinds": kinds, "n_struct": len(kinds), "composition": "+".join(sorted(kinds)),
                     "composition_seen_in_training": tuple(sorted(kinds)) in compositions,
                     "contains_unseen_kind": bool(set(kinds) - train_kinds),
                     "held_pair_present": any(frozenset(p) in v3.HELD for p in combinations(kinds, 2)),
                     "max_parameter": sizes, "max_occupied_height": max(heights),
                     "intervention_kinds": [iv["kind"] for iv in bench["ivs"][i]]})
    return rows, ranges


def load_source(value):
    source = source_path(value)
    print(f"[04] checking frozen experiment 03 inputs: {common.rel(source)}", flush=True)
    group, split_ids, sources, audited = import_module(f"{MAIN_PACKAGE}.analysis").load_results(source)
    bench = common.load_bench(common.resolve(audited["inputs"]["bench"]))
    training = common.load_json(common.resolve(audited["inputs"]["data"]) / "train_scenes.json")
    raw, identity = {}, {}
    for c in CONDITIONS:
        path = resolve_file(sources[c])
        with np.load(path / "eval_samples.npz", allow_pickle=False) as z:
            raw[c] = {key: z[key].copy() for key in (*METRICS, "n_cascade_true", "rank_valid", "split_ids", "pred_fg", "pred_bg")}
        validate_samples(raw[c], bench)
        identity[c] = {name: common.sha1_file(path / name) for name in
                       ("config.json", "done.json", "ckpt.pt", "eval.json", "eval_samples.npz")}
        print(f"[04] {c}: paired predictions and identities checked", flush=True)
    features, ranges = scene_features(bench, training)
    manifest = {"version": VERSION, "source_run": common.rel(source), "sources": sources,
                "source_audit_sha1": common.sha1_file(source / "audit.json"),
                "inputs": audited["inputs"], "hashes": audited["hashes"], "models": identity,
                "training_seeds": [0], "reference": "id", "reference_is_training_id": False}
    return {"source": source, "group": group, "raw": raw, "bench": bench, "training": training,
            "features": features, "ranges": ranges, "manifest": manifest, "audited": audited}
