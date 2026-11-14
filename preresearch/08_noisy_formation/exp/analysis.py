"""Six single-seed conditions, paired scene intervals and a dose-response chart."""
import argparse
import csv
from html import escape
from importlib import import_module
from io import StringIO

import numpy as np

from .shared import (METRICS, NOISE, PILOT, VARIANTS, common, evaluation_path,
                     integer, jobs, resolve_file, run_path, scale, validate_done,
                     validate_eval, verify_run, write_json)

stats = import_module("preresearch.04_ood_generalization.exp.statistics")
metrics = import_module(f"{PILOT}.metrics")


def summarize(samples, bench, n_boot=10000, seed=12345):
    integer(n_boot, "bootstrap count")
    integer(seed, "bootstrap seed", 0)
    groups = {"all": np.ones(len(bench["split_ids"]), bool)}
    groups.update({s: bench["split_ids"] == i for i, s in enumerate(bench["splits"])})
    groups["dependency_proxy_n_changed_ge3"] = groups["all"]
    membership = np.asarray(bench["n_changed"]) >= 3
    result = {"training_seeds": [0], "uncertainty": "paired scene bootstrap only; not training-seed uncertainty",
              "dependency_subset": "intervention n_changed>=3 proxy, not an exact support-removal label",
              "n_boot": n_boot, "bootstrap_seed": seed, "groups": {}}
    for group, mask in groups.items():
        values = {}
        for v in VARIANTS:
            values[v] = {}
            for metric in METRICS:
                raw = np.asarray(samples[v][metric])
                if group.startswith("dependency_proxy"):
                    values[v][metric] = (stats.cluster_values(raw, membership) if raw.ndim == 2
                                          else np.where(membership.any(axis=1), raw, np.nan))
                else:
                    values[v][metric] = metrics.sample_level(samples[v], metric)
        estimates = {v: {m: stats.estimate(values[v][m][mask], n_boot, seed, (group, v, m)) for m in METRICS} for v in VARIANTS}
        contrasts = [(noise, clean) for noise in ("process_noisy_low", "process_noisy_high") for clean in common.CONDITIONS]
        contrasts.append(("process_noisy_high", "process_noisy_low"))
        differences = {f"{a}_minus_{b}": {m: stats.paired(values[a][m], values[b][m], mask, n_boot, seed, (group, a, b, m))
                                        for m in METRICS} for a, b in contrasts}
        dose = [{"variant": v, "nominal_p": p, **estimates[v]["cascade_f1"]} for v, p in NOISE.items()]
        means = [d["mean"] for d in dose]
        result["groups"][group] = {"estimates": estimates, "paired_differences": differences,
                                    "dose_response": dose,
                                    "cascade_f1_strictly_increasing": all(a < b for a, b in zip(means, means[1:])) if all(x is not None for x in means) else None}
    return result


def dose_svg(result):
    groups = list(result["groups"])
    width, panel_w, height = 280 * len(groups), 280, 280
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="sans-serif" font-size="12" fill="#263238">']
    for i, group in enumerate(groups):
        ox = i * panel_w
        lines.append(f'<text x="{ox+12}" y="20">{escape(group)}</text>')
        lines.append(f'<path d="M {ox+40},40 V 230 H {ox+255}" fill="none" stroke="#607d8b"/>')
        points = []
        for d in result["groups"][group]["dose_response"]:
            x = ox + 40 + d["nominal_p"] / .3 * 200
            lines.append(f'<text x="{x-8}" y="250">{d["nominal_p"]:g}</text>')
            if d["mean"] is None:
                continue
            y = 230 - 180 * d["mean"]
            points.append(f"{x},{y}")
            if d["ci"]:
                lo, hi = d["ci"]
                lines.append(f'<path d="M {x},{230-180*lo} V {230-180*hi}" stroke="#1565c0"/>')
            lines.append(f'<circle cx="{x}" cy="{y}" r="4" fill="#1565c0"/>')
        lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#1565c0"/>')
        lines.append(f'<text x="{ox+45}" y="270">nominal p; cascade F1 (0..1)</text>')
    lines += ["</g>", "</svg>"]
    return "\n".join(lines)


def analyze(name, n_boot=10000, seed=12345):
    with scale.run_lock(run_path(name)):
        p, _ = verify_run(name)
        bench = common.load_bench(resolve_file(p["inputs"]["bench"]))
        samples, identities = {}, {}
        for row in jobs(name, p):
            done = validate_done(resolve_file(row["out"]), row, p)
            validate_eval(row, p, done)
            out = evaluation_path(row)
            identities[row["variant"]] = common.sha1_file(out / "eval_complete.json")
            with np.load(out / "eval_samples.npz", allow_pickle=False) as z:
                samples[row["variant"]] = {m: z[m] for m in METRICS}
        identity = {"evaluations": identities, "protocol_sha1": common.sha1_file(run_path(name) / "protocol.json"),
                    "n_boot": n_boot, "bootstrap_seed": seed}
        out = run_path(name) / "analysis"
        if (out / "complete.json").exists():
            recorded = common.load_json(out / "complete.json")
            hashes = {n: common.sha1_file(out / n) for n in ("results.json", "table.csv", "dose_response.svg")}
            if recorded != {"identity": identity, "hashes": hashes}:
                raise ValueError("analysis identity/parameters changed; refusing overwrite")
            return common.load_json(out / "results.json")
        result = summarize(samples, bench, n_boot, seed)
        result["input_identity"] = identity
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "results.json", result)
        stream = StringIO()
        writer = csv.writer(stream)
        writer.writerow(("group", "variant", "metric", "mean", "ci_low", "ci_high", "n_scenes"))
        for group, data in result["groups"].items():
            for v, estimates in data["estimates"].items():
                for metric, estimate in estimates.items():
                    writer.writerow((group, v, metric, estimate["mean"], *(estimate["ci"] or (None, None)), estimate["n_cases"]))
        (out / "table.csv").write_text(stream.getvalue())
        (out / "dose_response.svg").write_text(dose_svg(result))
        write_json(out / "complete.json", {"identity": identity,
                   "hashes": {n: common.sha1_file(out / n) for n in ("results.json", "table.csv", "dose_response.svg")}})
        print(f"[08] analysis complete: {out}", flush=True)
        return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--bootstrap-seed", type=int, default=12345)
    a = ap.parse_args(argv)
    analyze(a.run, a.bootstrap, a.bootstrap_seed)


if __name__ == "__main__":
    main()
