"""Q1: local-rule coverage of true FormBench transitions; CPU only.

python -m exp.analysis.coverage --run 20261105-... [--data DIR] [--bench PATH]
"""
from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..shared import EXP_NAME, common


def window_key(fg, bg, action, next_fg, next_bg, y, x, radius):
    """Stable 64-bit key, padded with 255 outside the grid."""
    size = 2 * radius + 1
    block = np.stack((fg, bg, action), axis=-1).astype(np.uint8)
    padded = np.pad(block, ((radius, radius), (radius, radius), (0, 0)), constant_values=255)
    payload = padded[y:y + size, x:x + size].tobytes() + bytes((int(next_fg[y, x]), int(next_bg[y, x])))
    return hashlib.blake2b(payload, digest_size=8).digest()


def changed_keys(fg, bg, action, next_fg, next_bg, radii=(1, 2)):
    """Yield (y,x,local_action,3x3 key,5x5 key) for cells changing in either layer."""
    yy, xx = np.nonzero((fg != next_fg) | (bg != next_bg))
    if not len(yy):
        return
    block = np.stack((fg, bg, action), axis=-1).astype(np.uint8)
    pads = {r: np.pad(block, ((r, r), (r, r), (0, 0)), constant_values=255) for r in radii}
    for y, x in zip(yy, xx):
        keys = []
        tail = bytes((int(next_fg[y, x]), int(next_bg[y, x])))
        for r in radii:
            size = 2 * r + 1
            keys.append(hashlib.blake2b(pads[r][y:y + size, x:x + size].tobytes() + tail,
                                        digest_size=8).digest())
        yield int(y), int(x), bool(action[y, x]), keys


def training_keys(data_dir):
    train = {r: {} for r in (1, 2)}
    phases = {r: {"formation": set(), "operation": set()} for r in (1, 2)}
    counts = {}
    for condition in common.CONDITIONS:
        d = common.load_condition(data_dir, condition)
        pairs = common.make_pairs(data_dir, condition, "video")
        keys = {r: set() for r in (1, 2)}
        for e, t in pairs:
            e, t = int(e), int(t)
            act = common.encode_actions(d["actions"][e][t])
            group = "formation" if d["phase"][e, t] == 0 else "operation"
            for _, _, _, row in changed_keys(d["fg"][e, t-1], d["bg"][e, t-1], act,
                                               d["fg"][e, t], d["bg"][e, t]):
                for r, key in zip((1, 2), row):
                    keys[r].add(key)
                    if condition == "process":
                        phases[r][group].add(key)
        counts[condition] = len(pairs)
        for r in (1, 2):
            train[r][condition] = keys[r]
    return train, phases, counts


def bench_keys(bench):
    rows = {r: defaultdict(list) for r in (1, 2)}
    for i, split_id in enumerate(bench["split_ids"]):
        sp = bench["splits"][int(split_id)]
        for k in range(2):
            prev_fg, prev_bg = bench["ctx_fg"][i], bench["ctx_bg"][i]
            mask = bench["A"][i, k]
            acts = common.iv_action_maps(bench["ivs"][i][k], common.HORIZON)
            for t in range(common.HORIZON):
                fg, bg, act = bench["fut_fg"][i, k, t], bench["fut_bg"][i, k, t], acts[t]
                step = "first" if t == 0 else "later"
                noop = not act.any()
                for y, x, _, keys in changed_keys(prev_fg, prev_bg, act, fg, bg):
                    for r, key in zip((1, 2), keys):
                        rows[r][(sp, step, "all")].append(key)
                        if not mask[y, x]:
                            rows[r][(sp, step, "cascade")].append(key)
                        if r == 1 and noop:
                            rows[r][(sp, step, "noop_physics")].append(key)
                prev_fg, prev_bg = fg, bg
    return rows


def summarize(rows, train, phases):
    out = {}
    for radius in (1, 2):
        by_split = {}
        cascade_all = []
        for sp in common.SPLITS:
            by_split[sp] = {}
            for step in ("first", "later"):
                by_split[sp][step] = {}
                for subset in (("all", "cascade", "noop_physics") if radius == 1 else ("all", "cascade")):
                    keys = rows[radius].get((sp, step, subset), ())
                    if subset == "cascade":
                        cascade_all.extend(keys)
                    by_split[sp][step][subset] = {
                        "n_changes": len(keys),
                        "coverage": {c: (sum(key in train[radius][c] for key in keys) / len(keys) if keys else None)
                                     for c in common.CONDITIONS},
                    }
        def coverage(keys, c):
            return sum(k in train[radius][c] for k in keys) / len(keys) if keys else None
        total = {c: coverage(cascade_all, c) for c in common.CONDITIONS}
        added = phases[radius]["formation"] - phases[radius]["operation"]
        out[str(radius)] = {
            "window": f"{2*radius+1}x{2*radius+1}", "training_unique_keys": {c: len(train[radius][c]) for c in common.CONDITIONS},
            "process_formation_unique": len(phases[radius]["formation"]),
            "process_operation_unique": len(phases[radius]["operation"]),
            "formation_only_unique": len(added),
            "formation_only_bench_cascade_fraction": (sum(k in added for k in cascade_all) / len(cascade_all)
                                                       if cascade_all else None),
            "by_split": by_split,
            "cascade_all": ({"n_changes": len(cascade_all), "coverage": total,
                             "delta_vs_state": {c: total[c] - total["state"] for c in common.CONDITIONS[1:]}}
                            if cascade_all else None),
        }
    return out


def table(result):
    lines = ["# Appendix table A2 — local-rule coverage (experiment 02)", "",
             "Each cell is the fraction of changed-cell occurrences whose deterministic local key appeared in training.",
             "Cascade excludes all intervention cells; noop_physics uses steps with an empty action map.", ""]
    for r, report in result["by_radius"].items():
        lines += [f"## {report['window']}", "", "| split | step | subset | n | state | process | partial | shuffle |",
                  "|---|---|---|---:|---:|---:|---:|---:|"]
        for sp, steps in report["by_split"].items():
            for step, subsets in steps.items():
                for subset, data in subsets.items():
                    def fmt(c):
                        x = data["coverage"][c]
                        return f"{x:.3f}" if x is not None else "n/a"
                    lines.append(f"| {sp} | {step} | {subset} | {data['n_changes']} | " +
                                 " | ".join(fmt(c) for c in common.CONDITIONS) + " |")
        lines += ["", "Training unique keys: " + ", ".join(f"{c}={n}" for c, n in report["training_unique_keys"].items()),
                  f"Formation-only unique keys: {report['formation_only_unique']}", ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=common.DEFAULT_DATA)
    ap.add_argument("--bench", type=Path, default=common.DEFAULT_BENCH)
    ap.add_argument("--run", default=None, help="experiment 02 run name; defaults to FW_RUN or latest")
    ap.add_argument("--out", type=Path, default=None, help="new result folder")
    a = ap.parse_args(argv)
    data, bench_path = common.resolve(a.data), common.resolve(a.bench)
    train, phases, pairs = training_keys(data)
    bench = common.load_bench(bench_path)
    report = {"by_radius": summarize(bench_keys(bench), train, phases), "train_pairs": pairs,
              "inputs": {"data": common.rel(data), "bench": common.rel(bench_path)}}
    run = a.run or common.current_run(EXP_NAME)
    out = common.resolve_out(a.out) if a.out else common.new_result_dir(EXP_NAME, "analysis", "pilot_coverage",
                                                                          f"bench-{common.bench_label(bench_path)}", run=run)
    out.mkdir(parents=True, exist_ok=True)
    common.write_meta(out, "analysis", "pilot_coverage", {"bench": common.bench_label(bench_path)}, inputs=report["inputs"])
    common.dump_json(report, out / "pilot_coverage.json")
    (out / "table_A2.md").write_text(table(report), encoding="utf-8")
    print(common.rel(out))


if __name__ == "__main__":
    main()
