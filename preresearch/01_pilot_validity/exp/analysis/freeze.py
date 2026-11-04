"""One-off hyper-parameter selection (plan 00 section 13, item 7): state condition, validation set only.

python -m exp.make_jobs tune && python -m exp.run_queue --jobs <run>/jobs/tune_steps6000.json --gpus 0,1,2,3
python -m exp.analysis.freeze            # writes exp/configs/frozen.json + a record folder <run>/tune/ (current run only)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import common


def last_val(run: Path):
    log = run / "train_log.jsonl"
    vals = [json.loads(l) for l in log.read_text().splitlines() if '"type": "val"' in l] if log.exists() else []
    return vals[-1] if vals else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=6000, help="only tuning runs trained with this many steps")
    ap.add_argument("--out", type=Path, default=common.FROZEN, help="frozen config to write")
    a = ap.parse_args(argv)
    newest = {}                                           # one entry per (lr, cw, depth): the newest finished run
    for d, m in common.iter_results(common.EXP_NAME, common.current_run(common.EXP_NAME, create=False)):
        if m["kind"] == "train" and m["id"].startswith("tune_") and (d / "done.json").exists() \
                and m["settings"].get("steps") == a.steps:
            v = last_val(d)
            if v:
                s = m["settings"]
                newest[(s["lr"], s["change_weight"], s["depth"])] = {
                    "run": d.name, "lr": s["lr"], "change_weight": s["change_weight"], "depth": s["depth"],
                    "val_cascade_acc": v["val_cascade_acc"], "val_chg_acc": v["val_chg_acc"]}
    rows = list(newest.values())
    if not rows:
        raise SystemExit("no finished tuning runs with validation records")
    best = max(rows, key=lambda r: (r["val_cascade_acc"], r["val_chg_acc"]))
    old = common.load_json(a.out) if Path(a.out).exists() else {}
    out = {"lr": best["lr"], "change_weight": best["change_weight"], "depth": best["depth"],
           "unroll": old.get("unroll", 1),
           "selected_by": "validation cascade-cell accuracy of the state model (bench not used)",
           "selected_run": best["run"], "n_candidates": len(rows), "table": rows}
    common.dump_json(out, a.out)
    rec = common.new_result_dir(common.EXP_NAME, "freeze", "tune", f"steps{a.steps}_best-lr{best['lr']:g}"
                                f"-cw{best['change_weight']:g}-d{best['depth']}")
    common.write_meta(rec, "freeze", "tune", {"steps": a.steps, "best": best})
    common.dump_json(out, rec / "frozen.json")
    for r in sorted(rows, key=lambda r: -r["val_cascade_acc"]):
        print(f"{r['run'][:70]:70s} val_cascade_acc {r['val_cascade_acc']:.4f}  val_chg_acc {r['val_chg_acc']:.4f}")
    print(f"\nfrozen -> {common.rel(a.out)}: lr={best['lr']} change_weight={best['change_weight']} depth={best['depth']}")
    print(f"record: {common.rel(rec)}")


if __name__ == "__main__":
    main()
