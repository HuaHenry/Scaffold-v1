"""Run experiment 04 without training; optional probes require explicit opt-in."""
import argparse
import os

from .analysis import run_analysis
from .evaluate import run_evaluations
from .load import load_source
from .probe_analysis import summarize_probes
from .probes import prepare_probe
from .shared import bind_source, common, forbid_source_output, output_path, start_run

DEFAULT_SOURCE = "20261111-044057_four_conditions"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-run", default=DEFAULT_SOURCE)
    ap.add_argument("--run", default=os.environ.get("FW_RUN"))
    ap.add_argument("--tag", default="ood_analysis")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345, help="bootstrap seed, not training seed")
    ap.add_argument("--dry-run", action="store_true", help="read and check sources; do not create result files")
    ap.add_argument("--with-probes", action="store_true", help="also create separate evaluation-only scale and component probes")
    ap.add_argument("--probe", choices=("all", "scale", "components"), default="all")
    ap.add_argument("--per-setting", type=int, default=32)
    ap.add_argument("--probe-seed", type=int, default=2027)
    ap.add_argument("--max-attempts", type=int, default=512)
    ap.add_argument("--gpus", default=os.environ.get("GPUS", "0,1,2,3"))
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args(argv)
    if min(a.boot, a.per_setting, a.max_attempts, a.batch) <= 0 or min(a.seed, a.probe_seed) < 0:
        ap.error("counts must be positive and random seeds nonnegative")
    if a.run:
        output_path(a.run)
    cards = [c.strip() for c in a.gpus.split(",")]
    if not cards or any(not c for c in cards) or len(set(cards)) != len(cards):
        ap.error("--gpus requires distinct nonempty GPU IDs or cpu")
    bundle = load_source(a.source_run)
    if a.run:
        out = output_path(a.run)
        forbid_source_output(out, bundle["source"], bundle["audited"])
        if (out / "source.json").exists() and common.load_json(out / "source.json") != bundle["manifest"]:
            raise ValueError("existing experiment 04 run has different source files")
    print("[04] four conditions, seed 0; reference id is hard transfer, not training ID", flush=True)
    if a.dry_run:
        print("[04] dry-run complete: no files written, no data generated, no model evaluated or trained", flush=True)
        return
    run = a.run or start_run(a.tag)
    out = output_path(run)
    if not (out / "run.json").exists():
        common.start_run("04_ood_generalization", a.tag, run)
    bind_source(out, bundle["manifest"])
    result = run_analysis(bundle, run, a.boot, a.seed)
    if a.with_probes:
        names = ("scale", "components") if a.probe == "all" else (a.probe,)
        for name in names:
            prepare_probe(bundle, out, name, a.per_setting, a.probe_seed, a.max_attempts)
        run_evaluations(run, names, a.gpus, a.batch)
        summarize_probes(out, result, names, a.batch, a.boot, a.seed)
    print(f"[04] complete: {common.rel(result)}", flush=True)


if __name__ == "__main__":
    main()
