"""Start (or select) the big result folder of one execution of the experiment.

export FW_RUN=$(python -m exp.new_run [--label smoke])    # new run: <results>/01_pilot_validity/<YYYYMMDD-HHMMSS>[_label]/
python -m exp.new_run --path                              # absolute path of the current run (FW_RUN, else the newest)

Everything one execution produces (checks, tuning, trainings, baselines, analysis, job lists) goes into that folder, so
a re-run is simply a new folder and never mixes with old results. Resume = export the old FW_RUN again.
"""
from __future__ import annotations

import argparse
import os

from . import common


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default=common.EXP_NAME)
    ap.add_argument("--label", default="", help="suffix of the folder name, e.g. smoke / seed5")
    ap.add_argument("--path", action="store_true", help="print the path of the current run instead of creating one")
    a = ap.parse_args(argv)
    if a.path:
        d = common.run_dir(a.exp, create=True)
        print(d)
        return
    if os.environ.get("FW_RUN"):                      # already chosen: make sure it exists, print it again
        print(common.start_run(a.exp, a.label, os.environ["FW_RUN"]).name)
        return
    base = "_".join(p for p in (common.now_ts(), common.slug(a.label)) if p)
    name, k = base, 1
    while (common.RESULTS_ROOT / a.exp / name).exists():      # never reuse an existing run folder
        k += 1
        name = f"{base}-{k}"
    print(common.start_run(a.exp, a.label, name).name)


if __name__ == "__main__":
    main()
