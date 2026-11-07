"""Select a new isolated result folder for experiment 02."""
from __future__ import annotations

import argparse
import os

from .shared import EXP_NAME, common


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="")
    ap.add_argument("--path", action="store_true")
    a = ap.parse_args(argv)
    if a.path:
        print(common.run_dir(EXP_NAME))
    elif os.environ.get("FW_RUN"):
        print(common.start_run(EXP_NAME, a.label, os.environ["FW_RUN"]).name)
    else:
        base = "_".join(p for p in (common.now_ts(), common.slug(a.label)) if p)
        name, i = base, 1
        while (common.RESULTS_ROOT / EXP_NAME / name).exists():
            i += 1
            name = f"{base}-{i}"
        print(common.start_run(EXP_NAME, a.label, name).name)


if __name__ == "__main__":
    main()
