"""Create a separate experiment 05 run, without data generation or training."""
import argparse
import os

from .main import fresh_name
from .shared import EXP_NAME, common, run_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="supervision_ratio")
    ap.add_argument("--path", action="store_true", help="print FW_RUN path without creating it")
    a = ap.parse_args(argv)
    name = os.environ.get("FW_RUN")
    if a.path:
        if not name:
            ap.error("set FW_RUN before --path")
        print(run_path(name))
        return
    name = name or fresh_name(a.label)
    run_path(name)
    print(common.start_run(EXP_NAME, a.label, name).name)


if __name__ == "__main__":
    main()
