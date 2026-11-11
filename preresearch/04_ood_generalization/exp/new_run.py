"""Create an independent experiment 04 run or print its path."""
import argparse
import os

from .shared import output_path, start_run


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="")
    ap.add_argument("--path", action="store_true")
    a = ap.parse_args(argv)
    if a.path:
        run = os.environ.get("FW_RUN")
        if not run:
            ap.error("set FW_RUN before --path")
        print(output_path(run))
    else:
        print(start_run(a.label))


if __name__ == "__main__":
    main()
