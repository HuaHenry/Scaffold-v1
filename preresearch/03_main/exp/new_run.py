"""Create or resume an independent main-experiment run."""
import argparse
import os

from .shared import EXP_NAME, common


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", action="store_true")
    ap.add_argument("--label", default="")
    a = ap.parse_args(argv)
    if a.path:
        print(common.run_dir(EXP_NAME))
    elif os.environ.get("FW_RUN"):
        print(common.start_run(EXP_NAME, a.label, os.environ["FW_RUN"]).name)
    else:
        name = "_".join(part for part in (common.now_ts(), common.slug(a.label)) if part)
        i, original = 1, name
        while (common.RESULTS_ROOT / EXP_NAME / name).exists():
            i += 1
            name = f"{original}-{i}"
        print(common.start_run(EXP_NAME, a.label, name).name)


if __name__ == "__main__":
    main()
