"""Reuse experiment 01's GPU queue and model evaluation."""
from importlib import import_module

from .shared import PACKAGE


def main(argv=None):
    return import_module(f"{PACKAGE}.run_queue").main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
