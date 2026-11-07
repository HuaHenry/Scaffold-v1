"""Reuse the tested training/evaluation interfaces of pre-experiment 01."""
from importlib import import_module
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

PACKAGE = "preresearch.01_pilot_validity.exp"
common = import_module(f"{PACKAGE}.common")
aggregate = import_module(f"{PACKAGE}.aggregate")
evaluate = import_module(f"{PACKAGE}.evaluate")
metrics = import_module(f"{PACKAGE}.metrics")

EXP_NAME = "02_pilot_signal"
