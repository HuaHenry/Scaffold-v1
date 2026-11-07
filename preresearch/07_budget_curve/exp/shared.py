"""Use experiment 01's common data/metric and training interfaces."""
from importlib import import_module
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

PACKAGE = "preresearch.01_pilot_validity.exp"
common = import_module(f"{PACKAGE}.common")
aggregate = import_module(f"{PACKAGE}.aggregate")
metrics = import_module(f"{PACKAGE}.metrics")
EXP_NAME = os.environ.get("FW_CURVE_EXP", "07_budget_curve")
LEVELS = (16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512,
          768, 1024, 1536, 2048, 3072, 4096)
CONDITIONS = ("state", "process")
