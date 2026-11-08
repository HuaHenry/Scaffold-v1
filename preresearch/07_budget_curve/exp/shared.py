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
if EXP_NAME in ("07_03_budget_epochs", "07_03_simple_to_complex"):
    LEVELS += (6144, 8192, 12288, 16384, 20480)
if EXP_NAME in ("07_03_simple_to_complex", "07_03_simple_to_complex_81920"):
    LEVELS += (28672, 32768, 40960)
if EXP_NAME == "07_03_simple_to_complex_81920":
    LEVELS += (49152, 57344, 65536, 73728, 81920)
CONDITIONS = ("state", "process")
