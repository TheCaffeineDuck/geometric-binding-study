"""Shared helpers for the Session 3 diagnostics (configuration-level drivers around swirl.family.wpl)."""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
from swirl.family import wpl  # noqa: E402

RUNS = os.path.join(ROOT, "outputs", "session3", "wpL", "runs")


def load_ctrl(start, T=2000.0, scale=1.0):
    return json.load(open(os.path.join(RUNS, wpl.spec_name(dict(start=start, kind="none", T=T, scale=scale)) + ".json")))


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)
