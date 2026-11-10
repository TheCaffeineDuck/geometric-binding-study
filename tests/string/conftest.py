import json
import os

import numpy as np
import pytest

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "string", "checks")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long run (minutes); deselect with -m 'not slow'")


def _default(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


@pytest.fixture
def record():
    """record(name, dict): write measured check values to outputs/string/checks/<name>.json."""
    def _record(name, data):
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, name + ".json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=1, default=_default)
        os.replace(tmp, path)
    return _record
