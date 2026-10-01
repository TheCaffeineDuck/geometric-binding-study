import json
import os

import pytest

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "engine")


@pytest.fixture
def record():
    """record(name, dict): write measured check values to outputs/engine/<name>.json."""
    def _record(name, data):
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, name + ".json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, path)
    return _record
