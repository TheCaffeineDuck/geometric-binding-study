import json
import os

import numpy as np
import pytest

from swirl.arrange.pairlaw import ROOT, load_params, make_laws

WPD = os.path.join(ROOT, "outputs", "session2", "wpD")
WPC_JSON = os.path.join(ROOT, "outputs", "session2", "wpC", "pair_law_predictions.json")


@pytest.fixture(scope="session")
def params():
    return load_params(os.path.join(WPD, "params_default.json"))


@pytest.fixture(scope="session")
def fsup():
    path = os.path.join(WPD, "fsup_table.json")
    t = None
    if os.path.exists(path):
        with open(path) as f:
            t = json.load(f)
    if t is None or "overlap_true" not in t:
        from swirl.arrange.pairlaw import build_fsup_table
        from swirl.arrange.scan import dump
        dump(build_fsup_table(), path)
        with open(path) as f:
            t = json.load(f)
    return t


@pytest.fixture(scope="session")
def laws(params, fsup):
    return make_laws(params, fsup)


@pytest.fixture(scope="session")
def wpc():
    with open(WPC_JSON) as f:
        return json.load(f)


@pytest.fixture
def record():
    """record(name, dict): measured check values to outputs/session2/wpD/checks/<name>.json."""
    def _record(name, data):
        d = os.path.join(WPD, "checks")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, name + ".json")
        with open(path + ".tmp", "w") as f:
            json.dump(data, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
        os.replace(path + ".tmp", path)
    return _record
