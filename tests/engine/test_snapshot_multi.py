"""A list of snapshot entries (one per source; WP-P3 unequal pairs) is the sum of the entries placed one by one,
and a one-entry list equals the original dict form bit for bit. Uses the Session 2 test source (N = 80, L = 64);
skipped when it is not on disk (it is never committed)."""
import os

import numpy as np
import pytest

from swirl.engine.initial import build_initial
from swirl.engine.lumps import check_lumps
from swirl.engine.snapshot import load_meta

REL = "outputs/session2/wpE/source/src_test_N80_L64.npz"
HERE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CANDIDATES = [os.path.join(HERE_ROOT, REL), os.path.join(HERE_ROOT, "..", "..", "..", REL)]
SRC = next((os.path.abspath(p) for p in CANDIDATES if os.path.exists(p)), None)


@pytest.mark.skipif(SRC is None, reason="test source not on disk")
def test_snapshot_list_is_sum_of_entries():
    meta = load_meta(SRC)
    N, L = meta["params"]["N"], meta["params"]["L"]
    coords = (np.arange(N) - N // 2) * (L / N)
    a = {"source": SRC, "centers": [[-6.0, 0.0, 0.0]], "phases": [0.0], "window": [14.0, 18.0], "margin": 2.0}
    b = {"source": SRC, "centers": [[6.0, 0.4, 0.0]], "phases": [1.3], "window": [14.0, 18.0], "margin": 2.0}
    pa, va = build_initial(coords, L, {"snapshot": a}, threads=2)
    pb, vb = build_initial(coords, L, {"snapshot": b}, threads=2)
    info = {}
    p1, v1 = build_initial(coords, L, {"snapshot": [a]}, threads=2)
    p2, v2 = build_initial(coords, L, {"snapshot": [a, b]}, info=info, threads=2)
    assert np.array_equal(p1, pa) and np.array_equal(v1, va)
    assert np.max(np.abs(p2 - (pa + pb))) < 1e-13 and np.max(np.abs(v2 - (va + vb))) < 1e-13
    assert isinstance(info["snapshot"], list) and len(info["snapshot"]) == 2
    g = check_lumps({"snapshot": [a, b]}, L, None)
    assert len(g) == 2
