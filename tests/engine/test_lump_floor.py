"""WP0 (Session 3, F-018): the absolute floor of the lump finder. Debris on a nearly empty field is not a lump.

Synthetic densities test find_lumps and refloor; one test recounts the saved Session 2 runs (E3, E7, the E3
checks, E1, E2, E14 and a lone lump) from their stored count windows (skipped if the external drive is absent)."""

import copy
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from swirl.engine.lumps import LUMP_DEFAULTS, LumpDiagnostics, find_lumps, refloor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
N, L = 48, 24.0
DX = L / N
COORDS = -L / 2 + DX * np.arange(N)
X, Y, Z = np.meshgrid(COORDS, COORDS, COORDS, indexing="ij")


def blob(c, amp, w=1.5):
    return amp * np.exp(-((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2) / (2 * w ** 2))


def test_relative_threshold_counts_debris_floor_does_not():
    """A field that has emptied: a faint remnant (0.4% of a lump) plus smaller bumps. 1% of the current
    maximum is below all of them, so all are counted; an absolute floor of 5% of a lump is not."""
    lump = 16.0
    e = blob((0, 0, 0), 0.004 * lump) + blob((6, 0, 0), 0.0025 * lump) + blob((-6, 5, 0), 0.0022 * lump) \
        + blob((0, -7, 3), 0.002 * lump)
    old = find_lumps(e, COORDS, L, DX)
    assert old["count"] == 4 and old["abs_floor"] == 0.0
    new = find_lumps(e, COORDS, L, DX, abs_floor=0.05 * lump)
    assert new["count"] == 0 and new["count_distinct"] == 0 and new["abs_floor"] == 0.05 * lump


def test_floor_keeps_real_lumps_and_leaves_the_rest_alone():
    lump = 16.0
    e = blob((-6, 0, 0), lump) + blob((6, 0, 0), 0.6 * lump) + blob((0, 7, 0), 0.01 * lump)
    assert find_lumps(e, COORDS, L, DX)["count"] == 3
    assert find_lumps(e, COORDS, L, DX, abs_floor=0.05 * lump)["count"] == 2
    # the floor never admits what the relative threshold excludes
    r = find_lumps(e, COORDS, L, DX, threshold=0.5, abs_floor=0.0)
    assert r["count"] == 2


def test_refloor_matches_find_lumps_with_the_floor():
    lump = 16.0
    e = blob((-3, 0, 0), lump) + blob((3, 0, 0), 0.8 * lump) + blob((0, 7, 0), 0.03 * lump) + blob((0, -6, 2), 0.02 * lump)
    entry = find_lumps(e, COORDS, L, DX)
    for floor in (0.0, 0.01 * lump, 0.05 * lump, 0.9 * lump):
        a = refloor(entry, floor)
        b = find_lumps(e, COORDS, L, DX, abs_floor=floor)
        assert (a["count"], a["count_distinct"]) == (b["count"], b["count_distinct"]), floor
        assert a["abs_floor"] == floor
    assert entry["count"] == 4 and entry["abs_floor"] == 0.0     # the input is not modified


def test_refloor_recomputes_distinct_from_saddles():
    """Two maxima joined through a third (a dimpled lump seen as a chain): dropping the middle one is a
    pair-level fact and must not invent a join. Here the third is the only link, so dropping it separates."""
    entry = {"count": 3, "count_distinct": 1, "labels": [0, 0, 0], "threshold": 0.01, "max": 10.0, "saddle_merge": 0.9,
             "peaks": [{"pos": [0, 0, 0], "value": 10.0}, {"pos": [1, 0, 0], "value": 9.0},
                       {"pos": [2, 0, 0], "value": 0.3}],
             "pairs": [{"pair": [0, 1], "distance": 1.0, "saddle_ratio": 0.95},
                       {"pair": [0, 2], "distance": 2.0, "saddle_ratio": 0.2},
                       {"pair": [1, 2], "distance": 1.0, "saddle_ratio": 0.95}]}
    out = refloor(entry, 1.0)
    assert out["count"] == 2 and out["count_distinct"] == 1 and out["labels"] == [0, 0]
    assert refloor(entry, 9.5)["count"] == 1


def test_floor_default_and_reference():
    assert LUMP_DEFAULTS["count_floor_frac"] == 0.05 and LUMP_DEFAULTS["count_floor_ref"] is None
    f = LumpDiagnostics.count_floor
    first = SimpleNamespace(cfg={"count_floor_frac": 0.05, "count_floor_ref": None}, counts=[])
    assert f(first) == 0.0                      # no reference before the first window completes
    after = SimpleNamespace(cfg=first.cfg, counts=[{"max": 16.0}, {"max": 3.0}])
    assert f(after) == pytest.approx(0.8)       # 5% of the FIRST window's maximum
    given = SimpleNamespace(cfg={"count_floor_frac": 0.05, "count_floor_ref": 20.0}, counts=[{"max": 16.0}])
    assert f(given) == pytest.approx(1.0)       # an explicit reference wins
    off = SimpleNamespace(cfg={"count_floor_frac": 0.0, "count_floor_ref": None}, counts=[{"max": 16.0}])
    assert f(off) == 0.0                        # frac 0 = the Session 2 behaviour


# ---------------------------------------------------------------------------
# Saved Session 2 runs: recount from the stored count windows
# ---------------------------------------------------------------------------

def _counts(path):
    p = os.path.join(ROOT, "outputs", "session2", path)
    if not os.path.exists(p):
        pytest.skip("saved result not available: " + p)
    return json.load(open(p))["lumps"]["counts"]


def _recount(counts, frac=0.05):
    ref = counts[0]["max"]
    return [counts[0]] + [refloor(c, frac * ref) for c in counts[1:]]


@pytest.mark.parametrize("run, K, end_before, end_after", [
    ("wpE/results/E03_ring5_m1_a8.11.json", 5, 5, 0),      # debris counted as 5 distinct (7 maxima); none is a lump
    ("wpE/results/E07_ring6_m1_a7.32.json", 6, 6, 0),      # 6 distinct (10 maxima)
    ("wpE/results/E15_res_E03.json", 5, 3, 0),
    ("wpE/results/E15_box_E03.json", 5, 2, 2),             # a real lump and a 0.078 remnant: kept at 0.05
])
def test_saved_debris_runs(run, K, end_before, end_after):
    c = _counts(run)
    assert c[-1]["count_distinct"] == end_before
    r = _recount(c)
    assert r[-1]["count_distinct"] == end_after
    assert end_after < K                     # "merged" by the Session 2 rule, as the dynamics say


@pytest.mark.parametrize("run, expect_end", [
    ("wpE/results/E01_ring5_m1_a16.json", 2), ("wpE/results/E02_ring5_m0_a16.json", 2),
    ("wpE/results/E14_ring5_m1_a30.json", 5), ("wpC/results/single_x12_t00.0.json", 1),
])
def test_saved_clean_runs_keep_their_end_count(run, expect_end):
    c = _counts(run)
    r = _recount(c)
    assert c[-1]["count_distinct"] == r[-1]["count_distinct"] == expect_end
    K = len(c[0]["peaks"])
    first_drop = lambda cc: next((x["t_mid"] for x in cc if x["count_distinct"] < K), None)
    # the floor can only bring the first count drop earlier (E1: 360.4 -> 335.4, small satellites of 0.03 to
    # 0.04 of a lump no longer pad the count); the contact time of the report, min over several measures, is 295.7 either way
    d0, d1 = first_drop(c), first_drop(r)
    assert (d0 is None and d1 is None) or d1 <= d0


def test_e14_and_lone_lump_counts_unchanged_in_every_window():
    for run in ("wpE/results/E14_ring5_m1_a30.json", "wpC/results/single_x12_t00.0.json"):
        c = _counts(run)
        r = _recount(c)
        assert [(a["count"], a["count_distinct"]) for a in c] == [(a["count"], a["count_distinct"]) for a in r]
