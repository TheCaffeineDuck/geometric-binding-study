"""Checkpoint/resume gives a bit-identical continuation (fields and every series)."""

import json
import os

import numpy as np
import pytest

from swirl.engine.run import run

CFG = {
    "N": 32, "L": 30.0, "m": 1.0, "g4": 0.30, "g6": 0.055, "dt": 0.1, "t_end": 12.0,
    "init": {"gaussians": [{"center": [0.4, 0, 0], "amp": 1.5, "R": 2.0}]},
    "absorber": {"gamma": 2.0, "width": 6.0, "power": 3},
    "kfilter": {"nu": 0.5, "power": 8},
    "center": [0.4, 0.0, 0.0], "r_c": [3.0, 6.0], "sample_every": 0.5,
    "avg_window": 4.0, "checkpoint_every": 2.0, "threads": 4,
}


def _strip(res):
    return {k: v for k, v in res.items() if k not in ("_evolver", "wall_seconds",
                                                       "steps_per_second_this_call")}


@pytest.mark.parametrize("first_chunk", [37, 40])  # off and on a sample/checkpoint boundary
def test_resume_bit_identical(tmp_path, first_chunk):
    ref = run(CFG, out_path=str(tmp_path / "ref.json"))
    ck = str(tmp_path / "ck")
    part = run(CFG, checkpoint_dir=ck, max_steps=first_chunk)
    assert part["completed"] is False and part["steps_done"] == first_chunk
    assert os.path.exists(os.path.join(ck, "checkpoint.npz"))
    # resume in two more chunks to cover a resume from a resumed run
    run(CFG, checkpoint_dir=ck, max_steps=23)
    res = run(CFG, checkpoint_dir=ck, out_path=str(tmp_path / "res.json"))
    assert res["completed"] is True
    a, b = ref["_evolver"], res["_evolver"]
    assert np.array_equal(a.phi, b.phi)
    assert np.array_equal(a.pi, b.pi)
    assert a.absorbed == b.absorbed and a.C_kin == b.C_kin
    assert json.dumps(_strip(ref)) == json.dumps(_strip(res))
    with open(tmp_path / "res.json") as f:
        saved = json.load(f)
    assert saved["params"]["absorber"] == CFG["absorber"]
    assert saved["series"]["t"] == ref["series"]["t"]


def test_resume_rejects_changed_config(tmp_path):
    ck = str(tmp_path / "ck")
    run(CFG, checkpoint_dir=ck, max_steps=10)
    with pytest.raises(ValueError):
        run({**CFG, "g4": 0.31}, checkpoint_dir=ck)
