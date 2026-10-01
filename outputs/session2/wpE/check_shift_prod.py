"""WP-E check 2b at the production dx: an off-lattice shifted copy (delta = (0.31, -0.17, 0.23) dx)
against the unshifted copy, both cut from the production source (N = 160, L = 88, dx = 0.55),
window [14, 18], T as given (default 40). Same measure as tests/engine/test_snapshot.py (shift_record).

Usage (from the repo root): NUMBA_NUM_THREADS=2 <python> outputs/session2/wpE/check_shift_prod.py [T]
Writes outputs/session2/wpE/checks/snapshot_offlattice_shift_prod.json.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from swirl.engine.snapshot import load_meta  # noqa: E402
from tests.engine.test_snapshot import shift_record  # noqa: E402

if __name__ == "__main__":
    T = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
    path = "outputs/session2/wpE/source/src_prod_N160_L88.npz"
    out = shift_record(load_meta(path), path, T=T)
    out["source"] = path
    p = os.path.join(HERE, "checks", "snapshot_offlattice_shift_prod.json")
    with open(p + ".tmp", "w") as f:
        json.dump(out, f, indent=1)
    os.replace(p + ".tmp", p)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
