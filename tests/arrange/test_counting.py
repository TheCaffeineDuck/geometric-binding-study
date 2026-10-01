"""Symmetry groups from coordinates and 0/180 class counts against Burnside's lemma and closed forms."""
import numpy as np

from swirl.arrange import geometry as geo

# closed forms first: group orders, and classic counts
ORDERS = {"tetra": 24, "octa": 48, "cube": 48, "icosa": 120}
BRACELETS_2COLOUR = {3: 4, 4: 6, 5: 8, 6: 13, 7: 18, 8: 30}      # OEIS A000029


def test_group_orders():
    for name, X in geo.arrangements().items():
        G = geo.point_group(X)
        want = 4 * int(name[4]) if name.startswith("ring") else ORDERS[name.rstrip("+c")]
        assert len(G) == want, name
        for Q, perm, det in G:
            assert np.allclose(X @ Q.T, X[perm], atol=1e-12)


def test_closed_form_counts():
    arr = geo.arrangements()
    rot = lambda G: [g for g in G if g[2] == 1]
    # cube vertices, 2 colours, rotations only: 23; with reflections: 22
    G = geo.point_group(arr["cube"])
    assert len(geo.binary_classes(rot(G), 8, flip=False)) == 23 == geo.burnside_count(rot(G), 8, flip=False)
    assert len(geo.binary_classes(G, 8, flip=False)) == 22
    # icosahedron vertices, rotations only: 96
    G = geo.point_group(arr["icosa"])
    assert geo.burnside_count(rot(G), 12, flip=False) == 96
    # rings: bracelets (rotations and reflections of the polygon) without the sign flip
    for K, b in BRACELETS_2COLOUR.items():
        G = geo.point_group(arr["ring%d" % K])
        assert len(geo.binary_classes(G, K, flip=False)) == b == geo.burnside_count(G, K, flip=False)


def test_bruteforce_equals_burnside_with_sign_flip(record):
    counts = {}
    for name, X in geo.arrangements().items():
        G = geo.point_group(X)
        cl = geo.binary_classes(G, len(X), flip=True)
        b = geo.burnside_count(G, len(X), flip=True)
        assert sum(c for _, c in cl) == 2 ** len(X)
        assert len(cl) == b, name
        counts[name] = {"bruteforce": len(cl), "burnside": b, "group_order": len(G)}
    record("class_counts", counts)
