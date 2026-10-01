"""Lump positions for WP-E arrangement families (centred at the origin; a = nearest-neighbour spacing
of the ring or shell). Used by design_cost.py and the example candidate lists; WP-D supplies the real
candidate positions."""
import itertools

import numpy as np


def ring(K, a, axis="z"):
    R = a / (2 * np.sin(np.pi / K))
    ang = 2 * np.pi * np.arange(K) / K
    return np.stack([R * np.cos(ang), R * np.sin(ang), np.zeros(K)], 1)


def tetrahedron(a):
    v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    return v * a / (2 * np.sqrt(2))


def octahedron(a):
    v = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    return v * a / np.sqrt(2)


def cube(a):
    return np.array(list(itertools.product((-0.5, 0.5), repeat=3))) * a


def icosahedron(a):
    g = (1 + np.sqrt(5)) / 2
    v = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            v += [[0, s1, s2 * g], [s1, s2 * g, 0], [s2 * g, 0, s1]]
    return np.array(v, float) * a / 2


FAMILIES = {**{"ring%d" % K: (lambda a, K=K: ring(K, a)) for K in range(3, 9)},
            "tetrahedron": tetrahedron, "octahedron": octahedron, "cube": cube, "icosahedron": icosahedron}


def positions(family, a, centre=False):
    p = FAMILIES[family](a)
    if centre:
        p = np.vstack([p, np.zeros(3)])
    return p
