"""Starting states. All at rest unless a push is added.

Gaussian plucks (tool c), centre x0:
  bump   u = A exp(-(x-x0)^2 / (2 sigma^2))
  sbend  u = A ((x-x0)/sigma) exp((1 - (x-x0)^2/sigma^2) / 2)   (odd about x0, peak value A)
On the grid the S-bend is centred on the bond x0 = -1/2 so it is odd under n -> -1-n, matching the
pattern (..., 0, -1, 1, 0, ...). In C-smooth both are centred on x0 = 0.
"""

import numpy as np

from swirl.string import orbits as O


def gaussian(S, obj, sigma, A=None, peak_stretch=None):
    x0 = -0.5 if (S.medium == "grid" and obj == "sbend") else 0.0
    y = (S.x - x0) / sigma
    u = np.exp(-0.5 * y * y) if obj == "bump" else y * np.exp(0.5 * (1.0 - y * y))
    if peak_stretch is not None:
        A = peak_stretch / float(np.max(np.abs(S.stretch(u))))
    return A * u, x0


def rwa(S, obj, A=None, omega=None):
    """The rotating-wave pattern as published (no tail): bump (-A/2, A, -A/2), sbend (-A, A).
    Give A, or omega to take A from the rotating-wave relation of the law."""
    if A is None:
        A = O.rwa_amplitude(obj, S.law, S.beta, omega)
    x0 = -0.5 if obj == "sbend" else 0.0
    return O.grid_pattern(S.N, obj, A), x0


def push_velocity(S, u, lam):
    """Velocity along the discrete translation direction: v = -lam du/dx (central difference on
    the grid and in 'fd', spectral derivative in 'spectral'). A state moving rigidly at speed c has
    u_t = -c u_x at a turning point (u_t = 0 there), so lam is the nominal speed."""
    if S.medium == "spectral":
        return -lam * S.D(u)
    return -lam * (np.roll(u, -1) - np.roll(u, 1)) / (2 * S.h)
