"""
engine/evolver.py
=================
Standalone reusable module containing SexticEvolver: a 3D spectral RK4 time-evolution
engine for a scalar field with sextic potential V(phi) = 1/2*m^2*phi^2 - (g4/24)*phi^4
+ (g6/720)*phi^6, with Kreiss-Oliger dissipation.

Extracted without modification from:
  swirl_taxonomy/nest_dynamics/validate_parity_rule.py (lines 12-90)
"""

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq


class SexticEvolver:
    """
    Evolves the field using RK4 with a sextic potential and Kreiss-Oliger dissipation.

    Potential: V(phi) = 1/2*m^2*phi^2 - (g4/24)*phi^4 + (g6/720)*phi^6
    EOM: phi_tt = nabla^2 phi - V'(phi) - dissipation
    """
    def __init__(self, N, L, m, g4, g6, dissipation_sigma=0.01):
        self.N = N
        self.L = L
        self.dx = L / N
        self.m_sq = m**2
        self.g4 = g4
        self.g6 = g6
        self.sigma = dissipation_sigma

        # Periodic grid
        coords = np.linspace(-L/2, L/2, N, endpoint=False)
        self.X, self.Y, self.Z = np.meshgrid(coords, coords, coords, indexing='ij')

        # Precompute spectral operators
        k = fftfreq(N, d=self.dx) * 2 * np.pi
        KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
        self.K2 = KX**2 + KY**2 + KZ**2
        self.K4 = self.K2**2

        self.phi = np.zeros((N, N, N))
        self.phi_dot = np.zeros((N, N, N))
        self.t = 0.0

    def set_initial_conditions(self, phi, phi_dot):
        """Set initial field and its time derivative."""
        self.phi = phi.copy()
        self.phi_dot = phi_dot.copy()
        self.t = 0.0

    def compute_rhs(self, phi, phi_dot):
        """Compute the RHS of the coupled first-order system."""
        phi_hat = fftn(phi)
        laplacian = np.real(ifftn(-self.K2 * phi_hat))

        # V'(phi) = m^2*phi - (g4/6)*phi^3 + (g6/120)*phi^5
        v_prime = self.m_sq * phi - (self.g4 / 6.0) * phi**3 + (self.g6 / 120.0) * phi**5

        # Kreiss-Oliger dissipation: -sigma * dx^4 * laplacian(laplacian(phi))
        dissipation = self.sigma * self.dx**4 * np.real(ifftn(self.K4 * phi_hat))

        phi_ddot = laplacian - v_prime - dissipation
        return phi_dot, phi_ddot

    def step_rk4(self, dt):
        """Advance one time step using RK4."""
        phi0 = self.phi.copy()
        dot0 = self.phi_dot.copy()

        p1, v1 = self.compute_rhs(phi0, dot0)
        p2, v2 = self.compute_rhs(phi0 + 0.5 * dt * p1, dot0 + 0.5 * dt * v1)
        p3, v3 = self.compute_rhs(phi0 + 0.5 * dt * p2, dot0 + 0.5 * dt * v2)
        p4, v4 = self.compute_rhs(phi0 + dt * p3, dot0 + dt * v3)

        self.phi = phi0 + (dt / 6.0) * (p1 + 2*p2 + 2*p3 + p4)
        self.phi_dot = dot0 + (dt / 6.0) * (v1 + 2*v2 + 2*v3 + v4)
        self.t += dt

    def compute_energy(self):
        """Compute total energy (Hamiltonian)."""
        dV = self.dx**3
        # Kinetic: 1/2 * integral(phi_dot^2)
        E_kin = 0.5 * np.sum(self.phi_dot**2) * dV

        # Gradient: 1/2 * integral(|grad phi|^2) - computed in Fourier space
        phi_hat = fftn(self.phi)
        E_grad = 0.5 * np.sum(self.K2 * np.abs(phi_hat)**2) * dV / (self.N**3)

        # Potential: integral(V(phi))
        v = 0.5 * self.m_sq * self.phi**2 - (self.g4 / 24.0) * self.phi**4 + (self.g6 / 720.0) * self.phi**6
        E_pot = np.sum(v) * dV

        return float(E_kin + E_grad + E_pot)
