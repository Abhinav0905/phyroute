"""Linearised DistFlow voltage proxy for radial feeders.

For a radial network the squared voltage magnitude obeys, to first order,
    v_j = v_0 - 2 * sum_k (R_jk P_k + X_jk Q_k),
where R_jk (X_jk) is the total resistance (reactance) on the shared path from
the root to buses j and k, and P_k, Q_k are net consumptions in per-unit.
The proxy ignores losses, so it is biased towards optimistic (higher) voltages
under heavy load, but it costs a single matrix-vector product and depends on
no learned parameters. We use it as a physics-derived estimate of how close a
scenario is to a voltage limit, i.e. the *consequence* of a wrong prediction.
"""
from __future__ import annotations

import numpy as np

from .feeder import Feeder


def path_matrices(feeder: Feeder) -> tuple[np.ndarray, np.ndarray]:
    """R and X matrices of shared root-path impedance, shape (n_bus, n_bus)."""
    nb = feeder.n_bus
    # A[j, e] = 1 if line e (indexed by its downstream bus) lies on the path root -> j.
    A = np.zeros((nb, nb))
    for j in range(1, nb):
        k = j
        while k != 0:
            A[j, k] = 1.0
            k = feeder.parent[k]
    r = feeder.z_pu.real
    x = feeder.z_pu.imag
    R = A @ np.diag(r) @ A.T
    X = A @ np.diag(x) @ A.T
    return R, X


class LinDistFlow:
    def __init__(self, feeder: Feeder):
        self.feeder = feeder
        self.R, self.X = path_matrices(feeder)

    def voltages(self, p_net_mw: np.ndarray, q_net_mvar: np.ndarray, v0) -> np.ndarray:
        p = np.atleast_2d(p_net_mw) / self.feeder.s_base_mva
        q = np.atleast_2d(q_net_mvar) / self.feeder.s_base_mva
        v0 = np.broadcast_to(np.asarray(v0, dtype=float), (p.shape[0],))
        v_sq = v0[:, None] ** 2 - 2.0 * (p @ self.R.T + q @ self.X.T)
        return np.sqrt(np.clip(v_sq, 1e-6, None))
