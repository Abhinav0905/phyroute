"""Exact AC power flow for radial feeders: vectorised backward/forward sweep.

Constant-power loads and generators. The solver iterates a backward sweep of
branch currents (leaves to root) and a forward sweep of voltages (root to
leaves) until the largest voltage update falls below `tol`. All scenarios in a
batch are solved simultaneously; the per-scenario cost is that of a dedicated
radial solver, which is the cheapest exact tier we can offer a router.
"""
from __future__ import annotations

import numpy as np

from .feeder import Feeder


class PowerFlowConvergenceError(RuntimeError):
    """The solver could not produce finite, converged voltages for the batch."""

    def __init__(self, message: str, *, iterations: int, max_voltage_update: float,
                 failed_scenarios: list[int]):
        super().__init__(message)
        self.iterations = iterations
        self.max_voltage_update = max_voltage_update
        self.failed_scenarios = failed_scenarios


def solve_bfs(feeder: Feeder, p_net_mw: np.ndarray, q_net_mvar: np.ndarray,
              v0: np.ndarray | float, tol: float = 1e-9, max_iter: int = 100,
              return_iters: bool = False, return_info: bool = False):
    """Solve the power flow for a batch of scenarios.

    p_net_mw, q_net_mvar: arrays of shape (N, n_bus), net *consumption* per bus
    (load minus generation) in MW / MVAr; the root entry is ignored.
    v0: slack-bus voltage magnitude (p.u.), scalar or shape (N,).
    One-dimensional power arrays are treated as a single scenario. Invalid
    inputs raise ValueError. A nonfinite iterate or failure to converge raises
    PowerFlowConvergenceError; unconverged arrays are never returned as labels.
    Returns |V| of shape (N, n_bus) in p.u. By request, returns (|V|, iteration
    count) or (|V|, diagnostics dict). These two options are mutually exclusive.
    """
    if return_iters and return_info:
        raise ValueError("return_iters and return_info are mutually exclusive")
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isscalar(tol) or not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    if not np.isrealobj(p_net_mw) or not np.isrealobj(q_net_mvar) or not np.isrealobj(v0):
        raise ValueError("power inputs and slack voltage must be real")
    p_net_mw = np.atleast_2d(np.asarray(p_net_mw, dtype=float))
    q_net_mvar = np.atleast_2d(np.asarray(q_net_mvar, dtype=float))
    if (p_net_mw.ndim != 2 or q_net_mvar.shape != p_net_mw.shape
            or p_net_mw.shape[0] == 0 or p_net_mw.shape[1] != feeder.n_bus):
        raise ValueError("power arrays must have matching nonempty shape (N, feeder.n_bus)")
    if not np.isfinite(p_net_mw).all() or not np.isfinite(q_net_mvar).all():
        raise ValueError("power arrays must contain only finite values")
    n, nb = p_net_mw.shape
    v0 = np.asarray(v0, dtype=float)
    if v0.ndim > 1 or (v0.ndim == 1 and v0.shape != (n,)):
        raise ValueError("v0 must be a scalar or have shape (N,)")
    if not np.isfinite(v0).all() or not (v0 > 0).all():
        raise ValueError("v0 must contain finite positive voltages")
    if not np.isfinite(feeder.s_base_mva) or feeder.s_base_mva <= 0:
        raise ValueError("feeder power base must be finite and positive")
    if feeder.z_pu.shape != (nb,) or not np.isfinite(feeder.z_pu).all():
        raise ValueError("feeder impedances must be finite and match n_bus")
    with np.errstate(over="ignore", invalid="ignore"):
        s_pu = (p_net_mw + 1j * q_net_mvar) / feeder.s_base_mva
    if not np.isfinite(s_pu).all():
        raise ValueError("power inputs overflow the feeder's per-unit base")
    v0 = np.broadcast_to(v0, (n,)).astype(complex)

    V = np.repeat(v0[:, None], nb, axis=1)
    order = feeder.order
    parent = feeder.parent
    z = feeder.z_pu
    it = 0
    for it in range(1, max_iter + 1):
        # backward sweep: current flowing into each bus's subtree
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            I = np.conj(s_pu / V)
            I[:, 0] = 0.0
            for j in order[::-1]:
                if j == 0:
                    continue
                I[:, parent[j]] += I[:, j]
            # forward sweep
            V_new = V.copy()
            V_new[:, 0] = v0
            for j in order:
                if j == 0:
                    continue
                V_new[:, j] = V_new[:, parent[j]] - z[j] * I[:, j]
            delta_by_scenario = np.max(np.abs(V_new - V), axis=1)
        finite = np.isfinite(V_new).all(axis=1) & np.isfinite(delta_by_scenario)
        if not finite.all():
            raise PowerFlowConvergenceError(
                f"nonfinite power-flow iterate at iteration {it}", iterations=it,
                max_voltage_update=float("inf"), failed_scenarios=np.flatnonzero(~finite).tolist())
        delta = float(delta_by_scenario.max())
        V = V_new
        if delta < tol:
            break
    else:
        failed = np.flatnonzero(delta_by_scenario >= tol).tolist()
        raise PowerFlowConvergenceError(
            f"power flow did not converge in {max_iter} iterations (max update {delta:.3g} p.u.)",
            iterations=it, max_voltage_update=delta, failed_scenarios=failed)
    vm = np.abs(V)
    if return_info:
        return vm, {"converged": True, "iterations": it, "max_voltage_update": delta, "tol": float(tol)}
    return (vm, it) if return_iters else vm


def voltage_margin(vm: np.ndarray, v_min: float, v_max: float) -> np.ndarray:
    """Signed distance to the nearest limit; negative means a violation."""
    vm = np.atleast_2d(vm)
    return np.minimum(v_max - vm.max(axis=1), vm.min(axis=1) - v_min)
