"""Reference labels with explicit numerical resolution status.

The fast path uses the strict radial sweep. Failed batches are subdivided, and
failed single scenarios receive two Newton-Raphson attempts. An unresolved row
is NaN with valid=False, never a voltage-limit label. Resolution is a numerical
status; failure is not proof that no AC solution exists.
"""
from __future__ import annotations

import copy

import numpy as np
import pandapower as pp

from .feeder import Feeder
from .powerflow import PowerFlowConvergenceError, solve_bfs


def _power_residual_pu(feeder: Feeder, voltage: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    """Maximum non-slack nodal complex-power mismatch for the radial model."""
    branch = np.zeros(feeder.n_bus, dtype=complex)
    for j in feeder.order[1:]:
        branch[j] = (voltage[feeder.parent[j]] - voltage[j]) / feeder.z_pu[j]
    delivered = branch.copy()
    for j in feeder.order[1:]:
        delivered[feeder.parent[j]] -= branch[j]
    specified = (p + 1j * q) / feeder.s_base_mva
    residual = voltage * np.conj(delivered) - specified
    return float(np.max(np.abs(residual[1:])))


def _newton_reference(feeder: Feeder, p: np.ndarray, q: np.ndarray, v0: float,
                      max_iteration: int, residual_tolerance_pu: float):
    """Try preset NR initializations, checking finite output and nodal residual."""
    attempts = []
    # Work on a copy: validation/fallback must not leave generators or load edits
    # in the shared feeder, including after pandapower raises an exception.
    for initialization in ('flat', 'auto'):
        net = copy.deepcopy(feeder.net)
        net.load['p_mw'] = p[net.load.bus.to_numpy()]
        net.load['q_mvar'] = q[net.load.bus.to_numpy()]
        if len(net.sgen):
            net.sgen.drop(net.sgen.index, inplace=True)
        net.ext_grid['vm_pu'] = v0
        attempt = {'initialization': initialization}
        try:
            pp.runpp(net, algorithm='nr', init=initialization, max_iteration=max_iteration,
                     tolerance_mva=1e-10, numba=False, calculate_voltage_angles=True)
            magnitude = net.res_bus.vm_pu.to_numpy(dtype=float)
            angle = np.deg2rad(net.res_bus.va_degree.to_numpy(dtype=float))
            voltage = magnitude * np.exp(1j * angle)
            if not net.converged or not np.isfinite(voltage).all() or not (magnitude > 0).all():
                raise ValueError('NR did not return finite positive converged voltages')
            residual = _power_residual_pu(feeder, voltage, p, q)
            attempt['max_power_residual_pu'] = residual
            if not np.isfinite(residual) or residual > residual_tolerance_pu:
                raise ValueError(f'NR nodal residual {residual:.3g} exceeds tolerance')
            attempt['accepted'] = True
            attempts.append(attempt)
            return magnitude.copy(), attempts
        except Exception as exc:
            attempt.update(accepted=False, error=f'{type(exc).__name__}: {exc}')
            attempts.append(attempt)
    return None, attempts


def solve_reference(feeder: Feeder, p_net_mw, q_net_mvar, v0, *, tol: float = 1e-9,
                    max_iter: int = 100, block_size: int = 256, nr_max_iteration: int = 100,
                    residual_tolerance_pu: float = 1e-8):
    """Return (voltages, valid_mask, diagnostics) without losing failed rows.

    Invalid shapes/NaN inputs still raise ValueError via solve_bfs. Numerical
    nonconvergence is handled per scenario. Diagnostics include method counts,
    unresolved indices and every NR attempt. Failed rows remain in input order.
    """
    if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError('block_size must be a positive integer')
    if isinstance(nr_max_iteration, bool) or not isinstance(nr_max_iteration, (int, np.integer)) or nr_max_iteration < 1:
        raise ValueError('nr_max_iteration must be a positive integer')
    if not np.isfinite(residual_tolerance_pu) or residual_tolerance_pu <= 0:
        raise ValueError('residual_tolerance_pu must be finite and positive')
    info = {'bfs_calls': 1, 'bfs_max_iterations': 0, 'nr_attempts': [],
            'residual_tolerance_pu': float(residual_tolerance_pu)}
    try:
        vm, diagnostic = solve_bfs(feeder, p_net_mw, q_net_mvar, v0, tol=tol, max_iter=max_iter, return_info=True)
        n = len(vm)
        info.update(n_generated=n, n_resolved=n, n_unresolved=0, n_bfs=n, n_nr=0,
                    unresolved_indices=[], bfs_max_iterations=diagnostic['iterations'])
        return vm, np.ones(n, dtype=bool), info
    except PowerFlowConvergenceError as exc:
        info['initial_batch_failure'] = str(exc)
        info['bfs_max_iterations'] = exc.iterations
    p = np.atleast_2d(np.asarray(p_net_mw, dtype=float))
    q = np.atleast_2d(np.asarray(q_net_mvar, dtype=float))
    n = len(p)
    slack = np.broadcast_to(np.asarray(v0, dtype=float), (n,))
    vm = np.full((n, feeder.n_bus), np.nan)
    valid = np.zeros(n, dtype=bool)
    methods = np.zeros(n, dtype=np.int8)  # 0 unresolved, 1 BFS, 2 NR

    def solve_subset(indices):
        info['bfs_calls'] += 1
        try:
            part, diagnostic = solve_bfs(feeder, p[indices], q[indices], slack[indices],
                                         tol=tol, max_iter=max_iter, return_info=True)
            vm[indices], valid[indices], methods[indices] = part, True, 1
            info['bfs_max_iterations'] = max(info['bfs_max_iterations'], diagnostic['iterations'])
        except PowerFlowConvergenceError as exc:
            info['bfs_max_iterations'] = max(info['bfs_max_iterations'], exc.iterations)
            if len(indices) > 1:
                midpoint = len(indices) // 2
                solve_subset(indices[:midpoint])
                solve_subset(indices[midpoint:])
                return
            i = int(indices[0])
            nr, attempts = _newton_reference(feeder, p[i], q[i], float(slack[i]),
                                             nr_max_iteration, residual_tolerance_pu)
            info['nr_attempts'].append({'scenario_index': i, 'bfs_failure': str(exc), 'attempts': attempts})
            if nr is not None:
                vm[i], valid[i], methods[i] = nr, True, 2

    for start in range(0, n, block_size):
        solve_subset(np.arange(start, min(start + block_size, n)))
    info.update(n_generated=n, n_resolved=int(valid.sum()), n_unresolved=int((~valid).sum()),
                n_bfs=int((methods == 1).sum()), n_nr=int((methods == 2).sum()),
                unresolved_indices=np.flatnonzero(~valid).tolist())
    return vm, valid, info
