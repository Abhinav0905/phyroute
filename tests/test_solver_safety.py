import numpy as np
import pytest

from phyroute.powerflow import PowerFlowConvergenceError, solve_bfs


def test_default_and_diagnostic_apis_preserve_converged_solution(feeder):
    p, q = feeder.p_load_mw, feeder.q_load_mvar
    vm = solve_bfs(feeder, p, q, 1.0)
    with_iters, iterations = solve_bfs(feeder, p, q, 1.0, return_iters=True)
    with_info, info = solve_bfs(feeder, p, q, 1.0, return_info=True)
    assert vm.shape == (1, feeder.n_bus)
    assert np.array_equal(vm, with_iters) and np.array_equal(vm, with_info)
    assert info['converged'] and info['iterations'] == iterations
    assert info['max_voltage_update'] < info['tol']


@pytest.mark.parametrize('fault', ['nan_power', 'inf_power', 'zero_voltage', 'nan_voltage',
                                    'empty_batch', 'wrong_bus_count', 'q_shape', 'v_shape',
                                    'zero_iterations', 'fractional_iterations', 'zero_tol', 'nan_tol'])
def test_invalid_inputs_cannot_become_voltage_labels(feeder, fault):
    p, q = feeder.p_load_mw[None].copy(), feeder.q_load_mvar[None].copy()
    v0, kwargs = 1.0, {}
    if fault == 'nan_power': p[0, 1] = np.nan
    elif fault == 'inf_power': q[0, 1] = np.inf
    elif fault == 'zero_voltage': v0 = 0.0
    elif fault == 'nan_voltage': v0 = np.nan
    elif fault == 'empty_batch': p, q = p[:0], q[:0]
    elif fault == 'wrong_bus_count': p, q = p[:, :-1], q[:, :-1]
    elif fault == 'q_shape': q = np.repeat(q, 2, axis=0)
    elif fault == 'v_shape': v0 = np.ones((1, 1))
    elif fault == 'zero_iterations': kwargs['max_iter'] = 0
    elif fault == 'fractional_iterations': kwargs['max_iter'] = 1.5
    elif fault == 'zero_tol': kwargs['tol'] = 0
    elif fault == 'nan_tol': kwargs['tol'] = np.nan
    with pytest.raises(ValueError):
        solve_bfs(feeder, p, q, v0, **kwargs)


def test_unconverged_batch_raises_instead_of_returning_labels(feeder):
    p = np.stack([np.zeros(feeder.n_bus), feeder.p_load_mw])
    q = np.stack([np.zeros(feeder.n_bus), feeder.q_load_mvar])
    with pytest.raises(PowerFlowConvergenceError) as failure:
        solve_bfs(feeder, p, q, 1.0, max_iter=1)
    assert failure.value.iterations == 1
    assert failure.value.failed_scenarios == [1]
    assert failure.value.max_voltage_update > 1e-9


def test_extreme_loads_raise_rather_than_silently_labeling(feeder):
    with pytest.raises(PowerFlowConvergenceError):
        solve_bfs(feeder, feeder.p_load_mw * 1e6, feeder.q_load_mvar * 1e6, 1.0, max_iter=20)


def test_nonfinite_iteration_is_reported(feeder):
    from dataclasses import replace
    # A two-bus constant-power system whose first step reaches exactly zero V.
    two = replace(feeder, n_bus=2, parent=np.array([-1, 0]), order=np.array([0, 1]),
                  z_pu=np.array([0j, 1 + 0j]), s_base_mva=1.0)
    with pytest.raises(PowerFlowConvergenceError, match='nonfinite') as failure:
        solve_bfs(two, np.array([0.0, 1.0]), np.zeros(2), 1.0)
    assert failure.value.iterations == 2
    assert failure.value.failed_scenarios == [0]
