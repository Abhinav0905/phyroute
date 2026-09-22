import numpy as np
import pytest

from phyroute.powerflow import solve_bfs
from phyroute.reference import _power_residual_pu, solve_reference


def test_reference_fast_path_preserves_solution_and_row_count(feeder):
    p = feeder.p_load_mw[None] * np.array([[0.4], [0.7]])
    q = feeder.q_load_mvar[None] * np.array([[0.4], [0.7]])
    vm, valid, info = solve_reference(feeder, p, q, 1.0)
    assert np.array_equal(vm, solve_bfs(feeder, p, q, 1.0))
    assert valid.tolist() == [True, True]
    assert info['n_generated'] == info['n_bfs'] == info['n_resolved'] == 2
    assert info['n_nr'] == info['n_unresolved'] == 0


def test_nr_fallback_checks_residual_and_leaves_feeder_unmodified(feeder):
    original_load = feeder.net.load.p_mw.to_numpy().copy()
    p, q = feeder.p_load_mw * 0.5, feeder.q_load_mvar * 0.5
    vm, valid, info = solve_reference(feeder, p, q, 1.01, max_iter=1)
    assert valid.tolist() == [True] and info['n_nr'] == 1
    assert np.allclose(vm, solve_bfs(feeder, p, q, 1.01), atol=1e-8)
    accepted = info['nr_attempts'][0]['attempts'][-1]
    assert accepted['accepted'] and accepted['max_power_residual_pu'] <= 1e-8
    assert np.array_equal(feeder.net.load.p_mw.to_numpy(), original_load)


def test_failed_points_remain_nan_and_keep_original_indices(feeder, monkeypatch):
    import phyroute.reference as reference
    monkeypatch.setattr(reference, '_newton_reference', lambda *args: (None, [{'accepted': False, 'error': 'forced NR failure'}]))
    p = np.stack([np.zeros(feeder.n_bus), feeder.p_load_mw])
    q = np.stack([np.zeros(feeder.n_bus), feeder.q_load_mvar])
    vm, valid, info = solve_reference(feeder, p, q, 1.0, max_iter=1, block_size=2)
    assert valid.tolist() == [True, False]
    assert np.isfinite(vm[0]).all() and np.isnan(vm[1]).all()
    assert info['n_generated'] == 2 and info['n_resolved'] == info['n_unresolved'] == 1
    assert info['unresolved_indices'] == [1]
    assert info['nr_attempts'][0]['scenario_index'] == 1


def test_invalid_data_does_not_enter_numerical_fallback(feeder):
    with pytest.raises(ValueError, match='finite'):
        solve_reference(feeder, feeder.p_load_mw * np.nan, feeder.q_load_mvar, 1.0)
