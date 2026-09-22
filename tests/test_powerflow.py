import numpy as np

from phyroute.feeder import V_MAX, V_MIN, build_feeder, pandapower_voltages
from phyroute.powerflow import solve_bfs, voltage_margin


def test_radial_structure(feeder):
    assert feeder.n_bus == 33
    assert feeder.parent[0] == -1
    assert (feeder.parent[1:] >= 0).all()
    assert len(set(feeder.order.tolist())) == 33
    # every non-root bus appears after its parent in BFS order
    pos = {b: i for i, b in enumerate(feeder.order)}
    assert all(pos[feeder.parent[j]] < pos[j] for j in range(1, 33))


def test_bfs_matches_pandapower_base_case(feeder):
    vm = solve_bfs(feeder, feeder.p_load_mw[None], feeder.q_load_mvar[None], 1.0)[0]
    ref = pandapower_voltages(feeder, feeder.p_load_mw, feeder.q_load_mvar,
                              np.zeros(len(feeder.pv_bus)), 1.0)
    assert np.abs(vm - ref).max() < 1e-8
    assert abs(vm.min() - 0.9131) < 1e-3  # published Baran-Wu minimum voltage


def test_bfs_matches_pandapower_random_scenarios(feeder, rng):
    for _ in range(15):
        mult = rng.uniform(0.2, 1.3) * rng.uniform(0.6, 1.4, size=feeder.n_bus)
        p, q = feeder.p_load_mw * mult, feeder.q_load_mvar * mult
        pv = feeder.pv_rated_mw * rng.uniform(0, 1.6, size=len(feeder.pv_bus))
        v0 = rng.uniform(0.95, 1.06)
        p_net = p.copy()
        p_net[feeder.pv_bus] -= pv
        vm = solve_bfs(feeder, p_net[None], q[None], v0)[0]
        ref = pandapower_voltages(feeder, p, q, pv, v0)
        assert np.abs(vm - ref).max() < 1e-8


def test_batch_equals_sequential(feeder, rng):
    P = feeder.p_load_mw[None] * rng.uniform(0.3, 1.1, size=(8, 1))
    Q = feeder.q_load_mvar[None] * np.ones((8, 1))
    v0 = rng.uniform(0.97, 1.05, size=8)
    batch = solve_bfs(feeder, P, Q, v0)
    seq = np.stack([solve_bfs(feeder, P[i:i + 1], Q[i:i + 1], v0[i:i + 1])[0] for i in range(8)])
    assert np.allclose(batch, seq, atol=1e-12)


def test_voltage_margin_sign():
    vm = np.array([[1.0, 0.96, 0.99], [1.0, 0.94, 1.0], [1.0, 1.06, 1.0]])
    m = voltage_margin(vm, V_MIN, V_MAX)
    assert np.isclose(m[0], 0.01)
    assert m[1] < 0 and np.isclose(m[1], -0.01)
    assert m[2] < 0 and np.isclose(m[2], -0.01)


def test_reconfigured_feeder_is_radial_and_matches_pandapower(rng):
    f2 = build_feeder(open_lines=(27,), close_lines=(36,))
    assert f2.n_bus == 33 and len(set(f2.order.tolist())) == 33
    assert f2.parent[28] == 24          # bus 28 is now fed from bus 24 through the closed tie
    mult = rng.uniform(0.5, 1.0, size=f2.n_bus)
    p, q = f2.p_load_mw * mult, f2.q_load_mvar * mult
    vm = solve_bfs(f2, p[None], q[None], 1.0)[0]
    ref = pandapower_voltages(f2, p, q, np.zeros(len(f2.pv_bus)), 1.0)
    assert np.abs(vm - ref).max() < 1e-8


def test_reconfiguration_must_stay_radial():
    import pytest
    with pytest.raises(ValueError):
        build_feeder(open_lines=(), close_lines=(36,))
