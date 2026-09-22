import numpy as np

from phyroute.lindistflow import LinDistFlow, path_matrices
from phyroute.powerflow import solve_bfs
from phyroute.scenarios import ID, sample


def test_path_matrices_properties(feeder):
    R, X = path_matrices(feeder)
    assert np.allclose(R, R.T) and np.allclose(X, X.T)
    assert (R >= -1e-12).all() and (X >= -1e-12).all()
    # diagonal equals the total series resistance from the root to the bus
    for j in [1, 5, 17, 32]:
        k, tot = j, 0.0
        while k != 0:
            tot += feeder.z_pu[k].real
            k = feeder.parent[k]
        assert np.isclose(R[j, j], tot)
    assert np.allclose(R[0], 0) and np.allclose(X[0], 0)


def test_lindistflow_close_to_exact(feeder, rng):
    ds = sample(feeder, ID, 500, rng)
    vl = LinDistFlow(feeder).voltages(ds.p_net, ds.q_net, ds.v0)
    err = np.abs(vl - ds.vm)
    assert err.max() < 0.012          # loss-free approximation stays within ~1% on this feeder
    assert np.median(err.max(axis=1)) < 0.003


def test_lindistflow_exact_at_zero_load(feeder):
    zero = np.zeros((1, feeder.n_bus))
    vl = LinDistFlow(feeder).voltages(zero, zero, 1.02)[0]
    ve = solve_bfs(feeder, zero, zero, 1.02)[0]
    assert np.allclose(vl, 1.02) and np.allclose(ve, 1.02)
