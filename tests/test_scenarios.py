import numpy as np

from phyroute.scenarios import ID, OOD, feature_names, sample


def test_shapes_and_labels(feeder, rng):
    ds = sample(feeder, ID, 300, rng)
    assert ds.X.shape == (300, 32 + 32 + len(feeder.pv_bus) + 1)
    assert len(feature_names(feeder)) == ds.X.shape[1]
    assert ds.vm.shape == (300, feeder.n_bus)
    assert np.array_equal(ds.violation, ds.margin < 0)
    assert np.allclose(ds.vmin, ds.vm.min(axis=1)) and np.allclose(ds.vmax, ds.vm.max(axis=1))


def test_reproducible_with_seed(feeder):
    a = sample(feeder, ID, 50, np.random.default_rng(5))
    b = sample(feeder, ID, 50, np.random.default_rng(5))
    assert np.array_equal(a.X, b.X) and np.array_equal(a.vm, b.vm)


def test_ood_has_more_pv(feeder, rng):
    a = sample(feeder, ID, 2000, rng)
    b = sample(feeder, OOD, 2000, rng)
    pv_cols = slice(64, 64 + len(feeder.pv_bus))
    assert b.X[:, pv_cols].mean() > a.X[:, pv_cols].mean()
    assert (b.vmax > 1.05).mean() > (a.vmax > 1.05).mean()   # overvoltage becomes more common
