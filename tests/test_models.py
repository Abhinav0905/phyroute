import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from phyroute.models import NumpyMLP


def test_numpy_forward_matches_sklearn_regressor():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 6))
    Y = np.stack([X[:, 0] * 0.01 + 1.0, X[:, 1] * 0.02 + 1.0], 1)
    xs, ys = StandardScaler().fit(X), StandardScaler().fit(Y)
    m = MLPRegressor(hidden_layer_sizes=(8, 8), max_iter=200, random_state=0).fit(xs.transform(X), ys.transform(Y))
    np_m = NumpyMLP.from_sklearn(m, xs, ys)
    ref = ys.inverse_transform(m.predict(xs.transform(X)))
    assert np.allclose(np_m.forward(X), ref, atol=1e-10)
    assert np_m.macs == 6 * 8 + 8 * 8 + 8 * 2
    assert np_m.n_params == np_m.macs + 8 + 8 + 2


def test_numpy_forward_matches_sklearn_classifier():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    xs = StandardScaler().fit(X)
    m = MLPClassifier(hidden_layer_sizes=(8,), max_iter=300, random_state=0).fit(xs.transform(X), y)
    np_m = NumpyMLP.from_sklearn(m, xs, None, classifier=True)
    assert np.allclose(np_m.forward(X), m.predict_proba(xs.transform(X))[:, 1], atol=1e-10)
