"""Surrogate tiers and auxiliary probes.

Tier 1: an ensemble of tiny MLPs predicting (V_min, V_max); its mean is the
        cheap prediction and its spread is the learned uncertainty signal.
Tier 2: a single larger MLP with the same targets.
Tier 3: the exact solver (see powerflow.py), used as the escalation target.
Probes: a violation classifier (decision-level confidence baseline) and a
        delegation-value probe that predicts whether the tiny decision is wrong
        (learned escalation benefit, in the spirit of Calibrate-Then-Delegate).
All inference used for timing is re-implemented as plain NumPy matrix products
so that the cost of a tier is its arithmetic, not scikit-learn's bookkeeping.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from .feeder import V_MAX, V_MIN


def _relu(a: np.ndarray) -> np.ndarray:
    return np.maximum(a, 0.0)


@dataclass
class NumpyMLP:
    """Weights lifted out of a fitted scikit-learn MLP for lean inference."""
    weights: list[np.ndarray]
    biases: list[np.ndarray]
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray | None = None
    y_scale: np.ndarray | None = None
    classifier: bool = False

    @classmethod
    def from_sklearn(cls, model, xs: StandardScaler, ys: StandardScaler | None, classifier=False):
        return cls([w.astype(np.float64) for w in model.coefs_],
                   [b.astype(np.float64) for b in model.intercepts_],
                   xs.mean_, xs.scale_,
                   None if ys is None else ys.mean_, None if ys is None else ys.scale_,
                   classifier)

    def forward(self, X: np.ndarray) -> np.ndarray:
        h = (X - self.x_mean) / self.x_scale
        n = len(self.weights)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w + b
            if i < n - 1:
                h = _relu(h)
        if self.classifier:
            return 1.0 / (1.0 + np.exp(-h[:, 0]))
        return h * self.y_scale + self.y_mean

    @property
    def macs(self) -> int:
        return int(sum(w.shape[0] * w.shape[1] for w in self.weights))

    @property
    def n_params(self) -> int:
        return int(sum(w.size for w in self.weights) + sum(b.size for b in self.biases))


@dataclass
class Tiers:
    tiny: list[NumpyMLP]
    large: NumpyMLP
    classifier: NumpyMLP
    delegation: NumpyMLP | None = None
    error_probe: NumpyMLP | None = None
    meta: dict = field(default_factory=dict)

    def tiny_predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ensemble mean (V_min, V_max), predicted margin, and margin spread."""
        preds = np.stack([m.forward(X) for m in self.tiny])          # (M, N, 2)
        margins = np.minimum(V_MAX - preds[:, :, 1], preds[:, :, 0] - V_MIN)  # (M, N)
        mean = preds.mean(axis=0)
        return mean, margins.mean(axis=0), margins.std(axis=0, ddof=1)

    def large_predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = self.large.forward(X)
        return p, np.minimum(V_MAX - p[:, 1], p[:, 0] - V_MIN)


def _fit_regressor(X, Y, hidden, seed, max_iter=400):
    xs = StandardScaler().fit(X)
    ys = StandardScaler().fit(Y)
    m = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", solver="adam",
                     learning_rate_init=1e-3, batch_size=256, max_iter=max_iter,
                     early_stopping=True, validation_fraction=0.1, n_iter_no_change=25,
                     random_state=seed, tol=1e-7)
    m.fit(xs.transform(X), ys.transform(Y))
    return NumpyMLP.from_sklearn(m, xs, ys), m.n_iter_


def _fit_classifier(X, y, hidden, seed, max_iter=400):
    xs = StandardScaler().fit(X)
    m = MLPClassifier(hidden_layer_sizes=hidden, activation="relu", solver="adam",
                      learning_rate_init=1e-3, batch_size=256, max_iter=max_iter,
                      early_stopping=True, validation_fraction=0.1, n_iter_no_change=25,
                      random_state=seed, tol=1e-7)
    m.fit(xs.transform(X), y.astype(int))
    return NumpyMLP.from_sklearn(m, xs, None, classifier=True), m.n_iter_


def train_tiers(X_train, Y_train, viol_train, X_calib, Y_calib, viol_calib,
                tiny_hidden=(32, 32), large_hidden=(256, 256, 256, 256),
                n_tiny=5, seed=0, tiny_max_iter=250, large_max_iter=120,
                probe_max_iter=200) -> Tiers:
    """Fit all tiers on the training split and the delegation probe on the calibration split."""
    tiny, iters = [], []
    for k in range(n_tiny):
        m, it = _fit_regressor(X_train, Y_train, tiny_hidden, seed + k, max_iter=tiny_max_iter)
        tiny.append(m)
        iters.append(it)
    large, it_large = _fit_regressor(X_train, Y_train, large_hidden, seed + 100, max_iter=large_max_iter)
    clf, it_clf = _fit_classifier(X_train, viol_train, tiny_hidden, seed + 200, max_iter=probe_max_iter)
    tiers = Tiers(tiny=tiny, large=large, classifier=clf,
                  meta={"tiny_iters": iters, "large_iters": it_large, "clf_iters": it_clf})

    # delegation-value probe: predicts whether the tiny decision is wrong, from the
    # inputs plus the tiny tier's own outputs (mean margin and spread).
    _, m_nn, sigma = tiers.tiny_predict(X_calib)
    wrong = ((m_nn >= 0) != (~viol_calib)).astype(int)      # tiny decision != truth
    Z = np.concatenate([X_calib, m_nn[:, None], sigma[:, None]], axis=1)
    dv, it_dv = _fit_classifier(Z, wrong, tiny_hidden, seed + 300, max_iter=probe_max_iter)
    tiers.delegation = dv
    tiers.meta["dv_iters"] = it_dv
    tiers.meta["dv_calib_positive_rate"] = float(wrong.mean())
    tiers.meta["dv_calib_positives"] = int(wrong.sum())

    # learned error probe: regress the log absolute margin error of the tiny tier.
    # The score used for routing is predicted error / |predicted margin|, i.e. a
    # learned estimate of how likely the error is to flip the decision.
    m_true = np.minimum(V_MAX - Y_calib[:, 1], Y_calib[:, 0] - V_MIN)
    log_err = np.log10(np.abs(m_nn - m_true) + 1e-6)[:, None]
    ep, it_ep = _fit_regressor(Z, log_err, tiny_hidden, seed + 400, max_iter=probe_max_iter)
    tiers.error_probe = ep
    tiers.meta["error_probe_iters"] = it_ep
    return tiers


def error_ratio_scores(tiers: Tiers, X, m_nn, sigma, eps: float = 1e-6) -> np.ndarray:
    Z = np.concatenate([X, m_nn[:, None], sigma[:, None]], axis=1)
    pred_err = 10.0 ** tiers.error_probe.forward(Z)[:, 0]
    return pred_err / (np.abs(m_nn) + eps)


def delegation_scores(tiers: Tiers, X, m_nn, sigma) -> np.ndarray:
    Z = np.concatenate([X, m_nn[:, None], sigma[:, None]], axis=1)
    return tiers.delegation.forward(Z)
