import numpy as np

from phyroute.metrics import auroc, bootstrap_ci, clopper_pearson, decision_metrics
from phyroute.routing import (POLICIES, Signals, cascade_decisions, escalation_to_zero_unsafe,
                              frontier, scores, threshold_for_rate)


def _signals(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m_true = rng.normal(0.01, 0.02, size=n)
    m_nn = m_true + rng.normal(0, 0.002, size=n)
    return Signals(m_nn=m_nn, sigma=np.abs(rng.normal(0.002, 0.001, size=n)),
                   m_lin=m_true + rng.normal(0.0005, 0.0015, size=n),
                   p_viol=1 / (1 + np.exp(m_nn / 0.005)), q_dv=rng.uniform(size=n),
                   q_ratio=rng.uniform(size=n), rng=rng), m_true < 0


def test_all_policies_produce_scores():
    sig, viol = _signals()
    for pol in POLICIES:
        s = scores(pol, sig)
        assert s.shape == (2000,) and np.isfinite(s).all()


def test_threshold_rate():
    s = np.random.default_rng(1).normal(size=10000)
    for r in [0.05, 0.2, 0.5]:
        tau = threshold_for_rate(s, r)
        assert abs((s > tau).mean() - r) < 0.01
    assert threshold_for_rate(s, 0.0) == np.inf and threshold_for_rate(s, 1.0) == -np.inf


def test_cascade_extremes():
    sig, viol = _signals()
    cheap, exact = sig.m_nn >= 0, ~viol
    s = scores("phy_c", sig)
    a0, e0 = cascade_decisions(s, threshold_for_rate(s, 0.0), cheap, exact)
    a1, e1 = cascade_decisions(s, threshold_for_rate(s, 1.0), cheap, exact)
    assert np.array_equal(a0, cheap) and not e0.any()
    assert np.array_equal(a1, exact) and e1.all()
    assert decision_metrics(a1, viol)["unsafe_rate"] == 0.0


def test_frontier_and_zero_unsafe():
    sig, viol = _signals()
    cheap, exact = sig.m_nn >= 0, ~viol
    s = scores("phy_c", sig)
    fr = frontier(s, cheap, exact, viol, np.array([0.0, 0.1, 0.5, 1.0]))
    assert fr["unsafe"][-1] == 0.0
    assert fr["unsafe"][0] >= fr["unsafe"][-1]
    r0 = escalation_to_zero_unsafe(s, cheap, exact, viol, np.arange(0, 1.0001, 0.01))
    assert 0 <= r0 <= 1
    # a near-boundary gate should need far fewer escalations than random routing
    r_rand = escalation_to_zero_unsafe(scores("random", sig), cheap, exact, viol, np.arange(0, 1.0001, 0.01))
    assert r0 < r_rand


def test_metrics():
    approve = np.array([True, True, False, False])
    viol = np.array([False, True, False, True])
    d = decision_metrics(approve, viol)
    assert d["unsafe_rate"] == 0.25 and d["miss_rate"] == 0.5 and d["false_alarm_rate"] == 0.5
    assert d["accuracy"] == 0.5
    assert auroc(np.array([0.1, 0.9, 0.2, 0.8]), np.array([0, 1, 0, 1])) == 1.0
    lo, hi = bootstrap_ci(np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1.0]), np.mean, n_boot=200)
    assert lo <= 0.5 <= hi


def test_clopper_pearson():
    lo, hi = clopper_pearson(0, 1000)
    assert lo == 0.0 and 0.002 < hi < 0.005
    lo, hi = clopper_pearson(50, 1000)
    assert lo < 0.05 < hi and hi - lo < 0.03
