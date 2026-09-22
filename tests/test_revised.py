import json
from pathlib import Path

import numpy as np
import pytest

from phyroute.revised import (budget_threshold, conformal_radius, decision_counts,
                              load_arrays, physics_variants, policy, save_arrays, tiny_predictions)
from phyroute.feeder import build_feeder
from phyroute.powerflow import solve_bfs


def test_signed_screening_never_spends_budget_on_rejections():
    pred = {"m_nn": np.array([.02, -.01, .01, -.02]), "sigma": np.ones(4)}
    cheap, score = policy("nn_approval_signed_physics", pred, np.array([-.1, -.5, .03, -.3]))
    assert cheap.tolist() == [True, False, True, False]
    tau = budget_threshold(score, .75)
    assert (score > tau).tolist() == [True, False, True, False]
    assert np.isfinite(tau)


def test_physics_only_and_hybrid_retain_different_decisions():
    pred = {"m_nn": np.array([.01, -.02, .03]), "sigma": np.ones(3)}
    lin = np.array([-.01, .02, .005])
    phy, _ = policy("physics_only_abs_margin", pred, lin)
    hybrid, score = policy("phy_d", pred, lin)
    assert phy.tolist() == [False, True, True]
    assert hybrid.tolist() == [True, False, True]
    assert (score > 0).tolist() == [True, True, False]
    conservative, _ = policy("conservative_and", pred, lin)
    assert conservative.tolist() == [False, False, True]


def test_conformal_finite_sample_order_statistic_and_edge():
    assert conformal_radius(np.arange(1, 11), .2) == 9
    assert conformal_radius(np.arange(1, 11), .01) == np.inf
    with pytest.raises(ValueError): conformal_radius(np.array([np.nan]), .1)


def test_conditional_risk_denominator_and_no_acceptance():
    d = decision_counts(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]), np.zeros(4))
    assert d["unsafe_rate"] == .25 and d["conditional_risk"] == .5
    assert d["false_alarm_rate"] == .5 and d["accepted"] == 2
    none = decision_counts(np.zeros(4), np.array([1, 0, 1, 0]), np.zeros(4))
    assert none["conditional_risk"] is None and none["unsafe_rate"] == 0


def test_unresolved_reference_rows_never_enter_risk_denominator():
    d = decision_counts(np.array([1, 1, 0]), np.array([1, 0, 0]), np.array([0, 0, 1]),
                        np.array([1, 0, 0]))
    assert d["generated"] == 3 and d["n"] == 1 and d["unresolved"] == 2
    assert d["accepted_unresolved"] == 1 and d["escalated_unresolved"] == 1
    assert d["conditional_risk"] == 1 and d["unsafe_rate"] == 1


def test_margin_of_mean_extrema_is_not_mean_of_member_margins():
    class Dummy:
        def __init__(self, out): self.out = np.array([out])
        def forward(self, X): return self.out
    models = {"tiny_0": Dummy([.94, 1.01]), "tiny_1": Dummy([.99, 1.06])}
    nn, sigma, legacy = tiny_predictions(models, np.zeros((1, 3)), details=True)
    assert nn[0] > 0 and legacy[0] < 0


def test_cache_checks_identity_and_corruption(tmp_path):
    p = tmp_path / "a.npz"
    save_arrays(p, {"x": np.arange(5)}, {"seed": 1})
    assert np.array_equal(load_arrays(p, {"seed": 1})["x"], np.arange(5))
    with pytest.raises(ValueError, match="identity"):
        load_arrays(p, {"seed": 2})
    p.write_bytes(p.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="content"):
        load_arrays(p, {"seed": 1})


def test_physics_perturbations_change_proxy_not_truth():
    f = build_feeder(open_lines=(31,), close_lines=(35,))
    data = {"p_net": f.p_load_mw[None] * .6, "q_net": f.q_load_mvar[None] * .6, "v0": np.array([1.02])}
    top = {"open_line": 31, "close_line": 35}
    cfg = {"impedance_scales": [.8, 1.2], "perturbation_seed": 88}
    original_z = f.z_pu.copy()
    truth = solve_bfs(f, data["p_net"], data["q_net"], data["v0"])
    a = physics_variants(f, data, cfg, top)
    b = physics_variants(f, data, cfg, top)
    assert np.array_equal(a["branch_noise_10pct"], b["branch_noise_10pct"])
    assert not np.array_equal(a["current"], a["stale"])
    assert not np.array_equal(a["impedance_0.8"], a["impedance_1.2"])
    assert np.array_equal(f.z_pu, original_z)
    assert np.array_equal(truth, solve_bfs(f, data["p_net"], data["q_net"], data["v0"]))
