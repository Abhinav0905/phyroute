"""Routing policies and cascade evaluation.

Every policy produces a score s(x); the cascade escalates the decision on x to
the exact solver when s(x) exceeds a threshold. Sweeping the threshold traces
the safety-versus-escalation frontier. Higher score = stronger case to escalate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-6

POLICIES = {
    "random":            "Random routing (budget-matched control)",
    "nn_uncertainty":    "Ensemble spread of the tiny tier",
    "nn_margin":         "Predicted margin of the tiny tier (consequence from the model)",
    "chance":            "Decision uncertainty |m_nn|/sigma (chance-constraint routing)",
    "clf_conf":          "Violation-classifier confidence (decision-level confidence)",
    "delegation_value":  "Learned delegation-value probe (CTD-style classifier)",
    "learned_ratio":     "Learned error probe: predicted |error| / |predicted margin|",
    "phy_c":             "PhyRoute-C: physics margin from LinDistFlow (consequence only)",
    "phy_min":           "PhyRoute-min: smaller of physics and model margins",
    "phy_d":             "PhyRoute-D: physics disagreement first, then physics margin",
}

PHYSICS_POLICIES = {"phy_c", "phy_min", "phy_d"}


@dataclass
class Signals:
    """Everything a policy may look at for a batch of scenarios."""
    m_nn: np.ndarray        # tiny-ensemble mean predicted margin
    sigma: np.ndarray       # tiny-ensemble spread of the margin
    m_lin: np.ndarray       # LinDistFlow margin
    p_viol: np.ndarray      # classifier probability of violation
    q_dv: np.ndarray        # delegation-value probe output
    q_ratio: np.ndarray     # learned error / margin ratio
    rng: np.random.Generator


def scores(policy: str, s: Signals) -> np.ndarray:
    if policy == "random":
        return s.rng.uniform(size=len(s.m_nn))
    if policy == "nn_uncertainty":
        return s.sigma
    if policy == "nn_margin":
        return -np.abs(s.m_nn)
    if policy == "chance":
        return -np.abs(s.m_nn) / (s.sigma + EPS)
    if policy == "clf_conf":
        return -np.abs(s.p_viol - 0.5)
    if policy == "delegation_value":
        return s.q_dv
    if policy == "learned_ratio":
        return s.q_ratio
    if policy == "phy_c":
        return -np.abs(s.m_lin)
    if policy == "phy_min":
        return -np.minimum(np.abs(s.m_lin), np.abs(s.m_nn))
    if policy == "phy_d":
        # the linear physics acts as an independent verifier: scenarios where its
        # safety verdict contradicts the surrogate's are escalated first (the more
        # confident the contradiction, the sooner), then agreements by closeness
        # to the limit.
        disagree = (s.m_lin >= 0) != (s.m_nn >= 0)
        return np.where(disagree, np.abs(s.m_lin), -np.abs(s.m_lin))
    raise KeyError(policy)


def threshold_for_rate(score: np.ndarray, rate: float) -> float:
    """Threshold such that a fraction `rate` of these scores exceeds it."""
    if rate <= 0:
        return np.inf
    if rate >= 1:
        return -np.inf
    return float(np.quantile(score, 1.0 - rate))


def cascade_decisions(score: np.ndarray, tau: float, cheap_approve: np.ndarray,
                      exact_approve: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two-tier cascade: use the exact decision where score > tau, else the cheap one."""
    esc = score > tau
    approve = np.where(esc, exact_approve, cheap_approve)
    return approve, esc


def frontier(score: np.ndarray, cheap_approve: np.ndarray, exact_approve: np.ndarray,
             violation: np.ndarray, rates: np.ndarray) -> dict[str, np.ndarray]:
    """Trace unsafe-decision and false-alarm rates against the escalation rate."""
    unsafe, false_alarm, realised = [], [], []
    n_safe = max(int((~violation).sum()), 1)
    for r in rates:
        tau = threshold_for_rate(score, r)
        approve, esc = cascade_decisions(score, tau, cheap_approve, exact_approve)
        unsafe.append(float((approve & violation).mean()))
        false_alarm.append(float((~approve & ~violation).sum() / n_safe))
        realised.append(float(esc.mean()))
    return {"rate": rates, "unsafe": np.array(unsafe), "false_alarm": np.array(false_alarm),
            "realised": np.array(realised)}


def escalation_to_zero_unsafe(score: np.ndarray, cheap_approve, exact_approve, violation,
                              grid: np.ndarray) -> float:
    """Smallest escalation rate on the grid at which no unsafe decision remains."""
    for r in grid:
        tau = threshold_for_rate(score, r)
        approve, _ = cascade_decisions(score, tau, cheap_approve, exact_approve)
        if not (approve & violation).any():
            return float(r)
    return float("nan")
