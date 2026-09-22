"""Decision metrics and bootstrap intervals."""
from __future__ import annotations

import numpy as np


def decision_metrics(approve: np.ndarray, violation: np.ndarray) -> dict[str, float]:
    n = len(approve)
    n_viol = max(int(violation.sum()), 1)
    n_safe = max(int((~violation).sum()), 1)
    unsafe = approve & violation
    false_alarm = ~approve & ~violation
    return {
        "unsafe_rate": float(unsafe.sum() / n),          # approved violations / all decisions
        "miss_rate": float(unsafe.sum() / n_viol),       # approved violations / true violations
        "false_alarm_rate": float(false_alarm.sum() / n_safe),
        "accuracy": float(((approve) == (~violation)).mean()),
    }


def bootstrap_ci(values: np.ndarray, stat, n_boot: int = 1000, alpha: float = 0.05,
                 rng: np.random.Generator | None = None) -> tuple[float, float]:
    rng = rng or np.random.default_rng(0)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.array([stat(values[i]) for i in idx])
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """Area under the ROC curve via the rank statistic (ties handled by average rank)."""
    from scipy.stats import rankdata
    pos = positive.astype(bool)
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided binomial confidence interval for a rate k/n."""
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi
