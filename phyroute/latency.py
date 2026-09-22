"""Per-decision latency of each tier, measured in streaming (single-scenario) mode."""
from __future__ import annotations

import time

import numpy as np

from .feeder import Feeder, pandapower_voltages
from .lindistflow import LinDistFlow
from .models import Tiers
from .powerflow import solve_bfs


def _time(fn, n: int) -> float:
    fn()  # warm-up
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1e6  # microseconds


def measure(feeder: Feeder, tiers: Tiers, ldf: LinDistFlow, X, p_net, q_net, v0, pv_mw,
            n_fast: int = 1000, n_pp: int = 100) -> dict[str, float]:
    x1 = X[:1]
    out = {
        "lindistflow_us": _time(lambda: ldf.voltages(p_net[:1], q_net[:1], v0[:1]), n_fast),
        "tiny_ensemble_us": _time(lambda: tiers.tiny_predict(x1), n_fast),
        "tiny_single_us": _time(lambda: tiers.tiny[0].forward(x1), n_fast),
        "large_us": _time(lambda: tiers.large.forward(x1), n_fast),
        "classifier_us": _time(lambda: tiers.classifier.forward(x1), n_fast),
        "exact_bfs_us": _time(lambda: solve_bfs(feeder, p_net[:1], q_net[:1], v0[:1]), n_fast // 2),
    }
    p_load = p_net[0].copy()
    p_load[feeder.pv_bus] += pv_mw[0]
    out["exact_pandapower_us"] = _time(
        lambda: pandapower_voltages(feeder, p_load, q_net[0], pv_mw[0], float(v0[0])), n_pp)
    # batched throughput for reference (per scenario)
    nb = min(2000, len(X))
    out["batched_tiny_ensemble_us_per_sample"] = _time(lambda: tiers.tiny_predict(X[:nb]), 5) / nb
    out["batched_exact_bfs_us_per_sample"] = _time(
        lambda: solve_bfs(feeder, p_net[:nb], q_net[:nb], v0[:nb]), 5) / nb
    return out
