"""Operating-scenario sampling for the feeder and feature construction.

A scenario is (per-bus load, PV output after curtailment, substation voltage
set-point). The controller's candidate action is the pair (set-point,
curtailment) and the surrogate's job is to predict whether the resulting
voltage profile stays inside the statutory band.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .feeder import Feeder, V_MAX, V_MIN
from .powerflow import solve_bfs, voltage_margin


@dataclass
class Distribution:
    """Ranges of the scenario generator. ID = in-distribution, OOD = shifted."""
    name: str
    load_factor: tuple[float, float] = (0.3, 1.1)   # global multiplier on base load
    bus_spread: tuple[float, float] = (0.7, 1.3)    # per-bus multiplier
    p_night: float = 0.25                           # probability of zero PV
    irradiance: tuple[float, float] = (0.1, 1.0)    # global PV factor when not night
    plant_spread: tuple[float, float] = (0.85, 1.0) # per-plant factor
    curtail_keep: tuple[float, float] = (0.0, 1.0)  # fraction of PV output retained
    v0: tuple[float, float] = (0.97, 1.05)          # slack voltage set-point (p.u.)
    pv_scale: float = 1.0                           # multiplier on rated PV capacity


ID = Distribution(name="in-distribution")
OOD = Distribution(name="pv-growth-shift", load_factor=(0.2, 0.8), pv_scale=1.6)


@dataclass
class Dataset:
    name: str
    X: np.ndarray            # standardised later; raw features here
    p_net: np.ndarray        # (N, n_bus) net MW consumption
    q_net: np.ndarray        # (N, n_bus)
    v0: np.ndarray           # (N,)
    vm: np.ndarray           # (N, n_bus) exact voltage magnitudes
    vmin: np.ndarray
    vmax: np.ndarray
    margin: np.ndarray       # signed exact margin to the nearest limit
    violation: np.ndarray    # bool

    def __len__(self) -> int:
        return len(self.v0)


def sample(feeder: Feeder, dist: Distribution, n: int, rng: np.random.Generator) -> Dataset:
    nb = feeder.n_bus
    L = rng.uniform(*dist.load_factor, size=(n, 1))
    spread = rng.uniform(*dist.bus_spread, size=(n, nb))
    p = feeder.p_load_mw[None, :] * L * spread
    q = feeder.q_load_mvar[None, :] * L * spread

    night = rng.uniform(size=(n, 1)) < dist.p_night
    g = np.where(night, 0.0, rng.uniform(*dist.irradiance, size=(n, 1)))
    plant = rng.uniform(*dist.plant_spread, size=(n, len(feeder.pv_bus)))
    keep = rng.uniform(*dist.curtail_keep, size=(n, 1))
    pv = feeder.pv_rated_mw[None, :] * dist.pv_scale * g * plant * keep   # (n, n_pv)

    v0 = rng.uniform(*dist.v0, size=n)

    p_net = p.copy()
    p_net[:, feeder.pv_bus] -= pv
    q_net = q.copy()

    vm = solve_bfs(feeder, p_net, q_net, v0)
    margin = voltage_margin(vm, V_MIN, V_MAX)
    X = np.concatenate([p[:, 1:], q[:, 1:], pv, v0[:, None]], axis=1)
    return Dataset(name=dist.name, X=X, p_net=p_net, q_net=q_net, v0=v0, vm=vm,
                   vmin=vm.min(axis=1), vmax=vm.max(axis=1), margin=margin,
                   violation=margin < 0)


def feature_names(feeder: Feeder) -> list[str]:
    names = [f"P_load_bus{b}" for b in range(1, feeder.n_bus)]
    names += [f"Q_load_bus{b}" for b in range(1, feeder.n_bus)]
    names += [f"PV_bus{b}" for b in feeder.pv_bus]
    names += ["V0"]
    return names
