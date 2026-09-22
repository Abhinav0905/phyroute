"""Radial feeder data: IEEE 33-bus (Baran & Wu) with added PV plants.

All electrical quantities are kept in per-unit on the pandapower base
(S_base = 10 MVA, V_base = 12.66 kV) so that results can be cross-checked
against pandapower directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandapower as pp
import pandapower.networks as pn

# PV plants: bus index (0-based, pandapower numbering) -> rated active power in MW.
# Placement favours feeder ends and laterals, where voltage rise is largest.
DEFAULT_PV_MW = {17: 1.0, 21: 0.6, 24: 0.8, 32: 1.0, 12: 0.6, 29: 0.5}

V_MIN = 0.95  # p.u. lower voltage limit chosen for this benchmark
V_MAX = 1.05  # p.u. upper voltage limit chosen for this benchmark


@dataclass
class Feeder:
    """Static description of a radial feeder rooted at bus 0."""

    n_bus: int
    parent: np.ndarray          # parent[j] for j >= 1, parent[0] = -1
    order: np.ndarray           # buses in breadth-first order from the root
    z_pu: np.ndarray            # complex series impedance of the line into bus j (p.u.), z[0] = 0
    p_load_mw: np.ndarray       # base active load per bus (MW), zero at the root
    q_load_mvar: np.ndarray     # base reactive load per bus (MVAr)
    pv_bus: np.ndarray          # buses hosting PV plants
    pv_rated_mw: np.ndarray     # rated MW per PV plant
    s_base_mva: float = 10.0
    v_base_kv: float = 12.66
    net: object = field(default=None, repr=False)  # the pandapower net (for validation)

    @property
    def z_base_ohm(self) -> float:
        return self.v_base_kv ** 2 / self.s_base_mva

    @property
    def load_bus(self) -> np.ndarray:
        return np.arange(1, self.n_bus)

    def children(self) -> list[list[int]]:
        ch: list[list[int]] = [[] for _ in range(self.n_bus)]
        for j in range(1, self.n_bus):
            ch[self.parent[j]].append(j)
        return ch


def build_feeder(pv_mw: dict[int, float] | None = None, pv_scale: float = 1.0,
                 open_lines: tuple[int, ...] = (), close_lines: tuple[int, ...] = ()) -> Feeder:
    """Load pandapower's case33bw and derive the radial structure.

    open_lines / close_lines are pandapower line indices to switch out / in,
    which lets us build a reconfigured (branch-exchanged) copy of the feeder.
    """
    pv_mw = dict(DEFAULT_PV_MW if pv_mw is None else pv_mw)
    net = pn.case33bw()
    for idx in open_lines:
        net.line.loc[idx, "in_service"] = False
    for idx in close_lines:
        net.line.loc[idx, "in_service"] = True
    if int(net.line.in_service.sum()) != len(net.bus) - 1:
        raise ValueError("reconfiguration must keep exactly n_bus - 1 lines in service")
    n_bus = len(net.bus)
    lines = net.line[net.line.in_service]
    z_base = net.bus.vn_kv.iloc[0] ** 2 / net.sn_mva

    parent = -np.ones(n_bus, dtype=int)
    z_pu = np.zeros(n_bus, dtype=complex)
    adj: dict[int, list[tuple[int, complex]]] = {i: [] for i in range(n_bus)}
    for _, ln in lines.iterrows():
        z = complex(ln.r_ohm_per_km, ln.x_ohm_per_km) * ln.length_km / z_base
        adj[int(ln.from_bus)].append((int(ln.to_bus), z))
        adj[int(ln.to_bus)].append((int(ln.from_bus), z))

    # breadth-first traversal from the slack bus
    order = [0]
    seen = {0}
    queue = [0]
    while queue:
        i = queue.pop(0)
        for j, z in adj[i]:
            if j not in seen:
                seen.add(j)
                parent[j] = i
                z_pu[j] = z
                order.append(j)
                queue.append(j)
    if len(order) != n_bus:
        raise ValueError("feeder is not a connected radial tree")

    p_load = np.zeros(n_bus)
    q_load = np.zeros(n_bus)
    for _, ld in net.load.iterrows():
        p_load[int(ld.bus)] += ld.p_mw
        q_load[int(ld.bus)] += ld.q_mvar

    pv_bus = np.array(sorted(pv_mw), dtype=int)
    pv_rated = np.array([pv_mw[b] for b in pv_bus]) * pv_scale
    return Feeder(
        n_bus=n_bus, parent=parent, order=np.array(order), z_pu=z_pu,
        p_load_mw=p_load, q_load_mvar=q_load, pv_bus=pv_bus, pv_rated_mw=pv_rated,
        s_base_mva=float(net.sn_mva), v_base_kv=float(net.bus.vn_kv.iloc[0]), net=net,
    )


def pandapower_voltages(feeder: Feeder, p_mw: np.ndarray, q_mvar: np.ndarray,
                        pv_mw: np.ndarray, v0: float) -> np.ndarray:
    """Reference solve with pandapower for one scenario. Returns |V| per bus (p.u.)."""
    net = feeder.net
    net.load["p_mw"] = p_mw[net.load.bus.values]
    net.load["q_mvar"] = q_mvar[net.load.bus.values]
    if "sgen" in net and len(net.sgen):
        net.sgen.drop(net.sgen.index, inplace=True)
    for b, p in zip(feeder.pv_bus, pv_mw):
        pp.create_sgen(net, bus=int(b), p_mw=float(p), q_mvar=0.0)
    net.ext_grid["vm_pu"] = v0
    pp.runpp(net, numba=False, tolerance_mva=1e-10)
    vm = net.res_bus.vm_pu.values.copy()
    net.sgen.drop(net.sgen.index, inplace=True)
    return vm
