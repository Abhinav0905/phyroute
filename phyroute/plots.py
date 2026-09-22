"""Figures for the paper. All figures are written as PNG at 200 dpi."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "random": "Random",
    "nn_uncertainty": "Ensemble spread",
    "nn_margin": "Model margin",
    "chance": "Chance-constraint (z-score)",
    "clf_conf": "Classifier confidence",
    "delegation_value": "Delegation-value probe",
    "learned_ratio": "Learned error/margin probe",
    "phy_c": "PhyRoute-C (physics margin only)",
    "phy_min": "PhyRoute-min (smaller margin)",
    "phy_d": "PhyRoute-D (physics disagreement first)",
}
STYLE = {
    "random": dict(color="0.6", ls=":"),
    "nn_uncertainty": dict(color="tab:orange", ls="--"),
    "nn_margin": dict(color="tab:blue", ls="--"),
    "chance": dict(color="tab:purple", ls="--"),
    "clf_conf": dict(color="tab:brown", ls="-."),
    "delegation_value": dict(color="tab:green", ls="-."),
    "learned_ratio": dict(color="darkgreen", ls="-."),
    "phy_c": dict(color="tab:red", ls="-", lw=1.4, alpha=0.8),
    "phy_min": dict(color="firebrick", ls="-", lw=1.4, alpha=0.6),
    "phy_d": dict(color="black", ls="-", lw=2.4),
}


def frontier_figure(fronts: dict[str, dict], title: str, path: str, xmax: float = 0.4):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for pol, fr in fronts.items():
        ax.plot(fr["rate"] * 100, fr["unsafe"] * 100, label=LABELS[pol], **STYLE[pol])
    ax.set_xlim(0, xmax * 100)
    ax.set_xlabel("Decisions escalated to the exact solver (%)")
    ax.set_ylabel("Unsafe decisions (% of all decisions)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, ncol=1, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def quadrant_figure(sigma, m_lin, miss, title: str, path: str, sigma_med: float, m_med: float,
                    n_show: int = 4000, rng=None):
    rng = rng or np.random.default_rng(0)
    idx = rng.choice(len(sigma), size=min(n_show, len(sigma)), replace=False)
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    ok = idx[~miss[idx]]
    bad = np.where(miss)[0]
    ax.scatter(np.abs(m_lin[ok]), sigma[ok], s=4, c="0.7", label="tiny decision correct")
    ax.scatter(np.abs(m_lin[bad]), sigma[bad], s=14, c="tab:red", marker="x",
               label="unsafe tiny decision")
    ax.axvline(m_med, color="k", lw=0.8, ls="--")
    ax.axhline(sigma_med, color="k", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|physics margin| from LinDistFlow (p.u.)")
    ax.set_ylabel("ensemble spread of predicted margin (p.u.)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower left")
    ax.text(0.02, 0.97, "hidden danger:\nlow spread, small margin", transform=ax.transAxes,
            va="top", fontsize=8, color="tab:red")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def deployed_figure(rows: list[dict], set_key: str, title: str, path: str):
    """Realised escalation and unsafe rate when thresholds fixed on ID data are used on another set."""
    pols = sorted({r["policy"] for r in rows}, key=lambda p: list(LABELS).index(p))
    targets = sorted({r["target_rate"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    w = 0.8 / len(targets)
    x = np.arange(len(pols))
    for k, t in enumerate(targets):
        esc = [next(r for r in rows if r["policy"] == p and r["target_rate"] == t)[f"realised_rate_{set_key}"] for p in pols]
        uns = [next(r for r in rows if r["policy"] == p and r["target_rate"] == t)[f"unsafe_rate_{set_key}"] for p in pols]
        axes[0].bar(x + k * w, np.array(esc) * 100, width=w, label=f"target {t:.0%}")
        axes[1].bar(x + k * w, np.array(uns) * 100, width=w, label=f"target {t:.0%}")
    for ax, yl in zip(axes, ["realised escalation (%)", "unsafe decisions (% of all decisions)"]):
        ax.set_xticks(x + w * (len(targets) - 1) / 2)
        ax.set_xticklabels([LABELS[p] for p in pols], rotation=35, ha="right", fontsize=7.5)
        ax.set_ylabel(yl)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def three_tier_figure(curves: dict[str, list[dict]], path: str):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, rows in curves.items():
        ax.plot([r["exact_rate"] * 100 for r in rows], [r["unsafe_rate"] * 100 for r in rows],
                marker="o", ms=3, label=name)
    ax.set_xlabel("Decisions escalated to the exact solver (%)")
    ax.set_ylabel("Unsafe decisions (% of all decisions)")
    ax.set_title("Two-tier versus three-tier cascades (voltage-worsening reconfiguration)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
