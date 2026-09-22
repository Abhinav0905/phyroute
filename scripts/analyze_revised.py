"""Aggregate the frozen evaluation without choosing new methods or thresholds."""
from pathlib import Path
import hashlib
import json
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revised" / "analysis"
FIG = ROOT / "figures" / "revised"
METRICS = ["unsafe_rate", "false_alarm_rate", "realized_escalation", "conditional_risk", "miss_rate"]
LABELS = {
    "nn_margin": "Neural margin", "nn_spread": "Ensemble spread", "spread_norm": "Spread-normalized margin",
    "classifier_conf": "Classifier confidence", "error_ratio": "Learned error ratio", "phy_d": "PhyRoute-D",
    "phy_min": "Minimum margin", "nn_approval_signed_physics": "Neural approval-only",
    "physics_only_abs_margin": "Physics only", "physics_approval_only": "Physics approval-only",
    "conservative_and": "Joint approval", "physics_residual_margin": "Physics-augmented MLP",
    "conformal_margin": "Conformal margin", "always_exact": "Always exact", "known_topology_fallback": "Known-change fallback",
}
MAIN = ["nn_margin", "nn_spread", "spread_norm", "classifier_conf", "error_ratio", "phy_d", "phy_min",
        "physics_only_abs_margin", "physics_approval_only", "conservative_and", "physics_residual_margin"]
COLORS = {"nn_margin": "#777777", "phy_d": "#0067a5", "physics_only_abs_margin": "#c24c19",
          "physics_residual_margin": "#298153"}


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, (np.floating, float)): return float(x) if math.isfinite(x) else None
    return x


def records(df):
    return clean(df.to_dict("records"))


def summarize(df, keys):
    rows = []
    for key, group in df.groupby(keys, dropna=False):
        if not isinstance(key, tuple): key = (key,)
        row = dict(zip(keys, key))
        for metric in METRICS:
            by_topology = group.groupby("topology_id")[metric].mean()
            row[metric] = float(by_topology.mean())
            row[metric + "_worst_topology"] = float(by_topology.max())
            by_seed = group.groupby("train_seed")[metric].mean()
            row[metric + "_seed_min"] = float(by_seed.min())
            row[metric + "_seed_max"] = float(by_seed.max())
        row.update({"topologies": group.topology_id.nunique(), "rows": len(group),
                    "unsafe_model_evaluations": int(group.unsafe.sum()),
                    "false_alarm_model_evaluations": int(group.false_alarms.sum()),
                    "resolved_model_evaluations": int(group.n.sum())})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    cfg = yaml.safe_load((ROOT / "configs/revised.yaml").read_text())
    inventory = json.loads((ROOT / cfg["topology_inventory"]).read_text())["topologies"]
    paths = sorted((ROOT / "results/revised/evaluation").glob("*.csv"))
    if not paths: raise SystemExit("No evaluation CSVs")
    for path in paths:
        meta=json.loads(path.with_suffix(".json").read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest()!=meta["sha256"]:
            raise ValueError(f"CSV hash mismatch: {path}")
        cell=pd.read_csv(path)
        if len(cell)!=549 or meta["rows"]!=549:
            raise ValueError(f"Incomplete policy matrix: {path}")
        keys=["policy","physics_variant","operating_mode","target"]
        if cell.duplicated(keys).any():
            raise ValueError(f"Duplicate policy rows: {path}")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    # Escalation consumes compute on every generated scenario, including rows
    # whose reference label is unresolved. Error rates retain resolved labels.
    df["resolved_escalation"] = df["realized_escalation"]
    df["realized_escalation"] = df["realized_escalation_all"]
    expected = {(tr, te, "id", top) for tr in cfg["training_seeds"] for te in cfg["test_seeds"]
                for top in ["base"] + [t["id"] for t in inventory]}
    expected |= {(tr, te, "pv_shift", "base") for tr in cfg["training_seeds"] for te in cfg["test_seeds"]}
    got = set(map(tuple, df[["train_seed", "test_seed", "condition", "topology_id"]].drop_duplicates().values.tolist()))
    if got != expected:
        raise SystemExit(f"Incomplete or unexpected evaluation: missing={len(expected-got)} extra={len(got-expected)}")
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((ROOT / "protocol/FROZEN.json").read_text())
    for f, h in frozen["files"].items():
        if hashlib.sha256((ROOT/f).read_bytes()).hexdigest() != h: raise ValueError(f"Frozen protocol changed: {f}")
    nominal = df[df.physics_variant.isin(["current", "not_applicable"])]
    primary = nominal[(nominal.partition == "heldout") & (nominal.operating_mode == "frozen_calibration")]
    summ = summarize(primary, ["policy", "target"])
    summ.to_csv(OUT/"nominal_primary.csv", index=False)
    mismatch = df[(df.partition == "heldout") & (df.operating_mode == "frozen_calibration")]
    mismatch_summary = summarize(mismatch, ["policy", "physics_variant", "target"])
    mismatch_summary.to_csv(OUT/"mismatch_primary.csv", index=False)
    by_topology = primary.groupby(["policy", "target", "topology_id"])[METRICS].mean().reset_index()
    by_topology.to_csv(OUT/"nominal_by_topology.csv", index=False)
    by_seed = primary.groupby(["policy", "target", "train_seed"])[METRICS].mean().reset_index()
    by_seed.to_csv(OUT/"nominal_by_training_seed.csv", index=False)
    conformal = nominal[(nominal.partition == "heldout") & (nominal.operating_mode == "conformal_calibration")]
    conformal_summary = summarize(conformal, ["policy", "target"])
    matched = nominal[(nominal.partition == "heldout") & (nominal.operating_mode == "matched_diagnostic")]
    matched_summary = summarize(matched, ["policy", "target"])
    matched_summary.to_csv(OUT/"matched_diagnostic.csv", index=False)
    conditions = nominal[(nominal.partition == "base") & (nominal.operating_mode.isin(["frozen_calibration", "conformal_calibration"]))]
    condition_summary = summarize(conditions, ["condition", "policy", "operating_mode", "target"])
    noroute = nominal[(nominal.partition == "heldout") & (nominal.operating_mode == "no_routing")]
    noroute_summary = summarize(noroute, ["policy"])
    coverage = df.groupby(["test_seed", "condition", "topology_id", "partition"])[["generated", "n", "unresolved"]].first().reset_index()
    coverage.to_csv(OUT/"reference_coverage.csv", index=False)
    # Fixed 10% thresholds: differences across 44 topology means are descriptive.
    at10 = by_topology[np.isclose(by_topology.target, .1)]
    comparisons = []
    for other in ["phy_d", "physics_residual_margin", "nn_margin"]:
        a = at10[at10.policy == other].set_index("topology_id")
        b = at10[at10.policy == "physics_only_abs_margin"].set_index("topology_id")
        delta = a[METRICS] - b[METRICS]
        comparisons.append({"method": other, "reference": "physics_only_abs_margin",
                            "mean_delta": delta.mean().to_dict(),
                            "topologies_lower_unsafe": int((delta.unsafe_rate < -1e-12).sum()),
                            "topologies_lower_false_alarm": int((delta.false_alarm_rate < -1e-12).sum()),
                            "topologies_no_worse_both_errors": int(((delta.unsafe_rate <= 1e-12) & (delta.false_alarm_rate <= 1e-12)).sum()),
                            "n_topologies": len(delta)})
    summary = {
        "interpretation": "Descriptive topology-macro means over 44 held-out exchanges; three training and three test seeds. Risk conditional on solver-resolved draws. Seed ranges are not confidence intervals.",
        "evaluation_files": len(paths), "rows": len(df), "expected_cells": len(expected),
        "nominal_primary": records(summ), "mismatch_primary": records(mismatch_summary),
        "conformal_heldout": records(conformal_summary), "matched_diagnostic": records(matched_summary),
        "base_conditions": records(condition_summary), "no_routing": records(noroute_summary),
        "paired_topology_comparisons": comparisons,
        "reference_coverage": {"generated": int(coverage.generated.sum()), "resolved": int(coverage.n.sum()),
                               "unresolved": int(coverage.unresolved.sum()),
                               "topologies_with_unresolved": coverage.loc[coverage.unresolved > 0, "topology_id"].unique().tolist()},
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }
    (OUT/"summary.json").write_text(json.dumps(clean(summary), indent=2, allow_nan=False)+"\n")

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans"})
    short_labels={"phy_d":"PhyRoute-D","physics_only_abs_margin":"Physics only","physics_residual_margin":"Residual MLP"}
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.75), layout="constrained")
    for pol in short_labels:
        color=COLORS[pol]
        x = at10[at10.policy == pol]
        for ax, metric, label in zip(axes, ["unsafe_rate", "false_alarm_rate"], ["Unsafe approvals (%)", "False rejections (%)"]):
            ax.scatter(x.realized_escalation*100, x[metric]*100, s=13, color=color, alpha=.35)
            ax.scatter(x.realized_escalation.mean()*100, x[metric].mean()*100, s=70,
                       marker="D", color=color, edgecolor="white", linewidth=.6, label=short_labels[pol])
            ax.set_xlabel("Exact-solver escalation (%)"); ax.set_ylabel(label); ax.grid(alpha=.18)
    axes[0].legend(fontsize=7, loc="upper right")
    fig.savefig(FIG/"nominal_tradeoff.pdf"); fig.savefig(FIG/"nominal_tradeoff.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.75), layout="constrained")
    scales = sorted(cfg["impedance_scales"] + [1.0])
    for pol in ["phy_d", "physics_only_abs_margin", "physics_residual_margin"]:
        ys = {metric: [] for metric in ["unsafe_rate", "false_alarm_rate"]}
        mins, maxs = {m: [] for m in ys}, {m: [] for m in ys}
        for scale in scales:
            variant = "current" if scale == 1 else f"impedance_{scale:g}"
            row = mismatch_summary[(mismatch_summary.policy == pol) & np.isclose(mismatch_summary.target, .1) & (mismatch_summary.physics_variant == variant)].iloc[0]
            for m in ys:
                ys[m].append(row[m]*100); mins[m].append(row[m+"_seed_min"]*100); maxs[m].append(row[m+"_seed_max"]*100)
        for ax, metric, label in zip(axes, ys, ["Unsafe approvals (%)", "False rejections (%)"]):
            ax.plot(scales, ys[metric], "o-", ms=3.5, color=COLORS[pol], label=short_labels[pol])
            ax.fill_between(scales, mins[metric], maxs[metric], color=COLORS[pol], alpha=.12)
            ax.set_xlabel("Verifier impedance multiplier"); ax.set_ylabel(label); ax.grid(alpha=.18)
    axes[0].legend(fontsize=7); fig.savefig(FIG/"impedance_sensitivity.pdf"); fig.savefig(FIG/"impedance_sensitivity.png", dpi=220); plt.close(fig)

    print(json.dumps({k:v for k,v in clean(summary).items() if k in ["evaluation_files","rows","reference_coverage","paired_topology_comparisons"]}, indent=2))
    print(summ[np.isclose(summ.target,.1)][["policy"]+METRICS].to_string(index=False))


if __name__ == "__main__": main()
