"""Frozen-protocol experiments with paired topologies and independent calibration.

This module never loads pickle checkpoints. All reusable arrays have manifests
covering their configuration, input sources and byte-level content hashes.
"""
from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import warnings

import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
import yaml

from .checkpoint import load_checkpoint, save_checkpoint
from .feeder import V_MAX, V_MIN, build_feeder
from .lindistflow import LinDistFlow
from .models import NumpyMLP
from .powerflow import solve_bfs, voltage_margin
from .scenarios import ID, OOD, sample
from .topology_model import predict_topology_model, train_topology_model

LEARNED_POLICIES = ("nn_margin", "nn_spread", "spread_norm", "classifier_conf", "error_ratio")
PHYSICS_POLICIES = ("phy_d", "phy_min", "nn_approval_signed_physics", "physics_only_abs_margin",
                    "physics_approval_only", "conservative_and", "physics_residual_margin")
SOURCE_FILES = ("feeder.py", "powerflow.py", "scenarios.py", "lindistflow.py", "models.py",
                "checkpoint.py", "revised.py", "topology_model.py", "reference.py")


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_hashes() -> dict:
    root = Path(__file__).resolve().parent
    return {name: file_hash(root / name) for name in SOURCE_FILES}


def environment() -> dict:
    versions = {}
    for package in ("numpy", "scipy", "pandas", "pandapower", "scikit-learn", "pyyaml", "threadpoolctl"):
        versions[package] = importlib.metadata.version(package)
    from threadpoolctl import threadpool_info
    return {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "packages": versions, "threadpools": threadpool_info(),
            "thread_environment": {k: os.environ.get(k) for k in
                                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def save_arrays(path, arrays, identity, details=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if any(np.asarray(v).dtype.hasobject for v in arrays.values()):
        raise ValueError("object arrays are forbidden")
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as f:
        np.savez(f, **arrays)
    temp.replace(path)
    atomic_json(path.with_suffix(".json"), {"identity": identity, "sha256": file_hash(path), "details": details or {}})


def load_arrays(path, identity):
    path = Path(path)
    if not path.exists():
        return None
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta["identity"] != identity:
        raise ValueError(f"cache identity mismatch: {path}; use a new output directory")
    if file_hash(path) != meta["sha256"]:
        raise ValueError(f"cache content hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def read_protocol(config_path):
    config_path = Path(config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text())
    root = config_path.parent.parent
    inventory_path = root / cfg["topology_inventory"]
    inventory = json.loads(inventory_path.read_text())
    topologies = inventory["topologies"] if isinstance(inventory, dict) else inventory
    base = {"id": "base", "open_line": None, "close_line": None, "partition": "base"}
    topologies = [base] + [t for t in topologies if t["id"] != "base"]
    identifiers = [t["id"] for t in topologies]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate topology IDs")
    return cfg, topologies, root


def feeder_for(topology):
    if topology["open_line"] is None:
        return build_feeder()
    return build_feeder(open_lines=(topology["open_line"],), close_lines=(topology["close_line"],))


def extrema_from_linear(feeder, data):
    vm = LinDistFlow(feeder).voltages(data["p_net"], data["q_net"], data["v0"])
    if not np.isfinite(vm).all():
        raise ValueError("nonfinite linear proxy")
    return np.column_stack([vm.min(axis=1), vm.max(axis=1)])


def margin_from_extrema(extrema):
    return np.minimum(V_MAX-extrema[:, 1], extrema[:, 0]-V_MIN)


def margin_from_linear(feeder, data):
    return margin_from_extrema(extrema_from_linear(feeder, data))


def physics_variants(feeder, data, cfg, topology, include_extrema=False):
    """Truth remains fixed; variants alter only the proxy network model."""
    extrema = {"current": extrema_from_linear(feeder, data), "stale": extrema_from_linear(build_feeder(), data)}
    for scale in cfg["impedance_scales"]:
        extrema[f"impedance_{scale:g}"] = extrema_from_linear(replace(feeder, z_pu=feeder.z_pu * scale), data)
    seed = np.random.SeedSequence([int(cfg["perturbation_seed"]),
                                  0 if topology["open_line"] is None else int(topology["open_line"]) + 1,
                                  0 if topology["close_line"] is None else int(topology["close_line"]) + 1])
    factors = 1 + np.random.default_rng(seed).uniform(-.1, .1, (2, feeder.n_bus))
    factors[:, 0] = 1
    noisy_z = feeder.z_pu.real * factors[0] + 1j * feeder.z_pu.imag * factors[1]
    extrema["branch_noise_10pct"] = extrema_from_linear(replace(feeder, z_pu=noisy_z), data)
    margins = {k: margin_from_extrema(v) for k, v in extrema.items()}
    if not include_extrema: return margins
    return {**{f"lin__{k}": v for k, v in margins.items()},
            **{f"extrema__{k}": v for k, v in extrema.items()}}


def scenario_arrays(feeder, dist, n, seed):
    ds = sample(feeder, dist, n, np.random.default_rng(seed))
    return {"X": ds.X, "p_net": ds.p_net, "q_net": ds.q_net, "v0": ds.v0,
            "margin": ds.margin, "Y": np.column_stack([ds.vmin, ds.vmax])}


def policy(name, pred, lin):
    nn, sigma = pred["m_nn"], pred["sigma"]
    a = nn >= 0
    if name == "nn_margin": return a, -np.abs(nn)
    if name == "nn_spread": return a, sigma
    if name == "spread_norm": return a, -np.abs(nn) / (sigma + 1e-6)
    if name == "classifier_conf": return a, -np.abs(pred["p_viol"] - .5)
    if name == "error_ratio": return a, pred["error_ratio"]
    if name == "physics_residual_margin":
        return pred["m_residual"] >= 0, -np.abs(pred["m_residual"])
    b = lin >= 0
    if name == "phy_d": return a, np.where(a != b, np.abs(lin), -np.abs(lin))
    if name == "phy_min": return a, -np.minimum(np.abs(lin), np.abs(nn))
    if name == "nn_approval_signed_physics": return a, np.where(a, -lin, -np.inf)
    if name == "physics_only_abs_margin": return b, -np.abs(lin)
    if name == "physics_approval_only": return b, np.where(b, -lin, -np.inf)
    if name == "conservative_and": return a & b, -np.abs(lin)
    raise KeyError(name)


def budget_threshold(score, rate):
    """Quantile threshold, with a finite sentinel for noneligible approvals.

    Target budgets exceeding the eligible fraction cannot force useless solves.
    Ties are never broken using labels; realized budgets are always recorded.
    """
    score = np.asarray(score)
    if np.isnan(score).any() or np.isposinf(score).any():
        raise ValueError("routing scores must be finite or negative infinity")
    if not 0 <= rate <= 1: raise ValueError("budget outside [0,1]")
    if rate == 0: return float("inf")
    finite = score[np.isfinite(score)]
    if len(finite) == 0: return float("inf")
    # This finite sentinel preserves the exact rank order, including all ties.
    sentinel = float(finite.min() - max(1.0, np.ptp(finite)))
    ranked = np.where(np.isfinite(score), score, sentinel)
    if rate == 1: return float(np.nextafter(sentinel, -np.inf))
    return float(np.quantile(ranked, 1-rate))


def conformal_radius(calibration_residuals, alpha):
    residuals = np.asarray(calibration_residuals)
    if not 0 < alpha < 1 or residuals.ndim != 1 or not len(residuals):
        raise ValueError("invalid conformal calibration")
    if not np.isfinite(residuals).all() or (residuals < 0).any():
        raise ValueError("invalid calibration residual")
    order = int(math.ceil((len(residuals) + 1) * (1-alpha)))
    return float("inf") if order > len(residuals) else float(np.partition(residuals, order-1)[order-1])


def decision_counts(approve, violation, escalate, valid=None):
    approve, violation, escalate = (np.asarray(v, dtype=bool) for v in (approve, violation, escalate))
    generated = len(approve)
    if generated == 0 or approve.shape != violation.shape or approve.shape != escalate.shape:
        raise ValueError("invalid decision arrays")
    valid = np.ones(generated, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if valid.shape != approve.shape: raise ValueError("invalid reference mask")
    unresolved = int((~valid).sum())
    extra = {"generated": generated, "unresolved": unresolved,
             "accepted_unresolved": int((approve & ~valid).sum()),
             "escalated_unresolved": int((escalate & ~valid).sum()),
             "realized_escalation_all": float(escalate.mean())}
    approve, violation, escalate = approve[valid], violation[valid], escalate[valid]
    n = len(approve)
    accepted = int(approve.sum()); viols = int(violation.sum()); safe = n-viols
    unsafe = int((approve & violation).sum()); fa = int((~approve & ~violation).sum())
    return {**extra, "n": n, "accepted": accepted, "violations": viols, "unsafe": unsafe, "false_alarms": fa,
            "escalated": int(escalate.sum()), "realized_escalation": float(escalate.mean()) if n else None,
            "acceptance_coverage": accepted/n if n else None,
            "unsafe_rate": unsafe/n if n else None, "conditional_risk": unsafe/accepted if accepted else None,
            "miss_rate": unsafe/viols if viols else None, "false_alarm_rate": fa/safe if safe else None,
            "accuracy": 1-(unsafe+fa)/n if n else None}


def _fit(X, Y, hidden, seed, max_iter, classifier=False):
    xs = StandardScaler().fit(X)
    ys = None if classifier else StandardScaler().fit(Y)
    cls = MLPClassifier if classifier else MLPRegressor
    m = cls(hidden_layer_sizes=tuple(hidden), activation="relu", solver="adam", learning_rate_init=1e-3,
            batch_size=256, max_iter=max_iter, early_stopping=True, validation_fraction=.1,
            n_iter_no_change=25, random_state=seed, tol=1e-7)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.fit(xs.transform(X), Y.astype(int) if classifier else ys.transform(Y))
    history = {"seed": int(seed), "iterations": int(m.n_iter_), "elapsed_s": time.perf_counter()-start,
               "loss_curve": [float(x) for x in m.loss_curve_],
               "validation_scores": [float(x) for x in m.validation_scores_],
               "best_validation_score": float(m.best_validation_score_),
               "warnings": [{"category": type(w.message).__name__, "message": str(w.message)} for w in caught]}
    return NumpyMLP.from_sklearn(m, xs, ys, classifier=classifier), history


def tiny_predictions(models, X, details=False):
    names = sorted(k for k in models if k.startswith("tiny_"))
    if len(names) < 2: raise ValueError("at least two ensemble members required")
    outputs = np.stack([models[k].forward(X) for k in names])
    margins = np.minimum(V_MAX-outputs[:, :, 1], outputs[:, :, 0]-V_MIN)
    canonical = margin_from_extrema(outputs.mean(0))
    if details:
        return canonical, margins.std(0, ddof=1), margins.mean(0)
    return canonical, margins.std(0, ddof=1)


def predictions(models, X):
    nn, sigma, legacy = tiny_predictions(models, X, details=True)
    features = np.column_stack([X, nn, sigma])
    err = 10 ** models["error_probe"].forward(features)[:, 0]
    out = {"m_nn": nn, "legacy_mean_margin": legacy, "sigma": sigma, "p_viol": models["classifier"].forward(X),
           "error_ratio": err/(np.abs(nn)+1e-6)}
    if not all(np.isfinite(x).all() for x in out.values()):
        raise ValueError("nonfinite model prediction")
    return out


class Experiment:
    def __init__(self, config_path, output=None):
        self.cfg, self.topologies, self.root = read_protocol(config_path)
        self.output = Path(output).resolve() if output else self.root / "results" / "revised"
        self.output.mkdir(parents=True, exist_ok=True)
        self.sources = source_hashes()
        self.identity = {"config": self.cfg, "topologies": self.topologies, "source_sha256": self.sources}
        self.run_hash = canonical_hash(self.identity)
        run_manifest = self.output / "run_manifest.json"
        if run_manifest.exists() and json.loads(run_manifest.read_text())["run_hash"] != self.run_hash:
            raise ValueError("output run manifest disagrees with code or protocol; use a new output directory")
        if not run_manifest.exists():
            atomic_json(run_manifest, {"run_hash": self.run_hash, **self.identity, "environment": environment()})

    def cache_identity(self, kind, **kw):
        return {"run_hash": self.run_hash, "kind": kind, **kw}

    def train_data(self, seed):
        if seed not in self.cfg["training_seeds"]: raise ValueError("training seed outside frozen protocol")
        out = {}
        for stream, (split, count) in enumerate((("train", self.cfg["n_train"]), ("probe", self.cfg["n_probe"]),
                                                ("calibration", self.cfg["n_calibration"]))):
            identity = self.cache_identity("training_data", seed=seed, split=split)
            path = self.output / "data" / f"train_{seed}__{split}.npz"
            d = load_arrays(path, identity)
            if d is None:
                d = scenario_arrays(build_feeder(), ID, count, np.random.SeedSequence([seed, stream]))
                save_arrays(path, d, identity, {"seed_sequence_entropy": [seed, stream]})
            out[split] = d
        return out

    def train(self, seed):
        path = self.output / "models" / f"train_{seed}.npz"
        checkpoint_cfg = self.cache_identity("models", seed=seed)
        if path.exists():
            models, _ = load_checkpoint(path, expected_config=checkpoint_cfg)
            print(f"[train {seed}] validated cached checkpoint", flush=True)
            return models
        data = self.train_data(seed); train, probe = data["train"], data["probe"]
        models, histories = {}, {}
        for i in range(self.cfg["n_tiny"]):
            name = f"tiny_{i}"
            print(f"[train {seed}] fitting {name}", flush=True)
            models[name], histories[name] = _fit(train["X"], train["Y"], self.cfg["tiny_hidden"],
                                                 seed+i, self.cfg["tiny_max_iter"])
            atomic_json(self.output / "models" / f"train_{seed}__history.json", histories)
        print(f"[train {seed}] fitting classifier and error probe", flush=True)
        models["classifier"], histories["classifier"] = _fit(train["X"], train["margin"] < 0,
                self.cfg["tiny_hidden"], seed+200, self.cfg["probe_max_iter"], classifier=True)
        nn, sigma = tiny_predictions(models, probe["X"])
        Z = np.column_stack([probe["X"], nn, sigma])
        target = np.log10(np.abs(nn-probe["margin"])+1e-6)[:, None]
        models["error_probe"], histories["error_probe"] = _fit(Z, target, self.cfg["tiny_hidden"],
                                                             seed+400, self.cfg["probe_max_iter"])
        details = {"source_sha256": self.sources, "environment": environment(), "histories": histories,
                   "training_seed": seed, "calibration_used_for_fitting": False}
        save_checkpoint(path, models, config=checkpoint_cfg, source_metadata=details)
        atomic_json(self.output / "models" / f"train_{seed}__history.json", details)
        print(f"[train {seed}] saved {path}", flush=True)
        return models

    def test_data(self, seed, condition="id"):
        if seed not in self.cfg["test_seeds"]: raise ValueError("test seed outside frozen protocol")
        identity = self.cache_identity("test_data", seed=seed, condition=condition)
        path = self.output / "data" / f"test_{seed}__{condition}.npz"
        d = load_arrays(path, identity)
        if d is None:
            d = scenario_arrays(build_feeder(), ID if condition == "id" else OOD, self.cfg["n_test"],
                                np.random.SeedSequence([seed, 0 if condition == "id" else 1]))
            save_arrays(path, d, identity)
        return d

    def topology_data(self, seed, topology, condition="id"):
        identity = self.cache_identity("topology_data", seed=seed, topology_id=topology["id"], condition=condition)
        path = self.output / "physics" / f"test_{seed}__{condition}__{topology['id']}.npz"
        d = load_arrays(path, identity)
        if d is not None: return d
        injections = self.test_data(seed, condition)
        f = feeder_for(topology)
        from .reference import solve_reference
        vm, valid, info = solve_reference(f, injections["p_net"], injections["q_net"], injections["v0"])
        d = {"margin": voltage_margin(vm, V_MIN, V_MAX), "valid": valid}
        d.update(physics_variants(f, injections, self.cfg, topology, include_extrema=True))
        noise_seed = [int(self.cfg["perturbation_seed"]),
                      0 if topology["open_line"] is None else int(topology["open_line"])+1,
                      0 if topology["close_line"] is None else int(topology["close_line"])+1]
        factors = 1 + np.random.default_rng(np.random.SeedSequence(noise_seed)).uniform(-.1, .1, (2, f.n_bus))
        factors[:, 0] = 1
        d["branch_noise_factors_rx"] = factors
        save_arrays(path, d, identity, {"solver": info, "paired_injections": f"test_{seed}__{condition}"})
        return d

    def prepare(self, seed, partition="all", topology_id=None):
        self.test_data(seed)
        for top in self.topologies:
            if topology_id and top["id"] != topology_id: continue
            if partition != "all" and top["partition"] not in (partition, "base"): continue
            self.topology_data(seed, top)
            print(f"[prepare {seed}] {top['id']}", flush=True)
        if topology_id in (None, "base"):
            self.topology_data(seed, self.topologies[0], "pv_shift")

    def cached_predictions(self, models, training_seed, test_seed, condition):
        path = self.output / "predictions" / f"train_{training_seed}__test_{test_seed}__{condition}.npz"
        identity = self.cache_identity("predictions", training_seed=training_seed, test_seed=test_seed, condition=condition)
        d = load_arrays(path, identity)
        if d is None:
            d = predictions(models, self.test_data(test_seed, condition)["X"])
            save_arrays(path, d, identity)
        return d

    def evaluate(self, training_seed, test_seed, partition="all", topology_id=None):
        models = self.train(training_seed)
        residual_model, _ = train_topology_model(self.root, self.cfg, training_seed)
        calibration = self.train_data(training_seed)["calibration"]
        cal_pred = predictions(models, calibration["X"])
        cal_extrema = extrema_from_linear(build_feeder(), calibration)
        cal_lin = margin_from_extrema(cal_extrema)
        _, cal_pred["m_residual"] = predict_topology_model(residual_model, calibration["X"], cal_extrema)
        thresholds = {name: {str(r): budget_threshold(policy(name, cal_pred, cal_lin)[1], r)
                            for r in self.cfg["calibration_budgets"]}
                      for name in LEARNED_POLICIES + PHYSICS_POLICIES}
        radii = {str(alpha): conformal_radius(np.abs(cal_pred["m_nn"]-calibration["margin"]), alpha)
                 for alpha in self.cfg["conformal_alpha"]}
        atomic_json(self.output / "models" / f"train_{training_seed}__calibration.json",
                    {"run_hash": self.run_hash, "thresholds": thresholds,
                     "conformal_radii": {k: v if np.isfinite(v) else "inf" for k, v in radii.items()},
                     "ensemble_mean_margin_disagreements": int(((cal_pred["m_nn"] >= 0) !=
                                                                 (cal_pred["legacy_mean_margin"] >= 0)).sum()),
                     "ensemble_mean_margin_max_difference": float(np.max(np.abs(cal_pred["m_nn"]-cal_pred["legacy_mean_margin"]))),
                     "calibration_n": len(calibration["X"]), "physics_model": "nominal_base"})
        jobs = [(top, "id") for top in self.topologies
                if (not topology_id or top["id"] == topology_id)
                and (partition == "all" or top["partition"] in (partition, "base"))]
        if topology_id in (None, "base"): jobs.append((self.topologies[0], "pv_shift"))
        paths = []
        for top, condition in jobs:
            name = f"train_{training_seed}__test_{test_seed}__{condition}__{top['id']}"
            path = self.output / "evaluation" / f"{name}.csv"
            identity = self.cache_identity("evaluation", training_seed=training_seed, test_seed=test_seed,
                                           topology_id=top["id"], condition=condition)
            if path.exists():
                meta = json.loads(path.with_suffix(".json").read_text())
                if meta["identity"] != identity or meta["sha256"] != file_hash(path):
                    raise ValueError(f"evaluation cache mismatch: {path}")
                paths.append(str(path)); continue
            d = self.topology_data(test_seed, top, condition)
            pred = self.cached_predictions(models, training_seed, test_seed, condition)
            valid = d["valid"]
            if not np.isfinite(d["margin"][valid]).all(): raise ValueError("nonfinite resolved truth")
            violation = valid & (d["margin"] < 0)
            X = self.test_data(test_seed, condition)["X"]
            base = {"train_seed": training_seed, "test_seed": test_seed, "condition": condition,
                    "topology_id": top["id"], "partition": top["partition"]}
            rows = []

            def append(policy_name, variant, mode, target, tau, cheap, esc):
                result = np.where(esc, valid & ~violation, cheap)
                # JSON/CSV do not rely on nonstandard Infinity values.
                threshold_value = tau if tau is None or np.isfinite(tau) else ("inf" if tau > 0 else "-inf")
                rows.append({**base, "physics_variant": variant, "policy": policy_name, "operating_mode": mode,
                             "target": target, "threshold": threshold_value, **decision_counts(result, violation, esc, valid)})

            for pol in LEARNED_POLICIES + PHYSICS_POLICIES:
                variants = [("not_applicable", cal_lin)] if pol in LEARNED_POLICIES else [
                    (k.removeprefix("lin__"), v) for k, v in d.items() if k.startswith("lin__")]
                for variant, lin in variants:
                    if pol == "physics_residual_margin":
                        _, pred["m_residual"] = predict_topology_model(residual_model, X, d[f"extrema__{variant}"])
                    cheap, score = policy(pol, pred, lin)
                    append(pol, variant, "no_routing", 0, None, cheap, np.zeros_like(violation))
                    for r in self.cfg["calibration_budgets"]:
                        tau = thresholds[pol][str(r)]
                        append(pol, variant, "frozen_calibration", r, tau, cheap, score > tau)
                    for r in self.cfg["matched_budgets"]:
                        tau = budget_threshold(score, r)
                        append(pol, variant, "matched_diagnostic", r, tau, cheap, score > tau)
            for alpha in self.cfg["conformal_alpha"]:
                radius = radii[str(alpha)]
                append("conformal_margin", "not_applicable", "conformal_calibration", alpha, radius,
                       pred["m_nn"] >= 0, np.abs(pred["m_nn"]) <= radius)
            append("always_exact", "not_applicable", "always", 1, None, ~violation, np.ones_like(violation))
            changed = top["id"] != "base"
            append("known_topology_fallback", "not_applicable", "always", int(changed), None,
                   pred["m_nn"] >= 0, np.full_like(violation, changed))
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(path.name + ".tmp")
            with temp.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            temp.replace(path)
            atomic_json(path.with_suffix(".json"), {"identity": identity, "sha256": file_hash(path), "rows": len(rows)})
            paths.append(str(path))
            print(f"[evaluate {training_seed}/{test_seed}] {condition}/{top['id']}: {len(rows)} rows", flush=True)
        return paths
