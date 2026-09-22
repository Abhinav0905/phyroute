"""Physics-augmented residual baseline fitted only on development topologies."""
from __future__ import annotations

import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .checkpoint import load_checkpoint, save_checkpoint
from .feeder import V_MAX, V_MIN, build_feeder
from .lindistflow import LinDistFlow
from .models import NumpyMLP
from .scenarios import ID, sample


def voltage_extrema(vm):
    vm = np.asarray(vm)
    return np.stack([vm.min(axis=1), vm.max(axis=1)], axis=1)


def predict_topology_model(model: NumpyMLP, X, lin_voltage_extrema):
    """Predict current extrema by correcting the supplied physics approximation.

    The supplied extrema must use the same verifier variant as the other
    physics methods; this function never looks up the true network.
    """
    lin = np.asarray(lin_voltage_extrema)
    pred = lin + model.forward(np.concatenate([X, lin], axis=1))
    margin = np.minimum(V_MAX - pred[:, 1], pred[:, 0] - V_MIN)
    return pred, margin


def train_topology_model(root, cfg, train_seed, force=False):
    """Fit or validate/reuse one 64x64 residual model; return model and metadata."""
    root = Path(root)
    path = root / "results" / "models" / f"topology_{train_seed}.npz"
    model_cfg = {"study_config": cfg, "train_seed": int(train_seed),
                 "model_kind": "development_topology_physics_residual"}
    if path.exists() and not force:
        models, meta = load_checkpoint(path, expected_config=model_cfg)
        return models["residual"], meta
    started = time.perf_counter()
    inventory_path = root / cfg["topology_inventory"]
    inventory = json.loads(inventory_path.read_text())["topologies"]
    topologies = [r for r in inventory if r["partition"] == "development"]
    if cfg["topology_model"]["include_original_topology"]:
        topologies = [{"id": "base", "open_line": None, "close_line": None}] + topologies
    features, targets, records = [], [], []
    for index, row in enumerate(topologies):
        options = {} if row["open_line"] is None else {
            "open_lines": (row["open_line"],), "close_lines": (row["close_line"],)}
        feeder = build_feeder(**options)
        seed_sequence = [int(train_seed), 50000, index]
        ds = sample(feeder, ID, cfg["topology_model"]["scenarios_per_development_topology"],
                    np.random.default_rng(np.random.SeedSequence(seed_sequence)))
        lin = voltage_extrema(LinDistFlow(feeder).voltages(ds.p_net, ds.q_net, ds.v0))
        exact = np.stack([ds.vmin, ds.vmax], axis=1)
        features.append(np.concatenate([ds.X, lin], axis=1))
        targets.append(exact - lin)
        records.append({"topology_id": row["id"], "n": len(ds),
                        "seed_sequence": seed_sequence,
                        "x_sha256": hashlib.sha256(ds.X.tobytes()).hexdigest(),
                        "target_sha256": hashlib.sha256((exact-lin).tobytes()).hexdigest()})
    X, Y = np.concatenate(features), np.concatenate(targets)
    xs, ys = StandardScaler().fit(X), StandardScaler().fit(Y)
    est = MLPRegressor(hidden_layer_sizes=tuple(cfg["topology_model"]["hidden"]),
                       activation="relu", solver="adam", learning_rate_init=1e-3,
                       batch_size=256, max_iter=cfg["topology_model"]["max_iter"],
                       early_stopping=True, validation_fraction=0.1,
                       n_iter_no_change=25, random_state=int(train_seed)+60000,
                       tol=1e-7)
    print(f"[topology train] seed={train_seed} n={len(X)} topologies={len(topologies)}", flush=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        est.fit(xs.transform(X), ys.transform(Y))
    model = NumpyMLP.from_sklearn(est, xs, ys)
    provenance = {
        "training_s": time.perf_counter()-started, "training_rows": len(X),
        "topologies": records, "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        "n_iter": int(est.n_iter_), "loss_curve": [float(v) for v in est.loss_curve_],
        "validation_scores": [float(v) for v in est.validation_scores_],
        "best_validation_score": float(est.best_validation_score_),
        "warnings": [str(w.message) for w in caught],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "n_params": model.n_params,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = save_checkpoint(path, {"residual": model}, model_cfg, provenance)
    (path.parent / f"topology_{train_seed}.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(f"[topology trained] seed={train_seed} iterations={est.n_iter_} seconds={provenance['training_s']:.1f}", flush=True)
    return model, meta
