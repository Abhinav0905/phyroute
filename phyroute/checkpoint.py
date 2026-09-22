"""Versioned NumPy-only MLP checkpoints; loading never enables pickle.

Hashes detect mismatched configurations or damaged arrays. They do not prove
who created an archive. Record dataset/code hashes in source_metadata as part
of the experiment record.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .models import NumpyMLP

FORMAT_VERSION = 1


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def configuration_hash(config: dict) -> str:
    """SHA256 of a JSON configuration with sorted keys and no NaN/Infinity."""
    if not isinstance(config, dict):
        raise ValueError("checkpoint configuration must be a dictionary")
    return hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()


def _array_hash(a: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(a.dtype.str.encode("ascii"))
    h.update(_canonical_json(list(a.shape)).encode("ascii"))
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _validate_model(model: NumpyMLP) -> None:
    if not isinstance(model, NumpyMLP):
        raise ValueError("all checkpoint values must be NumpyMLP models")
    if not model.weights or len(model.weights) != len(model.biases):
        raise ValueError("model must contain matching nonempty weight and bias lists")
    if not isinstance(model.classifier, (bool, np.bool_)):
        raise ValueError("classifier flag must be boolean")
    arrays = [*model.weights, *model.biases, model.x_mean, model.x_scale]
    if model.classifier:
        if model.y_mean is not None or model.y_scale is not None:
            raise ValueError("classifier output scaling must be None")
    else:
        if model.y_mean is None or model.y_scale is None:
            raise ValueError("regression model requires output scaling")
        arrays.extend([model.y_mean, model.y_scale])
    for a in arrays:
        if not isinstance(a, np.ndarray) or a.dtype.kind != "f" or not np.isfinite(a).all():
            raise ValueError("model arrays must be finite floating-point NumPy arrays")
    if model.x_mean.ndim != 1 or model.x_scale.shape != model.x_mean.shape or model.x_mean.size == 0:
        raise ValueError("input scaler arrays must be matching nonempty vectors")
    if not (model.x_scale > 0).all():
        raise ValueError("input scales must be positive")
    width = model.x_mean.size
    for w, b in zip(model.weights, model.biases):
        if w.ndim != 2 or w.shape[0] != width or w.shape[1] == 0 or b.shape != (w.shape[1],):
            raise ValueError("incompatible layer weight/bias dimensions")
        width = w.shape[1]
    if model.classifier:
        if width != 1:
            raise ValueError("binary classifier must have one output")
    elif model.y_mean.shape != (width,) or model.y_scale.shape != (width,) or not (model.y_scale > 0).all():
        raise ValueError("output scaling must match model outputs and have positive scales")


def save_checkpoint(path, models: dict[str, NumpyMLP], config: dict,
                    source_metadata: dict | None = None) -> dict:
    """Atomically save named models and JSON provenance to an NPZ checkpoint.

    The file extension is not modified. source_metadata can contain training
    history, dataset/code hashes and environment details. It must be JSON data.
    """
    if not isinstance(models, dict) or not models or not all(isinstance(k, str) and k for k in models):
        raise ValueError("models must be a nonempty dictionary with nonempty string names")
    if source_metadata is not None and not isinstance(source_metadata, dict):
        raise ValueError("source_metadata must be a dictionary or None")
    metadata = {"format": "phyroute.numpy_mlp", "format_version": FORMAT_VERSION,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "config": config, "config_sha256": configuration_hash(config),
                "source_metadata": source_metadata or {}, "models": {}, "array_sha256": {}}
    _canonical_json(metadata)  # Fail before writing if provenance is not valid JSON.
    arrays = {}
    for index, (name, model) in enumerate(sorted(models.items())):
        _validate_model(model)
        prefix = f"model_{index}"
        keys = {"weights": [], "biases": []}
        for field in ("weights", "biases"):
            for layer, value in enumerate(getattr(model, field)):
                key = f"{prefix}__{field}_{layer}"
                arrays[key] = np.asarray(value).copy()
                keys[field].append(key)
        for field in ("x_mean", "x_scale", "y_mean", "y_scale"):
            value = getattr(model, field)
            key = None if value is None else f"{prefix}__{field}"
            keys[field] = key
            if key is not None:
                arrays[key] = np.asarray(value).copy()
        metadata["models"][name] = {"classifier": bool(model.classifier), "arrays": keys}
    metadata["array_sha256"] = {key: _array_hash(value) for key, value in arrays.items()}
    arrays["metadata_json"] = np.asarray(_canonical_json(metadata))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + ".", suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
            np.savez_compressed(f, **arrays)
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return metadata


def load_checkpoint(path, expected_config: dict | None = None) -> tuple[dict[str, NumpyMLP], dict]:
    """Read and validate a checkpoint; reject a supplied configuration mismatch."""
    with np.load(path, allow_pickle=False) as archive:
        meta_array = archive["metadata_json"]
        if meta_array.ndim != 0 or meta_array.dtype.kind != "U":
            raise ValueError("metadata_json must be a scalar Unicode string")
        metadata = json.loads(str(meta_array))
        if metadata.get("format") != "phyroute.numpy_mlp" or metadata.get("format_version") != FORMAT_VERSION:
            raise ValueError("unsupported checkpoint format or version")
        config_hash = configuration_hash(metadata["config"])
        if config_hash != metadata.get("config_sha256"):
            raise ValueError("checkpoint configuration hash is invalid")
        if expected_config is not None and configuration_hash(expected_config) != config_hash:
            raise ValueError("checkpoint configuration does not match expected_config")
        hashes = metadata["array_sha256"]
        if set(archive.files) != set(hashes) | {"metadata_json"}:
            raise ValueError("checkpoint array inventory does not match metadata")
        arrays = {}
        for key, expected_hash in hashes.items():
            a = archive[key]
            if a.dtype.kind != "f" or not np.isfinite(a).all() or _array_hash(a) != expected_hash:
                raise ValueError(f"checkpoint array failed validation: {key}")
            arrays[key] = a.copy()
    descriptions = metadata["models"]
    if not isinstance(descriptions, dict) or not descriptions:
        raise ValueError("checkpoint contains no model descriptions")
    models = {}
    for name, description in descriptions.items():
        if not isinstance(name, str) or not name:
            raise ValueError("invalid checkpoint model name")
        keys = description["arrays"]
        model = NumpyMLP(weights=[arrays[k] for k in keys["weights"]],
                         biases=[arrays[k] for k in keys["biases"]],
                         x_mean=arrays[keys["x_mean"]], x_scale=arrays[keys["x_scale"]],
                         y_mean=None if keys["y_mean"] is None else arrays[keys["y_mean"]],
                         y_scale=None if keys["y_scale"] is None else arrays[keys["y_scale"]],
                         classifier=description["classifier"])
        _validate_model(model)
        models[name] = model
    return models, metadata
