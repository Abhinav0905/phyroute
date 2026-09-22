import json

import numpy as np
import pytest

from phyroute.checkpoint import configuration_hash, load_checkpoint, save_checkpoint
from phyroute.models import NumpyMLP


def _model(classifier=False):
    return NumpyMLP(weights=[np.array([[1.0], [-0.5]])], biases=[np.array([0.2])],
                    x_mean=np.array([0.0, 1.0]), x_scale=np.array([1.0, 2.0]),
                    y_mean=None if classifier else np.array([0.95]),
                    y_scale=None if classifier else np.array([0.01]), classifier=classifier)


def test_npz_roundtrip_preserves_named_model_predictions_and_provenance(tmp_path):
    path = tmp_path / 'models.npz'
    config = {'seed': 7, 'hidden': [32, 32]}
    models = {'regression': _model(), 'classification': _model(True)}
    source = {'data_sha256': 'abc', 'iterations': [100, 80]}
    metadata = save_checkpoint(path, models, config, source)
    restored, loaded_metadata = load_checkpoint(path, expected_config=config)
    x = np.array([[0.3, 0.1], [1.0, -1.0]])
    assert loaded_metadata == metadata and loaded_metadata['source_metadata'] == source
    for name in models:
        assert np.array_equal(models[name].forward(x), restored[name].forward(x))
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[key].dtype.kind != 'O' for key in archive.files)


def test_configuration_order_is_stable_but_changed_seed_is_rejected(tmp_path):
    assert configuration_hash({'seed': 1, 'n': 20}) == configuration_hash({'n': 20, 'seed': 1})
    path = tmp_path / 'models.npz'
    save_checkpoint(path, {'tiny': _model()}, {'seed': 1, 'n': 20})
    with pytest.raises(ValueError, match='does not match'):
        load_checkpoint(path, expected_config={'seed': 2, 'n': 20})


def test_changed_weight_without_matching_hash_is_rejected(tmp_path):
    path = tmp_path / 'models.npz'
    save_checkpoint(path, {'tiny': _model()}, {'seed': 1})
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    key = next(k for k in arrays if '__weights_' in k)
    arrays[key][0, 0] += 1
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match='failed validation'):
        load_checkpoint(path)


def test_pickle_object_metadata_is_never_loaded(tmp_path):
    path = tmp_path / 'bad.npz'
    np.savez(path, metadata_json=np.array({'unsafe': 'object'}, dtype=object))
    with pytest.raises(ValueError, match='allow_pickle=False'):
        load_checkpoint(path)


def test_invalid_model_fails_before_checkpoint_is_written(tmp_path):
    path = tmp_path / 'bad.npz'
    model = _model()
    model.x_scale[0] = 0
    with pytest.raises(ValueError, match='positive'):
        save_checkpoint(path, {'bad': model}, {'seed': 1})
    assert not path.exists()
