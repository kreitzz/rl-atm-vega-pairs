"""Semantic hashes for fitted Q policies, independent of pickle padding.

Sklearn tree nodes use structured arrays with unused padding bytes. Hashing
the raw object can change after an identical model is loaded from disk. This
module hashes each named node field separately, alongside tree values and the
policy's predictive configuration and audit. Unknown model types fail closed.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping

import numpy as np


def _class_name(value):
    return f'{type(value).__module__}.{type(value).__qualname__}'


def _write_blob(digest, tag, data):
    digest.update(tag.encode('ascii'))
    digest.update(str(len(data)).encode('ascii'))
    digest.update(b':')
    digest.update(data)


def _update(digest, value):
    """Length-framed, ordered encoding with no object-address representation."""
    if value is None:
        _write_blob(digest, 'none', b'')
    elif isinstance(value, (bool, np.bool_)):
        _write_blob(digest, 'bool', b'1' if value else b'0')
    elif isinstance(value, (int, np.integer)):
        _write_blob(digest, 'int', str(int(value)).encode('ascii'))
    elif isinstance(value, (float, np.floating)):
        number = float(value)
        text = 'nan' if math.isnan(number) else number.hex()
        _write_blob(digest, 'float', text.encode('ascii'))
    elif isinstance(value, str):
        _write_blob(digest, 'str', value.encode('utf-8'))
    elif isinstance(value, bytes):
        _write_blob(digest, 'bytes', value)
    elif isinstance(value, np.ndarray):
        _write_blob(digest, 'array', b'')
        _update(digest, value.shape)
        if value.dtype.names is not None:
            # Field offsets and raw record bytes are deliberately omitted.
            # A field view is made contiguous before writing numeric bytes.
            _update(digest, list(value.dtype.names))
            for name in value.dtype.names:
                _update(digest, value[name])
        elif value.dtype.hasobject:
            _update(digest, 'object')
            for item in value.flat:
                _update(digest, item)
        elif value.dtype.kind in ['U', 'S']:
            _update(digest, value.dtype.kind)
            for item in value.flat:
                _update(digest, str(item) if value.dtype.kind == 'U' else bytes(item))
        else:
            # Canonical byte order also prevents architecture-dependent noise.
            dtype = value.dtype.newbyteorder('<')
            normalized = np.ascontiguousarray(value.astype(dtype, copy=False))
            _update(digest, dtype.str)
            _write_blob(digest, 'data', normalized.tobytes(order='C'))
    elif isinstance(value, Mapping):
        _write_blob(digest, 'mapping', str(len(value)).encode('ascii'))
        keys = []
        for key in value:
            encoded_key = hashlib.sha256()
            _update(encoded_key, key)
            keys.append((encoded_key.digest(), key))
        for _, key in sorted(keys, key=lambda item: item[0]):
            _update(digest, key)
            _update(digest, value[key])
    elif isinstance(value, (list, tuple)):
        _write_blob(digest, 'list' if isinstance(value, list) else 'tuple',
                    str(len(value)).encode('ascii'))
        for item in value:
            _update(digest, item)
    else:
        raise TypeError(f'Unsupported model provenance value: {_class_name(value)}')


def _tree_state(estimator):
    tree = estimator.tree_
    state = tree.__getstate__()
    return dict(
        predictive_class=_class_name(estimator),
        parameters=estimator.get_params(deep=False),
        n_features=int(tree.n_features), n_outputs=int(tree.n_outputs),
        n_classes=np.asarray(tree.n_classes),
        max_depth=int(state['max_depth']), node_count=int(state['node_count']),
        nodes=state['nodes'], values=state['values'])


def _predictive_state(model):
    if model is None:
        return None
    if hasattr(model, 'model') and hasattr(model, 'audit'):
        policy = dict(predictive_class=_class_name(model), audit=model.audit,
                      critic=_predictive_state(model.model))
        for name in ['columns', 'actions', 'conservative']:
            if hasattr(model, name):
                policy[name] = getattr(model, name)
        return policy
    if hasattr(model, 'forests_'):
        return dict(
            predictive_class=_class_name(model),
            parameters={name: getattr(model, name) for name in
                        ['n_actions', 'n_estimators', 'max_depth', 'min_samples_leaf',
                         'random_state', 'n_jobs']},
            n_features_in=int(model.n_features_in_),
            present_actions=list(model.present_actions_),
            action_counts=dict(model.action_counts_),
            forests={action: _predictive_state(forest)
                     for action, forest in model.forests_.items()})
    if hasattr(model, 'estimators_') and all(hasattr(tree, 'tree_') for tree in model.estimators_):
        return dict(predictive_class=_class_name(model),
                    parameters=model.get_params(deep=False),
                    n_features_in=int(model.n_features_in_),
                    n_outputs=int(model.n_outputs_),
                    trees=[_tree_state(tree) for tree in model.estimators_])
    if hasattr(model, 'tree_'):
        return _tree_state(model)
    raise TypeError(f'Unsupported fitted model: {_class_name(model)}')


def stable_model_digest(model):
    """SHA-256 of predictive fields and provenance, stable across disk loads."""
    digest = hashlib.sha256()
    _update(digest, 'p25-rl-semantic-model-v1')
    _update(digest, _predictive_state(model))
    return digest.hexdigest()
