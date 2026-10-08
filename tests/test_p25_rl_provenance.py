import os
for _name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'ARROW_NUM_THREADS']:
    os.environ[_name] = '1'

from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from scripts.p25_rl.critic import ActionForest
from scripts.p25_rl.policy import QPolicy, action_inputs
from scripts.p25_rl.provenance import stable_model_digest


TEST_TMP = Path(__file__).resolve().parents[1] / 'artifacts/contracts/p25_rl_provenance_tests'


def fitted_policy(independent=False):
    state = np.column_stack([np.linspace(-1., 1., 180), np.sin(np.arange(180))])
    actions = np.tile([0, 1, 2], 60)
    inputs = action_inputs(state, actions)
    target = state[:, 0] + actions * .2
    model = (ActionForest(4, n_estimators=3, max_depth=3, min_samples_leaf=5)
             if independent else ExtraTreesRegressor(n_estimators=3, max_depth=3,
                                                      min_samples_leaf=5, n_jobs=1,
                                                      random_state=2026))
    model.fit(inputs, target)
    return QPolicy(model, [0, 1], [0, 1, 2], False,
                   {'cutoff': '2020-01-01', 'counts': {0: 60, 1: 60, 2: 60}})


class ProvenanceTests(unittest.TestCase):
    def test_both_critic_types_are_stable_across_repeated_disk_loads(self):
        TEST_TMP.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temporary:
            path = Path(temporary) / 'policy.joblib'
            for independent in [False, True]:
                original = fitted_policy(independent)
                expected = stable_model_digest(original)
                for _ in range(3):
                    joblib.dump(original, path)
                    original = joblib.load(path)
                    self.assertEqual(stable_model_digest(original), expected)
        self.assertEqual(stable_model_digest(None), stable_model_digest(None))

    def test_named_predictive_fields_and_audit_change_digest(self):
        original = fitted_policy()
        expected = stable_model_digest(original)
        threshold = deepcopy(original)
        threshold.model.estimators_[0].tree_.threshold[0] += .01
        self.assertNotEqual(stable_model_digest(threshold), expected)
        values = deepcopy(original)
        values.model.estimators_[0].tree_.value[0, 0, 0] += .01
        self.assertNotEqual(stable_model_digest(values), expected)
        audit = deepcopy(original)
        audit.audit['cutoff'] = '2022-01-01'
        self.assertNotEqual(stable_model_digest(audit), expected)
        actions = deepcopy(original)
        actions.actions = [0, 2]
        self.assertNotEqual(stable_model_digest(actions), expected)

    def test_structured_node_padding_is_excluded(self):
        policy = fitted_policy()
        expected = stable_model_digest(policy)
        nodes = policy.model.estimators_[0].tree_.__getstate__()['nodes']
        used = set()
        for name in nodes.dtype.names:
            dtype, offset = nodes.dtype.fields[name][:2]
            used.update(range(offset, offset + dtype.itemsize))
        padding = sorted(set(range(nodes.dtype.itemsize)) - used)
        if not padding:
            self.skipTest('This sklearn build stores no tree-node padding')
        raw = nodes.view(np.uint8).reshape(len(nodes), nodes.dtype.itemsize)
        raw[:, padding] = 173
        self.assertEqual(stable_model_digest(policy), expected)

    def test_management_wrapper_shape_and_mapping_order_are_stable(self):
        # The held-book wrapper has model/audit, with no entry-policy columns,
        # actions or conservative attributes. Exercise that distinct shape.
        managed = SimpleNamespace(model=fitted_policy(True).model,
                                  audit={'cutoff': '2020-01-01', 'spec': {'gamma': .99}})
        expected = stable_model_digest(managed)
        reordered = deepcopy(managed)
        reordered.audit = {'spec': {'gamma': .99}, 'cutoff': '2020-01-01'}
        self.assertEqual(stable_model_digest(reordered), expected)
        changed = deepcopy(managed)
        changed.model.forests_[0].estimators_[0].tree_.value[0, 0, 0] += .01
        self.assertNotEqual(stable_model_digest(changed), expected)


if __name__ == '__main__':
    unittest.main()
