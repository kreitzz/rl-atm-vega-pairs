import ast
import json
import unittest
from pathlib import Path
from scripts import run_atm65_rl_transcripts as runner


class RLOnlyWorkflowTest(unittest.TestCase):
    def test_single_feature_isolated_and_no_nn_training(self):
        runner.configure('call_qa_uncertainty')
        self.assertEqual(runner.MODELS, ['rl_transcripts'])
        self.assertEqual(runner.ADDED, ['transcript_call_qa_uncertainty_gap',
                                       'transcript_call_qa_uncertainty_available'])
        self.assertNotEqual(runner.OUT, Path('artifacts/contracts/atm65_transcripts'))
        tree = ast.parse(Path(runner.__file__).read_text())
        imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                   for alias in node.names}
        self.assertNotIn('fit_fixed_nn', imports)
        self.assertNotIn('predict_fixed_nn', imports)
        workflow = json.loads(runner.WORKFLOW.read_text())
        self.assertEqual(workflow['retrain_models'], ['rl'])
        self.assertFalse(workflow['retrain_nn'])


if __name__ == '__main__':
    unittest.main()
