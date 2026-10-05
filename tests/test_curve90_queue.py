"""Exercise the full scheduler without network access or numerical workers."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import finish_curve90 as pipeline


class QueueTests(unittest.TestCase):
    def test_every_history_and_training_dependency_precedes_use(self):
        jobs = [dict(year=year, secids=ids,
                     path=f'quotes/year={year}/batch={batch}.parquet')
                for year in [2021, 2020, 2019]
                for batch, ids in enumerate([[1, 2], [3, 4], [5, 6]])]
        definitions = [dict(pair='1__3', label='AMD/NVDA', secids=[1, 3]),
                       dict(pair='2__5', label='second', secids=[2, 5]),
                       dict(pair='4__6', label='third', secids=[4, 6])]
        read_text = Path.read_text

        def fixture_text(path, *args, **kwargs):
            if path == Path('data/curve_trading/puts_raw/requests.json'):
                return json.dumps(jobs)
            if path == Path('artifacts/contracts/curve_trading/inputs.json'):
                return json.dumps(dict(definitions=definitions))
            return read_text(path, *args, **kwargs)
        by_pair = {d['pair']: d for d in definitions}
        downloaded, selected, history, trained = set(), set(), set(), []
        checked = False

        def execute(module, arguments, *args, **kwargs):
            nonlocal checked
            if module.endswith('download_curve90'):
                downloaded.add(int(arguments[-1]))
            elif module.endswith('prepare_curve90'):
                stage, index = arguments[1], int(arguments[-1])
                own = jobs[index]
                if stage == 'select':
                    self.assertIn(index, downloaded)
                    selected.add(index)
                else:
                    required = {i for i, j in enumerate(jobs)
                                if j['year'] in [own['year']-1, own['year']]
                                and set(j['secids']).intersection(own['secids'])}
                    self.assertTrue(required.issubset(selected))
                    history.add(index)
            elif module.endswith('check_curve90'):
                pilot = next(d for d in definitions if d['label'] == 'AMD/NVDA')
                required = {i for i, j in enumerate(jobs) if set(j['secids']).intersection(pilot['secids'])}
                self.assertTrue(required.issubset(history))
                self.assertLess(len(history), len(jobs))
                checked = True
            elif module.endswith('run_curve90'):
                self.assertTrue(checked)
                d = by_pair[arguments[1]]
                required = {i for i, j in enumerate(jobs) if set(j['secids']).intersection(d['secids'])}
                self.assertTrue(required.issubset(history))
                if '--verify-existing' in arguments:
                    self.assertIn(d['pair'], trained)
                else:
                    trained.append(d['pair'])
            elif module.endswith('report_curve90'):
                self.assertEqual(set(trained), set(by_pair))
            else:
                self.fail(module)

        with tempfile.TemporaryDirectory() as temp, \
             patch.object(pipeline, 'DATA', Path(temp)/'data'), \
             patch.object(pipeline, 'OUTPUT', Path(temp)/'output'), \
             patch.object(Path, 'read_text', fixture_text), \
             patch.object(pipeline, 'digest', return_value='fixture'), \
             patch.object(pipeline.shutil, 'disk_usage', return_value=SimpleNamespace(free=20*1024**3)), \
             patch.object(pipeline.signal, 'signal'), \
             patch.object(pipeline, 'run', side_effect=execute), \
             patch.object(pipeline.time, 'sleep'), \
             patch('sys.argv', ['finish_curve90', '--max-jobs', '2000']), \
             contextlib.redirect_stdout(io.StringIO()):
            pipeline.main()
        self.assertEqual(len(trained), len(definitions))
        self.assertEqual(by_pair[trained[0]]['label'], 'AMD/NVDA')


if __name__ == '__main__':
    unittest.main()
