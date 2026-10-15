import tempfile
from pathlib import Path
import unittest
import joblib

from scripts.research_atomic_io import atomic_dump,atomic_text


class AtomicResearchIO(unittest.TestCase):
    def test_restart_cannot_extend_absolute_study_deadline(self):
        from unittest.mock import patch
        from scripts import launch_straddle_overnight as runner
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(runner,'OUT',Path(folder)),patch.object(runner.time,'time',return_value=1000):
                self.assertEqual(runner.Supervisor(8,deadline=2000).deadline,2000)

    def test_eof_traceback_is_not_mistaken_for_a_memory_stop(self):
        from scripts.launch_straddle_overnight import retryable_failure
        self.assertFalse(retryable_failure('pause_low_headroom_seconds\nEOFError\nCalledProcessError'))
        self.assertTrue(retryable_failure('MemoryError: System swap increased by more than 256.0 MiB; stopped safely'))

    def test_interrupted_dump_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'checkpoint.joblib';joblib.dump({'value':7},p)
            def failed(value,path,*a,**k):
                Path(path).write_bytes(b'partial');raise OSError('interrupted')
            with self.assertRaises(OSError):atomic_dump({'value':8},p,failed)
            self.assertEqual(joblib.load(p),{'value':7})

    def test_checkpoint_published_only_after_complete_dump(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'checkpoint.joblib'
            def write(value,path,*a,**k):
                self.assertFalse(p.exists());return joblib.dump(value,path,*a,**k)
            self.assertEqual(atomic_dump({'value':9},p,write),[str(p)])
            self.assertEqual(joblib.load(p),{'value':9})

    def test_interrupted_json_preserves_previous_model(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'model.json';p.write_text('{"value":1}')
            def failed(path,text,*a,**k):
                Path(path).write_bytes(b'{');raise OSError('interrupted')
            with self.assertRaises(OSError):atomic_text(p,'{"value":2}',failed)
            self.assertEqual(p.read_text(),'{"value":1}')
