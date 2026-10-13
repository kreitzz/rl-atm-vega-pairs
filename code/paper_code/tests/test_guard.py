"""Check Windows worker supervision and cleanup on cancellation/failure."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

from paper_code.guard import control_windows_process, run


class ProcessTeardownTests(unittest.TestCase):
    def test_access_denied_is_ignored_for_exited_worker(self):
        import psutil
        process = Mock()
        process.suspend.side_effect = psutil.AccessDenied(pid=123)
        process.wait.return_value = 0
        control_windows_process(process, "suspend")
        process.wait.assert_called_once_with(timeout=1)

    def test_access_denied_is_raised_for_live_worker(self):
        import psutil
        process = Mock()
        process.suspend.side_effect = psutil.AccessDenied(pid=123)
        process.wait.side_effect = psutil.TimeoutExpired(1, pid=123)
        with self.assertRaises(psutil.AccessDenied):
            control_windows_process(process, "suspend")


@unittest.skipUnless(sys.platform == "win32", "Windows process supervision")
class WindowsGuardTests(unittest.TestCase):
    def assert_no_new_children(self, before):
        import psutil
        after = {p.pid for p in psutil.Process().children(recursive=True)
                 if p.is_running()}
        self.assertFalse(after - before, "Guard left a worker running")

    def test_timeout_reaps_worker(self):
        import psutil
        before = {p.pid for p in psutil.Process().children(recursive=True)}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as folder:
            with self.assertRaises(InterruptedError):
                run(["fit"], folder, timeout=0)
        self.assert_no_new_children(before)

    def test_stop_file_reaps_worker(self):
        import psutil
        before = {p.pid for p in psutil.Process().children(recursive=True)}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as folder:
            stop = Path(folder) / "artifacts/paper_code/STOP"
            stop.parent.mkdir(parents=True)
            stop.touch()
            with self.assertRaises(InterruptedError):
                run(["fit"], folder)
        self.assert_no_new_children(before)

    def test_worker_failure_is_returned(self):
        root = Path(__file__).resolve().parents[2]
        self.assertEqual(run(["invalid-command"], root), 2)
