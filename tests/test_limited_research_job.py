"""The resource guard must distinguish an exiting child from a live failure."""
from pathlib import Path
import subprocess
import unittest
from unittest.mock import Mock, patch

from scripts.limited_research_job import run


class ResourceGuardTests(unittest.TestCase):
    def exercise_permission_race(self, finished):
        child = Mock(pid=999999, returncode=None)
        child.poll.side_effect = lambda: child.returncode

        def reap(timeout=None):
            if not finished:
                raise subprocess.TimeoutExpired('child', timeout)
            child.returncode = 0
            return 0

        child.wait.side_effect = reap
        with patch('scripts.limited_research_job.subprocess.Popen', return_value=child), \
             patch('scripts.limited_research_job.os.killpg', side_effect=PermissionError), \
             patch('scripts.limited_research_job.time.sleep'), \
             patch.object(Path, 'exists', return_value=False):
            run('scripts.download_curve90', [], Path('STOP'))
        child.wait.assert_called_with(timeout=1)

    def test_group_teardown_can_precede_waitpid(self):
        self.exercise_permission_race(finished=True)

    def test_live_inaccessible_child_is_not_ignored(self):
        with self.assertRaises(PermissionError):
            self.exercise_permission_race(finished=False)

    def test_stop_file_prevents_launch(self):
        with patch.object(Path, 'exists', return_value=True), \
             patch('scripts.limited_research_job.subprocess.Popen') as launch:
            with self.assertRaises(InterruptedError):
                run('scripts.download_curve90', [], Path('STOP'))
            launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
