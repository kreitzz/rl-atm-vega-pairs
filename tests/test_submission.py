"""Submission integrity checks that do not require market data or downloads."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.submission import digest, relative_path, require_supervisor


class SubmissionTest(unittest.TestCase):
    def test_bundle_paths_reject_escape(self):
        for value in ['/tmp/outside', '../outside', 'artifacts/../../outside', '']:
            with self.assertRaises(ValueError):
                relative_path(value)
        self.assertEqual(relative_path('artifacts/submission/test'), Path('artifacts/submission/test'))

    def test_digest_hashes_file_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cache'
            path.write_bytes(b'abc')
            self.assertEqual(digest(path), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')

    def test_numerical_commands_require_guard(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'supervisor|supervised|guarded'):
                require_supervisor()


if __name__ == '__main__':
    unittest.main()
