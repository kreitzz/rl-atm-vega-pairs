"""The serial launcher recognizes frozen native Windows path identities."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.p25_rl import research


class LauncherTests(unittest.TestCase):
    def test_frozen_selection_accepts_both_path_separators_and_rejects_changes(self):
        temp_parent=Path(__file__).resolve().parents[1]/'artifacts/contracts/p25_rl_test_tmp'
        temp_parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_parent) as folder:
            root=Path(folder);selection=root/'selection.json';sources={}
            self.assertEqual(root.resolve().parent,temp_parent.resolve())
            for name in research.SELECTION_SOURCES:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(name,encoding='utf-8');sources[name]=research._digest(path)
            body=dict(validation_only=True,validation_start='2020-01-01',
                validation_end='2021-12-31',proposal_cutoff='2021-10-31',
                candidates=sorted(research.DECLARED_CANDIDATES),
                selected='action_q_capacity',best_q_by_validation='action_q_capacity')
            with patch.object(research,'ROOT',root),patch.object(research,'SELECTION',selection):
                for separator in ['/', '\\']:
                    body['source_hashes']={name.replace('/',separator):digest for name,digest in sources.items()}
                    selection.write_text(json.dumps(body),encoding='utf-8')
                    self.assertEqual(research.check_frozen_selection()['selected'],'action_q_capacity')
                first=next(iter(sources));(root/first).write_text('changed',encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'input changed'):
                    research.check_frozen_selection()


if __name__=='__main__':unittest.main()
