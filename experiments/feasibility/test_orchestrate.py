import json
from pathlib import Path
import tempfile
import unittest
from orchestrate import next_stage


class OrchestrationTests(unittest.TestCase):
    def fixture(self, root, eligible):
        (root / 'INITIAL_COMPLETE.json').write_text('{}')
        (root / 'BRANCH_GATE.json').write_text(json.dumps({'eligible_states': eligible}))
        (root / 'plan.json').write_text(json.dumps({'jobs': {'initial': '123'}}))

    def test_only_approved_branches_sequentially_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, ['object_00', 'object_02'])
            self.assertEqual(next_stage(root), 'branch_object_00')
            (root / 'branch_object_00_COMPLETE.json').write_text('{}')
            self.assertEqual(next_stage(root), 'branch_object_02')
            (root / 'plan.json').write_text(json.dumps({'jobs': {'initial': '123', 'branch_object_02': '124'}}))
            with self.assertRaisesRegex(RuntimeError, 'Prior branch attempt'):
                next_stage(root)

    def test_failure_missing_evidence_and_missing_export_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, 'Required evidence'):
                next_stage(root)
            self.fixture(root, [])
            with self.assertRaisesRegex(RuntimeError, 'complete exported evidence'):
                next_stage(root)
            (root / 'COMPLETE.json').write_text('{}')
            (root / 'report').mkdir()
            (root / 'report/REPORT_COMPLETE.json').write_text('{}')
            self.assertIsNone(next_stage(root))
            (root / 'STOP.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError, 'Scientific root stopped'):
                next_stage(root)

    def test_unknown_duplicate_and_control_states_not_submitted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for states in [['object_00', 'object_00'], ['control_spatial_01'], ['arbitrary']]:
                self.fixture(root, states)
                with self.assertRaisesRegex(RuntimeError, 'Invalid branch eligibility'):
                    next_stage(root)


if __name__ == '__main__':
    unittest.main()
