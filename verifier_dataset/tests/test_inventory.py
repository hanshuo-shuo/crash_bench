import json
from pathlib import Path
import tempfile
import unittest
from verifier_dataset.inventory import candidate, smoke_selection


class EvidenceRouting(unittest.TestCase):
    def row(self, success=True, collision=None, exited=None):
        return dict(spec=dict(scene=dict(suite='object', level='I', task=2, episode=0),
                             method='aegis', repeat=0), status='complete', success=success,
                    collision_step=collision, exited=exited, seed=7, actions=100,
                    outcome='safe_success' if success and collision is None else 'failure')

    def test_policy_success_is_still_unverified(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            (root/'RESULT.json').write_text('{}')
            (root/'manifest.json').write_text(json.dumps({'code_commit':'abc'}))
            for success, collision, exited in [(True,None,None),(False,None,None),(True,4,None),(True,None,'failure')]:
                result=candidate(self.row(success,collision,exited),root)
                self.assertEqual(result['feasibility_label'],'unknown')
                self.assertEqual(result['needs_contact_replay'], success and collision is None and exited is None)

    def test_smoke_does_not_fill_coverage_with_policy_repeats(self):
        rows=[]
        for task in range(3):
            for state in range(2):
                for repeat in range(4):
                    rows.append(dict(proxy_safe_success=True,has_action_tape=True,has_settled_state=True,
                        task_group=str(task),source_id=f'{task}/{state}',repeat=repeat,method='nominal'))
        selected=smoke_selection(rows,5)
        self.assertEqual(len(selected),5)
        self.assertEqual(len({r['source_id'] for r in selected}),5)
        self.assertEqual(len({r['task_group'] for r in selected[:3]}),3)


if __name__ == '__main__':
    unittest.main()
