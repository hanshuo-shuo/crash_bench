import gzip
import json
from pathlib import Path
import tempfile
import unittest

from verifier_dataset.contact_contract import VERSION, task_targets, prohibited, verify_saved


class ContactScope(unittest.TestCase):
    def test_targets_include_both_manipulated_objects_but_not_receiver(self):
        goals=[['on','mug','plate'],['on','pudding','table_goal_region']]
        self.assertEqual(task_targets(goals, {'mug':1,'pudding':2,'plate':3}), ['mug','pudding'])
        with self.assertRaisesRegex(RuntimeError,'Unsupported'):
            task_targets([['unknown_predicate','mug']], {'mug':1})

    def test_native_support_allowed_robot_and_target_still_prohibited(self):
        self.assertFalse(prohibited((10,30), {10}, {1,2}))
        for pair in ((10,1),(1,10),(10,2),(2,10)):
            self.assertTrue(prohibited(pair,{10},{1,2}))

    def saved(self, folder, bad=False, missing=False):
        scope=dict(version=VERSION,protected=[10],actors=[1,2],robot=[1],targets={'milk':[2]})
        (folder/'CONTACT_CONTRACT.json').write_text(json.dumps(scope))
        rows=[dict(action=0,substep=0,phase='initial',contacts=[[10,30,-.001]],prohibited_contact_count=0),
              dict(action=0,substep=0,phase='synchronized_endpoint',contacts=[],prohibited_contact_count=0),
              dict(action=1,substep=1,phase='integration',contacts=[[10,2,-.001]] if bad else [],prohibited_contact_count=0),
              dict(action=1,substep=1,phase='synchronized_endpoint',contacts=[],prohibited_contact_count=0)]
        if missing:rows.pop()
        with gzip.open(folder/'CONTACT_SAMPLES.jsonl.gz','wt') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
        return dict(contact_contract_version=VERSION,actions=1,integration_steps=1,
                    protected_contact_events=0,out_of_scope_contact_events=1)

    def test_saved_samples_reclassify_contacts_and_check_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);verify_saved(root,self.saved(root))
            with self.assertRaisesRegex(RuntimeError,'misclassified'):
                verify_saved(root,self.saved(root,bad=True))
            with self.assertRaisesRegex(RuntimeError,'synchronized endpoint'):
                verify_saved(root,self.saved(root,missing=True))


if __name__=='__main__':unittest.main()
