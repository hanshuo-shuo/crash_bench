import gzip
import json
from pathlib import Path
import tempfile
import unittest
from verifier_dataset.fixture_smoke import check_saved,prohibited_contact
from verifier_dataset.fixture_design import walls
from verifier_dataset.inventory import write
from experiments.reach_avoid.certificate import certify


class TaskCertificateScope(unittest.TestCase):
    def fixture(self,root,scope):
        geo=dict(boxes=walls([-.1,-.1,0.],[.1,.1,.2]),lower=[-.1,-.1,0.],upper=[.1,.1,.2],
            initial_root=[0.,0.,.1],goal_bounds=[[.3,.3,.1],[.4,.4,.2]],radius=.013,
            initial_safe=True,static=True,material_verified=True,dynamic=[1],protected=[2],
            rigid_groups={'1':1,'2':0},
            material_evidence=dict(collision_boxes=[dict(half_sizes=[.013,.02,.03],root_local=[0.,0.,0.])]))
        cert=certify(geo['boxes'],geo['lower'],geo['upper'],geo['initial_root'],geo['goal_bounds'],
                     geo['radius'],True,True,True,True)
        write(root/'EXECUTED_GEOMETRY.json',geo);write(root/'PUDDING_CERTIFICATE.json',cert)
        write(root/'SUMMARY.json',dict(samples=1,safe_history=True,label='conditionally_infeasible',task_scope=scope))
        with gzip.open(root/'SAMPLES.jsonl.gz','wt') as f:
            f.write(json.dumps(dict(native_contacts=[],synchronized_contacts=[],native_root=[0.,0.,.1],
                                    synchronized_root=[0.,0.,.1],safe_history=True))+'\n')

    def test_pudding_exclusion_cannot_label_an_unrelated_mug_task(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);self.fixture(root,'mug_only')
            with self.assertRaisesRegex(RuntimeError,'does not apply'):check_saved(root)

    def test_full_task_requires_and_recomputes_its_certificate(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);self.fixture(root,'full');check_saved(root)
            saved=json.loads((root/'PUDDING_CERTIFICATE.json').read_text());saved['label']='unknown'
            write(root/'PUDDING_CERTIFICATE.json',saved)
            with self.assertRaisesRegex(RuntimeError,'separator mismatch'):check_saved(root)

    def test_movable_protected_object_contact_with_protected_wall_is_forbidden(self):
        dynamic={1,2,4};protected={1,2,3};rigid={'1':8,'2':8,'3':0,'4':9,'5':0}
        self.assertTrue(prohibited_contact(1,3,-.001,dynamic,protected,rigid))
        self.assertTrue(prohibited_contact(3,1,-.001,dynamic,protected,rigid))
        self.assertTrue(prohibited_contact(4,3,-.001,dynamic,protected,rigid))
        self.assertFalse(prohibited_contact(1,2,-.001,dynamic,protected,rigid))
        self.assertFalse(prohibited_contact(1,5,-.001,dynamic,protected,rigid))


if __name__=='__main__':unittest.main()
