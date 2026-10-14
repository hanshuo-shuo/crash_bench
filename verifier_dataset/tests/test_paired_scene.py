import gzip
import json
from pathlib import Path
import tempfile
import unittest
from verifier_dataset.fixture_design import walls
from verifier_dataset.inventory import write,read
from verifier_dataset.paired_geometry import CONTRACT,point_certificate
from verifier_dataset.paired_scene import scan_saved


class PairedEvidenceAdmission(unittest.TestCase):
    def fixture(self,root):
        bounds=[[-.1,-.1,0.],[.1,.1,.2]];region=[[.4,-.1,0.],[.6,.1,.2]]
        boxes=[dict(b,world_body=0,geom_type=6,contype=1,conaffinity=1,
                    rotation=[1.,0.,0.,0.,1.,0.,0.,0.,1.]) for b in walls(*bounds)]
        materials={role:dict(object=name,geom_type=6,primitive_size=[.01,.01,.01],radius=.01,
            center_in_root=[0.,0.,0.],initial_world_center=point,rigid_primitive_verified=True)
            for role,name,point in [('primary','apple',[.5,0.,.1]),('enclosed','orange',[0.,0.,.1])]}
        goal=['on','apple','plate'];spec=dict(goal=goal,kind='object',target_contact_geoms=[11],receiver_contact_geoms=[20])
        geo=dict(contract=CONTRACT,proposal=dict(primary='apple',enclosed='orange'),boxes=boxes,
            lower=bounds[0],upper=bounds[1],materials=materials,destination=dict(fixed_root_region=region),
            initial_safe=dict(primary=True,enclosed=True),initial_fixture_valid=True,wall_geoms=list(range(1,7)),
            static=True,protected=list(range(1,7))+[100],actors=dict(primary=[10,11],enclosed=[10,12]),goal_definition=spec)
        cert=point_certificate(boxes,*bounds,materials['enclosed'],region,True,True)
        write(root/'GEOMETRY.json',geo);write(root/'CERTIFICATE.json',cert)
        write(root/'SUMMARY.json',dict(primary_goal=goal,enclosed_goal=['on','orange','plate'],fixed_destination=region,
            samples=3,actions=1,integration_steps=1,safe_history=dict(primary=True,enclosed=True),positive_witness=True,matched_pair=True))
        write(root/'ACTIONS.json',[dict(action=[0.]*7)])
        rows=[]
        for phase,step,substep,success in [('initial',0,0,False),('integration',1,1,True),('endpoint',1,1,True)]:
            row=dict(phase=phase,step=step,substep=substep,primary_native_goal=success,safe_history=dict(primary=True,enclosed=True))
            for mode in ('native','synchronized'):
                row[mode+'_contacts']=[[11,20,.0001]] if success else []
                row[mode+'_points']={r:m['initial_world_center'] for r,m in materials.items()}
                row[mode+'_roots']=row[mode+'_points']
                row[mode+'_goal_state']=dict(target_position=[.5,0.,.1],receiver_position=[.5,0.,0.])
                row[mode+'_goal_success']=success
            rows.append(row)
        self.samples(root,rows);return rows

    def samples(self,root,rows):
        with gzip.open(root/'SAMPLES.jsonl.gz','wt') as f:
            for row in rows:f.write(json.dumps(row)+'\n')

    def test_independently_validated_pair_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);scan_saved(root)
            self.assertTrue(read(root/'VERIFIED.json')['matched_pair'])

    def test_unrelated_wall_overlap_is_not_a_valid_constructed_initial_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=self.fixture(root);rows[0]['native_contacts']=[[1,30,-.001]];self.samples(root,rows)
            with self.assertRaisesRegex(RuntimeError,'initial validity'):scan_saved(root)

    def test_missing_integration_or_false_goal_cannot_supply_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=self.fixture(root);self.samples(root,[rows[0],rows[2]])
            with self.assertRaisesRegex(RuntimeError,'Endpoint'):scan_saved(root)
            rows=self.fixture(root);rows[-1]['synchronized_contacts']=[];self.samples(root,rows)
            with self.assertRaisesRegex(RuntimeError,'goal formula'):scan_saved(root)

    def test_certificate_for_wrong_task_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);s=read(root/'SUMMARY.json');s['enclosed_goal'][1]='apple';write(root/'SUMMARY.json',s)
            with self.assertRaisesRegex(RuntimeError,'another target'):scan_saved(root)


if __name__=='__main__':unittest.main()
