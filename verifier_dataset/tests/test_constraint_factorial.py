import copy
import itertools
import unittest
from collections import Counter
import numpy as np
from verifier_dataset.constraint_sides import EXIT,ENTRY,BOTH,NEITHER,VERSION,cell,factors_for,factorial_regions,verify_sides,selection_quotas
from verifier_dataset.constraint_visuals import balanced_poses
from verifier_dataset.constraint_evidence import verify_layout,rows
from verifier_dataset.finalize_constraints import final_rationale
from verifier_dataset.constraint_selection import select_units,selection_cell
from verifier_dataset.tests import test_constraint_pairs as evidence_fixtures
from verifier_dataset.tests import test_constraint_selection as selection_fixtures


class FactorialEndpointControl(unittest.TestCase):
    def fixture(self):
        layout,geometry,samples,actions=evidence_fixtures.ConstraintEvidence().fixture()
        layout['target']=[[-.01,-.01,.09],[.01,.01,.11]]
        return layout,samples,actions

    def test_all_four_cells_and_exact_factor_margins_are_preassigned(self):
        for source in ('source_a','source_b','source_c'):
            factors=[factors_for(source,t) for t in range(0,50,5)]
            self.assertEqual(len({f['construction_cell'] for f in factors[:4]}),4)
            self.assertEqual(Counter(f['construction_mode'] for f in factors),{EXIT:5,ENTRY:5})
            self.assertEqual(Counter(f['positive_side'] for f in factors),{BOTH:5,NEITHER:5})
        self.assertEqual(sum(sum(v.values()) for v in selection_quotas().values()),250)
        for split,quota in selection_quotas().items():
            self.assertEqual(quota[cell(EXIT,BOTH)]+quota[cell(EXIT,NEITHER)],quota[cell(ENTRY,BOTH)]+quota[cell(ENTRY,NEITHER)])
            self.assertEqual(quota[cell(EXIT,BOTH)]+quota[cell(ENTRY,BOTH)],quota[cell(EXIT,NEITHER)]+quota[cell(ENTRY,NEITHER)])
            self.assertLessEqual(max(quota.values())-min(quota.values()),1)

    def test_each_cell_has_a_real_witness_certificate_and_correct_positive_explanation(self):
        original,samples,actions=self.fixture();seen=set()
        source=dict(source_id='s',task_group='t',family='f',method='aegis',seed=7)
        for t in (0,5,10,15):
            for design in factorial_regions(original['target'],original['goal'],original['material'],'s',t):
                layout=dict(original,**design);layout['construction_version']=VERSION
                self.assertTrue(all(verify_sides(layout).values()))
                proof=verify_layout(layout,samples,actions);self.assertTrue(proof['matched_pair'])
                self.assertTrue(proof['blocked_boundary_crossed_by_positive_witness'])
                seen.add(design['construction_cell'])
                for row in rows(layout,proof,source,[[{'sha256':'0'}],[{'sha256':'1'}]]):
                    text=final_rationale(row)
                    self.assertNotIn('recorded legal trajectory',text)
                    if row['evidence']['protected_boundary']=='b':
                        self.assertIn('are inside the protected' if design['positive_side']==BOTH else 'are outside the protected',text)
                        if design['positive_side']==NEITHER:
                            self.assertIn('enter the destination' if design['construction_mode']==ENTRY else 'leave the target',text)
        self.assertEqual(len(seen),4)

    def test_unoccluded_region_projections_exchange_between_opposed_views(self):
        original,_,_=self.fixture()
        for t in (0,5,10,15):
            for design in factorial_regions(original['target'],original['goal'],original['material'],'s',t):
                layout=dict(original,**design,construction_version=VERSION,camera_seed_sha256='projection-control')
                poses=balanced_poses(layout,45.);focus=np.asarray(poses[0]['focus'])
                a=np.asarray(list(itertools.product(*zip(*layout['a']))))
                b=a.copy();b[:,:2]=2*focus[:2]-a[:,:2]
                wanted=np.asarray(list(itertools.product(*zip(*layout['b']))))
                self.assertTrue(all(any(np.allclose(x,y,rtol=0,atol=1e-12) for y in wanted) for x in b))
                projected=[]
                for points,pose in zip((a,b),poses):
                    local=(points-focus)@np.asarray(pose['rotation']).reshape(3,3)
                    projected.append(local[:,:2]/(pose['distance']-local[:,2,None])/np.tan(np.radians(pose['fovy'])/2))
                np.testing.assert_allclose(*projected,rtol=0,atol=1e-12)

    def test_incorrect_side_or_height_claim_is_rejected(self):
        original,_,_=self.fixture()
        layout=dict(original,**factorial_regions(original['target'],original['goal'],original['material'],'s',0)[0],construction_version=VERSION)
        changed=copy.deepcopy(layout);changed['a'][0][2]-=.01;changed['a'][1][2]-=.01
        with self.assertRaisesRegex(RuntimeError,'sides disagree'):verify_sides(changed)
        changed=copy.deepcopy(layout);changed['positive_side']=BOTH if layout['positive_side']==NEITHER else NEITHER
        with self.assertRaisesRegex(RuntimeError,'factor cell'):verify_sides(changed)

    def test_no_horizontal_separator_is_not_an_infeasibility_label(self):
        layout,_,_=self.fixture()
        vertical_goal=[[-.005,-.005,.4],[.005,.005,.5]]
        with self.assertRaisesRegex(RuntimeError,'separation'):
            factorial_regions(layout['target'],vertical_goal,layout['material'],'s',0)

    def test_endpoint_membership_does_not_replace_a_safe_inside_witness(self):
        original,samples,actions=self.fixture()
        for t in (0,5,10,15):
            design=factorial_regions(original['target'],original['goal'],original['material'],'s',t)[0]
            if design['positive_side']!=BOTH:continue
            layout=dict(original,**design,construction_version=VERSION);bad=copy.deepcopy(samples)
            for mode in ('native','synchronized'):
                bad[1][mode+'_point'][0]=layout['b'][1][0]+.1
                bad[1][mode+'_root'][0]=layout['b'][1][0]+.1
            proof=verify_layout(layout,bad,actions)
            self.assertTrue(proof['conditional_negative']);self.assertFalse(proof['positive_witness'])
            self.assertFalse(proof['matched_pair'])

    def test_selection_balances_the_joint_cells_without_splitting_units(self):
        units=[];fixture=selection_fixtures.ContrastSelection()
        for source in ('a','b','c'):
            for t in (0,5,10,15):
                unit=fixture.unit(source,t)
                for row in unit:
                    row['split']='train';row['metadata'].update(factors_for(source,t),construction_version=VERSION)
                units.append(unit)
        quota={k:2 for k in selection_quotas()['train']}
        chosen=select_units(units,target=8,quotas={'train':quota})
        self.assertEqual(Counter(selection_cell(u[0]) for u in chosen),quota)
        self.assertEqual(len({u[0]['metadata']['source_id'] for u in chosen}),3)


if __name__=='__main__':unittest.main()
