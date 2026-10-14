import copy
import unittest
from verifier_dataset.constraint_sides import EXIT,ENTRY,VERSION,mode_for,entry_regions,exit_regions,verify_sides
from verifier_dataset.constraint_evidence import certificate,verify_layout,rows
from verifier_dataset.finalize_constraints import final_rationale
from verifier_dataset.tests import test_constraint_pairs as fixtures


class EndpointSides(unittest.TestCase):
    def case(self):
        layout,geometry,samples,actions=fixtures.ConstraintEvidence().fixture()
        layout['target']=[[-.01,-.01,.09],[.01,.01,.11]]
        layout.update(entry_regions(layout['target'],layout['goal'],layout['material'],'source',0))
        layout.update(construction_mode=ENTRY,construction_version=VERSION)
        return layout,geometry,samples,actions

    def test_initial_geometry_supports_entry_certificate_and_actual_inside_witness(self):
        layout,_,samples,actions=self.case();self.assertTrue(all(verify_sides(layout).values()))
        result=verify_layout(layout,samples,actions)
        self.assertTrue(result['matched_pair']);self.assertEqual(result['witness_commands'],1)
        self.assertTrue(result['blocked_boundary_crossed_by_positive_witness'])
        for proof in result['certificates'].values():
            self.assertEqual(proof['direction'],'inward')
            self.assertTrue(proof['separator']['conditions']['entire_necessary_material_goal_strictly_inside'])

    def test_goal_center_inside_is_insufficient_if_some_goal_points_are_outside(self):
        layout,_,samples,_=self.case();layout['a'][1][0]=.52
        proof=certificate(layout,samples[0],'native')
        self.assertEqual(proof['label'],'unknown')
        self.assertFalse(proof['separator']['conditions']['entire_necessary_material_goal_strictly_inside'])

    def test_initial_inside_or_wrong_side_claim_cannot_be_certified_as_entry(self):
        layout,_,samples,_=self.case();layout['a']=copy.deepcopy(layout['b'])
        self.assertEqual(certificate(layout,samples[0],'native')['label'],'unknown')
        with self.assertRaisesRegex(RuntimeError,'sides disagree'):verify_sides(layout)

    def test_both_endpoints_inside_does_not_replace_the_actual_safe_path(self):
        layout,_,samples,actions=self.case()
        for mode in ('native','synchronized'):
            samples[1][mode+'_point'][0]=.9
            samples[1][mode+'_root'][0]=.9
        result=verify_layout(layout,samples,actions)
        self.assertFalse(result['positive_witness']);self.assertFalse(result['matched_pair'])
        self.assertTrue(result['conditional_negative'])

    def test_four_endpoint_side_cells_have_the_correct_evidence_and_explanations(self):
        entry,_,samples,actions=self.case()
        outward,_,samples2,actions2=fixtures.ConstraintEvidence().fixture()
        outward.update(target=[[-.01,-.01,.09],[.01,.01,.11]],construction_mode=EXIT,construction_version=VERSION)
        source=dict(source_id='s',task_group='t',family='f',method='aegis',seed=7)
        cells=set()
        for layout,history,commands in ((outward,samples2,actions2),(entry,samples,actions)):
            records=rows(layout,verify_layout(layout,history,commands),source,[[{'sha256':'0'}],[{'sha256':'1'}]])
            for r in records:
                inward=r['metadata']['construction_mode']==ENTRY
                a=r['evidence']['protected_boundary']=='a'
                target_inside=(not a) if inward else a;goal_inside=inward
                feasible=r['evidence']['feasibility']=='feasible'
                self.assertEqual(feasible,target_inside==goal_inside);cells.add((target_inside,goal_inside,feasible))
                text=final_rationale(r)
                self.assertNotIn('recorded legal trajectory',text)
                if inward and a:self.assertIn('enters through a face',text)
                if inward and not a:self.assertIn('are inside the protected',text)
        self.assertEqual(len(cells),4)

    def test_checkpoint_modes_are_preassigned_and_balanced_per_source(self):
        for source in ('a','b','c'):
            modes=[mode_for(source,t) for t in range(0,50,5)]
            self.assertEqual(modes.count(EXIT),5);self.assertEqual(modes.count(ENTRY),5)
            self.assertNotEqual(modes[0],modes[1])

    def test_region_size_cannot_identify_the_construction_mode_at_a_fixed_state(self):
        entry,_,samples,actions=self.case()
        for outside in exit_regions(entry['target'],entry['goal'],entry['material'],'source',0):
            layout=dict(entry,**outside);layout['construction_mode']=EXIT
            self.assertTrue(all(verify_sides(layout).values()))
            self.assertTrue(verify_layout(layout,samples,actions)['matched_pair'])
            for role in ('a','b'):
                for axis in range(3):
                    self.assertAlmostEqual(layout[role][1][axis]-layout[role][0][axis],entry['b'][1][axis]-entry['b'][0][axis])


if __name__=='__main__':unittest.main()
