import copy
import unittest
from verifier_training.v4_export import metadata_facts,geometry_facts,relation,check_evidence,stratified_shortcut,sample_units
from verifier_dataset.constraint_sides import EXIT,ENTRY,BOTH,NEITHER,VERSION,cell

COLORS={'a':'orange','b':'purple'}


def row(mode,side,protected,split='train',unit='unit_000',checkpoint=0,sample='s0'):
    feasible=protected=='b'
    return dict(sample_id=sample,split=split,metadata=dict(construction_mode=mode,positive_side=side,appearance_assignment=COLORS,
        unit_id=unit,checkpoint=checkpoint,construction_cell=cell(mode,side)),
        evidence=dict(protected_boundary=protected,feasibility='feasible' if feasible else 'conditionally_infeasible',contract='C'),
        target=dict(verdict='feasible' if feasible else 'infeasible'))


def box(x0,x1):return [[x0,0.,0.],[x1,1.,1.]]


def exit_neither_layout():
    """Target inside a, goal outside a, box b away from both (PLAN.md EXIT x NEITHER)."""
    return dict(contract='C',checkpoint=0,remaining_horizon=300,construction_mode=EXIT,positive_side=NEITHER,construction_version=VERSION,
        construction_cell=cell(EXIT,NEITHER),separation_axis=0,a=box(0.,1.),b=box(5.,6.),
        target=[[.4,.4,.4],[.6,.6,.6]],goal=[[2.4,.4,.4],[2.6,.6,.6]],necessary_material_goal=[[2.39,.39,.39],[2.61,.61,.61]])


def verified(layout,negative=True,witness=120):
    from verifier_dataset.constraint_sides import verify_sides
    sep=dict(label='infeasible',contract='RA-SEP-1',conditions={'all_six_faces_covered':True},covered_faces=[True]*6)
    return dict(contract='C',checkpoint=0,matched_pair=True,endpoint_side_checks=verify_sides(layout),conditional_negative=negative,
        positive_witness=True,witness_commands=witness,
        certificates={k:dict(label='conditionally_infeasible',separator=copy.deepcopy(sep)) for k in ('native','synchronized')})


class StructuredFacts(unittest.TestCase):
    def test_plan_table_for_every_cell_and_rule(self):
        expected={(EXIT,'a'):('yes','no','yes'),(ENTRY,'a'):('no','yes','yes'),(BOTH,'b'):('yes','yes','no'),(NEITHER,'b'):('no','no','no')}
        for mode in (EXIT,ENTRY):
            for side in (BOTH,NEITHER):
                for protected in ('a','b'):
                    f=metadata_facts(row(mode,side,protected))
                    key=(mode,'a') if protected=='a' else (side,'b')
                    self.assertEqual((f['target_inside'],f['goal_inside'],f['separated']),expected[key])
                    self.assertEqual(f['protected'],COLORS[protected])

    def test_geometry_must_agree_and_partial_overlap_fails(self):
        layout=exit_neither_layout()
        self.assertEqual(geometry_facts(layout,'a','orange'),metadata_facts(row(EXIT,NEITHER,'a')))
        self.assertEqual(geometry_facts(layout,'b','purple'),metadata_facts(row(EXIT,NEITHER,'b')))
        with self.assertRaisesRegex(ValueError,'neither strictly inside nor disjoint'):relation(box(0.,1.),[[.5,.4,.4],[1.5,.6,.6]])
        broken=dict(layout,necessary_material_goal=[[.9,.4,.4],[2.6,.6,.6]])
        with self.assertRaises(ValueError):geometry_facts(broken,'a','orange')

    def test_labels_need_certificate_or_witness(self):
        layout=exit_neither_layout()
        self.assertEqual(check_evidence(row(EXIT,NEITHER,'a'),layout,verified(layout))['contract'],'RA-SEP-1')
        self.assertEqual(check_evidence(row(EXIT,NEITHER,'b'),layout,verified(layout))['witness_commands'],120)
        with self.assertRaisesRegex(ValueError,'certificate'):check_evidence(row(EXIT,NEITHER,'a'),layout,verified(layout,negative=False))
        with self.assertRaisesRegex(ValueError,'horizon'):check_evidence(row(EXIT,NEITHER,'b'),layout,verified(layout,witness=301))
        wrong=verified(layout);wrong['certificates']['native']['separator']['contract']='FD1000-INWARD-SEP-1'
        with self.assertRaisesRegex(ValueError,'certificate incomplete'):check_evidence(row(EXIT,NEITHER,'a'),layout,wrong)

    def test_shortcut_strata_use_validation_only(self):
        rows=[row(EXIT,NEITHER,p,split='validation',checkpoint=c,sample=f'{p}{c}') for p in 'ab' for c in (0,5)]
        rows.append(row(EXIT,NEITHER,'a',split='test',sample='t'))
        predictions=dict(sample_ids=[r['sample_id'] for r in rows[:4]],predictions={n:[.2,.1,.9,.8] for n in
            ('color_statistics_rule_interactions_linear','rgb_color_rule_interactions_linear')})
        result=stratified_shortcut(rows,predictions)['color_statistics_rule_interactions_linear']
        self.assertEqual(result['all']['auroc'],1.)
        self.assertEqual(result['EXIT x checkpoint 0']['rows'],2)
        predictions['sample_ids'][0]='t'
        with self.assertRaises(ValueError):stratified_shortcut(rows,predictions)

    def test_sample_pack_takes_two_training_units_per_cell(self):
        units={}
        for i,(mode,side) in enumerate([(m,s) for m in (EXIT,ENTRY) for s in (BOTH,NEITHER)]*3):
            units[f'unit_{i:03d}']=[row(mode,side,'a',unit=f'unit_{i:03d}')]
        units['unit_999']=[row(EXIT,BOTH,'a',split='test',unit='unit_999')]
        names=sample_units(units)
        self.assertEqual(len(names),8);self.assertNotIn('unit_999',names)


if __name__=='__main__':unittest.main()
