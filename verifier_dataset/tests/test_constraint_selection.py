import copy
import unittest
from verifier_dataset.constraint_selection import freeze_sources,EXPOSED,check_unit,select_units
from verifier_dataset.tests import test_constraint_pairs as fixture_tests
from verifier_dataset.constraint_evidence import rows,verify_layout


class ContrastSelection(unittest.TestCase):
    def unit(self,source='a',step=0):
        layout,_,samples,actions=fixture_tests.ConstraintEvidence().fixture()
        result=rows(layout,verify_layout(layout,samples,actions),
            dict(source_id=source,task_group='t',family='f',method='aegis',seed=7),
            [[{'sha256':source+str(step)+'0'}],[{'sha256':source+str(step)+'1'}]])
        for r in result:r['metadata'].update(ancestry_component=source,checkpoint=step)
        return result

    def test_full_inputs_cannot_contradict_or_leak(self):
        unit=self.unit();self.assertTrue(check_unit(unit))
        unit[0]['student']['source_id']='leak'
        with self.assertRaisesRegex(RuntimeError,'Privileged'):check_unit(unit)
        unit=self.unit();unit[2]['student']['images']=copy.deepcopy(unit[0]['student']['images']);unit[3]['student']['images']=copy.deepcopy(unit[1]['student']['images'])
        with self.assertRaisesRegex(RuntimeError,'Duplicate'):check_unit(unit)

    def test_selection_covers_sources_before_extra_checkpoints(self):
        units=[self.unit(s,t) for s in ('a','b','c') for t in (0,5,10)]
        chosen=select_units(units,target=4)
        self.assertEqual(len({u[0]['metadata']['source_id'] for u in chosen}),3)
        with self.assertRaisesRegex(RuntimeError,'inflate'):select_units(units+[units[0]],target=4)
        with self.assertRaisesRegex(RuntimeError,'Insufficient'):select_units(units,target=10)

    def test_split_is_order_independent_and_keeps_ancestry(self):
        data=[r for i in range(10) for r in self.unit(str(i))]
        data+=self.unit(sorted(EXPOSED)[0]);data+=self.unit('shared')
        for r in data:
            if r['metadata']['source_id']=='shared':r['metadata']['ancestry_component']='0'
        a=freeze_sources(data);b=freeze_sources(list(reversed(data)));self.assertEqual(a,b)
        splits={r['source_id']:r['split'] for r in a['sources']}
        self.assertEqual(splits['0'],splits['shared']);self.assertEqual(splits[sorted(EXPOSED)[0]],'train')
        self.assertEqual(set(splits.values()),{'train','validation','test'})

    def test_mode_quotas_preserve_whole_units_and_prioritize_source_coverage(self):
        from verifier_dataset.constraint_sides import EXIT,ENTRY
        units=[self.unit(s,t) for s in ('a','b','c') for t in (0,5)]
        for u in units:
            for r in u:
                r['split']='train';r['metadata']['construction_mode']=EXIT if r['metadata']['checkpoint']==0 else ENTRY
        chosen=select_units(units,target=4,quotas={'train':{EXIT:2,ENTRY:2}})
        self.assertEqual(sum(u[0]['metadata']['construction_mode']==ENTRY for u in chosen),2)
        self.assertEqual(len({u[0]['metadata']['source_id'] for u in chosen}),3)
        with self.assertRaisesRegex(RuntimeError,'Insufficient'):
            select_units(units[:2],target=4,quotas={'train':{EXIT:2,ENTRY:2}})


if __name__=='__main__':unittest.main()
