from collections import defaultdict
import json
import unittest
import xml.etree.ElementTree as ET
from verifier_dataset.paired_inputs import drafts,visual_xml


class CounterbalancedInputs(unittest.TestCase):
    def test_no_image_only_or_text_only_label_lookup_and_no_early_training(self):
        summary=dict(matched_pair=True,synchronized_native_goal_required=True,primary_goal=['on','a','plate_1'],
            enclosed_goal=['on','b','plate_1'],remaining_horizon=200,contract='paired',checkpoint=0,
            source=dict(source_id='source',task_group='task',family='family'))
        images=[[dict(camera=c,path=str(a)+'/'+c+'.png',sha256=str(a)+c) for c in ('external','wrist')] for a in (0,1)]
        rows=drafts(summary,{},images,'red mug');by_image=defaultdict(list);by_text=defaultdict(list)
        self.assertEqual(len(rows),4)
        for row in rows:
            self.assertEqual(set(row['student']),{'images','task','constraints'})
            self.assertFalse(row['training_eligible']);self.assertIsNone(row['reasoning_target'])
            label=row['evidence']['feasibility']
            by_image[tuple(x['sha256'] for x in row['student']['images'])].append(label)
            by_text[(row['student']['task'],row['student']['constraints'])].append(label)
        for labels in list(by_image.values())+list(by_text.values()):
            self.assertEqual(sorted(labels),['conditionally_infeasible','feasible'])

    def test_colored_object_outlines_do_not_change_collision_masks(self):
        xml='<mujoco><worldbody><geom name="fd_wall_0_0" type="box" pos="0 0 0" size=".1 .1 .1" contype="1" conaffinity="1" rgba="0 0 1 1"/></worldbody></mujoco>'
        proposal=dict(destination=dict(fixed_root_region=[[1,1,1],[2,2,2]]),
                      primary_bounds=[[3,3,3],[4,4,4]],enclosed_bounds=[[0,0,0],[.1,.1,.1]])
        root=ET.fromstring(visual_xml(xml,proposal));visual=[g for g in root.findall('.//geom') if g.get('name').startswith('fd_visual_')]
        self.assertTrue(all(g.get('contype')=='0' and g.get('conaffinity')=='0' for g in visual))
        self.assertEqual(len([g for g in visual if g.get('name').startswith('fd_visual_primary_')]),12)
        self.assertEqual(root.find('.//geom').get('contype'),'1')


if __name__=='__main__':unittest.main()
