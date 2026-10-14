import unittest
import xml.etree.ElementTree as ET
from verifier_dataset.fixture_visuals import ground


class GroundedRendering(unittest.TestCase):
    def test_render_changes_cannot_modify_wall_collisions_or_original_bodies(self):
        xml='<mujoco><worldbody><body name="target"><joint type="free"/></body><geom name="fd_wall_0_0" type="box" pos="1 2 3" size=".01 .1 .2" rgba=".2 .4 .7 1" contype="1" conaffinity="1" friction="1 .005 .0001" solref=".02 1"/></worldbody></mujoco>'
        before=ET.fromstring(xml);after=ET.fromstring(ground(xml,[[4,5,6],[5,6,7]]))
        old=before.find('.//geom');new=after.find('.//geom')
        self.assertEqual({k:v for k,v in old.attrib.items() if k!='rgba'},
                         {k:v for k,v in new.attrib.items() if k!='rgba'})
        self.assertEqual(ET.tostring(before.find('.//body')),ET.tostring(after.find('.//body')))
        decorations=[g for g in after.findall('.//geom') if g.get('name').startswith('fd_visual_')]
        self.assertEqual(len(decorations),24)
        self.assertTrue(all(g.get('contype')=='0' and g.get('conaffinity')=='0' for g in decorations))


if __name__=='__main__':unittest.main()
