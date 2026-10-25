import unittest
import xml.etree.ElementTree as ET

import numpy as np

from verifier_dataset import v5_render


class Render(unittest.TestCase):
    def test_lines_are_parked_noncolliding_display_geoms(self):
        xml = v5_render.add_lines('<mujoco><worldbody><geom name="table" type="box" size="1 1 .1"/></worldbody></mujoco>')
        geoms = {g.get('name'): g for g in ET.fromstring(xml).find('worldbody').findall('geom')}
        for index in range(len(v5_render.LINE_COLORS)):
            line = geoms['v5_line_%d' % index]
            self.assertEqual((line.get('contype'), line.get('conaffinity'), line.get('pos')), ('0', '0', '99 99 99'))

    def test_poses_are_opposing_equal_distance_and_seeded(self):
        regions = [([0, 0, .9], [.05, .05, 1.]), ([.4, .1, .9], [.5, .2, .95]), ([-.2, .3, .9], [-.1, .4, 1.1])]
        a, b = v5_render.poses(regions, 'src|5', 45.)
        self.assertAlmostEqual(a['distance'], b['distance'])
        focus = np.asarray(a['focus'])
        da, db = np.asarray(a['position']) - focus, np.asarray(b['position']) - focus
        self.assertAlmostEqual(da[0], -db[0]); self.assertAlmostEqual(da[1], -db[1]); self.assertAlmostEqual(da[2], db[2])
        self.assertLess(max(a['maximum_projected_abs'], b['maximum_projected_abs']), .8)
        self.assertEqual(v5_render.poses(regions, 'src|5', 45.), [a, b])
        self.assertNotEqual(v5_render.poses(regions, 'src|10', 45.)[0]['azimuth'], a['azimuth'])

    def test_top_view_looks_straight_down_with_the_robot_at_the_bottom(self):
        regions = [([0, 0, .0], [.05, .05, .1]), ([.2, .3, 0], [.3, .4, .05]), ([-.66, -.06, 0], [-.54, .06, .05])]
        pose = v5_render.top_pose(regions, 45.)
        R = np.asarray(pose['rotation']).reshape(3, 3)
        np.testing.assert_allclose(R[:, 2], [0, 0, 1]); np.testing.assert_allclose(R[:, 1], [1, 0, 0])
        local = (np.asarray([-.6, 0., 0.]) - np.asarray(pose['position'])) @ R  # camera frame: x right, y up, looking along -z
        self.assertLess(local[1] / -local[2], 0)  # below the image centre
        self.assertLess(pose['maximum_projected_abs'], .8)


if __name__ == '__main__':
    unittest.main()
