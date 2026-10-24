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


if __name__ == '__main__':
    unittest.main()
