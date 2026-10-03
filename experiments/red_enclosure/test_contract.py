import unittest
import xml.etree.ElementTree as ET
from contract import *

class ContractTests(unittest.TestCase):
    def test_tunneling_and_tangency(self):
        self.assertTrue(segment_box([-2,0,0],[2,0,0],[-1,-1,-1],[1,1,1]))
        self.assertTrue(segment_box([-2,1,0],[2,1,0],[-1,-1,-1],[1,1,1]))
        self.assertFalse(segment_box([-2,1.01,0],[2,1.01,0],[-1,-1,-1],[1,1,1]))
        self.assertTrue(segment_box([0,0,0],[0,0,0],[-1,-1,-1],[1,1,1]))
    def test_top_exit_only_open(self):
        a=CENTER+[.07];b=CENTER+[.35]
        self.assertFalse(any(segment_box(a,b,w['lower'],w['upper']) for w in walls(False)))
        self.assertTrue(any(segment_box(a,b,w['lower'],w['upper']) for w in walls(True)))
    def test_all_faces_blocked(self):
        a=CENTER+[.07]
        for axis in range(3):
            for sign in [-1,1]:
                b=a.copy();b[axis]+=sign*2
                self.assertTrue(any(segment_box(a,b,w['lower'],w['upper']) for w in walls(True)))
    def test_invalid_certificate_is_unknown(self):
        args=(True,CENTER+[.07],([-.05,.2,-.01],[.07,.33,.14]),True,True)
        self.assertEqual(certify(walls(True),*args)['label'],'infeasible')
        self.assertEqual(certify(walls(False),*args)['label'],'unknown')
        broken=walls(True);broken[0]['upper'][0]-=.001
        self.assertEqual(certify(broken,*args)['label'],'unknown')
        self.assertEqual(certify(walls(True),True,CENTER+[.07],(LOWER,UPPER),True,True)['label'],'unknown')
        self.assertEqual(certify(walls(True),True,CENTER+[.07],args[2],False,True)['label'],'unknown')
    def test_xml_difference_only_lid(self):
        a=ET.fromstring(modify_xml('<mujoco><worldbody/></mujoco>',False))
        b=ET.fromstring(modify_xml('<mujoco><worldbody/></mujoco>',True))
        lid=b.find('.//*[@name="red_z_high"]');b.find('worldbody').remove(lid)
        self.assertEqual(ET.tostring(a),ET.tostring(b))
        self.assertIsNone(a.find('.//joint'));self.assertIsNone(a.find('.//freejoint'))
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):segment_box([float('nan'),0,0],[0,0,0],[0,0,0],[1,1,1])

if __name__=='__main__':unittest.main()
