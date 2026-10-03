import unittest
from fixture import variants,certificate,decide
class FixtureTests(unittest.TestCase):
 def setUp(self):self.lo=[0.,0.,0.];self.hi=[1.,1.,1.];self.v=variants(self.lo,self.hi,2.)
 def cert(self,boxes):return certificate(boxes,self.lo,self.hi,True,[.5]*3,[[2.]*3,[3.]*3],True,True)
 def test_closed_only(self):
  self.assertEqual([self.cert(self.v[k])['label'] for k in ('open','sealed','parked')],['unknown','infeasible','unknown'])
 def test_same_solid_lid_and_common_walls(self):
  a=self.v['sealed'][-1];b=self.v['parked'][-1]
  self.assertEqual([round(y-x,12) for x,y in zip(a['lower'],a['upper'])],[round(y-x,12) for x,y in zip(b['lower'],b['upper'])])
  self.assertEqual(self.v['open'],self.v['parked'][:-1]);self.assertEqual(self.v['open'],self.v['sealed'][:-1])
 def test_failures_unknown(self):self.assertEqual(decide(False,self.cert(self.v['open'])),'unknown')
 def test_contradiction_rejected(self):
  with self.assertRaises(RuntimeError):decide(True,self.cert(self.v['sealed']))
 def test_one_face_gap_invalidates(self):
  self.v['sealed'][-1]['lower'][0]+=.01
  self.assertEqual(self.cert(self.v['sealed'])['label'],'unknown')
if __name__=='__main__':unittest.main()
