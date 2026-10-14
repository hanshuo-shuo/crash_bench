import unittest
from verifier_dataset.fixture_design import walls
from experiments.reach_avoid.certificate import certify


class FixtureProposal(unittest.TestCase):
    def test_open_roof_is_not_a_separator_even_when_sides_look_closed(self):
        lower=[-.1,-.1,0.];upper=[.1,.1,.2]
        geometry=walls(lower,upper)
        args=(lower,upper,[0.,0.,.1],[[.3,.3,.1],[.4,.4,.2]],.013,True,True,True,True)
        self.assertEqual(certify(geometry,*args)['label'],'infeasible')
        opened=[w for w in geometry if w['name']!='fd_wall_2_1']
        self.assertEqual(certify(opened,*args)['label'],'unknown')


if __name__=='__main__':unittest.main()
