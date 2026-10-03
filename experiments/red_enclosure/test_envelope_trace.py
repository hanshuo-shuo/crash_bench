import unittest
from envelope_trace import recorded_rows
class TraceTests(unittest.TestCase):
    def test_duplicate_initial_not_an_action(self):
        z=dict(stage='suffix',step=0,action=None,physics_sha256='same')
        a=dict(stage='suffix',step=1,action=[0]*7,physics_sha256='next')
        self.assertEqual(recorded_rows([z,z.copy(),a]),[z,a])
    def test_mismatch_not_silently_removed(self):
        z=dict(stage='suffix',step=0,action=None,physics_sha256='same')
        with self.assertRaises(ValueError):recorded_rows([z,dict(z,physics_sha256='different')])
        with self.assertRaises(ValueError):recorded_rows([z,dict(stage='suffix',step=2,action=[0]*7)])
if __name__=='__main__':unittest.main()
