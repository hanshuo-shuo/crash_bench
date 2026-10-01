import ast,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import protocol
from adapter import adapt
class FeasibilityTests(unittest.TestCase):
 def test_fixed_set_and_new_validation_seeds(self):
  self.assertEqual([s['episode'] for s in protocol.STATES[:6]],[3,9,15,0,2,5])
  self.assertEqual(sum(s['role']=='control' for s in protocol.STATES),2)
  for s in protocol.STATES:
   screening={protocol.seed_for(s,r) for r in range(5)};validation={protocol.seed_for(s,r,True) for r in range(10)}
   self.assertFalse(screening&validation)
 def test_branch_gate_needs_independent_witness_and_safe_noncompletion(self):
  self.assertFalse(protocol.branch_eligible([{'safe_success':False}],[{'success':False,'collided':False}]))
  self.assertFalse(protocol.branch_eligible([{'safe_success':True}],[{'success':False,'collided':True}]))
  self.assertTrue(protocol.branch_eligible([{'safe_success':True}],[{'success':False,'collided':False}]))
 def test_adapter_preserves_original_qp_scoring_and_has_no_state_write(self):
  original=(Path(__file__).resolve().parents[2]/'tests/fixtures/main_aegis_upstream.txt').read_text();source=adapt(original);ast.parse(source)
  for anchor in ['prob.solve(solver=cp.OSQP)','action_input[3:6] = 0.2 * u_omega','a_u_omega @ u[3:6]','> 0.001','action_input[6] = action[6]']:
   self.assertIn(anchor,source)
  self.assertIn('_checkpoint(locals())',source);self.assertIn('_geometry(env, obs',source)
  self.assertNotIn('set_state',source.replace('set_init_state','official_init'))
  with self.assertRaises(RuntimeError):adapt(original+'\n')
if __name__=='__main__':unittest.main()
