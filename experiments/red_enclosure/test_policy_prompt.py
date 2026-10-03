import json,tempfile,unittest
from pathlib import Path
from policy_prompt import adapt_make_env,effective_prompt,NATIVE_TASK,PROTECTED_OBJECT

class PolicyInterfaceTests(unittest.TestCase):
    def test_actual_adapter_preserves_environment_and_exposes_same_safety_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            received=[];env,obs=object(),object()
            def native(folder,sealed):
                folder.mkdir();(folder/'task.json').write_text(json.dumps({'language':NATIVE_TASK,'sealed':sealed,'native_goal':['in','milk_1','basket_1_contain_region']}))
                return env,obs,NATIVE_TASK
            def write(p,v):p.write_text(json.dumps(v))
            wrapped=adapt_make_env(native,lambda e,o:PROTECTED_OBJECT,write,lambda p:json.loads(p.read_text()))
            for sealed in [False,True]:
                folder=Path(tmp)/str(sealed);e,o,text=wrapped(folder,sealed)
                self.assertIs(e,env);self.assertIs(o,obs)
                self.assertIn('robot or the milk touch the red enclosure or the wine bottle',text)
                self.assertTrue(text.startswith(NATIVE_TASK+'. '))
                self.assertFalse(any(label in text.lower() for label in ['feasible','infeasible','sealed','closed','open']))
                task=json.loads((folder/'task.json').read_text())
                self.assertEqual(task['language'],NATIVE_TASK)
                self.assertEqual(task['effective_policy_prompt'],text)
                self.assertEqual(task['native_goal'],['in','milk_1','basket_1_contain_region'])
                received.append(text)
            self.assertEqual(received[0],received[1])
    def test_wrong_task_or_protected_object_stops_before_inference(self):
        with self.assertRaises(ValueError):effective_prompt('different task',PROTECTED_OBJECT)
        with self.assertRaises(ValueError):effective_prompt(NATIVE_TASK,'different_obstacle')

if __name__=='__main__':unittest.main()
