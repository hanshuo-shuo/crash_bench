import copy
import unittest
from verifier_training.common import rationale, student_messages, target_text, TASK, CONSTRAINTS
from verifier_training.common import label_contract

class Contracts(unittest.TestCase):
    def test_failure_is_not_negative(self):
        summary={'initial_valid':True,'execution_horizon_T':300,'label':'infeasible','outcome':'collision'}
        with self.assertRaises(ValueError):label_contract(summary,{'label':'unknown','contract':'RA-SEP-1','conditions':{}})
    def test_unknown_rejected(self):
        with self.assertRaises(ValueError):label_contract({'initial_valid':True,'execution_horizon_T':300,'label':'unknown'}, {})
    def test_empty_certificate_rejected(self):
        with self.assertRaises(ValueError):label_contract({'initial_valid':True,'execution_horizon_T':300,'label':'infeasible'},
            {'contract':'RA-SEP-1','label':'infeasible','conditions':{}})
    def test_positive_requires_witness(self):
        s={'initial_valid':True,'execution_horizon_T':300,'label':'feasible','outcome':'safe_timeout','safe_success':False,'safe_history':True,'steps':300}
        with self.assertRaises(ValueError):label_contract(s,{})
        s.update(outcome='safe_completion',safe_success=True,steps=200);label_contract(s,{})
    def test_failed_certificate_condition_rejected(self):
        with self.assertRaises(ValueError):label_contract({'initial_valid':True,'execution_horizon_T':300,'label':'infeasible'},
            {'contract':'RA-SEP-1','label':'infeasible','conditions':{'covered':False}})
    def test_input_invariant_to_metadata_and_target(self):
        r={'student':{'task':TASK,'constraints':CONSTRAINTS},'metadata':{'source':'SECRET_ANCESTRY'},'target':{'verdict':'infeasible','rationale':'SECRET_EVIDENCE'}}
        a=student_messages(r,'B');r['metadata']={};r['target']={'verdict':'feasible'}
        self.assertEqual(a,student_messages(r,'B'))
        self.assertEqual(student_messages(r,'A')[1],a[1])
        self.assertNotIn('SECRET',str(a))
    def test_hidden_target_never_invented(self):
        v={'views':[{'camera':c,'entities':{k:{'pixels_224':0 if k=='target' else 20} for k in ('target','goal','panels')}} for c in ('agentview','robot0_eye_in_hand')]}
        text,evidence=rationale(v)
        self.assertIn('milk is not visible in either view',text)
        self.assertIn('do not establish exact clearance',text)
        self.assertNotIn('inside',text)
        self.assertEqual(len(evidence),4)
    def test_grounding_tracks_each_camera(self):
        v={'views':[{'camera':c,'entities':{k:{'pixels_224':10 if c=='agentview' else 0} for k in ('target','goal','panels')}} for c in ('agentview','robot0_eye_in_hand')]}
        self.assertIn('milk is visible only in the external view',rationale(v)[0])
    def test_binary_output(self):
        r={'target':{'verdict':'unknown','rationale':'x'}}
        with self.assertRaises(ValueError):target_text(r,'B')

if __name__=='__main__':unittest.main()
