import unittest
from verifier_training.fd1000_metrics import parse,rationale_grade


class FreeGeneration(unittest.TestCase):
    def test_format_and_verdict_are_separate_measurements(self):
        self.assertEqual(parse('Verdict: 1','A'),(1,True))
        self.assertEqual(parse('Verdict: 1','B'),(1,False))
        self.assertEqual(parse('Rationale: It is enclosed.\nVerdict: 0','B'),(0,True))
        self.assertEqual(parse('Verdict: 10','A'),(None,False))
        self.assertEqual(parse('Verdict: feasible','A'),(None,False))
        self.assertEqual(parse('Rationale: First guess.\nVerdict: 0\nVerdict: 1','B'),(None,False))

    def test_color_contradiction_is_not_a_fluent_correct_explanation(self):
        row=dict(metadata=dict(appearance_assignment={'a':'orange','b':'purple'}),
            evidence=dict(protected_boundary='a'),target=dict(rationale='Expected grounded explanation.'))
        wrong=rationale_grade('Rationale: It is inside the purple boundary.\nVerdict: 0',row)
        self.assertEqual(wrong['status'],'contradiction')
        unknown=rationale_grade('Rationale: A novel paraphrase.\nVerdict: 0',row)
        self.assertEqual(unknown['status'],'needs_manual_review')


if __name__=='__main__':unittest.main()
