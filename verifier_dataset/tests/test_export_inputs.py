import unittest
from verifier_dataset.export_candidates import public_fields


class InputLeakage(unittest.TestCase):
    def test_archive_future_actions_and_proprioception_cannot_enter_student(self):
        saved=dict(image='external',wrist_image='wrist',prompt='task',state='private_state',
                   actions='future_actions',diagnostic_actions='future_samples',sample_id='private_id')
        self.assertEqual(public_fields(saved),dict(external_rgb='external',wrist_rgb='wrist',task='task'))


if __name__=='__main__':unittest.main()
