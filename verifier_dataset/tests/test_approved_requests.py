import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.openrouter_perception import export_request, make_request, request_hash
from verifier_dataset.approved_requests import (
    AUTH_PATH, ApprovedBudget, authorization, validate_payload, validate_response)
from verifier_dataset.budget import DatasetBudget
from verifier_dataset.inventory import sha


class ApprovedPerception(unittest.TestCase):
    def setup_budget(self, root):
        request = make_request(b'png', 'move bowl', 'suite', budgeted=True)
        identity = request_hash(request)
        raw = DatasetBudget(root); raw.refresh_prices = lambda: None
        return raw, ApprovedBudget(raw, {'rows': [{'request_identity': identity}]}), identity, request

    def test_approval_is_exact_and_cannot_be_extended(self):
        self.assertEqual(len(authorization()['rows']), 32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'approval.json'
            value = json.loads(AUTH_PATH.read_text()); value['rows'].append(value['rows'][0])
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, 'allowlist changed'):
                authorization(path)

    def test_payload_changed_image_prompt_or_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = export_request(b'png', 'move bowl', 'suite', tmp, budgeted=True)
            row = dict(request_identity=directory.name, request_file_sha256=sha(directory/'request.json'),
                       image_sha256=sha(directory/'input.png'))
            validate_payload(directory, row)
            (directory/'input.png').write_bytes(b'different')
            with self.assertRaisesRegex(RuntimeError, 'reviewed file hashes'):
                validate_payload(directory, row)
            (directory/'input.png').write_bytes(b'png')
            request = json.loads((directory/'request.json').read_text()); request['messages'][0]['content'][1]['text'] = 'new'
            (directory/'request.json').write_text(json.dumps(request))
            with self.assertRaisesRegex(RuntimeError, 'reviewed file hashes'):
                validate_payload(directory, row)

    def test_unknown_charge_blocks_second_request_and_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, guard, identity, request = self.setup_budget(tmp)
            try:
                guard.reserve(identity, request)
                with self.assertRaisesRegex(RuntimeError, 'Unresolved prior charge'):
                    guard.reserve(identity+':attempt2', request)
            finally:
                raw.close()
            with self.assertRaisesRegex(RuntimeError, 'Unknown prior charge'):
                DatasetBudget(tmp)

    def test_two_attempt_ceiling_and_durable_duplicate_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, guard, identity, request = self.setup_budget(tmp)
            try:
                for key in (identity, identity+':attempt2'):
                    guard.reserve(key, request)
                    guard.settle(key, dict(id=key, usage=dict(cost='.001'), model='z-ai/glm-4.5v', provider='Z.AI'))
                with self.assertRaisesRegex(RuntimeError, 'third attempt'):
                    guard.reserve(identity+':attempt3', request)
                with self.assertRaisesRegex(RuntimeError, 'already charged'):
                    guard.reserve(identity, request)
            finally:
                raw.close()

    def test_batch_reservation_ceiling_does_not_reset_after_cheap_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, guard, identity, request = self.setup_budget(tmp)
            try:
                for i in range(64):
                    raw.state['calls'][guard.prefix+str(i)] = dict(status='settled', reserved_usd='.10', cost_usd='.0001')
                with self.assertRaisesRegex(RuntimeError, '64-attempt boundary'):
                    guard.reserve(identity, request)
            finally:
                raw.close()

    def test_new_payload_and_wrong_provider_are_rejected_before_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, guard, identity, request = self.setup_budget(tmp)
            try:
                changed = json.loads(json.dumps(request)); changed['provider']['allow_fallbacks'] = True
                with patch('urllib.request.urlopen', side_effect=AssertionError('Network forbidden')):
                    with self.assertRaisesRegex(RuntimeError, 'Unapproved payload'):
                        guard.reserve(identity, changed)
                self.assertEqual(raw.state['calls'], {})
            finally:
                raw.close()

    def test_cache_response_requires_same_generation_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, guard, identity, request = self.setup_budget(root/'budget')
            directory = export_request(b'png', 'move bowl', 'suite', root/'cache', budgeted=True)
            response = dict(id='generation', usage=dict(cost='.001'), model='z-ai/glm-4.5v', provider='Z.AI',
                            choices=[dict(finish_reason='stop', message=dict(content='blue moka pot'))])
            (directory/'response.json').write_text(json.dumps(dict(request_sha256=identity, response=response)))
            try:
                with self.assertRaisesRegex(RuntimeError, 'accounting'):
                    validate_response(directory, guard)
                guard.reserve(identity, request); guard.settle(identity, response)
                self.assertEqual(validate_response(directory, guard), 'blue moka pot')
            finally:
                raw.close()


if __name__ == '__main__':
    unittest.main()
