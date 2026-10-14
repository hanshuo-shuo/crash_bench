import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from verifier_dataset.collect import prepare_response,CacheMiss,check_expected_request
from verifier_dataset.approved_collection import check_cache
from verifier_dataset.approved_requests import AUTH_SHA256
from scripts.openrouter_perception import export_request


class OfflineCollection(unittest.TestCase):
    def test_unapproved_reconstructed_screenshot_remains_offline_unknown(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);request=export_request(b'different pixels','task','suite',root/'requests',budgeted=True)
            with patch('urllib.request.urlopen',side_effect=AssertionError('Network forbidden')):
                with self.assertRaises(CacheMiss):check_expected_request(request,'approved_other_input')
            self.assertFalse((request/'NEEDS_API.json').exists())
            self.assertFalse(json.loads((request/'CACHE_MISS.json').read_text())['paid_request_allowed'])

    def test_changed_staged_response_blocks_offline_collection(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);cached=root/'perception_cache/request';cached.mkdir(parents=True)
            (cached/'response.json').write_text('{}')
            (root/'APPROVED_CACHE.json').write_text(json.dumps(dict(authorization_sha256=AUTH_SHA256,
                rows=[dict(identity='request',files={'response.json':'wrong hash'})])))
            with self.assertRaisesRegex(RuntimeError,'Staged exact cache changed'):check_cache(root)

    def test_cache_only_miss_cannot_publish_paid_work(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);request=export_request(b'pixels','task','suite',root/'requests',budgeted=True)
            with patch('urllib.request.urlopen',side_effect=AssertionError('Network forbidden')):
                with self.assertRaises(CacheMiss):prepare_response(request,[root/'cache'],True)
            self.assertTrue((request/'CACHE_MISS.json').exists())
            self.assertFalse((request/'NEEDS_API.json').exists())

    def test_exact_cached_response_requires_no_network(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);request=export_request(b'pixels','task','suite',root/'requests',budgeted=True)
            cached=export_request(b'pixels','task','suite',root/'cache',budgeted=True)
            (cached/'response.json').write_text(json.dumps(dict(request_sha256=cached.name,response=dict(
                model='z-ai/glm-4.5v',provider='Z.AI',choices=[dict(message=dict(content='blue moka pot'))]))))
            with patch('urllib.request.urlopen',side_effect=AssertionError('Network forbidden')):
                prepare_response(request,[root/'cache'],True)
            self.assertTrue((request/'response.json').exists())
            self.assertFalse((request/'NEEDS_API.json').exists())


if __name__=='__main__':unittest.main()
