import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from verifier_dataset.collect import prepare_response,CacheMiss
from scripts.openrouter_perception import export_request


class OfflineCollection(unittest.TestCase):
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
