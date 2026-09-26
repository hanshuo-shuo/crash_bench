import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import openrouter_perception as perception
import run_safelibero as runner

class ReproductionContract(unittest.TestCase):
    def test_retained_evidence_is_byte_identical(self):
        rows=json.loads((ROOT/'retained/manifest.json').read_text())
        self.assertEqual(len(rows),59)
        for row in rows:
            self.assertEqual(hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest(),row['sha256'],row['path'])
    def test_nominal_and_aegis_preserve_upstream_horizon_and_scoring(self):
        original=(ROOT/'tests/fixtures/main_aegis_upstream.txt').read_text()
        for mode in ['nominal','prepare','aegis']:
            code=runner.adapt_source(original,mode)
            ast.parse(code)
            self.assertIn('np.sum(np.abs(then_obstacle_pos - initial_obstacle_pos)) > 0.001',code)
            self.assertIn('max_steps = 300',code)
            self.assertIn('max_steps = 550',code)
            self.assertNotIn('logging.error(f"Caught exception: {e}")',code)
        nominal=runner.adapt_source(original,'nominal')
        self.assertNotIn('obstacle_detection(agentview_img',nominal)
        self.assertNotIn('load_model(CONFIG_PATH',nominal)
        aegis=runner.adapt_source(original,'aegis')
        self.assertIn('prob.solve(solver=cp.OSQP)',aegis)
        self.assertIn('action_input[3:6] = 0.2 * u_omega',aegis)
        self.assertIn('action_input[6] = action[6]',aegis)
    def test_unreviewed_upstream_change_fails_closed(self):
        original=(ROOT/'tests/fixtures/main_aegis_upstream.txt').read_text()
        with self.assertRaises(RuntimeError):runner.adapt_source(original+'\n','aegis')
    def test_cache_identity_includes_image_prompt_and_suite(self):
        baseline=perception.make_request(b'png','put bowl on plate','safelibero_spatial')
        for image,prompt,suite in [(b'other','put bowl on plate','safelibero_spatial'),(b'png','different task','safelibero_spatial'),(b'png','put bowl on plate','safelibero_long')]:
            self.assertNotEqual(perception.request_hash(baseline),perception.request_hash(perception.make_request(image,prompt,suite)))
        self.assertEqual(baseline['provider']['allow_fallbacks'],False)
        self.assertEqual(baseline['temperature'],0.1)
    def test_cache_refuses_tampered_or_wrong_model_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=perception.export_request(b'png','task','safelibero_spatial',tmp)
            response={'request_sha256':directory.name,'response':{'model':perception.MODEL,'choices':[{'message':{'content':'blue moka pot'}}]}}
            (directory/'response.json').write_text(json.dumps(response))
            self.assertEqual(perception.read_response(directory),'blue moka pot')
            response['response']['model']='another-model'
            (directory/'response.json').write_text(json.dumps(response))
            with self.assertRaises(RuntimeError):perception.read_response(directory)
    def test_missing_key_never_sends_a_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=perception.export_request(b'png','task','safelibero_spatial',tmp)
            with patch.dict('os.environ',{},clear=True), patch.object(perception.Path,'home',return_value=Path(tmp)/'home'), patch('urllib.request.urlopen') as network:
                with self.assertRaises(RuntimeError):perception.fill(directory)
                network.assert_not_called()

if __name__=='__main__':unittest.main()

class ArchiveSafety(unittest.TestCase):
    def test_external_relative_links_survive_and_result_is_verified(self):
        import archive_legacy_outputs as archive
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);src=base/'work/results';src.mkdir(parents=True)
            (base/'work/data').write_text('external immutable data')
            (src/'raw.json').write_text('{"success":false}')
            (src/'link').symlink_to('../data')
            target=base/'store/results';target.parent.mkdir()
            archive.archive_one(src,target)
            self.assertFalse(src.exists())
            self.assertEqual((target/'link').read_text(),'external immutable data')
            self.assertEqual((target/'raw.json').read_text(),'{"success":false}')
    def test_copy_corruption_cannot_remove_original(self):
        import archive_legacy_outputs as archive
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);src=base/'original';src.write_text('irreplaceable')
            target=base/'archive'
            def corrupt(a,b):Path(b).write_text('corrupt')
            with patch.object(archive.shutil,'copy2',side_effect=corrupt):
                with self.assertRaises(AssertionError):archive.archive_one(src,target)
            self.assertEqual(src.read_text(),'irreplaceable')
