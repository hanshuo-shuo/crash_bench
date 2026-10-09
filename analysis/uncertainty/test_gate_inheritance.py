"""Recovery guards: reject evidence mutation, incomplete coverage and wrong cases."""
import copy,json,tempfile,unittest,shutil
from pathlib import Path
from common import config,sha
from gate import schedule
from gate_inheritance import read_inherited


class InheritanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        cfg=config('full');cfg['repeats']=list(range(5));self.expected=schedule(cfg,0)
        results=[dict(spec=s,status='complete') for s in self.expected[:4]];files=[]
        for r in results:
            p=self.root/'runs'/r['spec']['name'];p.mkdir(parents=True)
            for n in ['RESULT.json','settled_state.npz','steps.jsonl','rows.jsonl','inferences.jsonl','CONDITIONAL_ACTION_EQUALITY.json','ZERO_COMMAND_SEMANTICS.json']:
                (p/n).write_text(json.dumps(r) if n=='RESULT.json' else 'saved bytes')
                files.append(dict(path=str((p/n).relative_to(self.root)),sha256=sha(p/n)))
        self.saved=dict(passed=True,results=results,files=files);self.write()
    def write(self): (self.root/'INHERITED_SMOKE.json').write_text(json.dumps(self.saved))
    def test_exact_prefix_retained_and_next_case_is_fifth(self):
        got=read_inherited(self.root,self.expected);self.assertEqual(len(got),4);self.assertEqual(self.expected[len(got):][0],self.expected[4])
    def test_linked_original_directories_verified_without_rewriting(self):
        shutil.move(str(self.root/'runs'),str(self.root/'original_runs'));(self.root/'runs').mkdir()
        for r in self.saved['results']:(self.root/'runs'/r['spec']['name']).symlink_to(self.root/'original_runs'/r['spec']['name'],target_is_directory=True)
        self.assertEqual(len(read_inherited(self.root,self.expected)),4)
    def test_modified_bytes_rejected(self):
        (self.root/self.saved['files'][0]['path']).write_text('changed')
        with self.assertRaises(RuntimeError):read_inherited(self.root,self.expected)
    def test_missing_coverage_and_wrong_order_rejected(self):
        original=copy.deepcopy(self.saved);self.saved['files']=self.saved['files'][1:];self.write()
        with self.assertRaises(RuntimeError):read_inherited(self.root,self.expected)
        self.saved=original;self.saved['results'].reverse();self.write()
        with self.assertRaises(RuntimeError):read_inherited(self.root,self.expected)
    def test_duplicate_and_traversal_rejected(self):
        self.saved['files'].append(self.saved['files'][0]);self.write()
        with self.assertRaises(RuntimeError):read_inherited(self.root,self.expected)
        self.saved['files'][-1]=dict(path='../other',sha256='bad');self.write()
        with self.assertRaises(RuntimeError):read_inherited(self.root,self.expected)


if __name__=='__main__':unittest.main()
