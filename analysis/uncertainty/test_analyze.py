"""Synthetic end-to-end checks for leakage guards and retrospective windows."""
from pathlib import Path
import copy
import csv
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from analyze import Analysis,load

AVAILABLE=all(importlib.util.find_spec(x) is not None for x in ['numpy','scipy'])


def synthetic():
    scenes=['T1/episode00','T1/episode01','T2/episode00','T2/episode01'];runs=[]
    for index,scene in enumerate(scenes):
        for repeat in [0,1]:
            for arm in ['nominal','aegis']:
                C=(3 if index==2 else 27) if repeat==0 and arm=='nominal' else None
                success=(repeat==1 and arm=='nominal') or (repeat==0 and arm=='aegis') or (index==3 and arm=='aegis')
                outcome='crash' if C is not None else ('safe_success' if success else 'safe_incomplete')
                rows=[]
                for t in range(1,33):
                    phase=(t-1)//5
                    rows.append(dict(step=t,infer_boundary=(t-1)%5==0,disagreement=.05+.02*index+.04*repeat+.001*phase+(.01 if arm=='aegis' else 0),
                        churn=None if phase==0 else .01+.003*phase,act_norm=.2+.01*phase,min_dist=.1-.002*phase))
                runs.append(dict(id=scene+'-'+str(repeat)+'-'+arm,scene=scene,task=scene.rsplit('/',1)[0],arm=arm,repeat=repeat,
                    split='train' if index%2==0 else 'test',C=C,L=32,success=int(success),safe_success=int(outcome=='safe_success'),outcome=outcome,
                    rows=rows,inferences=[r for r in rows if r['infer_boundary']]))
    return runs,dict(train=[scenes[0],scenes[2]],test=[scenes[1],scenes[3]])


class GuardTests(unittest.TestCase):
    def test_partial_matrix_cannot_reach_optional_data_libraries_or_fit(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'INDEPENDENT_AUDIT.json').write_text(json.dumps(dict(passed=True,rollouts=1199,states=60,all_diagnostic_sample_tensors_independently_reduced=True)))
            with self.assertRaisesRegex(RuntimeError,'Complete independent1200'):load(p)


@unittest.skipUnless(AVAILABLE,'Pinned Quest NumPy/SciPy environment required')
class PipelineTests(unittest.TestCase):
    def test_all_outputs_and_train_only_invariance(self):
        runs,split=synthetic()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);first=p/'first';second=p/'second';first.mkdir();second.mkdir()
            with contextlib.redirect_stdout(io.StringIO()):Analysis(runs,split,first).run()
            read=lambda name:list(csv.DictReader((first/name).open()))
            self.assertEqual(len(read('collision_metrics.csv')),192)
            self.assertEqual(len(read('completion_metrics.csv')),96)
            self.assertEqual(len(read('confident_failure.csv')),36)
            self.assertEqual(len(read('temporal_curves.csv')),960)
            for row in read('confident_failure_cases.csv'):self.assertEqual(int(row['window_end']),int(row['C'])-1)
            self.assertTrue(any(r['reason']=='short_event_free_incomplete_censored' for r in read('collision_eligibility.csv')))
            self.assertTrue(all(r['eligible']=='False' for r in read('completion_eligibility.csv') if r['stage']=='first_infer' and r['proxy']=='churn'))
            modified=copy.deepcopy(runs)
            for run in modified:
                if run['split']=='test':
                    for row in run['rows']:row['disagreement']=1e6+row['step'];row['min_dist']=-1e6
            with contextlib.redirect_stdout(io.StringIO()):Analysis(modified,split,second).run()
            for name in ['PLATT_MODELS.json','TRAIN_THRESHOLDS.json','confident_failure_reference.csv']:
                self.assertEqual((first/name).read_bytes(),(second/name).read_bytes(),name)

    def test_fit_rejects_test_population(self):
        runs,split=synthetic()
        with tempfile.TemporaryDirectory() as d:
            analysis=Analysis(runs,split,Path(d))
            with self.assertRaisesRegex(RuntimeError,'Test rows reached fit'):
                analysis.fit_model('forbidden',[dict(split='test')])


if __name__=='__main__':unittest.main()
