from pathlib import Path
import unittest
from verifier_training.v5_gate_contract import (configuration,questions,prediction,attach,stops,outcome,readouts,reading,interval,
    truth,FILES,VERIFIERS,GROUPS,STATES,INFEASIBLE,FEASIBLE,G1_MAX,G2_RATIO)
from verifier_training.v5_self_contract import facts
from verifier_training.tests.test_v5_contract import v5_row,RULES
from verifier_training.tests.test_v5_self_contract import points_for

RUNS=Path('/projects/p33100/siosio/crashbench_safelibero/verifier_training/runs')
HIL=Path('/projects/p33100/siosio/crashbench_safelibero/verifier_dataset/20261028T024002Z_v5-hil_8d06be7c60a4')
ROOTS=dict(hil=HIL,s3=RUNS/'20261025T055227Z_v5-s3_bdcf277164b6',s4=RUNS/'20261025T180525Z_v5-s4_20913c71c3f3',
    seeds=RUNS/'20261026T001735Z_v5-seeds_ee20b2033465',s5=RUNS/'20261027T050745Z_v5-s5_f20304e222b9')
HAVE_DATA=all(p.exists() for p in ROOTS.values())
P_SEPARATED='Line: x\nLine points: [137, 193] [283, 212]\nTarget point: [282, 263]\nGoal point: [203, 153]\nSeparated: yes\nVerdict: 1'
P_APART='Line: x\nLine points: [137, 193] [283, 212]\nTarget point: [282, 263]\nGoal point: [300, 240]\nSeparated: yes\nVerdict: 0'


def state(i,split='validation',source='src0',time_bin=0):
    return dict(state='s%03d'%i,image='i%04d'%i,split=split,base_source=source,time_bin=time_bin,phase='resting',
        rules=dict(C2=dict(key='line:a%d|b'%i,label=0),C3=dict(key='line:c%d|d'%i,label=1)))


def runs_for(states,c2=(True,False),c3=(True,True)):
    """c2 and c3 are (completed, crossed own line) for every state."""
    out={}
    for s in states:
        for c,(done,crossed) in (('C2',c2),('C3',c3)):
            out[(s['state'],c)]=dict(base_source=s['base_source'],time_bin=s['time_bin'],completed=done,**{'crossed_'+c:crossed})
    return out


def record(q,generated='Verdict: 1',predicted=1,**extra):
    return dict(image_id=q['image'],rule=q['rule'],split='validation',truth=q['truth'],source_id=q['base_source'],
        sample_id='v-'+q['image']+q['condition'],student_input_sha256='h-'+q['image']+q['condition'],generated=generated,
        predicted=predicted,program=extra.get('program'),model=extra.get('model'))


class Questions(unittest.TestCase):
    def test_two_per_validation_state(self):
        states=[state(0),state(1,split='train'),state(2)];qs=questions(states,runs_for(states,c2=(True,False),c3=(False,True)))
        self.assertEqual([(q['state'],q['condition']) for q in qs],[('s000','C2'),('s000','C3'),('s002','C2'),('s002','C3')])
        self.assertEqual([q['truth'] for q in qs],[FEASIBLE,INFEASIBLE]*2);self.assertEqual((truth(0),truth(1)),(1,0))
        self.assertEqual([(q['completed'],q['violated']) for q in qs],[(True,False),(False,True)]*2)

    def test_label_must_match_condition(self):
        s=state(0);s['rules']['C2']['label']=1
        with self.assertRaisesRegex(ValueError,'label'):questions([s],runs_for([s]))

    def test_run_must_match_state(self):
        s=state(0);runs=runs_for([s]);runs[('s000','C3')]['time_bin']=2
        with self.assertRaisesRegex(ValueError,'disagrees'):questions([s],runs)


class Predictions(unittest.TestCase):
    def setUp(self):
        self.q=questions([state(0)],runs_for([state(0)]))[0]

    def test_each_reading_reparses(self):
        self.assertEqual(prediction(record(self.q,'Verdict: 0',0),'direct'),0)
        self.assertIsNone(prediction(record(self.q,'Verdict: maybe',None),'direct'))
        # P is read by its program on the points, not by its written verdict.
        self.assertEqual(prediction(record(self.q,P_SEPARATED,None,program=0),'program'),0)
        self.assertEqual(prediction(record(self.q,P_APART,None,program=1),'program'),1)
        r=v5_row(rule=RULES[0],target=True,goal=False);_,_,text=facts(r,points_for(r,(282,263),(203,153)))
        self.assertEqual(prediction(record(self.q,text,None,model=int(r['target']['digit'])),'model'),int(r['target']['digit']))

    def test_stored_and_reparsed_must_agree(self):
        with self.assertRaisesRegex(ValueError,'Re-parsed'):prediction(record(self.q,'Verdict: 0',1),'direct')
        with self.assertRaisesRegex(ValueError,'Re-parsed'):prediction(record(self.q,P_SEPARATED,None,program=1),'program')
        with self.assertRaisesRegex(ValueError,'Re-parsed'):prediction(record(self.q,'Verdict: 1',None,model=0),'model')


class Attach(unittest.TestCase):
    def setUp(self):
        self.states=[state(0),state(1)];self.qs=questions(self.states,runs_for(self.states))

    def test_keys_only_reads_no_verdict(self):
        attach(self.qs,'A_seed713',[record(q) for q in self.qs])
        self.assertTrue(all('predictions' not in q and q['sample_id'].startswith('v-') for q in self.qs))

    def test_verdicts_by_key(self):
        rs=[record(q,'Verdict: %d'%(i%2),i%2) for i,q in enumerate(self.qs)][::-1]+[dict(record(self.qs[0]),rule='line:other|x')]
        attach(self.qs,'A_seed713',rs,'direct');self.assertEqual([q['predictions']['A_seed713'] for q in self.qs],[0,1,0,1])

    def test_coverage_and_agreement(self):
        with self.assertRaisesRegex(ValueError,'missing'):attach(self.qs,'A_seed713',[record(q) for q in self.qs[1:]])
        with self.assertRaisesRegex(ValueError,'Duplicate'):attach(self.qs,'A_seed713',[record(q) for q in self.qs+self.qs[:1]])
        bad=[record(q) for q in self.qs];bad[0]['truth']=1-bad[0]['truth']
        with self.assertRaisesRegex(ValueError,'disagrees'):attach(self.qs,'A_seed713',bad)
        attach(self.qs,'A_seed713',[record(q) for q in self.qs]);other=[record(q) for q in self.qs];other[2]['sample_id']='x'
        with self.assertRaisesRegex(ValueError,'sample id'):attach(self.qs,'P_seed713',other)
        other=[record(q) for q in self.qs];other[1]['student_input_sha256']='x'
        with self.assertRaisesRegex(ValueError,'inputs'):attach(self.qs,'P_seed713',other)


class Gate(unittest.TestCase):
    def test_decisions(self):
        self.assertEqual([stops(v) for v in (0,1,None)],[True,False,False])
        self.assertEqual([stops(v,'closed') for v in (0,1,None)],[True,False,True])
        with self.assertRaises(ValueError):stops(0,'shut')

    def test_outcome(self):
        c2=dict(condition='C2',violated=False,completed=True);c3=dict(condition='C3',violated=True,completed=True)
        self.assertEqual(outcome(c2,False),dict(stopped=False,violated=False,completed=True,correct=True))
        self.assertEqual(outcome(c2,True),dict(stopped=True,violated=False,completed=False,correct=False))
        self.assertEqual(outcome(c3,False),dict(stopped=False,violated=True,completed=True,correct=False))
        self.assertEqual(outcome(c3,True),dict(stopped=True,violated=False,completed=False,correct=True))
        self.assertFalse(outcome(dict(c2,violated=True),False)['correct'])

    def test_reading_boundaries(self):
        none=dict(G2=.9)
        self.assertEqual(reading(dict(G1=G1_MAX,G2=G2_RATIO*.9),none),'gate_prevents')
        self.assertEqual(reading(dict(G1=G1_MAX,G2=G2_RATIO*.9-1e-9),none),'gate_costly')
        self.assertEqual(reading(dict(G1=G1_MAX+1e-9,G2=.9),none),'gate_leaks')
        self.assertEqual(reading(dict(G1=0.,G2=0.),dict(G2=0.)),'gate_prevents')

    def test_interval(self):
        self.assertEqual(interval([1.]*6,list('aabbcc')),[1.,1.]);self.assertIsNone(interval([0.,1.],['a','a']))
        v=[0.,1.,1.,0.,1.,1.];g=list('abcdef');self.assertEqual(interval(v,g),interval(v,g))
        lo,hi=interval(v,g);self.assertTrue(0<=lo<=4/6<=hi<=1)


class Readouts(unittest.TestCase):
    def setUp(self):
        # Four states over two sources; pi0.5 completes every feasible task and crosses every infeasible line,
        # except state 3's feasible run, which does not complete.
        self.states=[state(i,source='src%d'%(i%2),time_bin=i%2) for i in range(4)];runs=runs_for(self.states)
        runs[('s003','C2')]['completed']=False;self.qs=questions(self.states,runs)
        verdicts=dict(A_seed713=[1]*8,A_seed714=[0]*8,A_seed715=[1]*8,P_seed713=[1,0]*4,P_seed714=[1,0]*4,P_seed715=[1,0,1,0,0,0,1,0],
            S_seed713=[1,0,1,1,0,0,1,None])
        for name,(_,how) in VERIFIERS.items():
            rs=[]
            for q,v in zip(self.qs,verdicts[name]):
                text='Verdict: %d'%v if v is not None else 'no verdict'
                rs.append(record(q,text,v,model=v) if how!='program' else record(q,P_APART if v==1 else P_SEPARATED,None,program=v))
            attach(self.qs,name,rs,how)
        self.r=readouts(self.qs)

    def test_rows(self):
        rows=self.r['rows']
        self.assertEqual({k:(rows['none'][k]) for k in ('G1','G2','G3')},dict(G1=1.,G2=.75,G3=.375))
        self.assertEqual({k:rows['truth'][k] for k in ('G1','G2','G3','stop_infeasible','stop_feasible')},
                         dict(G1=0.,G2=.75,G3=.875,stop_infeasible=1.,stop_feasible=0.))
        # S: state 1's infeasible question passes, state 2's feasible one stops, state 3's infeasible verdict is unparsable (passes).
        self.assertEqual({k:rows['S_seed713'][k] for k in ('G1','G2','G3')},dict(G1=.5,G2=.5,G3=.5))
        self.assertEqual(self.r['rows_fail_closed']['S_seed713']['G1'],.25)
        # Seed means average each question's outcome over the three seeds.
        self.assertAlmostEqual(rows['A_mean']['G1'],2/3);self.assertAlmostEqual(rows['A_mean']['G2'],.5)
        self.assertEqual(rows['P_seed715']['G2'],.5);self.assertAlmostEqual(rows['P_mean']['G2'],(1+1+2/3+0)/4)
        self.assertNotIn('S_mean',rows);self.assertEqual(self.r['rows_fail_closed']['A_seed713'],rows['A_seed713'])

    def test_differences_and_reading(self):
        d=self.r['differences']['S_seed713'];self.assertEqual((d['G1']['value'],d['G2']['value']),(-.5,-.25))
        self.assertEqual(self.r['differences']['truth']['G1']['value'],-1.);self.assertEqual(self.r['reading'],'gate_leaks')
        self.assertEqual(self.r['accuracy']['S_seed713'],dict(correct=5,unparsed=1,predicted_infeasible=3,by_condition=dict(C2=3,C3=2),questions=8))
        self.assertEqual((self.r['states'],self.r['sources'],self.r['questions']),(4,2,8))

    def test_smoke_rows_need_no_verdict(self):
        qs=[{k:v for k,v in q.items() if k!='predictions'} for q in self.qs];r=readouts(qs,verdicts=False,intervals=False)
        self.assertEqual(set(r['rows']),{'none','truth','stop_all'});self.assertNotIn('reading',r)
        self.assertEqual(r['rows']['stop_all']['G1'],0.);self.assertEqual(r['rows']['stop_all']['G2'],0.)

    def test_each_state_needs_both_questions(self):
        with self.assertRaisesRegex(ValueError,'one C2 and one C3'):readouts(self.qs[:-1])


class Configuration(unittest.TestCase):
    def test_contract(self):
        c=configuration();self.assertEqual(c['primary'],'S');self.assertEqual(c['bootstrap']['replicates'],10000)
        self.assertEqual(c['bootstrap']['seed'],1027);self.assertEqual((c['api_calls'],c['execution'],c['training']),(0,0,0))
        for name,(root,_) in VERIFIERS.items():self.assertIn(name+'/validation.jsonl',FILES[root])
        self.assertEqual({n for v in GROUPS.values() for n in v},set(VERIFIERS))
        self.assertFalse(any('test' in f for files in FILES.values() for f in files))


@unittest.skipUnless(HAVE_DATA,'Quest run directories not mounted')
class RealKeys(unittest.TestCase):
    """The real 60 validation-source HIL states join every verifier's validation records by key; no verdict is read."""
    def test_coverage(self):
        from verifier_training.v5_gate_run import build
        qs=build(ROOTS);self.assertEqual(len(qs),2*STATES);self.assertEqual(len({q['base_source'] for q in qs}),4)
        self.assertTrue(all('predictions' not in q for q in qs));self.assertEqual({q['phase'] for q in qs},{'resting'})
        self.assertEqual(sum(q['truth']==INFEASIBLE for q in qs),STATES)


if __name__=='__main__':
    unittest.main()
