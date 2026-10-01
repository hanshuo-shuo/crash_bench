"""Stages have fail-closed evidence gates; every attempt stays recoverable."""
import argparse,json,os,sys,time
from pathlib import Path
from protocol import STATES,CONDITIONS,CANDIDATES,CHECKPOINTS,REPEATS,VALIDATION_REPEATS,branch_eligible
from runtime import Runner,InfrastructureError
from api_budget import atomic_json

def execute(root,port,stage):
 if (root/'STOP.json').exists():raise RuntimeError('Stopped root is immutable; use a new root')
 runner=Runner(root,port);rows=[]
 def record(row):
  rows.append(row);atomic_json(root/'rows.json',rows)
  atomic_json(root/'progress.json',{'stage':stage,'completed_runs':len(rows),'last_run':row['run_id'],'updated_unix':time.time()})
 # First run shared seeds through all arms; store every mismatched native inference rather than discard.
 for state in ([STATES[0],STATES[3]] if stage=='smoke' else STATES):
  for condition in CONDITIONS:record(runner.run(state,0,condition))
  for variant in ['center','rim','side']:
   record(runner.run(state,0,'reference',name=state['id']+'_screen_'+variant,variant=variant))
 atomic_json(root/'SMOKE_COMPLETE.json',{'rows':len(rows),'initial_observation_and_chunk_gate':'passed','no_paid_API_calls':True})
 if stage=='smoke':return
 # New policy seeds; same fixed geometry and caption, no cherry-picking repeats.
 for state in STATES:
  for repeat in range(1,REPEATS):
   for condition in CONDITIONS:record(runner.run(state,repeat,condition))
 reference_variants={}
 for state in STATES:
  screened=[r for r in rows if r['state']==state['id'] and r['condition']=='reference' and r['safe_success']]
  variant=screened[0]['variant'] if screened else 'center'
  reference_variants[state['id']]=variant
  # Validate the fixed selected reference on fresh seeded executions; this is not probability certification.
  for repeat in range(VALIDATION_REPEATS):
   record(runner.run(state,repeat,'reference',name=state['id']+'_validate_%02d'%repeat,variant=variant,validation=True))
 atomic_json(root/'INITIAL_COMPLETE.json',{'rows':len(rows),'reference_variants':reference_variants})
 eligible=[]
 for state in STATES:
  reference=[r for r in rows if r['state']==state['id'] and r['condition']=='reference' and r['validation']]
  aegis=[r for r in rows if r['state']==state['id'] and r['condition']=='identity_geometry']
  if state['role']=='diagnostic' and branch_eligible(reference,aegis):eligible.append(state)
 atomic_json(root/'BRANCH_GATE.json',{'eligible_states':[s['id'] for s in eligible],
  'rule':'at least one independent validation safe witness AND corrected identity+geometry AEGIS observed safe noncompletion; report all counts',
  'screening_only':True})
 for state in eligible:
  # Prefix is frozen raw AEGIS; require a noncollided, unsuccessful base prefix for each seed.
  for repeat in range(REPEATS):
   base=next(r for r in rows if r['state']==state['id'] and r['repeat']==repeat and r['condition']=='raw')
   if base['success']:continue
   for step in CHECKPOINTS:
    cp=root/'runs'/base['run_id']/('checkpoint_%03d.json'%step)
    if not cp.exists() or json.loads(cp.read_text())['collided']:continue
    for extra in sorted(set([0,step])):
     for candidate in ['raw']+CANDIDATES:
      name='%s_r%02d_t%03d_b%03d_%s'%(state['id'],repeat,step,extra,candidate)
      record(runner.run(state,repeat,candidate,name=name,variant=reference_variants[state['id']],branch=step,extra=extra))
 atomic_json(root/'COMPLETE.json',{'runs':len(rows),'eligible_states':[s['id'] for s in eligible],'finished_unix':time.time()})
 from report import report
 report(root)

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','full']);p.add_argument('root',type=Path);p.add_argument('--port',type=int,required=True);a=p.parse_args();root=a.root.resolve()
 try:execute(root,a.port,a.stage)
 except BaseException as e:
  atomic_json(root/'STOP.json',{'stage':a.stage,'reason':type(e).__name__+': '+str(e),'unix':time.time()});raise
if __name__=='__main__':main()
