"""Published isolated source, project-only job check, no live checkout mutation."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
SOURCE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(SOURCE/'scripts'))
from api_budget import atomic_json
ASSETS=Path('/projects/p33100/siosio/crashbench_safelibero')
def launch(root,stage):
 checkout=Path.home()/'crash_bench'
 def git(args):return subprocess.check_output(['git']+args,cwd=checkout,text=True).strip()
 if SOURCE!=root/'source' or root.parent.resolve()!=(ASSETS/'feasibility').resolve():raise RuntimeError('Wrong source/root')
 if git(['remote','get-url','origin']) not in ['git@github.com:hanshuo-shuo/crash_bench.git','https://github.com/hanshuo-shuo/crash_bench.git']:raise RuntimeError('Wrong project')
 if git(['status','--porcelain=v1','--untracked-files=all']):raise RuntimeError('Dirty live checkout')
 commit=(root/'SOURCE_COMMIT').read_text().strip()
 if git(['rev-parse','origin/codex/feasibility'])!=commit:raise RuntimeError('Unpublished commit')
 checks=[]
 for relative in git(['ls-tree','-r','--name-only',commit]).splitlines():
  expected=subprocess.check_output(['git','show',commit+':'+relative],cwd=checkout)
  actual=(SOURCE/relative).read_bytes()
  if expected!=actual:raise RuntimeError('Archive mismatch '+relative)
  checks.append(hashlib.sha256(actual).hexdigest()+'  '+relative)
 prior=json.loads((root/'plan.json').read_text()) if (root/'plan.json').exists() else None
 if prior and (not stage.startswith('branch_') or stage in prior['jobs']):raise RuntimeError('Existing stage cannot be resubmitted')
 preflight=root/'PREFLIGHT.json'
 if not preflight.exists() or json.loads(preflight.read_text()).get('status')!='passed':raise RuntimeError('CPU preflight not passed')
 # Restrict inspection to this project's recorded job IDs; other projects are untouched.
 jobs=[]
 for p in (ASSETS/'batches').glob('*/batch.json'):jobs+=json.loads(p.read_text()).get('slurm_arrays',[])
 for p in (ASSETS/'paired_repeats').glob('*/plan.json'):jobs+=list(json.loads(p.read_text()).get('jobs',{}).values())
 for p in (ASSETS/'feasibility').glob('*/plan.json'):jobs+=list(json.loads(p.read_text()).get('jobs',{}).values())
 queued=set(subprocess.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%A'],text=True).split())
 for job in jobs:
  if str(job) in queued:raise RuntimeError('Recorded CrashBench job active: '+str(job))
 (root/'source.sha256').write_text('\n'.join(checks)+'\n')
 plan=prior or {'code_commit':commit,'upstream_commit':'2457feed5968ae803926e178c8ce8243b9ecdcf9','source':str(SOURCE),'stage':stage,'jobs':{},'max_gpus':1,'API_calls':0,'live_checkout_commit':git(['rev-parse','HEAD']),'created_unix':time.time()}
 atomic_json(root/'plan.json',plan)
 env=os.environ.copy();env.pop('OPENROUTER_API_KEY',None);env.update(CB_DIAGNOSTIC_ROOT=str(root),CB_DIAGNOSTIC_STAGE=stage)
 args=['sbatch','--parsable','--output='+str(root/'slurm_%j.log')]
 if stage=='smoke':args+=['--time=01:00:00']
 elif stage=='initial':args+=['--time=08:00:00']
 elif stage.startswith('branch_'):args+=['--time=06:00:00']
 args+=[str(SOURCE/'experiments/feasibility/run_gpu.sbatch')]
 job=subprocess.check_output(args,env=env,cwd=checkout,text=True).strip().split(';')[0]
 if not job.isdigit():raise RuntimeError('Invalid job receipt')
 plan['jobs'][stage]=job;atomic_json(root/'plan.json',plan);print(json.dumps(plan,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','full','initial']+['branch_'+s['id'] for s in __import__('protocol').STATES if s['role']=='diagnostic']);p.add_argument('root',type=Path);a=p.parse_args();launch(a.root.resolve(),a.stage)
