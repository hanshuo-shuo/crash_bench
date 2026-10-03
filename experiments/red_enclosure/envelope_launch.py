"""Clean published source, unique immutable root, bounded CPU-only envelope replay."""
import hashlib,json,os,subprocess,tarfile,time
from pathlib import Path

BASE=Path(__file__).resolve().parents[2]
ASSETS=Path('/projects/p33100/siosio/crashbench_safelibero')
def call(args):return subprocess.check_output(args,cwd=BASE,text=True).strip()
def put(path,data):path.write_text(json.dumps(data,indent=2)+'\n')

def main():
    if BASE.resolve()!=(Path.home()/'crash_bench').resolve():raise RuntimeError('Wrong project checkout')
    if call(['git','remote','get-url','origin']) not in ['git@github.com:hanshuo-shuo/crash_bench.git','https://github.com/hanshuo-shuo/crash_bench.git']:raise RuntimeError('Wrong project identity')
    if call(['git','status','--porcelain=v1','--untracked-files=all']):raise RuntimeError('Dirty source')
    commit=call(['git','rev-parse','HEAD'])
    if commit!=call(['git','rev-parse','origin/codex/feasibility']):raise RuntimeError('Source not published')
    if call(['git','-C',str(BASE/'third_party/vlsa-aegis'),'rev-parse','HEAD'])!='2457feed5968ae803926e178c8ce8243b9ecdcf9':raise RuntimeError('Upstream mismatch')
    recorded=[]
    for folder,pattern in [('batches','*/batch.json'),('paired_repeats','*/plan.json'),('feasibility','*/plan.json'),('feasibility_contract','*/receipt.json'),('red_enclosure','*/receipt.json'),('red_enclosure_geometry','*/receipt.json')]:
        for p in (ASSETS/folder).glob(pattern):
            data=json.loads(p.read_text());recorded+=data.get('slurm_arrays',[])+list(data.get('jobs',{}).values())
            if data.get('job_id'):recorded.append(data['job_id'])
    active=set(call(['squeue','-h','-u',os.environ['USER'],'-o','%A']).split())&{str(x) for x in recorded}
    if active:raise RuntimeError('Recorded CrashBench jobs active: '+str(active))
    existing=list((ASSETS/'red_enclosure_geometry').glob('*/receipt.json'))
    # At most two bounded CPU geometry attempts; no silent experimental expansion.
    if len(existing)>=2:raise RuntimeError('Red enclosure submission cap reached')
    for p in existing:
        if (p.parent/'COMPLETE.json').exists():raise RuntimeError('Pilot already complete; do not duplicate')
        if json.loads(p.read_text())['code_commit']==commit:raise RuntimeError('This source already attempted; inspect its result')
    root=ASSETS/'red_enclosure_geometry'/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+commit[:12])
    root.mkdir(parents=True,exist_ok=False);source=root/'source';source.mkdir()
    archive=root/'source.tar';subprocess.check_call(['git','archive','--format=tar','--output='+str(archive),commit],cwd=BASE)
    with tarfile.open(archive) as t:t.extractall(source)
    checks=[]
    for p in call(['git','ls-tree','-r','--name-only',commit]).splitlines():
        data=subprocess.check_output(['git','show',commit+':'+p],cwd=BASE)
        if (source/p).read_bytes()!=data:raise RuntimeError('Archive mismatch')
        checks.append(hashlib.sha256(data).hexdigest()+'  '+p)
    (root/'source.sha256').write_text('\n'.join(checks)+'\n');(root/'SOURCE_COMMIT').write_text(commit+'\n')
    receipt=dict(root=str(root),source=str(source),code_commit=commit,upstream_commit='2457feed5968ae803926e178c8ce8243b9ecdcf9',
        resources=dict(account='p33100',partition='short',gpus=0,cpus=1,mem_gib=8,time_minutes=10),
        API_calls=0,independent_layouts=1,max_policy_rollouts=0,source_replay_actions=227,
        status='prepared_not_submitted',created_unix=time.time())
    put(root/'receipt.json',receipt)
    env=os.environ.copy();env.pop('OPENROUTER_API_KEY',None);env['CB_RED_ROOT']=str(root)
    job=subprocess.check_output(['sbatch','--parsable','--output='+str(root/'slurm_%j.log'),str(source/'experiments/red_enclosure/envelope.sbatch')],cwd=BASE,env=env,text=True).strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Invalid submission result')
    receipt.update(job_id=job,status='submitted');put(root/'receipt.json',receipt);print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
