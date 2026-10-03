"""Validate the single measured fixture on CPU, then at most two policy rollouts."""
import argparse,hashlib,json,os,shutil,subprocess,tarfile,time
from pathlib import Path
from contract import LOWER,UPPER
BASE=Path(__file__).resolve().parents[2]
ASSETS=Path('/projects/p33100/siosio/crashbench_safelibero')
DESIGN=ASSETS/'red_enclosure_geometry/20261003T022243Z_d7f8787b3174'
SCI=['experiments/red_enclosure/'+x for x in ['contract.py','run.py','derived_fixture.json']]+['experiments/feasibility/'+x for x in ['reference.py','geometry.py','reset_forward.py','serve.py']]
POLICY_INTERFACE=['experiments/red_enclosure/'+x for x in ['policy_prompt.py','policy_entry.py']]
def call(a):return subprocess.check_output(a,cwd=BASE,text=True).strip()
def put(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(stage,gate):
    if BASE.resolve()!=(Path.home()/'crash_bench').resolve():raise RuntimeError('Wrong checkout')
    if call(['git','remote','get-url','origin']) not in ['git@github.com:hanshuo-shuo/crash_bench.git','https://github.com/hanshuo-shuo/crash_bench.git']:raise RuntimeError('Wrong origin')
    if call(['git','status','--porcelain=v1','--untracked-files=all']):raise RuntimeError('Dirty source')
    commit=call(['git','rev-parse','HEAD'])
    if commit!=call(['git','rev-parse','origin/codex/feasibility']):raise RuntimeError('Unpublished source')
    if call(['git','-C',str(BASE/'third_party/vlsa-aegis'),'rev-parse','HEAD'])!='2457feed5968ae803926e178c8ce8243b9ecdcf9':raise RuntimeError('Wrong upstream')
    if call(['squeue','-h','-u',os.environ['USER'],'-n','cb_redwall,cb_red_envelope,cb_red_derived','-o','%A']):raise RuntimeError('Enclosure job still active')
    design=json.loads((DESIGN/'GEOMETRY_DESIGN.json').read_text())
    if not design['compatible_candidate'] or LOWER!=design['interior_lower'] or UPPER!=design['interior_upper']:raise RuntimeError('Not the measured compatible fixture')
    if sha(BASE/'experiments/red_enclosure/derived_fixture.json')!=sha(DESIGN/'GEOMETRY_DESIGN.json'):raise RuntimeError('Design changed')
    if call(['sacct','-X','-j','8358624','--noheader','--parsable2','--format=State,ExitCode'])!='COMPLETED|0:0':raise RuntimeError('Geometry replay incomplete')
    hashes={p:sha(BASE/p) for p in SCI};folder=ASSETS/'red_enclosure_derived'
    previous=list(folder.glob('*/receipt.json'))
    if len([p for p in previous if json.loads(p.read_text())['stage']==stage])>=2:raise RuntimeError('Bounded stage attempt cap reached')
    for p in previous:
        r=json.loads(p.read_text())
        if r['stage']==stage and (r['code_commit']==commit or (p.parent/'COMPLETE.json').exists()):raise RuntimeError('Do not duplicate stage')
        if stage=='policy' and r['stage']=='policy':
            if r['job_id']!='8359568' or not (p.parent/'CANCELLED_PROMPT_FIX.json').exists():raise RuntimeError('Unreviewed prior policy attempt')
            if not call(['sacct','-X','-j',r['job_id'],'--noheader','--parsable2','--format=State']).startswith('CANCELLED'):raise RuntimeError('Earlier policy job not canceled')
            if any((p.parent/'policy_cache').glob('*.npz')) or any((p.parent/'open_pi05').iterdir()):raise RuntimeError('Earlier policy attempt has execution records; review before rerunning')
    if stage=='policy':
        if not gate or gate.parent.resolve()!=folder.resolve():raise RuntimeError('Wrong gate root')
        receipt=json.loads((gate/'receipt.json').read_text())
        if receipt['scientific_hashes']!=hashes:raise RuntimeError('Scientific source changed')
        if not json.loads((gate/'GATE_COMPLETE.json').read_text())['passed']:raise RuntimeError('No safe witness/certificate')
        if call(['sacct','-X','-j',receipt['job_id'],'--noheader','--parsable2','--format=State,ExitCode'])!='COMPLETED|0:0':raise RuntimeError('Gate not terminal successful')
    root=folder/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+stage+'_'+commit[:12]);root.mkdir(parents=True)
    source=root/'source';source.mkdir();archive=root/'source.tar'
    subprocess.check_call(['git','archive','--format=tar','--output='+str(archive),commit],cwd=BASE)
    with tarfile.open(archive) as t:t.extractall(source)
    checks=[]
    for p in call(['git','ls-tree','-r','--name-only',commit]).splitlines():
        data=subprocess.check_output(['git','show',commit+':'+p],cwd=BASE)
        if (source/p).read_bytes()!=data:raise RuntimeError('Archive mismatch')
        checks.append(hashlib.sha256(data).hexdigest()+'  '+p)
    (root/'source.sha256').write_text('\n'.join(checks)+'\n');(root/'SOURCE_COMMIT').write_text(commit+'\n')
    if stage=='policy':
        for name in ['open_reference','sealed_reference','open_replay']:(root/name).symlink_to(gate/name,target_is_directory=True)
        for name in ['GATE_COMPLETE.json','PAIR_VERIFIED.json']:shutil.copyfile(gate/name,root/name)
        put(root/'INHERITED_GATE.json',dict(root=str(gate),receipt=receipt,
            hashes={str(p.relative_to(gate)):sha(p) for name in ['open_reference','sealed_reference','open_replay'] for p in (gate/name).iterdir() if p.is_file()}))
    resources=dict(account='p33100',partition='short' if stage=='gate' else 'gengpu',gpus=0 if stage=='gate' else 1,
            cpus=1 if stage=='gate' else 4,mem_gib=8 if stage=='gate' else 32,time_minutes=10 if stage=='gate' else 30)
    receipt=dict(stage=stage,root=str(root),code_commit=commit,scientific_hashes=hashes,
          policy_interface_hashes={p:sha(BASE/p) for p in POLICY_INTERFACE} if stage=='policy' else {},
          design_root=str(DESIGN),gate_root=str(gate) if gate else None,resources=resources,API_calls=0,
          max_policy_rollouts=0 if stage=='gate' else 2,max_actions_each=300,status='prepared')
    put(root/'receipt.json',receipt);env=os.environ.copy();env.pop('OPENROUTER_API_KEY',None);env['CB_RED_ROOT']=str(root)
    script='derived_gate.sbatch' if stage=='gate' else 'derived_policy.sbatch'
    job=subprocess.check_output(['sbatch','--parsable','--output='+str(root/'slurm_%j.log'),str(source/'experiments/red_enclosure'/script)],cwd=BASE,env=env,text=True).strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Submission ID invalid')
    receipt.update(job_id=job,status='submitted');put(root/'receipt.json',receipt);print(json.dumps(receipt,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['gate','policy']);p.add_argument('--gate',type=Path);a=p.parse_args();main(a.stage,a.gate)
