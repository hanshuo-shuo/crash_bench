"""Publish-first, isolated, bounded diagnostic launcher. Never submits bulk."""
import argparse,hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]
ASSETS=Path('/projects/p33100/siosio/crashbench_safelibero')
FOLDER=ASSETS/'reach_avoid_matrix_gpu'
SCI=['experiments/reach_avoid/'+x for x in ('matrix_protocol.json','baseline.py','observe.py','layer_features.py','matrix_policy.py','attention_features.py','matrix_cues.py','matrix_gpu.py')]+['experiments/red_enclosure/'+x for x in ('run.py','contract.py')]+['experiments/feasibility_risk/'+x for x in ('diagnostic.py','features.py','fixture.py')]+['experiments/feasibility/'+x for x in ('reference.py','geometry.py','reset_forward.py','serve.py')]
def call(args):return subprocess.check_output(args,cwd=BASE,text=True).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def main(stage,gate):
    if BASE.resolve()!=(Path.home()/'crash_bench').resolve():raise RuntimeError('Wrong checkout')
    if call(['git','remote','get-url','origin']) not in ['git@github.com:hanshuo-shuo/crash_bench.git','https://github.com/hanshuo-shuo/crash_bench.git']:raise RuntimeError('Wrong origin')
    if call(['git','status','--porcelain=v1','--untracked-files=all']):raise RuntimeError('Dirty checkout')
    commit=call(['git','rev-parse','HEAD'])
    if commit!=call(['git','rev-parse','origin/codex/feasibility']):raise RuntimeError('Not published')
    upstream=call(['git','-C',str(BASE/'third_party/vlsa-aegis'),'rev-parse','HEAD'])
    if upstream!='2457feed5968ae803926e178c8ce8243b9ecdcf9':raise RuntimeError('Wrong upstream')
    if call(['squeue','-h','-u',os.environ['USER'],'-n','cb_ra_matrix_gpu','-o','%A']):raise RuntimeError('Project job active')
    previous=list(FOLDER.glob('*/receipt.json'))
    attempts=[json.loads(p.read_text()) for p in previous if json.loads(p.read_text())['stage']==stage]
    if len(attempts)>=1 or any(a['code_commit']==commit for a in attempts):raise RuntimeError('Diagnostic attempt cap or duplicate')
    config=json.loads((BASE/'experiments/reach_avoid/matrix_gpu_protocol.json').read_text())
    if call(['sacct','-X','-j',config['adapter_job'],'--noheader','--parsable2','--format=State,ExitCode'])!='COMPLETED|0:0':raise RuntimeError('Adapter pilot not terminal successful')
    audit=json.loads((Path(config['adapter_root'])/'output/COMPLETE.json').read_text())
    if len(audit['rows'])!=2 or not all(r['layer_accepted'] and r['attention_accepted'] for r in audit['rows']):raise RuntimeError('Adapter audit unresolved')
    hashes={p:sha(BASE/p) for p in SCI}
    root=FOLDER/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+stage+'_'+commit[:12]);root.mkdir(parents=True)
    archive=root/'source.tar';source=root/'source';source.mkdir()
    subprocess.check_call(['git','archive','--format=tar','--output='+str(archive),commit],cwd=BASE)
    with tarfile.open(archive) as t:t.extractall(source)
    checks=[]
    for p in call(['git','ls-tree','-r','--name-only',commit]).splitlines():
        data=subprocess.check_output(['git','show',commit+':'+p],cwd=BASE)
        if data!=(source/p).read_bytes():raise RuntimeError('Archive mismatch')
        checks.append(hashlib.sha256(data).hexdigest()+'  '+p)
    (root/'source.sha256').write_text('\n'.join(checks)+'\n');(root/'SOURCE_COMMIT').write_text(commit+'\n')
    receipt=dict(stage=stage,root=str(root),code_commit=commit,upstream_commit=upstream,scientific_hashes=hashes,protocol_sha256=sha(BASE/'experiments/reach_avoid/observe_protocol.json'),gate_root=str(gate) if gate else None,status='prepared',resources=dict(account='p33100',partition='short' if stage=='cpu' else 'gengpu',gpus=0 if stage=='cpu' else 1,cpus=1 if stage=='cpu' else 4,mem_gib=8 if stage=='cpu' else 32,max_minutes=30,array_tasks=6,max_concurrent=1,expected_total_minutes=[45,70],monitor_checkpoint_minutes=60),API_calls=0)
    put(root/'receipt.json',receipt);env=os.environ.copy();env.pop('OPENROUTER_API_KEY',None);env['CB_RA_ROOT']=str(root);env['CB_RA_LABELS']=config['label_root'];env['CB_RA_VISION']=config['vision_asset']
    if gate:env['CB_RA_GATE']=str(gate)
    job=subprocess.check_output(['sbatch','--dependency=aftercorr:'+config['label_job'],'--kill-on-invalid-dep=yes','--parsable','--export=ALL,CB_RA_ROOT='+str(root)+(',CB_RA_GATE='+str(gate) if gate else ''),'--output='+str(root/'slurm_%A_%a.log'),str(source/'experiments/reach_avoid'/'matrix_gpu.sbatch')],cwd=BASE,env=env,text=True).strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Invalid submission response')
    receipt.update(status='submitted',job_id=job);put(root/'receipt.json',receipt);print(json.dumps(receipt,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['gpu']);p.add_argument('--gate',type=Path);a=p.parse_args();main(a.stage,a.gate)
