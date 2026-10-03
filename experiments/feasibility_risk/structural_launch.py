"""Publish-first, isolated, bounded diagnostic launcher. Never submits bulk."""
import argparse,ast,hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]
ASSETS=Path('/projects/p33100/siosio/crashbench_safelibero')
FOLDER=ASSETS/'feasibility_risk_structural'
SCI=['experiments/feasibility_risk/'+x for x in ('protocol.json','fixture.py','diagnostic.py','structural_protocol.json','structural.py','features.py','visibility.py')]+['experiments/red_enclosure/'+x for x in ('run.py','contract.py')]+['experiments/feasibility/'+x for x in ('reference.py','geometry.py','reset_forward.py','serve.py')]
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
    if call(['squeue','-h','-u',os.environ['USER'],'-n','cb_fr1_diag,cb_fr1b,cb_red_derived,cb_feasibility','-o','%A']):raise RuntimeError('Project job active')
    previous=list(FOLDER.glob('*/receipt.json'))
    attempts=[json.loads(p.read_text()) for p in previous if json.loads(p.read_text())['stage']==stage]
    if len(attempts)>=2 or any(a['code_commit']==commit for a in attempts):raise RuntimeError('Diagnostic attempt cap or duplicate')
    hashes={p:sha(BASE/p) for p in SCI}
    protocol=json.loads((BASE/'experiments/feasibility_risk/structural_protocol.json').read_text())
    for key,expected_job in [('cpu_parent','8368811'),('gpu_parent','8369470')]:
        parent=Path(protocol[key]);receipt=json.loads((parent/'receipt.json').read_text())
        if receipt['job_id']!=expected_job or call(['sacct','-X','-j',expected_job,'--noheader','--parsable2','--format=State,ExitCode'])!='COMPLETED|0:0':raise RuntimeError('Parent evidence incomplete')
    if stage=='gpu':
        if not gate or gate.parent.resolve()!=FOLDER.resolve():raise RuntimeError('Wrong gate directory')
        old=json.loads((gate/'receipt.json').read_text())
        differences=[p for p in hashes if old['scientific_hashes'][p]!=hashes[p]]
        if differences!=['experiments/feasibility_risk/structural.py']:raise RuntimeError('Unexpected CPU source change')
        # Explicit approved measurement amendment; preserve every function used
        # by the completed physical label experiment, checked against its archive.
        def functions(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        before=functions(gate/'source/experiments/feasibility_risk/structural.py');after=functions(BASE/'experiments/feasibility_risk/structural.py')
        for name in ('geometry','DecoyAudit','make','cpu','gpu'):
            if before[name]!=after[name]:raise RuntimeError('Frozen physical/failed-gate function changed: '+name)
        if call(['sacct','-X','-j','8371672','--noheader','--parsable2','--format=State,ExitCode'])!='FAILED|1:0':raise RuntimeError('Failed gate provenance differs')
        if not json.loads((gate/'CPU_GATE.json').read_text())['passed']:raise RuntimeError('No valid triplet')
        if call(['sacct','-X','-j',old['job_id'],'--noheader','--parsable2','--format=State,ExitCode'])!='COMPLETED|0:0':raise RuntimeError('CPU job incomplete')
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
    receipt=dict(stage=stage,root=str(root),code_commit=commit,upstream_commit=upstream,scientific_hashes=hashes,protocol_sha256=sha(BASE/'experiments/feasibility_risk/structural_protocol.json'),gate_root=str(gate) if gate else None,status='prepared',resources=dict(account='p33100',partition='short' if stage=='cpu' else 'gengpu',gpus=0 if stage=='cpu' else 1,cpus=1 if stage=='cpu' else 4,mem_gib=8 if stage=='cpu' else 32,max_minutes=10),API_calls=0)
    if stage=='gpu':receipt.update(analysis_amendment_sha256=sha(BASE/'experiments/feasibility_risk/structural_amendment.json'),inherited_physical_functions_ast_equal=True,prior_failed_job='8371672')
    put(root/'receipt.json',receipt);env=os.environ.copy();env.pop('OPENROUTER_API_KEY',None);env['CB_FR_ROOT']=str(root)
    if gate:env['CB_FR_GATE']=str(gate)
    job=subprocess.check_output(['sbatch','--parsable','--export=ALL,CB_FR_ROOT='+str(root)+(',CB_FR_GATE='+str(gate) if gate else ''),'--output='+str(root/'slurm_%j.log'),str(source/'experiments/feasibility_risk'/('structural_'+stage+'.sbatch'))],cwd=BASE,env=env,text=True).strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Invalid submission response')
    receipt.update(status='submitted',job_id=job);put(root/'receipt.json',receipt);print(json.dumps(receipt,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['cpu','gpu']);p.add_argument('--gate',type=Path);a=p.parse_args();main(a.stage,a.gate)
