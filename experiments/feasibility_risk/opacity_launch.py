"""Single-attempt, publish-first FR-1C launcher; no bulk or retry path."""
import argparse,hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]
FOLDER=Path('/projects/p33100/siosio/crashbench_safelibero/feasibility_risk_opacity')
SCI=['experiments/feasibility_risk/'+p for p in ('opacity.py','opacity_protocol.json','features.py','visibility.py','fixture.py','diagnostic.py','provenance.py')]+['experiments/red_enclosure/'+p for p in ('run.py','contract.py')]
def call(args):return subprocess.check_output(args,cwd=BASE,text=True).strip()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def put(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def main(stage,gate):
    assert BASE.resolve()==(Path.home()/'crash_bench').resolve()
    assert call(['git','remote','get-url','origin']) in ['git@github.com:hanshuo-shuo/crash_bench.git','https://github.com/hanshuo-shuo/crash_bench.git']
    assert not call(['git','status','--porcelain=v1','--untracked-files=all'])
    commit=call(['git','rev-parse','HEAD']);assert commit==call(['git','rev-parse','origin/codex/feasibility'])
    upstream=call(['git','-C','third_party/vlsa-aegis','rev-parse','HEAD']);assert upstream=='2457feed5968ae803926e178c8ce8243b9ecdcf9'
    assert not call(['squeue','-h','-u',os.environ['USER'],'-n','cb_fr1_diag,cb_fr1b,cb_fr1c,cb_red_derived,cb_feasibility','-o','%A'])
    assert not any(json.loads(p.read_text())['stage']==stage for p in FOLDER.glob('*/receipt.json')),'Single stage attempt already used'
    protocol=json.loads((BASE/'experiments/feasibility_risk/opacity_protocol.json').read_text())
    for key,job in [('cpu_parent','8368811'),('gpu_parent','8369470'),('decoy_cpu_parent','8371202'),('decoy_gpu_parent','8372919')]:
        parent=Path(protocol[key]);assert json.loads((parent/'receipt.json').read_text())['job_id']==job
        assert call(['sacct','-X','-j',job,'--noheader','--parsable2','--format=State,ExitCode'])=='COMPLETED|0:0'
    hashes={p:sha(BASE/p) for p in SCI}
    if stage=='gpu':
        assert gate and gate.parent.resolve()==FOLDER.resolve()
        prior=json.loads((gate/'receipt.json').read_text());assert prior['scientific_hashes']==hashes and prior['code_commit']==commit
        assert json.loads((gate/'CPU_GATE.json').read_text())['passed']
        assert call(['sacct','-X','-j',prior['job_id'],'--noheader','--parsable2','--format=State,ExitCode'])=='COMPLETED|0:0'
    root=FOLDER/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+stage+'_'+commit[:12]);root.mkdir(parents=True)
    source=root/'source';source.mkdir();archive=root/'source.tar'
    subprocess.check_call(['git','archive','--format=tar','--output='+str(archive),commit],cwd=BASE)
    with tarfile.open(archive) as t:t.extractall(source)
    lines=[]
    for p in call(['git','ls-tree','-r','--name-only',commit]).splitlines():
        data=subprocess.check_output(['git','show',commit+':'+p],cwd=BASE);assert data==(source/p).read_bytes()
        lines.append(hashlib.sha256(data).hexdigest()+'  '+p)
    (root/'source.sha256').write_text('\n'.join(lines)+'\n');(root/'SOURCE_COMMIT').write_text(commit+'\n')
    receipt=dict(stage=stage,root=str(root),code_commit=commit,upstream_commit=upstream,scientific_hashes=hashes,protocol_sha256=sha(BASE/'experiments/feasibility_risk/opacity_protocol.json'),gate_root=str(gate) if gate else None,status='prepared',resources=dict(account='p33100',partition='short' if stage=='cpu' else 'gengpu',gpus=0 if stage=='cpu' else 1,cpus=1 if stage=='cpu' else 4,mem_gib=8 if stage=='cpu' else 32,max_minutes=5 if stage=='cpu' else 10),api_calls=0,environment_actions=0)
    put(root/'receipt.json',receipt);env=os.environ.copy();env.pop('OPENROUTER_API_KEY',None);env['CB_FR_ROOT']=str(root)
    job=subprocess.check_output(['sbatch','--parsable','--export=ALL,CB_FR_ROOT='+str(root),'--output='+str(root/'slurm_%j.log'),str(source/'experiments/feasibility_risk'/('opacity_'+stage+'.sbatch'))],cwd=BASE,env=env,text=True).strip().split(';')[0]
    assert job.isdigit();receipt.update(status='submitted',job_id=job);put(root/'receipt.json',receipt);print(json.dumps(receipt,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['cpu','gpu']);p.add_argument('--gate',type=Path);a=p.parse_args();main(a.stage,a.gate)
