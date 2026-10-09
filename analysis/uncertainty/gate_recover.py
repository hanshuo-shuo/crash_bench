"""Recover only the four completed smoke cases after a baseline mount failure."""
import argparse,ast,copy,json,os,subprocess,time
from pathlib import Path
from common import ASSETS,atomic_json
from preflight import stream_sha
from gate import check,schedule,adapt_gate
from gate_launch import submit
from audit_shard import terminal_accounting


def same_science(old,new):
    names=['runtime.py','adapter.py','sampling.py','serve.py','geometry.py','common.py','config.yaml','states.json','split.json','STATS_PLAN.md','stat_rules.py','cluster_stats.py']
    for name in names:
        if (old/name).read_bytes()!=(new/name).read_bytes():raise RuntimeError('Scientific source changed: '+name)
    def functions(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    a,b=functions(old/'gate.py'),functions(new/'gate.py')
    for name in ['command','adapt_gate','schedule','install','check']:
        if a[name]!=b[name]:raise RuntimeError('Gate scientific function changed: '+name)
    return dict(byte_unchanged_files=names,ast_unchanged_functions=['command','adapt_gate','schedule','install','check'])


def failed_accounting(job):
    text=subprocess.check_output(['sacct','-j',job,'--format=JobIDRaw,State,ExitCode,ElapsedRaw','-P','-n'],text=True)
    rows=[r.split('|') for r in text.splitlines() if r.split('|')[0]==job]
    if len(rows)!=1 or rows[0][1:3]!=['FAILED','1:0']:raise RuntimeError('Exact failed mount-validation job required')
    return dict(job=job,state='FAILED',exit_code='1:0',elapsed_seconds=int(rows[0][3]),reason='Four complete smoke cases; baseline path unavailable inside original container')


def prepare(root,failed):
    repo=Path.home()/'crash_bench';source=root/'source';commit=(root/'SOURCE_COMMIT').read_text().strip()
    if root.parent!=(ASSETS/'uncertainty').resolve() or failed.parent!=root.parent or (root/'RECOVERY_PLAN.json').exists():raise RuntimeError('New immutable own-project root required')
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo) or subprocess.check_output(['git','rev-parse','origin/codex/feasibility'],cwd=repo,text=True).strip()!=commit:raise RuntimeError('Clean published recovery source required')
    old=json.loads((failed/'SUBMITTED.json').read_text());job=json.loads((failed/'shards/0/SUBMITTED.json').read_text())['job']
    if (failed/'shards/1/SUBMITTED.json').exists():raise RuntimeError('Recovery only before second shard release')
    failure=json.loads((failed/'shards/0/STOP.json').read_text())
    if failure['completed']!=4 or failure['error']!='FileNotFoundError' or 'settled_state.npz' not in failure['reason']:raise RuntimeError('Different failure requires review')
    accounting=failed_accounting(job);science=same_science(failed/'source/analysis/uncertainty',source/'analysis/uncertainty')
    files=[]
    for name in subprocess.check_output(['git','ls-tree','-r','--name-only',commit],cwd=repo,text=True).splitlines():
        if (source/name).read_bytes()!=subprocess.check_output(['git','show',commit+':'+name],cwd=repo):raise RuntimeError('Recovery archive differs')
        files.append(dict(path=name,sha256=stream_sha(source/name)))
    atomic_json(root/'SOURCE_FILES.json',files)
    (root/'FINGERPRINTS.json').write_bytes((failed/'FINGERPRINTS.json').read_bytes())
    plan=copy.deepcopy(old);plan.update(root=str(root),code_commit=commit,recovery_from=str(failed),recovery_scientific_identity=science)
    plan['prior_gpu_accounting'].append(accounting);plan['prior_gpu_seconds']+=accounting['elapsed_seconds'];plan['maximum_series_gpu_seconds']=plan['prior_gpu_seconds']+plan['requested_gpu_seconds']
    if plan['configuration']['resources']!=dict(account='p33100',partition='gengpu',gpu='a100:1',cpus=8,memory_gb=64,minutes=360,workers=2) or plan['maximum_series_gpu_seconds']>plan['authorized_ceiling_gpu_seconds']:raise RuntimeError('Original resources/48GPUh ceiling violated')
    (root/'shards').mkdir()
    for shard in [0,1]:
        p=root/'shards'/str(shard);p.mkdir();atomic_json(p/'plan.json',dict(configuration=plan['configuration'],code_commit=commit,upstream_root=plan['upstream_root'],shard=shard))
    atomic_json(root/'RECOVERY_PLAN.json',plan)
    env=dict(os.environ,CB_UNCERTAINTY_ROOT=str(root))
    cpu=subprocess.check_output(['sbatch','--parsable','--output='+str(root/'reaudit_%j.log'),str(source/'analysis/uncertainty/gate_smoke_reaudit.sbatch')],env=env,cwd=repo,text=True).strip().split(';')[0]
    if not cpu.isdigit():raise RuntimeError('Invalid CPU audit receipt')
    atomic_json(root/'REAUDIT_SUBMITTED.json',dict(job=cpu,code_commit=commit,failed_job=job,scope='Read-only four-case raw re-audit and report fixture; no model/simulator/new rollout/API',resources=dict(account='p33100',partition='short',cpus=2,memory_gb=8,minutes=60)))
    print(json.dumps(dict(root=str(root),cpu_job=cpu,prior_gpu_seconds=plan['prior_gpu_seconds'],maximum_series_gpu_seconds=plan['maximum_series_gpu_seconds']),indent=2))


def audit(root):
    plan=json.loads((root/'RECOVERY_PLAN.json').read_text());failed=Path(plan['recovery_from']);old=failed/'shards/0';results=json.loads((old/'progress.json').read_text())['results']
    if len(results)!=4 or [r['spec'] for r in results]!=schedule(plan['configuration'],0)[:4]:raise RuntimeError('Exact complete smoke prefix required')
    science=same_science(failed/'source/analysis/uncertainty',root/'source/analysis/uncertainty')
    for parent in [root,failed]:
        for item in json.loads((parent/'SOURCE_FILES.json').read_text()):
            if stream_sha(parent/'source'/item['path'])!=item['sha256']:raise RuntimeError('Frozen source changed')
    fingerprints=json.loads((root/'FINGERPRINTS.json').read_text())
    for item in fingerprints['upstream_files']:
        if stream_sha(Path(plan['upstream_root'])/item['path'])!=item['sha256']:raise RuntimeError('Upstream bytes changed')
    for item in fingerprints['assets']+[fingerprints['container']]+plan['configuration']['controller_files']:
        if stream_sha(Path(item['path']))!=item['sha256']:raise RuntimeError('Consumed asset/controller bytes changed')
    source=adapt_gate((Path(plan['upstream_root'])/'main/main_aegis.py').read_text(),'gate');files=[]
    for result in results:
        p=old/'runs'/result['spec']['name']
        if (p/'adapted_main_aegis.py').read_text()!=source:raise RuntimeError('Executed gate adapter changed')
        semantics=json.loads((p/'ZERO_COMMAND_SEMANTICS.json').read_text())
        expected={x['sha256'] for x in plan['configuration']['controller_files']}
        if semantics['gripper']!='PandaGripper' or not semantics['use_delta'] or semantics['impedance_mode']!='fixed' or not set(semantics['source_sha256'].values())<=expected:raise RuntimeError('Actual zero-command controller evidence differs')
        for f in sorted(p.rglob('*')):
            if f.is_symlink():raise RuntimeError('Unexpected linked smoke evidence')
            if f.is_file():files.append(dict(path=str(f.relative_to(old)),sha256=stream_sha(f)))
    proof,_=check(old,results)
    proof.update(results=results,files=files,failed_root=str(failed),failed_job=json.loads((failed/'shards/0/SUBMITTED.json').read_text())['job'],original_gate_commit=json.loads((failed/'SUBMITTED.json').read_text())['code_commit'],audit_code_commit=(root/'SOURCE_COMMIT').read_text().strip(),scientific_identity=science,actual_controller_evidence_verified=True,readonly_baseline_mount_verified=True,new_rollouts=0,api_calls=0,job=os.environ['SLURM_JOB_ID'])
    atomic_json(root/'REAUDIT_SMOKE_PASS.json',proof)
    print(json.dumps({k:v for k,v in proof.items() if k not in ['files','results']},indent=2))


def launch(root):
    if (root/'SUBMITTED.json').exists():raise RuntimeError('Never resubmit recovery')
    cpu=json.loads((root/'REAUDIT_SUBMITTED.json').read_text());terminal_accounting(cpu['job'])
    proof=json.loads((root/'REAUDIT_SMOKE_PASS.json').read_text());plan=json.loads((root/'RECOVERY_PLAN.json').read_text())
    if not proof['passed'] or proof['runs']!=4 or proof['job']!=cpu['job'] or proof['audit_code_commit']!=plan['code_commit'] or not proof['independent_all_sample_reduction'] or not proof['paired_settled_physical_arrays_exact']:raise RuntimeError('Four-case allocated re-audit required')
    p=root/'shards/0';(p/'runs').mkdir()
    old=Path(proof['failed_root'])/'shards/0'
    for result in proof['results']:
        target=old/'runs'/result['spec']['name'];(p/'runs'/result['spec']['name']).symlink_to(target,target_is_directory=True)
    inherited=dict(passed=True,results=proof['results'],files=proof['files'],original_gate_commit=proof['original_gate_commit'],original_gpu_job=proof['failed_job'],cpu_reaudit_job=cpu['job'],cpu_reaudit_sha256=stream_sha(root/'REAUDIT_SMOKE_PASS.json'))
    atomic_json(p/'INHERITED_SMOKE.json',inherited)
    from gate_inheritance import read_inherited
    read_inherited(p,schedule(plan['configuration'],0))
    plan.update(inherited_smoke_cases=4,fresh_rollouts_remaining=296,inherited_smoke=inherited)
    atomic_json(root/'SUBMITTED.json',plan);job=submit(root,0)
    atomic_json(root/'FIRST_SHARD_SUBMITTED.json',dict(job=job,inherited_complete_smoke_cases=4,fresh_shard0_cases=146,second_shard_held_until_smoke=True))
    print(json.dumps(dict(root=str(root),job=job,inherited=4,remaining=296),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','audit','launch']);p.add_argument('root',type=Path);p.add_argument('--failed',type=Path);a=p.parse_args()
    if a.action=='prepare':prepare(a.root.resolve(),a.failed.resolve())
    elif a.action=='audit':audit(a.root.resolve())
    else:launch(a.root.resolve())
