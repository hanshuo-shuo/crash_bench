"""Launch exact-commit collection stages from the verified project checkout."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from .inventory import ASSETS, sha, write, read


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def main(stage, inventory=None, canonicalize=False, limit=12, source_plan=None, replay_smoke=None, collection_root=None, batch_shard=None, candidate_exports=None, fixture_design=None, drift_roots=None):
    repo = Path.cwd().resolve()
    if repo != (Path.home()/'crash_bench').resolve():
        raise RuntimeError('Wrong project checkout')
    if command('git', 'status', '--porcelain', '--untracked-files=all'):
        raise RuntimeError('Dirty source')
    if command('git', 'remote', 'get-url', 'origin') not in (
            'git@github.com:hanshuo-shuo/crash_bench.git', 'https://github.com/hanshuo-shuo/crash_bench.git'):
        raise RuntimeError('Wrong repository identity')
    commit = command('git', 'rev-parse', 'HEAD')
    if 'origin/' not in command('git', 'branch', '-r', '--contains', commit):
        raise RuntimeError('Source must be published')
    if stage in ('replay-smoke','replay-spatial-batch','replay-object-sources','replay-spatial-sources'):
        if not 1<=limit<=12: raise RuntimeError('Smoke limit must be 1..12')
        if inventory is None or not read(inventory/'inventory/SUMMARY.json')['passed']:
            raise RuntimeError('Passed inventory required')
        inventory_job=read(inventory/'RECEIPT.json')['job']
        accounting=command('sacct','-X','-n','-P','-j',inventory_job,'--format=JobIDRaw,State,ExitCode')
        if accounting.splitlines()!=[inventory_job+'|COMPLETED|0:0']:
            raise RuntimeError('Inventory must be terminal COMPLETED/0:0')
        hashes=read(inventory/'inventory/HASHES.json')
        for name, expected in hashes.items():
            if sha(inventory/'inventory'/name)!=expected: raise RuntimeError('Inventory hash changed')
    if stage in ('replay-spatial-batch','replay-object-sources','replay-spatial-sources'):
        if not canonicalize or replay_smoke is None:raise RuntimeError('Canonicalized passing 12-source smoke required')
        smoke=read(replay_smoke/'REPLAY_COMPLETE.json');smoke_job=read(replay_smoke/'RECEIPT.json')['job']
        if not smoke['passed'] or smoke['completed']!=12 or smoke['errors']:
            raise RuntimeError('Full replay smoke did not pass')
        if command('sacct','-X','-n','-P','-j',smoke_job,'--format=JobIDRaw,State,ExitCode').splitlines()!=[smoke_job+'|COMPLETED|0:0']:
            raise RuntimeError('Smoke must terminate normally; a JSON pass alone is insufficient')
        if not read(replay_smoke/'CAMPAIGN.json')['canonicalize']:raise RuntimeError('Wrong smoke action contract')
        if stage in ('replay-object-sources','replay-spatial-sources') and read(replay_smoke/'CAMPAIGN.json').get('contact_contract')!='FD1000-SC-CONTACT-v2':
            raise RuntimeError('Distinct-source replay requires the new user-approved contact smoke')
    if stage=='export-candidates':
        if replay_smoke is None:raise RuntimeError('Explicit completed replay source required')
        source_receipt=read(replay_smoke/'RECEIPT.json')
        if source_receipt['stage'] not in ('replay-smoke','replay-spatial-batch','replay-object-sources','replay-spatial-sources','replay-collection'):
            raise RuntimeError('Wrong witness source stage')
        if not source_receipt['canonicalize']:raise RuntimeError('Canonical legal witness required')
    if stage=='design-fixture':
        if replay_smoke is None or not canonicalize:raise RuntimeError('Explicit legal replay source required')
        proof=read(replay_smoke/'REPLAY_COMPLETE.json');proof_receipt=read(replay_smoke/'RECEIPT.json')
        proof_job=proof_receipt['job']
        if not proof['passed'] or proof['errors'] or command('sacct','-X','-n','-P','-j',proof_job,'--format=JobIDRaw,State,ExitCode').splitlines()!=[proof_job+'|COMPLETED|0:0']:
            raise RuntimeError('Completed exact source replay required')
        eligible=[x for x in proof['rows'] if x['eligible_feasible_witness'] and
                  x['source']['scene']['suite']=='safelibero_long' and x['source']['scene']['task']==3]
        if not eligible:raise RuntimeError('No safe Long pudding witness')
        chosen_design=sorted(eligible,key=lambda x:x['source']['source_id'])[0]['source']
    if stage=='fixture-smoke':
        if fixture_design is None:raise RuntimeError('Explicit checked fixture proposal required')
        design_receipt=read(fixture_design/'RECEIPT.json');design_job=design_receipt['job']
        if design_receipt['stage']!='design-fixture' or command('sacct','-X','-n','-P','-j',design_job,'--format=JobIDRaw,State,ExitCode').splitlines()!=[design_job+'|COMPLETED|0:0']:
            raise RuntimeError('Design pass must terminate normally')
        if not read(fixture_design/'REPLAY_COMPLETE.json')['passed']:raise RuntimeError('Failed design replay')
        for name,wanted in read(fixture_design/'case_000/HASHES.json').items():
            if sha(fixture_design/'case_000'/name)!=wanted:raise RuntimeError('Design evidence changed')
        proposal=read(fixture_design/'case_000/FIXTURE_PROPOSAL.json')
        if not proposal['compatible_proposal'] or proposal['label']!='unknown':raise RuntimeError('Invalid proposal for bounded fixture test')
    if stage in ('fixture-render','fixture-grounding'):
        if collection_root is None:raise RuntimeError('Explicit completed fixture smoke required')
        old=read(collection_root/'RECEIPT.json');old_job=old['job']
        if old['stage']!='fixture-smoke' or command('sacct','-X','-n','-P','-j',old_job,'--format=JobIDRaw,State,ExitCode').splitlines()!=[old_job+'|COMPLETED|0:0']:
            raise RuntimeError('CPU fixture check must terminate normally')
        fixture_proof=read(collection_root/'FIXTURE_COMPLETE.json')
        if not fixture_proof['passed'] or not fixture_proof['same_initial_physics']:
            raise RuntimeError('CPU fixture evidence did not pass')
        inherited=read(collection_root/'INHERITED.json')
        for name,key in [('FIXTURE_PROPOSAL.json','proposal_sha256'),('CANONICAL_CONTEXT.json','canonical_context_sha256'),
                         ('CANONICAL_PHYSICS.npz','canonical_physics_sha256')]:
            if sha(collection_root/name)!=inherited[key]:raise RuntimeError('CPU fixture construction input changed')
        for case in ('closed_full','open_full','closed_mug'):
            for name,wanted in read(collection_root/case/'HASHES.json').items():
                if sha(collection_root/case/name)!=wanted:raise RuntimeError('CPU fixture evidence changed')
        active=command('squeue','-h','-u',os.environ['USER'],'-A','p33100','-p','gengpu','-o','%j|%Z')
        if sum('crash_bench' in x or 'crashbench_safelibero' in x or x.startswith('cb_') for x in active.splitlines())>=2:
            raise RuntimeError('Two CrashBench GPU jobs already active')
    if stage=='replay-collection':
        if collection_root is None or not canonicalize:raise RuntimeError('Explicit collection and canonical action replay required')
        from .collection_evidence import replay_candidates
        collection_rows,collection_inheritance=replay_candidates(collection_root)
        source_job=collection_inheritance['source_job']
        if command('sacct','-X','-n','-P','-j',source_job,'--format=JobIDRaw,State,ExitCode').splitlines()!=[source_job+'|COMPLETED|0:0']:
            raise RuntimeError('Collection job must finish normally before replay')
    if stage=='audit-candidates':
        if not candidate_exports or len(set(candidate_exports))!=len(candidate_exports):
            raise RuntimeError('Distinct explicit candidate export roots required')
        for export in candidate_exports:
            old=read(export/'RECEIPT.json');job=old['job']
            if old['stage']!='export-candidates' or not read(export/'CANDIDATE_EXPORT.json')['passed']:
                raise RuntimeError('Candidate export prerequisite did not pass')
            if command('sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode').splitlines()!=[job+'|COMPLETED|0:0']:
                raise RuntimeError('Export must terminate normally before audit')
    if stage in ('collect-smoke','collect-cached-smoke','collect-cached-batch'):
        if source_plan is None or not read(source_plan/'source_plan/SUMMARY.json')['passed']:
            raise RuntimeError('Passed broader-source inventory required')
        source_job=read(source_plan/'RECEIPT.json')['job']
        if command('sacct','-X','-n','-P','-j',source_job,'--format=JobIDRaw,State,ExitCode').splitlines()!=[source_job+'|COMPLETED|0:0']:
            raise RuntimeError('Source inventory must be terminal COMPLETED/0:0')
        for name,expected in read(source_plan/'source_plan/HASHES.json').items():
            if sha(source_plan/'source_plan'/name)!=expected:raise RuntimeError('Broader source manifest changed')
        active=command('squeue','-h','-u',os.environ['USER'],'-A','p33100','-p','gengpu','-o','%j|%Z')
        if sum('crash_bench' in x or 'crashbench_safelibero' in x or x.startswith('cb_') for x in active.splitlines())>=2:
            raise RuntimeError('Two CrashBench GPU jobs already active')
        if stage=='collect-cached-batch':
            if batch_shard not in (0,1) or collection_root is None:
                raise RuntimeError('Two-shard index and terminal cache-only smoke required')
            from .collection_evidence import replay_candidates
            _, prior=replay_candidates(collection_root)
            if read(collection_root/'RECEIPT.json')['stage']!='collect-cached-smoke':
                raise RuntimeError('Cache-only smoke prerequisite changed')
            prior_job=prior['source_job']
            if command('sacct','-X','-n','-P','-j',prior_job,'--format=JobIDRaw,State,ExitCode').splitlines()!=[prior_job+'|COMPLETED|0:0']:
                raise RuntimeError('Offline smoke must terminate normally before extending')
    if stage in ('collect-approved-cache','collect-frozen-perception'):
        from .approved_collection import sources
        if stage=='collect-approved-cache':
            if collection_root is None:raise RuntimeError('Explicit completed approved API batch required')
            approved_sources,approved_inheritance=sources(collection_root,batch_shard)
        else:
            if not drift_roots or len(drift_roots)!=2:raise RuntimeError('Two explicit drift diagnostics required')
            for drift in drift_roots:
                receipt=read(drift/'RECEIPT.json');job=receipt['job']
                if receipt['stage']!='audit-request-drift' or command('sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode').splitlines()!=[job+'|COMPLETED|0:0']:
                    raise RuntimeError('Drift diagnostics must finish normally')
            from .frozen_perception import select_sources
            approved_sources,collection_root,approved_inheritance=select_sources(drift_roots)
            first,_=sources(collection_root,0);second,_=sources(collection_root,1)
            if not {r['source_id'] for r in approved_sources}<={r['source_id'] for r in first+second}:
                raise RuntimeError('Repair includes an unapproved source')
        active=command('squeue','-h','-u',os.environ['USER'],'-A','p33100','-p','gengpu','-o','%j|%Z')
        if sum('crash_bench' in x or 'crashbench_safelibero' in x or x.startswith('cb_') for x in active.splitlines())>=2:
            raise RuntimeError('Two CrashBench GPU jobs already active')
    if stage=='audit-request-drift':
        if collection_root is None:raise RuntimeError('Explicit approved-cache collection required')
        old=read(collection_root/'RECEIPT.json');job=old['job']
        if old['stage']!='collect-approved-cache' or command('sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode').splitlines()!=[job+'|COMPLETED|0:0']:
            raise RuntimeError('Approved-cache collection must terminate normally')
    root = ASSETS/'verifier_dataset'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+stage+'_'+commit[:12])
    root.mkdir(parents=True, exist_ok=False); (root/'source').mkdir()
    with (root/'source.tar').open('wb') as stream:
        subprocess.run(['git', 'archive', commit], stdout=stream, check=True)
    subprocess.run(['tar', '-xf', str(root/'source.tar'), '-C', str(root/'source')], check=True)
    (root/'SOURCE_COMMIT').write_text(commit+'\n')
    (root/'source.sha256').write_text(sha(root/'source.tar')+'  '+str(root/'source.tar')+'\n')
    manifest = dict(stage=stage, run_root=str(root), code_commit=commit,
        upstream_commit='2457feed5968ae803926e178c8ce8243b9ecdcf9', seed=1014,
        protocol_sha256=sha(repo/'verifier_dataset/protocol.json'),
        authorization='2026-10-14 user: 1000 evidence-backed rows on Quest; train only after audit; API cap $20',
        api_calls=0, api_limit_usd='20.00', account='p33100', partition='short',canonicalize=canonicalize,smoke_limit=limit)
    manifest['contact_contract']=read(repo/'verifier_dataset/protocol.json').get('contact_contract')
    if stage in ('fixture-smoke','fixture-render','fixture-grounding'):
        manifest['contact_contract']='FD1000-fixture-v1 (development-only inherited strict contact and swept-root guard)'
    write(root/'CAMPAIGN.json', manifest)
    if stage in ('fixture-render','fixture-grounding'):
        manifest.update(partition='gengpu',api_mode='disabled',max_paid_attempts=0,max_logical_requests=0)
        write(root/'CAMPAIGN.json',manifest)
        for name in ('FIXTURE_PROPOSAL.json','CANONICAL_CONTEXT.json','CANONICAL_PHYSICS.npz'):
            shutil.copyfile(collection_root/name,root/name)
        write(root/'CPU_FIXTURE_COMPLETE.json',fixture_proof)
        write(root/'INHERITED.json',dict(fixture_root=str(collection_root),job=old_job,
            completion_sha256=sha(collection_root/'FIXTURE_COMPLETE.json'),
            purpose='Actual initial simulator images and CPU/GPU consistency only; no policy inference, API or training'))
        if stage=='fixture-grounding':
            from .fixture_visuals import STYLE
            write(root/'VISUAL_GROUNDING.json',dict(style=STYLE,labels_created=False,
                purpose='Check target visibility and goal grounding while retaining real collision geometry'))
    if stage=='audit-candidates':write(root/'AUDIT_SOURCES.json',[str(x) for x in candidate_exports])
    if stage=='design-fixture':
        write(root/'CASES.json',[chosen_design])
        write(root/'INHERITED.json',dict(replay_root=str(replay_smoke),replay_job=proof_job,
            completion_sha256=sha(replay_smoke/'REPLAY_COMPLETE.json'),
            purpose='One read-only enclosure proposal, no labels or modified-scene execution'))
    if stage=='fixture-smoke':
        write(root/'FIXTURE_PROPOSAL.json',proposal)
        canonical=fixture_design/'case_000/checkpoint_000'
        shutil.copyfile(canonical/'context.json',root/'CANONICAL_CONTEXT.json')
        shutil.copyfile(canonical/'physics.npz',root/'CANONICAL_PHYSICS.npz')
        write(root/'INHERITED.json',dict(design_root=str(fixture_design),design_job=design_job,
            proposal_sha256=sha(fixture_design/'case_000/FIXTURE_PROPOSAL.json'),
            canonical_context_sha256=sha(root/'CANONICAL_CONTEXT.json'),canonical_physics_sha256=sha(root/'CANONICAL_PHYSICS.npz'),
            purpose='Three mechanism/debug cases only; no batch labels or training; explicit digital swept-root contract'))
    if stage == 'replay-smoke':
        write(root/'CASES.json',read(inventory/'inventory/REPLAY_SMOKE.json')[:limit])
        write(root/'INHERITED.json',dict(inventory=str(inventory),hashes=hashes))
    if stage=='replay-collection':
        write(root/'CASES.json',collection_rows)
        write(root/'INHERITED.json',collection_inheritance)
    if stage in ('replay-spatial-batch','replay-object-sources','replay-spatial-sources'):
        family='safelibero_object' if stage=='replay-object-sources' else 'safelibero_spatial'
        rows=[r for r in read(inventory/'inventory/CANDIDATES.json') if r['family']==family
              and r['proxy_safe_success'] and r['has_action_tape'] and r['has_settled_state']]
        if stage in ('replay-object-sources','replay-spatial-sources'):
            from .inventory import smoke_selection
            rows=smoke_selection(rows,limit=60)
        if not 0<len(rows)<=200:raise RuntimeError('Unexpected bounded source replay pool')
        rows.sort(key=lambda r:(r['source_id'],r['repeat'],r['method']))
        for index in range(8):
            shard=root/'shards'/str(index);shard.mkdir(parents=True)
            write(shard/'CASES.json',rows[index::8]);write(shard/'CAMPAIGN.json',manifest)
        write(root/'CASES.json',rows)
        write(root/'INHERITED.json',dict(inventory=str(inventory),hashes=hashes,replay_smoke=str(replay_smoke),
            smoke_sha256=sha(replay_smoke/'REPLAY_COMPLETE.json'),smoke_job=smoke_job))
    if stage in ('collect-smoke','collect-cached-smoke','collect-cached-batch','collect-approved-cache','collect-frozen-perception'):
        if stage in ('collect-approved-cache','collect-frozen-perception'):
            chosen=approved_sources
            write(root/'COLLECTION_PLAN.json',approved_inheritance)
            from .approved_collection import stage_cache
            stage_cache(root,collection_root,chosen)
            if stage=='collect-frozen-perception':manifest['perception_variant']=approved_inheritance['mode']
        elif stage=='collect-cached-batch':
            from .cached_pool import select_pool
            pool=select_pool(read(source_plan/'source_plan/OFFICIAL_OUTCOMES.json'))
            chosen=pool[batch_shard::2]
            write(root/'COLLECTION_PLAN.json',dict(pool=pool,shard=batch_shard,shards=2,
                smoke_root=str(collection_root),smoke_hashes=prior['hashes'],
                source_plan_hashes=read(source_plan/'source_plan/HASHES.json'),
                selection_conditioned_on_historical_proxy_success=True,
                feasibility_labels_created=False))
        else:
            entries=read(source_plan/'source_plan/GOAL20_CANDIDATES.json')
            chosen=[next(r for r in entries if r['scene']['task']==task) for task in range(4)]
            if len({r['source_id'] for r in chosen})!=4:raise RuntimeError('Four unique Goal source states required')
        write(root/'COLLECTION_SOURCES.json',chosen)
        sys.path.insert(0,str(repo/'analysis/uncertainty'))
        from common import config
        cfg=config();cfg['save_diagnostic_samples']=False
        write(root/'plan.json',dict(configuration=cfg))
        manifest.update(partition='gengpu',budget_root=str(ASSETS/'verifier_dataset/budget_fd1000_20261014'),
            source_plan=str(source_plan),max_logical_requests=4 if stage=='collect-smoke' else 0,
            max_paid_attempts=8 if stage=='collect-smoke' else 0,
            api_mode='paid_guarded' if stage=='collect-smoke' else 'cache_only')
        write(root/'CAMPAIGN.json',manifest)
    env = dict(os.environ, CB_FD_ROOT=str(root), CB_FD_STAGE=stage)
    if stage=='audit-request-drift':
        env['CB_FD_COLLECTION']=str(collection_root)
        write(root/'INHERITED.json',dict(collection_root=str(collection_root),receipt_sha256=sha(collection_root/'RECEIPT.json')))
    if stage=='export-candidates':
        env['CB_FD_REPLAY']=str(replay_smoke)
        write(root/'EXPORT_SOURCE.json',dict(root=str(replay_smoke),receipt_sha256=sha(replay_smoke/'RECEIPT.json')))
    script='collect.sbatch' if stage in ('collect-smoke','collect-cached-smoke','collect-cached-batch','collect-approved-cache','collect-frozen-perception') else ('cpu.sbatch' if stage in ('inventory','source-plan') else 'replay.sbatch')
    if stage=='export-candidates':script='export.sbatch'
    if stage=='audit-candidates':script='audit.sbatch'
    if stage in ('fixture-render','fixture-grounding'):script='fixture_render.sbatch'
    if stage=='audit-request-drift':script='request_drift.sbatch'
    submit=['sbatch','--parsable','--output='+str(root/'slurm_%j.log')]
    if stage in ('replay-spatial-batch','replay-object-sources','replay-spatial-sources'):
        submit=['sbatch','--parsable','--array=0-7%4','--output='+str(root/'slurm_%A_%a.log')]
    if stage=='collect-smoke':submit.append('--hold')
    if stage in ('collect-cached-batch','collect-approved-cache','collect-frozen-perception'):submit.append('--time=01:00:00')
    job = subprocess.check_output(submit+[str(root/'source/verifier_dataset'/script)],env=env,text=True).strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError('Invalid Slurm receipt')
    receipt = dict(manifest, job=job)
    write(root/'RECEIPT.json', receipt)
    local = repo/'results/verifier_dataset'/root.name; local.mkdir(parents=True, exist_ok=False)
    write(local/'RECEIPT.json', receipt)
    print(receipt, flush=True)
    if stage in ('collect-cached-smoke','collect-cached-batch','collect-approved-cache','collect-frozen-perception'):
        write(root/'API_FINAL.json',dict(api_mode='cache_only',paid_calls=0,committed_usd='0.00',worker_started=False))
    if stage=='collect-smoke':
        try:
            worker=shlex.join(['/projects/p33100/siosio/envs/openpi/bin/python','-u','-m','verifier_dataset.api_worker',str(root)])
            worker+=' > '+shlex.quote(str(root/'api_worker.log'))+' 2>&1'
            exists=subprocess.run(['tmux','has-session','-t','=crashbench'],capture_output=True).returncode==0
            tmux=['tmux','new-window','-d','-t','crashbench','-n','fd1000_api'] if exists else ['tmux','new-session','-d','-s','crashbench','-n','fd1000_api']
            subprocess.run(tmux+['-c',str(root/'source'),worker],check=True)
            deadline=time.monotonic()+55
            while not (root/'WORKER_READY.json').exists():
                if (root/'STOP.json').exists() or time.monotonic()>deadline:raise RuntimeError('API worker readiness failed')
                time.sleep(1)
            subprocess.run(['scontrol','release',job],check=True)
            write(root/'RELEASED.json',dict(job=job,pricing_and_budget_ready=True))
        except BaseException as error:
            write(root/'STOP.json',dict(stage='launch',error=type(error).__name__,reason=str(error)))
            subprocess.run(['scancel',job],check=False)
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['inventory','source-plan','replay-smoke','replay-spatial-batch','replay-object-sources','replay-spatial-sources','collect-smoke','collect-cached-smoke','collect-cached-batch','collect-approved-cache','collect-frozen-perception','replay-collection','export-candidates','audit-candidates','design-fixture','fixture-smoke','fixture-render','fixture-grounding','audit-request-drift'])
    p.add_argument('--inventory',type=Path);p.add_argument('--canonicalize',action='store_true')
    p.add_argument('--limit',type=int,default=12)
    p.add_argument('--source-plan',type=Path)
    p.add_argument('--replay-smoke',type=Path)
    p.add_argument('--collection-root',type=Path)
    p.add_argument('--batch-shard',type=int,choices=(0,1))
    p.add_argument('--candidate-export',type=Path,action='append')
    p.add_argument('--fixture-design',type=Path)
    p.add_argument('--drift-root',type=Path,action='append')
    a=p.parse_args();main(a.stage,a.inventory,a.canonicalize,a.limit,a.source_plan,a.replay_smoke,a.collection_root,a.batch_shard,a.candidate_export,a.fixture_design,a.drift_root)
