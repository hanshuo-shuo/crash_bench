"""Launch V5 stages from the clean, published project checkout.

scan: step 1 CPU geometry scan. render SCAN_ROOT: step 2 render pilot on one A100.
"""
import argparse
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import subprocess
from .inventory import ASSETS, read, sha, write
from .v5_scan import SHARDS, V4_ROOTS, identity
from . import v5_geometry as g

UPSTREAM = '2457feed5968ae803926e178c8ce8243b9ecdcf9'
PROTOCOLS = dict(scan='docs/generative_verifier/V5_STEP1_PROTOCOL.md', render='docs/generative_verifier/V5_STEP2_PROTOCOL.md')
AUTHORIZATION = dict(
    scan='2026-10-24 user: V5 step 1 CPU geometry scan; no rendering, training, API or dataset rows',
    render='2026-10-24 user: V5 step 2 render pilot of about 20 checkpoints; no training, API or dataset rows')


def command(*args, env=None):
    return subprocess.check_output(args, text=True, env=env).strip()


def completed(job):
    lines = command('sacct', '-X', '-n', '-P', '-j', job, '--format=JobIDRaw,State,ExitCode').splitlines()
    return bool(lines) and all(x.split('|')[1:] == ['COMPLETED', '0:0'] for x in lines)


def submit(env, *args):
    out = command('sbatch', '--parsable', *args, env=env).split(';')[0]
    if not out.isdigit():
        raise RuntimeError('Invalid Slurm receipt')
    return out


def prepare(stage, repo):
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
    if command('git', '-C', str(repo/'third_party/vlsa-aegis'), 'rev-parse', 'HEAD') != UPSTREAM:
        raise RuntimeError('Upstream checkout changed')
    for base in V4_ROOTS:
        if read(base/'CONSTRAINT_COMPLETE.json').get('passed') is not True:
            raise RuntimeError('V4 input root did not complete: ' + str(base))
    root = ASSETS/'verifier_dataset'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_v5-'+stage+'_'+commit[:12])
    root.mkdir(parents=True, exist_ok=False); (root/'source').mkdir()
    with (root/'source.tar').open('wb') as stream:
        subprocess.run(['git', 'archive', commit], stdout=stream, check=True)
    subprocess.run(['tar', '-xf', str(root/'source.tar'), '-C', str(root/'source')], check=True)
    (root/'SOURCE_COMMIT').write_text(commit+'\n')
    (root/'source.sha256').write_text(sha(root/'source.tar')+'  '+str(root/'source.tar')+'\n')
    manifest = dict(stage='v5-'+stage, run_root=str(root), code_commit=commit, upstream_commit=UPSTREAM, seed=1014,
        protocol=PROTOCOLS[stage], protocol_sha256=sha(repo/PROTOCOLS[stage]), authorization=AUTHORIZATION[stage],
        api_calls=0, account='p33100', v4_inputs={str(b): sha(b/'CONSTRAINT_COMPLETE.json') for b in V4_ROOTS})
    return root, manifest


def v4_cases():
    out = {}
    for base in V4_ROOTS:
        for case in sorted(base.glob('case_*')):
            row = read(case/'INPUT.json')
            scene = read(Path(row['evidence']['witness_root'])/'SUMMARY.json')['source']['scene']
            out[identity(scene['suite'], scene['level'], scene['task'], scene['episode'])] = (case, row['student']['task'])
    return out


def render_plan(scan):
    receipt = read(scan/'RECEIPT.json')
    if receipt['stage'] != 'v5-scan' or not read(scan/'V5_SCAN_COMPLETE.json')['passed']:
        raise RuntimeError('A passed V5 scan is required')
    if not (completed(receipt['job']) and completed(receipt['summarize_job'])):
        raise RuntimeError('Scan jobs must finish normally')
    for name, digest in read(scan/'HASHES.json').items():
        if sha(scan/name) != digest:
            raise RuntimeError('Scan output changed: ' + name)
    with gzip.open(scan/'CHECKPOINT_ROWS.jsonl.gz', 'rt') as stream:
        rows = [json.loads(x) for x in stream]
    cases = v4_cases(); units = []
    for unit in sorted(g.select_units(rows), key=lambda u: g.digest(u['source'], u['unit'])):
        case, task = cases[unit['source']]
        unit.update(review_id='u%02d' % len(units), v4_case=str(case), student_task=task)
        for rule in sorted(unit['rules'], key=lambda r: g.digest(unit['source'], unit['unit'], 'review', r['key'])):
            rule.update(review_rule='r%d' % (1 + sum('review_rule' in r for r in unit['rules'])), text=g.rule_text(rule['key']))
        units.append(unit)
    sheet = [dict(review_id=u['review_id'], task=u['student_task'],
                  images=['units/%s/overview_a.png' % u['review_id'], 'units/%s/overview_b.png' % u['review_id']],
                  rules=sorted((dict(id=r['review_rule'], text=r['text']) for r in u['rules']), key=lambda r: r['id']))
             for u in units]
    return units, sheet


def main(stage, scan=None):
    repo = Path.cwd().resolve()
    if stage == 'render':
        units, sheet = render_plan(scan)
        active = command('squeue', '-h', '-u', os.environ['USER'], '-A', 'p33100', '-p', 'gengpu', '-o', '%j')
        if sum(x.startswith('cb_') for x in active.splitlines()) >= 2:
            raise RuntimeError('Two CrashBench GPU jobs already active')
    root, manifest = prepare(stage, repo)
    env = dict(os.environ, CB_FD_ROOT=str(root))
    if stage == 'scan':
        manifest.update(partition='short', shards=len(SHARDS)); write(root/'CAMPAIGN.json', manifest)
        script = str(root/'source/verifier_dataset/v5_scan.sbatch')
        job = submit(env, '--array=0-%d%%4' % (len(SHARDS)-1), '--output='+str(root/'slurm_%A_%a.log'), script)
        receipt = dict(manifest, job=job, summarize_job=submit(env, '--dependency=afterok:'+job, '--output='+str(root/'slurm_%j.log'), script))
    else:
        manifest.update(partition='gengpu', scan_root=str(scan), scan_hashes=read(scan/'HASHES.json'), units=len(units))
        write(root/'CAMPAIGN.json', manifest); write(root/'UNITS.json', units); write(root/'REVIEW_SHEET.json', sheet)
        receipt = dict(manifest, job=submit(env, '--output='+str(root/'slurm_%j.log'), str(root/'source/verifier_dataset/v5_render.sbatch')))
    write(root/'RECEIPT.json', receipt)
    local = repo/'results/verifier_dataset'/root.name; local.mkdir(parents=True, exist_ok=False)
    write(local/'RECEIPT.json', receipt)
    print(receipt, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['scan', 'render']); p.add_argument('scan_root', type=Path, nargs='?')
    a = p.parse_args()
    if (a.stage == 'render') != (a.scan_root is not None):
        raise SystemExit('render needs exactly one scan root; scan takes none')
    main(a.stage, a.scan_root)
