"""Launch V5 stages from the clean, published project checkout.

scan: step 1 CPU geometry scan. render SCAN_ROOT: step 2 render pilot on one A100.
trace, layout/screen TRACE_ROOT: C0 retrace and counterfactual layouts. s1-layout TRACE_ROOT:
eight layouts per object/II/task1 source. s1-render S1_LAYOUT_ROOT: blind-review renders.
s2 S1_LAYOUT_ROOT: render the object/II/task1 dataset on two A100 shards, then build it on CPU.
hil-smoke S2_ROOT / hil S2_ROOT: the HIL policy twins (4 smoke states, or the frozen 160-state batch on two
A100 shards), then the CPU summary.
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
PROTOCOLS = dict(scan='docs/generative_verifier/V5_STEP1_PROTOCOL.md', render='docs/generative_verifier/V5_STEP2_PROTOCOL.md',
                 trace='docs/generative_verifier/V5_COLLECTION_PROTOCOL.md', layout='docs/generative_verifier/V5_LAYOUT_PROTOCOL.md',
                 screen='docs/generative_verifier/V5_SCREEN_PROTOCOL.md',
                 **{'s1-layout': 'docs/generative_verifier/V5_S1_PROTOCOL.md', 's1-render': 'docs/generative_verifier/V5_S1_PROTOCOL.md',
                    's2': 'docs/generative_verifier/V5_S2_PROTOCOL.md', 'hil': 'docs/generative_verifier/V5_HIL_PROTOCOL.md',
                    'hil-smoke': 'docs/generative_verifier/V5_HIL_PROTOCOL.md'})
HIL_AUTHORIZATION = ('2026-10-27 user approved the HIL policy-twin design (V5_HIL_PROTOCOL.md) and asked to implement it, smoke '
                     '16 runs in a draft directory, freeze, then run the formal 640 and report ("写运行器和测试 -> 草稿目录 smoke 16 次 -> '
                     '填第 10 节、冻结 -> 正式 640 次 -> V5_HIL_REPORT.md"); pi0.5 alone, no verifier, no API, test sources untouched')
AUTHORIZATION = dict(
    scan='2026-10-24 user: V5 step 1 CPU geometry scan; no rendering, training, API or dataset rows',
    render='2026-10-24 user: V5 step 2 render pilot of about 20 checkpoints; no training, API or dataset rows',
    trace='2026-10-24 user: V5 design fixes and the V5 plan; CPU retrace of the 67 V4 sources, no rendering, training, API or dataset rows',
    layout='2026-10-24 user chose counterfactual layouts (option 3); CPU smoke of relocated anchors on 8 V4 sources, no rendering, training, API or dataset rows',
    screen='2026-10-24 user: restrict V5 to twin-capable compositions, no expansion; CPU screen of all 67 V4 sources, no rendering, training, API or dataset rows',
    **{'s1-layout': '2026-10-24 user: train on object/II/task1 only and add all three S1 items; CPU, eight layouts per object source, no rendering, training, API or dataset rows',
       's1-render': '2026-10-24 user: S1 render with a top view and a two-pass blind review on validation sources; no training, API or dataset rows',
       's2': '2026-10-25 user: continue with S2 (build the object/II/task1 dataset); two A100 render shards and a CPU build, no training or API',
       'hil-smoke': HIL_AUTHORIZATION + '; smoke: 4 states x 4 prompts in a draft run, results only in protocol section 10',
       'hil': HIL_AUTHORIZATION + '; formal: 160 states x 4 prompts after the protocol is frozen'})


def command(*args, env=None):
    return subprocess.check_output(args, text=True, env=env).strip()


def completed(job):
    # Slurm job numbers recur across years on Quest; keep only this project's records.
    lines = [x for x in command('sacct', '-X', '-n', '-P', '-j', job, '-u', os.environ['USER'],
                                '--format=JobIDRaw,State,ExitCode,Account').splitlines() if x.endswith('|p33100')]
    return bool(lines) and all(x.split('|')[1:3] == ['COMPLETED', '0:0'] for x in lines)


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


def trace_plan():
    specs = []
    for source, (case, task) in sorted(v4_cases().items()):
        geometry = read(case/'execution/GEOMETRY.json'); goal = geometry['goal_definition']['goal']
        suite, level, task_id = source.split('/')[:3]
        specs.append(dict(source=source, family=suite, task='/'.join((suite, level, task_id)), student_task=task,
                          v4_case=str(case), goal=goal, target=goal[1], receiver=goal[2],
                          material_geom=geometry['material']['geom_name']))
    return specs


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


def layout_render_plan(root):
    from .v5_layouts import frozen_split, select_review_units
    entries = read(root/'LAYOUTS.json')
    cases = {'%s#%d' % (e['spec']['source'], e['plan']['layout']): (root/('layout_%03d' % i), e['spec']) for i, e in enumerate(entries)}
    with gzip.open(root/'LAYOUT_ROWS.jsonl.gz', 'rt') as stream:
        rows = [json.loads(x) for x in stream]
    units = []
    for unit in sorted(select_review_units(rows, frozen_split()), key=lambda u: g.digest(u['source'], u['unit'])):
        case, spec = cases[unit['source']]
        if not read(case/'LAYOUT.json')['valid']:
            raise RuntimeError('Selected unit from an invalid layout')
        unit.update(review_id='u%02d' % len(units), v4_case=spec['v4_case'], layout_case=str(case), student_task=spec['student_task'])
        for rule in sorted(unit['rules'], key=lambda r: g.digest(unit['source'], unit['unit'], 'review', r['key'])):
            rule.update(review_rule='r%d' % (1 + sum('review_rule' in r for r in unit['rules'])), text=g.rule_text(rule['key']))
        units.append(unit)
    images = lambda u, names: ['units/%s/%s.png' % (u['review_id'], n) for n in names]
    sheet = [dict(review_id=u['review_id'], task=u['student_task'], pass1_images=images(u, ('overview_a', 'overview_b')),
                  pass2_images=images(u, ('overview_a', 'overview_b', 'top')),
                  rules=sorted((dict(id=r['review_rule'], text=r['text']) for r in u['rules']), key=lambda r: r['id']))
             for u in units]
    return units, sheet


def finished(root, complete):
    receipt = read(root/'RECEIPT.json')
    if not read(root/complete)['passed'] or not (completed(receipt['job']) and completed(receipt['summarize_job'])):
        raise RuntimeError('Prerequisite run did not finish normally: ' + str(root))
    for name, digest in read(root/'HASHES.json').items():
        if sha(root/name) != digest:
            raise RuntimeError('Prerequisite output changed: ' + name)
    return receipt


def prompt_tokens(texts):
    """Prompt length in pi0.5's own tokenizer (PaliGemma, BOS + text + newline, max_token_len 200)."""
    import sys
    os.environ.setdefault('OPENPI_DATA_HOME', str(ASSETS/'openpi_assets'))
    sys.path.insert(0, str(Path.cwd()/'third_party/vlsa-aegis/openpi/src'))
    from openpi.models.tokenizer import PaligemmaTokenizer
    tokenizer = PaligemmaTokenizer(200)
    return {t: int(tokenizer.tokenize(t)[1].sum()) for t in sorted(set(texts))}


def hil_plan(s2, mode):
    """Check the S2 and S1a runs, then pick the HIL states and record every prompt's token count."""
    from . import v5_hil as H
    receipt = read(s2/'RECEIPT.json')
    if receipt['stage'] != 'v5-s2' or not read(s2/'S2_COMPLETE.json')['passed'] or not (completed(receipt['job']) and completed(receipt['build_job'])):
        raise RuntimeError('A finished V5 S2 run is required')
    for name, digest in read(s2/'HASHES.json').items():
        if sha(s2/name) != digest:
            raise RuntimeError('S2 output changed: ' + name)
    campaign = read(s2/'CAMPAIGN.json'); layout_root = Path(campaign['layout_root'])
    if finished(layout_root, 'LAYOUT_COMPLETE.json')['stage'] != 'v5-s1-layout' or read(layout_root/'HASHES.json') != campaign['layout_hashes']:
        raise RuntimeError('S2 layout root changed')
    rows = []
    with (s2/'dataset.jsonl').open() as stream:
        for line in stream:
            if '"split": "test' in line:
                continue  # the test splits are skipped unparsed
            row = json.loads(line)
            if row['split'].startswith('test'):
                raise RuntimeError('Test row passed the filter')
            rows.append(H.flatten(row))
    units = {u['review_id']: u for u in read(s2/'UNITS.json') if u['split'] in H.SPLITS}
    specs = {e['spec']['source']: e['spec'] for e in read(layout_root/'LAYOUTS.json')}
    candidates = H.pool(rows); groups = H.strata(candidates); states = H.plan(candidates, mode)
    for s in states:
        u = units[s['image']]
        if u['source'] != s['image_source'] or u['unit'] != s['checkpoint'] or u['v4_case'] != specs[s['base_source']]['v4_case']:
            raise RuntimeError('S2 unit disagrees with its rows: ' + s['image'])
        s.update(layout_case=u.get('layout_case'), spec=specs[s['base_source']])
        if (s['layout'] is None) != (s['layout_case'] is None) or s['base_source'] in H.TEST_SOURCES:
            raise RuntimeError('Bad state: ' + s['image'])
    tokens = prompt_tokens(p for s in states for p in s['prompts'].values())
    for s in states:
        s['prompt_tokens'] = {c: tokens[s['prompts'][c]] for c in H.CONDITIONS}
    if max(tokens.values()) >= H.MAX_TOKENS:
        raise RuntimeError('A prompt reaches max_token_len and would be truncated')
    return dict(mode=mode, plan_salt=H.PLAN_SALT, seed_salt=H.SEED_SALT, candidates=len(candidates),
                supply={'%s/%s/bin%d' % k: len(v) for k, v in sorted(groups.items())},
                quotas={'%s/%s/bin%d' % k: n for k, n in sorted(H.quotas(groups).items())},
                prompt_tokens={c: dict(min=min(s['prompt_tokens'][c] for s in states), max=max(s['prompt_tokens'][c] for s in states))
                               for c in H.CONDITIONS},
                tokenizer_sha256=sha(ASSETS/'openpi_assets/big_vision/paligemma_tokenizer.model'),
                s2_root=str(s2), layout_root=str(layout_root), states=states)


def main(stage, scan=None):
    repo = Path.cwd().resolve()
    if stage in ('layout', 'screen', 's1-layout') and finished(scan, 'TRACE_COMPLETE.json')['stage'] != 'v5-trace':
        raise RuntimeError('Counterfactual layouts start from a V5 trace run')
    if stage == 's1-render' and finished(scan, 'LAYOUT_COMPLETE.json')['stage'] != 'v5-s1-layout':
        raise RuntimeError('S1 renders start from an S1 layout run')
    if stage == 's2' and finished(scan, 'LAYOUT_COMPLETE.json')['stage'] != 'v5-s1-layout':
        raise RuntimeError('S2 starts from the S1 layout run')
    if stage == 'hil' and '状态：已冻结' not in (repo/PROTOCOLS['hil']).read_text():
        raise RuntimeError('Freeze the HIL protocol before the formal batch')
    if stage in ('hil', 'hil-smoke'):
        plan_record = hil_plan(scan, 'formal' if stage == 'hil' else 'smoke'); shards = 2 if stage == 'hil' else 1
        active = command('squeue', '-h', '-u', os.environ['USER'], '-A', 'p33100', '-p', 'gengpu', '-o', '%j')
        if sum(x.startswith('cb_') for x in active.splitlines()) + shards > 2:
            raise RuntimeError('At most two CrashBench GPUs at a time')
    if stage in ('render', 's1-render', 's2'):
        if stage == 's2':
            from .v5_dataset import plan as dataset_plan
            units = dataset_plan(scan, Path(read(scan/'CAMPAIGN.json')['trace_root']))
        else:
            units, sheet = render_plan(scan) if stage == 'render' else layout_render_plan(scan)
        active = command('squeue', '-h', '-u', os.environ['USER'], '-A', 'p33100', '-p', 'gengpu', '-o', '%j')
        if sum(x.startswith('cb_') for x in active.splitlines()) >= 2:
            raise RuntimeError('Two CrashBench GPU jobs already active')
    root, manifest = prepare(stage, repo)
    env = dict(os.environ, CB_FD_ROOT=str(root))
    if stage == 'trace':
        from .v5_trace import SHARDS as TRACE_SHARDS
        cases = trace_plan(); manifest.update(partition='short', shards=TRACE_SHARDS, cases=len(cases))
        write(root/'CAMPAIGN.json', manifest); write(root/'CASES.json', cases)
        script = str(root/'source/verifier_dataset/v5_trace.sbatch')
        job = submit(env, '--array=0-%d' % (TRACE_SHARDS-1), '--output='+str(root/'slurm_%A_%a.log'), script)
        receipt = dict(manifest, job=job, summarize_job=submit(env, '--dependency=afterok:'+job, '--output='+str(root/'slurm_%j.log'), script))
    elif stage in ('layout', 'screen', 's1-layout'):
        from .v5_layouts import SHARDS as LAYOUT_SHARDS, plan
        entries = plan(scan, dict(screen='screen', layout='smoke').get(stage, 's1'))
        manifest.update(partition='short', shards=LAYOUT_SHARDS, layouts=len(entries), trace_root=str(scan))
        write(root/'CAMPAIGN.json', manifest); write(root/'LAYOUTS.json', entries)
        script = str(root/'source/verifier_dataset/v5_layouts.sbatch')
        job = submit(env, '--array=0-%d' % (LAYOUT_SHARDS-1), '--output='+str(root/'slurm_%A_%a.log'), script)
        receipt = dict(manifest, job=job, summarize_job=submit(env, '--dependency=afterok:'+job, '--output='+str(root/'slurm_%j.log'), script))
    elif stage == 's2':
        shards = 2
        manifest.update(partition='gengpu', shards=shards, units=len(units), layout_root=str(scan),
                        layout_hashes=read(scan/'HASHES.json'), trace_root=read(scan/'CAMPAIGN.json')['trace_root'])
        write(root/'CAMPAIGN.json', manifest); write(root/'UNITS.json', units)
        env['CB_RENDER_SHARDS'] = str(shards)
        job = submit(env, '--array=0-%d' % (shards - 1), '--time=02:00:00', '--output='+str(root/'slurm_%A_%a.log'),
                     str(root/'source/verifier_dataset/v5_render.sbatch'))
        receipt = dict(manifest, job=job, build_job=submit(env, '--dependency=afterok:'+job, '--output='+str(root/'slurm_%j.log'),
                                                           str(root/'source/verifier_dataset/v5_dataset.sbatch')))
    elif stage in ('hil', 'hil-smoke'):
        manifest.update(partition='gengpu', mode=plan_record['mode'], shards=shards, states=len(plan_record['states']),
                        runs=4 * len(plan_record['states']), seed='per state: v5_hil.policy_seed, shared by the four prompts',
                        policy='pi05_libero', policy_checkpoint=str(ASSETS/'openpi_assets/openpi-assets/checkpoints/pi05_libero'),
                        s2_root=str(scan), s2_hashes=read(scan/'HASHES.json'), layout_root=plan_record['layout_root'],
                        layout_hashes=read(Path(plan_record['layout_root'])/'HASHES.json'))
        write(root/'CAMPAIGN.json', manifest); write(root/'PLAN.json', plan_record)
        env['CB_HIL_SHARDS'] = str(shards)
        job = submit(env, '--array=0-%d' % (shards - 1), '--time=%s' % ('12:00:00' if stage == 'hil' else '02:00:00'),
                     '--output='+str(root/'slurm_%A_%a.log'), str(root/'source/verifier_dataset/v5_hil.sbatch'))
        receipt = dict(manifest, job=job, summarize_job=submit(env, '--dependency=afterok:'+job, '--output='+str(root/'slurm_%j.log'),
                                                               str(root/'source/verifier_dataset/v5_hil_summary.sbatch')))
    elif stage == 'scan':
        manifest.update(partition='short', shards=len(SHARDS)); write(root/'CAMPAIGN.json', manifest)
        script = str(root/'source/verifier_dataset/v5_scan.sbatch')
        job = submit(env, '--array=0-%d%%4' % (len(SHARDS)-1), '--output='+str(root/'slurm_%A_%a.log'), script)
        receipt = dict(manifest, job=job, summarize_job=submit(env, '--dependency=afterok:'+job, '--output='+str(root/'slurm_%j.log'), script))
    else:
        manifest.update(partition='gengpu', units=len(units), **{('scan_root' if stage == 'render' else 'layout_root'): str(scan),
                        ('scan_hashes' if stage == 'render' else 'layout_hashes'): read(scan/'HASHES.json')})
        write(root/'CAMPAIGN.json', manifest); write(root/'UNITS.json', units); write(root/'REVIEW_SHEET.json', sheet)
        receipt = dict(manifest, job=submit(env, '--output='+str(root/'slurm_%j.log'), str(root/'source/verifier_dataset/v5_render.sbatch')))
    write(root/'RECEIPT.json', receipt)
    local = repo/'results/verifier_dataset'/root.name; local.mkdir(parents=True, exist_ok=False)
    write(local/'RECEIPT.json', receipt)
    print(receipt, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['scan', 'render', 'trace', 'layout', 'screen', 's1-layout', 's1-render', 's2', 'hil-smoke', 'hil'])
    p.add_argument('scan_root', type=Path, nargs='?')
    a = p.parse_args()
    if (a.stage not in ('scan', 'trace')) != (a.scan_root is not None):
        raise SystemExit('render and layout need exactly one prior run root; scan and trace take none')
    main(a.stage, a.scan_root)
