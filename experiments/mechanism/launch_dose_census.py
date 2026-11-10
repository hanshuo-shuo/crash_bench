"""Published launcher for the dose batch and the task-pair census (docs/generative_verifier/HIL_MECHANISM_DOSE.md,
docs/generative_verifier/HIL_MECHANISM_CENSUS.md). Run from the repo root on the login node with the openpi environment:
  /projects/p33100/siosio/envs/openpi/bin/python experiments/mechanism/launch_dose_census.py
It prepares five run roots from the same published commit (census prefixes, census smoke, census formal, dose smoke, dose
formal), writes the plans it can write now, and submits one chain that never holds more than two GPUs:
  census prefixes (2 GPUs) -> prefix summary -> census plans -> census smoke (2 GPUs) -> census gate
  dose smoke (2 GPUs, after the census smoke ends) -> dose gate -> dose formal (4 shards, 2 at a time) -> dose summary
  census formal (6 shards, 2 at a time; after the census gate passes and the dose formal ends) -> census summary
A failed gate cancels its own formal batch through the dependency and leaves the other experiment running. Both protocols'
readings must be frozen before launch. pi0.5 alone, no training, no API.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch import BRANCH_ROOT  # noqa: E402  (the mechanism launch.py, before anything puts experiments/feasibility first)
from verifier_dataset.inventory import read, write  # noqa: E402
from verifier_dataset.v5_launch import command, prompt_tokens, submit  # noqa: E402
from verifier_dataset.hil_suite_launch import checkpoint_path, finished_branch, gpu_room, prepare  # noqa: E402
from verifier_dataset import hil_suite as S  # noqa: E402
import census as K  # noqa: E402
import dose as D  # noqa: E402

PROTOCOLS = dict(dose='docs/generative_verifier/HIL_MECHANISM_DOSE.md', census='docs/generative_verifier/HIL_MECHANISM_CENSUS.md')
FROZEN = '第 5 节状态：已冻结'
AUTHORIZATION = ('2026-11-09: after Claude listed the results still missing for a minimal paper story (1: the reverse dose - weaken '
                 'the layer 12-14 image route in states that switch natively; 2: a census of LIBERO-Goal task pairs with native and the '
                 'fixed amp2 repair), the user replied "1，2 都开始干吧 然后给我结果就可以" (start both, then give me the results). '
                 'pi0.5 alone, reruns of HIL suite stage 1 switch branches (dose) and new prefixes and switch branches on unchanged '
                 'standard LIBERO-Goal (census), no training, no verifier, no paid API, SafeLIBERO untouched, at most two GPUs.')
POLICY = 'pi05_libero'
DOSE_SHARDS = dict(smoke=2, formal=4)
HOURS = dict(prefix='02', census_smoke='02', census_formal='10', dose_smoke='03', dose_formal='08')
STATE_CHECK = dict(smoke=1000, formal=400)


def env(root, **extra):
    return dict(os.environ, CB_FD_ROOT=str(root), CB_POLICY_CONFIG=POLICY, CB_POLICY_CHECKPOINT=checkpoint_path(POLICY),
                **{k: str(v) for k, v in extra.items()})


def after(kind, *jobs):
    return ['--dependency=' + ','.join('%s:%s' % (kind, j) for j in jobs)] + (['--kill-on-invalid-dep=yes'] if kind == 'afterok' else [])


def roots(repo, names):
    finished_branch(BRANCH_ROOT, Path(read(BRANCH_ROOT/'PLAN.json')['prefix_root']))
    out, manifests = {}, {}
    for key, protocol in (('census-prefix', 'census'), ('census-smoke', 'census'), ('census-formal', 'census'),
                          ('dose-smoke', 'dose'), ('dose-formal', 'dose')):
        root, manifest = prepare('mech-' + key, POLICY, repo, protocol=PROTOCOLS[protocol], authorization=AUTHORIZATION, family='mechanism')
        manifest.update(stage='mechanism-' + key, partition='gengpu', stage1_root=str(BRANCH_ROOT))
        out[key], manifests[key] = root, manifest
    prefixes = K.prefix_plan(names)
    write(out['census-prefix']/'NAMES.json', names)
    write(out['census-prefix']/'PLAN.json', dict(mode='formal', experiment='mechanism-census-prefix', shards=2, prefixes=prefixes))
    manifests['census-prefix'].update(prefixes=len(prefixes), shards=2, inits=list(K.INITS), tasks=list(K.TASKS),
                                      seed='per prefix: hil_suite.policy_seed (salt %s)' % S.SEED_SALT)
    for mode in ('smoke', 'formal'):
        manifests['census-' + mode].update(mode=mode, shards=K.SHARDS[mode], prefix_root=str(out['census-prefix']), arms=K.ARMS,
                                           phase_arms={str(k): v for k, v in K.PHASE_ARMS.items()}, layers=K.LAYERS,
                                           plan='written by census_plan.sbatch after the prefix summary',
                                           seed='per run: the prefix seed, shared by both arms of a state')
        body = D.plan(BRANCH_ROOT, mode, DOSE_SHARDS[mode]); write(out['dose-' + mode]/'PLAN.json', body)
        manifests['dose-' + mode].update(mode=mode, runs=len(body['branches']), shards=body['shards'], layers=body['layers'], arms=body['arms'],
                                         cell_arms=body['cell_arms'], cells=body['cells'], inits=body['inits'],
                                         seed='per run: the stage 1 SW unit seed (prefix seed), shared by every arm of the state; rnd0 adds '
                                              'closed_loop.direction_seed(SW id) folded with the request index; act0 draws its fixed push '
                                              'directions from repair.push_seed(SW id)')
    for key, root in out.items():
        write(root/'CAMPAIGN.json', manifests[key])
    return out, manifests


def gpu_job(root, name, array, hours, deps, script, **extra):
    return submit(env(root, **extra), '--job-name=' + name, '--array=' + array, '--time=%s:00:00' % hours, *deps,
                  '--output=' + str(root/'slurm_%A_%a.log'), str(root/'source'/script))


def cpu_job(root, name, deps, script, log='slurm_%j.log', **extra):
    return submit(env(root, **extra), '--job-name=' + name, *deps, '--output=' + str(root/log), str(root/'source'/script))


def main():
    repo = Path.cwd().resolve()
    active = command('squeue', '-h', '-u', os.environ['USER'], '-o', '%j')
    if any(n.startswith('cb_') for n in active.splitlines()):
        raise RuntimeError('Finish existing CrashBench jobs before publishing/syncing and launching')
    for path in PROTOCOLS.values():
        if FROZEN not in (repo/path).read_text():
            raise RuntimeError('Freeze the readings (protocol section 5) before launching: ' + path)
    names = S.task_names((repo/'third_party/LIBERO/libero/libero/benchmark/libero_suite_task_map.py').read_text())
    tokens = prompt_tokens([S.language(n) for n in names[K.SUITE]])
    if max(tokens.values()) >= S.MAX_TOKENS:
        raise RuntimeError('A prompt would be truncated')
    gpu_room(2)
    r, manifests = roots(repo, names)
    mech = 'experiments/mechanism/'
    jobs = {}
    try:
        p = r['census-prefix']
        jobs['census_prefix'] = gpu_job(p, 'cb_mech_census_prefix', '0-1', HOURS['prefix'], [], 'verifier_dataset/hil_suite.sbatch',
                                        CB_SUITE_SHARDS=2, CB_SUITE_PHASE='prefix')
        jobs['census_prefix_summary'] = cpu_job(p, 'cb_mech_census_prefix_summary', after('afterok', jobs['census_prefix']),
                                                'verifier_dataset/hil_suite_summary.sbatch', CB_SUITE_SUMMARY='summarize-prefix')
        jobs['census_plan'] = cpu_job(p, 'cb_mech_census_plan', after('afterok', jobs['census_prefix_summary']), mech + 'census_plan.sbatch',
                                      log='slurm_plan_%j.log', CB_CENSUS_PREFIX_ROOT=p, CB_CENSUS_SMOKE_ROOT=r['census-smoke'],
                                      CB_CENSUS_FORMAL_ROOT=r['census-formal'], CB_STAGE1_ROOT=BRANCH_ROOT)
        jobs['census_smoke'] = gpu_job(r['census-smoke'], 'cb_mech_census', '0-1', HOURS['census_smoke'], after('afterok', jobs['census_plan']),
                                       mech + 'closed_loop.sbatch', CB_MECH_SHARDS=K.SHARDS['smoke'])
        jobs['census_gate'] = cpu_job(r['census-smoke'], 'cb_mech_census_gate', after('afterok', jobs['census_smoke']), mech + 'census_summary.sbatch',
                                      CB_CENSUS_GATE=1, CB_STATE_CHECK=STATE_CHECK['smoke'])
        jobs['dose_smoke'] = gpu_job(r['dose-smoke'], 'cb_mech_dose', '0-1', HOURS['dose_smoke'], after('afterany', jobs['census_smoke']),
                                     mech + 'closed_loop.sbatch', CB_MECH_SHARDS=DOSE_SHARDS['smoke'])
        jobs['dose_gate'] = cpu_job(r['dose-smoke'], 'cb_mech_dose_gate', after('afterok', jobs['dose_smoke']), mech + 'dose_summary.sbatch',
                                    CB_DOSE_GATE=1)
        jobs['dose_formal'] = gpu_job(r['dose-formal'], 'cb_mech_dose', '0-3%2', HOURS['dose_formal'], after('afterok', jobs['dose_gate']),
                                      mech + 'closed_loop.sbatch', CB_MECH_SHARDS=DOSE_SHARDS['formal'])
        jobs['dose_summary'] = cpu_job(r['dose-formal'], 'cb_mech_dose_summary', after('afterok', jobs['dose_formal']), mech + 'dose_summary.sbatch',
                                       CB_DOSE_GATE=0)
        jobs['census_formal'] = gpu_job(r['census-formal'], 'cb_mech_census', '0-5%2', HOURS['census_formal'],
                                        ['--dependency=afterok:%s,afterany:%s' % (jobs['census_gate'], jobs['dose_formal']), '--kill-on-invalid-dep=yes'],
                                        mech + 'closed_loop.sbatch', CB_MECH_SHARDS=K.SHARDS['formal'])
        jobs['census_summary'] = cpu_job(r['census-formal'], 'cb_mech_census_summary', after('afterok', jobs['census_formal']),
                                         mech + 'census_summary.sbatch', CB_CENSUS_GATE=0, CB_STATE_CHECK=STATE_CHECK['formal'])
    except BaseException:
        if jobs:
            command('scancel', *jobs.values())
        raise
    pipeline = dict(roots={k: str(v) for k, v in r.items()}, jobs=jobs)
    for key, root in r.items():
        info = dict(manifests[key], pipeline=pipeline)
        write(root/'RECEIPT.json', info); write(root/'PIPELINE.json', pipeline)
        local = REPO/'results/mechanism'/root.name; local.mkdir(parents=True, exist_ok=False); write(local/'RECEIPT.json', info)
    print(pipeline, flush=True)
    return pipeline


if __name__ == '__main__':
    main()
