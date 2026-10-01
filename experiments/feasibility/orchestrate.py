"""Small Slurm dependency controller; never runs simulation on a login node."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ASSETS = Path('/projects/p33100/siosio/crashbench_safelibero/feasibility')
SOURCE = Path(__file__).resolve().parents[2]


def next_stage(target):
    target = Path(target)
    if (target / 'STOP.json').exists():
        raise RuntimeError('Scientific root stopped; no automatic recovery')
    for name in ['INITIAL_COMPLETE.json', 'BRANCH_GATE.json', 'plan.json']:
        if not (target / name).is_file():
            raise RuntimeError('Required evidence missing: ' + name)
    gate = json.loads((target / 'BRANCH_GATE.json').read_text())
    plan = json.loads((target / 'plan.json').read_text())
    states = gate['eligible_states']
    allowed = {'spatial_03', 'spatial_09', 'spatial_15', 'object_00', 'object_02', 'object_05'}
    if len(set(states)) != len(states) or not set(states) <= allowed:
        raise RuntimeError('Invalid branch eligibility')
    for state in states:
        stage = 'branch_' + state
        if (target / (stage + '_COMPLETE.json')).is_file():
            continue
        if stage in plan['jobs']:
            raise RuntimeError('Prior branch attempt has no completion receipt: ' + stage)
        return stage
    if not (target / 'COMPLETE.json').is_file() or not (target / 'report/REPORT_COMPLETE.json').is_file():
        raise RuntimeError('All stages ended without complete exported evidence')
    return None


def terminal_success(job):
    if not str(job).isdigit():
        raise RuntimeError('Invalid predecessor job')
    for attempt in range(3):
        raw = subprocess.check_output([
            'sacct', '--noheader', '--parsable2', '--allocations', '--jobs', str(job),
            '--format=JobIDRaw,State,ExitCode'], text=True)
        entries = [line.split('|') for line in raw.splitlines() if line.strip()]
        match = [x for x in entries if x[0] == str(job)]
        if match:
            if match[0][1:3] != ['COMPLETED', '0:0']:
                raise RuntimeError('Predecessor failed: ' + '|'.join(match[0]))
            return
        if attempt < 2:
            time.sleep(5)
    raise RuntimeError('Predecessor accounting unavailable')


def main(target, controller, predecessor):
    target, controller = Path(target).resolve(), Path(controller).resolve()
    job = os.environ['SLURM_JOB_ID']
    if target.parent != ASSETS.resolve() or controller.parent != ASSETS.resolve():
        raise RuntimeError('Wrong project output root')
    if SOURCE != controller / 'source':
        raise RuntimeError('Controller source is not the frozen archive')
    receipts = controller / 'receipts'
    receipts.mkdir(exist_ok=True)
    receipt = {'controller_job': job, 'predecessor_job': str(predecessor), 'target': str(target),
               'controller_commit': (controller / 'SOURCE_COMMIT').read_text().strip(), 'unix': time.time()}
    try:
        terminal_success(predecessor)
        stage = next_stage(target)
        receipt['next_stage'] = stage
        if stage is None:
            receipt['status'] = 'all_scientific_stages_complete'
        else:
            launcher = target / 'source/experiments/feasibility/launch.py'
            result = subprocess.check_output([sys.executable, str(launcher), stage, str(target)], text=True)
            plan = json.loads((target / 'plan.json').read_text())
            gpu = plan['jobs'][stage]
            receipt['gpu_job'] = gpu
            env = os.environ.copy()
            env.pop('OPENROUTER_API_KEY', None)
            env.update(CB_FEASIBILITY_TARGET=str(target), CB_FEASIBILITY_CONTROLLER=str(controller),
                       CB_FEASIBILITY_PREDECESSOR=str(gpu))
            successor = subprocess.check_output([
                'sbatch', '--parsable', '--dependency=afterany:' + str(gpu),
                '--output=' + str(controller / 'controller_%j.log'),
                str(SOURCE / 'experiments/feasibility/orchestrate.sbatch')], env=env, text=True).strip().split(';')[0]
            if not successor.isdigit():
                raise RuntimeError('Invalid successor controller receipt')
            receipt.update(status='next_branch_submitted', successor_controller_job=successor)
        (receipts / (job + '.json')).write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(receipt, indent=2))
    except BaseException as error:
        receipt.update(status='stopped', reason=type(error).__name__ + ': ' + str(error))
        (controller / ('CONTROLLER_STOP_' + job + '.json')).write_text(json.dumps(receipt, indent=2) + '\n')
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('target', type=Path)
    p.add_argument('controller', type=Path)
    p.add_argument('predecessor')
    args = p.parse_args()
    main(args.target, args.controller, args.predecessor)
