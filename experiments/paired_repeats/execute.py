"""Compute-side checks/full execution; no network access or key here."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from protocol import SCENARIOS, schedule, scenario_name, summarize, write_csv
from runtime import Runner
from api_budget import atomic_json

def checks(root, port):
    directory = root/'checks'; directory.mkdir()
    runner = Runner(directory, port)
    report = {'status': 'running', 'idle': {'status': 'running'}, 'consistency': {'status': 'pending'}, 'smoke': {'status': 'pending'}}
    rows = []
    def save():
        atomic_json(root/'self_checks.json', report)
        write_csv(directory/'raw.csv', rows)
    save()
    # Finish the 60-state count before deciding the idle gate; no later check on failure.
    idle_rows = []
    for scenario in SCENARIOS:
        for episode in range(20):
            row = runner.run(scenario, episode, 0, 'nominal', 'idle', idle=True)
            idle_rows.append(row); rows.append(row)
            report['idle']['completed'] = len(idle_rows); save()
    drift = [dict(scenario=r['scenario'], episode=r['episode'], obstacle=r['active_obstacle'], max_l1_m=r['max_obstacle_l1_m'])
             for r in idle_rows if r['max_obstacle_l1_m'] > .001]
    if any(r['end_step'] != 300 or r['exited'] for r in idle_rows):
        raise RuntimeError('Idle run did not execute 300 steps')
    report['idle'] = {'status': 'failed' if drift else 'passed', 'states': 60, 'steps_per_state': 300,
                      'threshold_m': .001, 'drift_count': len(drift), 'drifting_states': drift}
    save()
    if drift:
        raise RuntimeError('Idle self-check failed: %d/60 states drift > 1 mm L1' % len(drift))
    report['consistency']['status'] = 'running'; save()
    mismatches = []
    for scenario in SCENARIOS:
        for episode in range(20):
            base = next(r for r in idle_rows if r['scenario'] == scenario_name(scenario) and r['episode'] == episode)
            hashes = [base['qpos_sha256']]
            for repeat in [1,2]:
                row = runner.run(scenario, episode, repeat, 'nominal', 'consistency', settle_only=True)
                rows.append(row); hashes.append(row['qpos_sha256']); save()
            if len(set(hashes)) != 1:
                mismatches.append({'scenario': scenario_name(scenario), 'episode': episode, 'hashes': hashes})
    report['consistency'] = {'status': 'failed' if mismatches else 'passed', 'states': 60, 'fresh_environments_per_state': 3,
                              'environment_seed': 7, 'comparison': 'exact float64 qpos bytes after 20 dummy steps', 'mismatches': mismatches}
    save()
    if mismatches:
        raise RuntimeError('Settled qpos differs despite rebuilding and seeding every environment')
    report['smoke']['status'] = 'running'; save()
    smoke = []
    for episode in [0,1]:
        for method in ['nominal','aegis']:
            row = runner.run(SCENARIOS[0], episode, 0, method, 'smoke')
            smoke.append(row); rows.append(row); save()
    for episode in [0,1]:
        pair = [r for r in smoke if r['episode'] == episode]
        for field in ['seed','policy_reset_key','qpos_sha256','first_action_chunk_sha256']:
            if not pair[0][field] or pair[0][field] != pair[1][field]:
                raise RuntimeError('Smoke pair mismatch: '+field)
        if any(r['exited'] or not r['video_frames'] for r in pair):
            raise RuntimeError('Smoke run exited early or video is empty')
    report['smoke'] = {'status': 'awaiting_visual_and_delta_review', 'runs': 4,
                       'durations_seconds': {r['run_id']:r['elapsed_seconds'] for r in smoke},
                       'rows': [str(directory/'runs'/r['run_id']/'row.json') for r in smoke],
                       'qpos_and_first_policy_chunk_equal': True, 'all_videos_decoded': True}
    report['status'] = 'awaiting_review'; save()
    atomic_json(root/'CHECKS_AWAIT_REVIEW.json', report)

def review_gate(root):
    report = json.loads((root/'self_checks.json').read_text())
    review = json.loads((root/'review.json').read_text())
    if report['idle']['status'] != 'passed' or report['consistency']['status'] != 'passed' or report['smoke']['status'] != 'awaiting_visual_and_delta_review':
        raise RuntimeError('Self-checks incomplete or failed')
    if review.get('self_checks_sha256') != hashlib.sha256((root/'self_checks.json').read_bytes()).hexdigest():
        raise RuntimeError('Review does not cover these checks')
    threshold = review.get('action_delta_linf_threshold')
    if not isinstance(threshold, (int,float)) or not 0 < threshold <= .001:
        raise RuntimeError('Missing reviewed small action threshold')
    if review.get('videos_normal') is not True or not review.get('distribution_rationale') or not review.get('reviewed_files'):
        raise RuntimeError('Missing visual/distribution evidence')
    for path, digest in review['reviewed_files'].items():
        file = (root/path).resolve()
        if root.resolve() not in file.parents or hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Reviewed evidence changed or lies outside experiment')
    return threshold

def full(root, port):
    threshold = review_gate(root)
    plan = json.loads((root/'plan.json').read_text())
    # Scheduling also checks Slurm; require authoritative completion on compute.
    for parent in plan['baseline_roots']:
        p = Path(parent)
        if not ((p/'COMPLETE.json').exists() or (p/'STOP.json').exists()):
            raise RuntimeError('Original reproduction still active')
    latest = Path(plan['latest_baseline'])
    if not (latest/'COMPLETE.json').exists():
        raise RuntimeError('Latest baseline did not finish successfully')
    directory = root/'full'; directory.mkdir()
    runner = Runner(directory, port)
    rows = []
    for scenario,episode,repeat,method,seed in schedule():
        if (root/'STOP.json').exists():
            raise RuntimeError('Paired batch stopped')
        row = runner.run(scenario,episode,repeat,method,'full',threshold=threshold)
        rows.append(row); write_csv(directory/'raw.csv', rows)
        atomic_json(root/'progress.json', {'completed_runs': len(rows), 'expected_runs': 600, 'updated_unix': time.time()})
    summarize(rows, root/'summary')
    atomic_json(root/'COMPLETE.json', {'runs': 600, 'summary': str(root/'summary'), 'finished_unix': time.time()})

def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['checks','full']); p.add_argument('root', type=Path); p.add_argument('--port', type=int, required=True)
    a = p.parse_args(); root = a.root.resolve()
    try:
        if (root/'STOP.json').exists():
            raise RuntimeError('Existing STOP cannot be removed to restart; use a new output root')
        (checks if a.stage == 'checks' else full)(root, a.port)
    except BaseException as error:
        atomic_json(root/'STOP.json', {'stage': a.stage, 'reason': type(error).__name__+': '+str(error), 'unix': time.time()})
        raise

if __name__ == '__main__':
    main()
