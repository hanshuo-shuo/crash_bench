from collections import deque
from types import SimpleNamespace
import copy
import gzip
import pickle
import numpy as np
import pytest
from scripts.expansion.selection_retest import (TimedQueue, select_panel, freeze, order,
                                                validate_records, step_record, rollout)
from scripts.expansion.analyze_selection_retest import evaluate, render


def test_panel_training_only_balanced_without_outcomes():
    assignments = [{'role': role, 'task_id': f'libero_spatial:{task}', 'physical_source_id': f'{task}:{i:02d}'}
                   for role in ('train', 'confirmatory_id_test') for task in (0, 2) for i in range(12)]
    panel = select_panel({'assignments': assignments[::-1]})
    assert len(panel) == 24 and all(r['role'] == 'train' for r in panel)
    assert len({r['physical_source_id'] for r in panel}) == 12
    assert [r['delay_steps'] for r in panel[:12:2]] == [1, 3, 5, 3, 5, 1]
    assert all(r['anchor_steps'] % 5 == 0 for r in panel)
    for i in range(24):
        assert [order(i, r)[0] for r in range(4)].count(0) == 2
        assert [order(i, r)[0] for r in range(4, 8)].count(0) == 2


def test_timing_snapshot_matches_mechanism_and_refresh():
    queue = TimedQueue(3)
    for step in range(5):
        delivered, timing = queue.deliver({'state': np.array([step])}, 'stale', step, step*.05)
    assert delivered['state'][0] == 1 and timing['age_steps'] == 3
    snap = queue.snapshot()
    queue.deliver({'state': np.array([5])}, 'stale', 5, .25, True)
    queue.restore(snap)
    delivered, timing = queue.deliver({'state': np.array([5])}, 'stale', 5, .25)
    assert delivered['state'][0] == 2 and timing['age_steps'] == 3
    delivered, timing = queue.deliver({'state': np.array([6])}, 'matched_buffer_control', 6, .30)
    assert delivered['state'][0] == 6 and timing['age_steps'] == 0


def synthetic(phase):
    anchors = [dict(panel_id=f'b{i}', physical_source_id=f's{i}', task_id=f't{i//2}',
                    condition='matched_buffer_control' if i%2 else 'stale', age_steps=i%2*3, anchor_steps=5)
               for i in range(4)]
    records = []
    for i, anchor in enumerate(anchors):
        for repeat in (range(4) if phase == 'A' else range(4, 8)):
            for option in (0, 1):
                # A selects Refresh for state 0; B reverses it, exposing winner optimism.
                success = int(option == (1 if phase == 'A' and i == 0 else 0))
                records.append(dict(panel_id=anchor['panel_id'], phase=phase, repeat=repeat, option=option,
                                    task_success=success, catastrophe=0, steps=10, path_length_m=.1))
    return anchors, records


def test_freeze_rejects_leakage_duplicates_and_missing():
    anchors, a = synthetic('A')
    _, b = synthetic('B')
    for bad in (a[:-1], a+[a[0]], b, a+b):
        with pytest.raises(ValueError):
            freeze(anchors, bad)
    frozen = freeze(anchors, a)
    assert frozen['reference']['b0'] == 1 and frozen['best_fixed'] == 'Base'
    for r in a:
        r['task_success'] = 1
    assert set(freeze(anchors, a)['reference'].values()) == {0}


def test_b_evaluation_does_not_reselect_and_bootstrap_is_paired(tmp_path):
    anchors, a = synthetic('A')
    _, b = synthetic('B')
    frozen = freeze(anchors, a)
    result = evaluate(anchors, b, frozen, bootstrap=100)
    rows = {r['method']: r for r in result['table']}
    assert rows['A_reference']['success_delta'] == -.25
    assert rows['B_posthoc_single_winner']['success_delta'] == 0
    assert rows['Base']['success_delta_ci95'] == [0, 0]
    assert rows['A_reference']['paired_harms'] == 4
    assert result == evaluate(anchors, b, frozen, bootstrap=100)
    render(result, tmp_path)
    assert (tmp_path/'main_figure.png').stat().st_size > 1000


class FakePolicy:
    resize_size = 2
    def __init__(self):
        self.queue = deque()
        self.calls = 0
        self._policy = SimpleNamespace(infer=self.infer)
    def infer(self, element):
        self.calls += 1
        return {'actions': np.repeat(element['state'][:1, None], 5, axis=0)}
    def act(self, obs, instruction):
        if not self.queue:
            self.queue.extend(self._policy.infer(obs)['actions'])
        return self.queue.popleft()
    def reset(self):
        self.queue.clear()
    def snapshot_continuation(self):
        return SimpleNamespace(payload={'queue': copy.deepcopy(tuple(self.queue))})


class FakeEnv:
    task_description = 'fake'
    def __init__(self):
        self.t = 0
        self.sim_view = SimpleNamespace(_live_mj=lambda: (None, SimpleNamespace(time=self.t*.05)),
                                        max_contact_force=lambda _: 0, _robot_bodies=lambda: [],
                                        _obs={'robot0_eef_pos': np.zeros(3), 'state': np.array([1.])})
    def policy_observation(self, obs, size):
        return {'state': np.asarray(obs['state']).copy()}
    def controller_state(self):
        return {'t': np.array([self.t])}
    def flat_state(self):
        return np.array([self.t])
    def episode_terminated(self):
        return False
    def step(self, action):
        self.t += 1
        return {'robot0_eef_pos': np.zeros(3), 'state': np.array([float(self.t)])}, 0, self.t == 8, {}


def test_neutral_closed_loop_and_logger_does_not_add_inference(tmp_path):
    paths = []
    for option in (0, 1):
        env, policy, queue = FakeEnv(), FakePolicy(), TimedQueue(3)
        trace = tmp_path/f'{option}.gz'
        result = rollout(env, policy, queue, {'anchor_steps': 5, 'condition': 'matched_buffer_control'}, option, trace)
        assert policy.calls == 2 and result['task_success'] == 1 and result['steps'] == 8
        records = []
        with gzip.open(trace, 'rb') as f:
            try:
                while True:
                    records.append(pickle.load(f))
            except EOFError:
                pass
        assert records[0]['nominal_proposal_status'] == 'unavailable'
        assert sum(len(r['inference']) for r in records) == 2
        paths.append(records)
    assert all(np.array_equal(a['action'], b['action']) for a, b in zip(*paths))


def test_resume_only_missing_execution_suffix():
    from scripts.expansion.resume_selection_retest import missing_cells
    anchors, b = synthetic('B')
    execution = [(a['panel_id'], r, o) for i, a in enumerate(anchors) for r in range(4, 8) for o in order(i, r)]
    lookup = {(r['panel_id'], r['repeat'], r['option']): r for r in b}
    completed = [lookup[key] for key in execution[:11]]
    assert missing_cells(anchors, completed) == execution[11:]
    assert not set(execution[:11]) & set(missing_cells(anchors, completed))
    assert missing_cells(anchors, b) == []
    for bad in (completed+[completed[0]], completed[1:], [lookup[execution[12]]]):
        with pytest.raises(ValueError):
            missing_cells(anchors, bad)


def test_cross_job_pair_diagnostic_preserves_primary_and_freeze():
    anchors, a = synthetic('A')
    _, b = synthetic('B')
    frozen = freeze(anchors, a)
    before = copy.deepcopy(frozen)
    primary = evaluate(anchors, b, frozen, bootstrap=50)
    diagnostic = evaluate(anchors, b, frozen, bootstrap=50, excluded_pairs=[('b0', 4)])
    assert primary['table'][0]['n_B_pairs'] == 16
    assert diagnostic['table'][0]['n_B_pairs'] == 15
    assert diagnostic['table'][0]['success_delta_ci95'] == [0, 0]
    assert frozen == before
    ref = next(r for r in diagnostic['table'] if r['method'] == 'A_reference')
    assert ref['paired_harms'] == 3
    with pytest.raises(ValueError):
        evaluate(anchors, b, frozen, excluded_pairs=[('b0', 100)])
