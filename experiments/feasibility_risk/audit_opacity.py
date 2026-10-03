"""Audit only persisted FR-1C evidence from the one-attempt integrity stop."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image

def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def fingerprint(base, check, rgba, red):
    after = check['model_fingerprint']
    assert base['numbers'] == after['numbers']
    assert base['arrays'].keys() == after['arrays'].keys()
    expected = rgba.copy(); expected[red, 3] = check['alpha']
    assert after['arrays']['geom_rgba']['sha256'] == hashlib.sha256(expected.tobytes()).hexdigest()
    for key in base['arrays']:
        if key != 'geom_rgba': assert base['arrays'][key] == after['arrays'][key], key

def main(cpu, gpu, parents, out):
    protocol = read(cpu/'protocol.json')
    assert protocol == read(gpu/'protocol.json')
    assert read(cpu/'CPU_GATE.json')['passed']
    assert read(cpu/'COMPLETE.json')['step_attempts'] == []
    stop = read(gpu/'STOP.json')
    assert stop == dict(error='AssertionError', reason='', step_attempts=[])
    mapping = {protocol['cpu_parent']: parents/'20261003T051248Z_cpu_3b76293292d0',
               protocol['decoy_cpu_parent']: parents/'structural/20261003T055338Z_cpu_be5b79d49e97'}
    rows = []
    for episode, variant in protocol['states']:
        folder = cpu/('e%d'%episode)/variant
        source = read(folder/'SOURCE_LABEL.json')
        old = Path(source['label_root']); prefix = next(k for k in mapping if str(old).startswith(k+'/'))
        local = mapping[prefix]/old.relative_to(prefix)
        for name, value in source['source_files'].items(): assert sha(local/name) == value
        assert read(folder/'INITIAL_RESTORE.json') == read(local/'initial_restore.json')
        base = read(folder/'MODEL_BEFORE.json'); checks = read(folder/'CPU_PHYSICS_CHECK.json')
        assert checks['passed'] and checks['zero_step_attempts'] == 0
        with np.load(folder/'RENDER_MODEL_BEFORE.npz') as z:
            for c in checks['checks']: fingerprint(base, c, z['geom_rgba'], z['red'])
        a, b = checks['checks']
        assert a['alpha'] == 1.0 and b['alpha'] == .25
        assert a['changed_arrays'] == [] and b['changed_arrays'] == ['geom_rgba']
        assert a['full_state_hash'] == b['full_state_hash']
        rows.append(dict(episode=episode, variant=variant, inherited_label=source['label'],
                         model_array_count=len(base['arrays']), alpha_only=True, full_state_exact=True))
    sub = gpu/'e10/open/opaque'
    assert list(gpu.glob('e*/**/FEATURE_AUDIT.json')) == [sub/'FEATURE_AUDIT.json']
    assert not list((gpu/'e10/open/transparent').iterdir())
    missing = ['COMPLETE.json','COUNTS.json','VISIBILITY_GATE.json','END_ANCHOR_REPEAT.json','CHECKPOINT_POST_VERIFY.json']
    assert all(not (gpu/name).exists() for name in missing)
    assert read(gpu/'e10/open/INITIAL_RESTORE.json') == read(cpu/'e10/open/INITIAL_RESTORE.json')
    feature = read(sub/'FEATURE_AUDIT.json')
    assert feature['passed'] and feature['feature_repeat_linf'] == feature['action_before_after_linf'] == 0
    assert feature['rng_before'] == feature['rng_after'] and feature['training_steps'] == 0
    with np.load(sub/'initial_policy_input.npz') as inputs, np.load(parents/'20261003T052306Z_gpu_11804f5cc2ad/e10/open/initial_policy_input.npz') as old:
        assert inputs.files == old.files and all(np.array_equal(inputs[k], old[k]) for k in inputs.files)
        counts = {}
        for camera, key in [('agentview','observation/image'),('robot0_eye_in_hand','observation/wrist_image')]:
            normal = np.asarray(Image.open(sub/(camera+'_policy_224.png')))
            assert np.array_equal(normal, inputs[key])
            a = np.asarray(Image.open(sub/(camera+'_cyan_224.png'))).astype(int)
            b = np.asarray(Image.open(sub/(camera+'_magenta_224.png'))).astype(int)
            delta = np.max(np.abs(a-b), axis=2).astype(np.uint8)
            roi = np.asarray(Image.open(sub/(camera+'_target_roi_224.png'))) > 0
            assert np.array_equal(delta, np.asarray(Image.open(sub/(camera+'_target_influence_224.png'))))
            counts[camera] = int(((delta >= 1) & roi).sum())
    influence = read(sub/'TARGET_INFLUENCE.json')
    for view in influence['views']: assert counts[view['camera']] == view['target_influence_pixels']
    assert counts == dict(agentview=69, robot0_eye_in_hand=0)
    assert influence['full_state_unchanged'] and influence['model_restored_exact'] and not influence['marker_images_sent_to_policy']
    log = (gpu/'slurm_8376030.log').read_text()
    assert 'line 116, in influence' in log and 'assert np.array_equal(normal,data[key])' in log
    lines = (out.parent/'TERMINAL_ACCOUNTING.txt').read_text().splitlines()
    fields = next(x for x in lines if x.startswith('JobIDRaw|')).split('|')
    jobs = [dict(zip(fields, line.split('|'))) for line in lines if line.split('|')[0].isdigit() and len(line.split('|')) == len(fields)]
    assert {j['JobIDRaw'] for j in jobs} == {'8368811','8369470','8371202','8371672','8372919','8375743','8376030'}
    assert all(j['State'] in ('COMPLETED','FAILED') for j in jobs)
    assert not any(line.split('|')[0].isdigit() and len(line.split('|')) == 2 for line in lines)
    seconds = {j['JobIDRaw']: int(j['ElapsedRaw']) for j in jobs}
    assert seconds['8375743'] == 109 and seconds['8376030'] == 209
    result = dict(persisted_evidence_audit_passed=True, experiment_completed=False,
                  cpu_rows=rows, opaque_completed_inputs=1, opaque_target_influence=counts,
                  input_sha256=feature['input_sha256'], zero_feature_repeat_error=True,
                  zero_action_bracketing_error=True, transparent_result=None,
                  visibility_gate=None, contrasts=None, heldout_metrics=None,
                  failure=dict(state='e10/open/transparent', location='opacity.py:116',
                               check='direct preprocessed camera RGB equals refreshed observation RGB',
                               exception='AssertionError', cause='unresolved', failed_compared_images_saved=False,
                               camera='agentview inferred from fixed camera order and empty output directory; not explicitly logged'),
                  missing_completion_artifacts=missing,
                  counts=dict(planned_inputs=17, completed_inputs=1, feature_forwards=2,
                              unexecuted_local_action_inferences=6, environment_actions=0,
                              step_attempts=0, paid_api_calls=0, trained_readout=False,
                              new_labels=0, new_independent_layouts=0),
                  resources=dict(fr1c_cpu_seconds=109, fr1c_gpu_seconds=209,
                                 all_study_cpu_job_seconds=sum(seconds[k] for k in ('8368811','8371202','8375743')),
                                 all_study_gpu_allocation_seconds=sum(seconds[k] for k in ('8369470','8371672','8372919','8376030'))),
                  jobs=jobs, all_current_study_jobs_terminal=True,
                  limitations=['Headless CPU alpha-only checks do not validate pixels.',
                               'Missing transparent pair prevents mismatch magnitude or root-cause reconstruction.',
                               'No post-job checkpoint verification or end anchor ran after the stop.',
                               'Opaque colour influence is not target recognition or a transparency result.'])
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('persisted_evidence_audit_passed','experiment_completed','counts','resources','all_current_study_jobs_terminal')}, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('cpu','gpu','parents','out'): p.add_argument(name, type=Path)
    a = p.parse_args(); main(a.cpu,a.gpu,a.parents,a.out)
