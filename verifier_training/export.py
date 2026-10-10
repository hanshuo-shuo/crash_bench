"""Read-only evidence exporter. Run on a Quest CPU allocation."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys
import numpy as np
from PIL import Image, ImageDraw
from .common import *



def check_hashes(root, names=None):
    manifest = read(root/'VERIFIED.json')
    require(manifest['passed'], 'Original verification failed')
    hashes = manifest['hashes'] if names is None else {n: manifest['hashes'][n] for n in names}
    for name, expected in hashes.items():
        require(sha(root/name) == expected, 'Evidence hash mismatch: '+str(root/name))
    return hashes


def export(cpu, gpu, out):
    out.mkdir(parents=True, exist_ok=False)
    # Reuse the frozen, pure persisted-evidence checkers, never import simulation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/reach_avoid'))
    from validation import verify_certificate, verify_steps
    rows, excluded, pairs = [], [], {}
    for folder in sorted(cpu.glob('shard_*/L*')):
        if not folder.is_dir():
            continue
        summary = read(folder/'summary.json')
        if summary['label'] == 'unknown':
            excluded.append({'case': folder.name, 'reason': 'unknown independent label'})
            continue
        cert = read(folder/'CERTIFICATE.json')
        label_contract(summary, cert)
        hashes = check_hashes(folder)
        steps = [json.loads(x) for x in (folder/'steps.jsonl').read_text().splitlines()]
        geometry = read(folder/'geometry.json')
        recomputed = verify_certificate(geometry, read(folder/'MATERIAL_BALL.json'),
            read(folder/'fixture.json'), steps[0], summary['initial_valid'], cert)
        verify_steps(steps, summary, geometry['fixed_goal'], recomputed, 300)
        obs = gpu/folder.parent.name/'output'/folder.name
        inherited = read(obs/'INHERITED.json')
        require(inherited['label'] == summary['label'] and inherited['environment_actions'] == 0, 'Bad inherited label')
        for name, expected in inherited['hashes'].items():
            require(sha(folder/name) == expected, 'Label inheritance changed')
        require(sha(obs/'post_restore.json') == sha(folder/'initial_restore.json'), 'Initial state mismatch')
        visibility = read(obs/'VISIBILITY.json')
        require(visibility['physics_unchanged'] and all(v['exact_policy_rgb_match'] for v in visibility['views']), 'Visibility alignment failed')
        used = ['actual_policy_input.npz', 'VISIBILITY.json', 'INHERITED.json', 'post_restore.json']
        used += [c+'_policy_224.png' for c in CAMERAS]
        used += [c+'_segmentation_224.png' for c in CAMERAS]
        # One GPU endpoint is invalid; that never invalidates independently restored t=0 evidence.
        anchored = (obs/'VERIFIED.json').exists()
        if anchored:
            check_hashes(obs, used)
        images = []
        with np.load(obs/'actual_policy_input.npz', allow_pickle=False) as actual:
            for c, key in zip(CAMERAS, ('observation/image', 'observation/wrist_image')):
                src = obs/(c+'_policy_224.png')
                with Image.open(src) as im:
                    require(im.mode == 'RGB' and im.size == (224, 224), 'Invalid RGB image')
                    require(np.array_equal(np.asarray(im), actual[key]), 'Not the original input RGB')
                    require(np.asarray(im).std() > 1, 'Blank input')
                rel = Path('images')/f'{len(rows):03d}'/(c+'.png')
                (out/rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, out/rel)
                images.append({'camera': c, 'path': str(rel), 'sha256': sha(src), 'width': 224, 'height': 224})
        text, grounding = rationale(visibility)
        task = read(folder/'task.json')
        require(task['state']['layout_group'] == 'object_I_t2_e0', 'Unexpected source ancestry')
        require(task['code_commit'] == CPU_COMMIT and task['upstream_commit'] == UPSTREAM, 'Unpinned evidence source')
        row = {'schema_version': 1, 'sample_id': f'ra_{len(rows):03d}',
            'student': {'images': images, 'task': TASK, 'constraints': CONSTRAINTS},
            'target': {'verdict': summary['label'], 'rationale': text},
            'metadata': {'case': folder.name, 'source_scene': 'object_I_t2_e0',
                'layout': summary['layout'], 'constructor': summary['mechanism'], 'variant': summary['variant'],
                'label_basis': 'safe_witness' if summary['label']=='feasible' else 'conditional_RA-SEP-1_certificate',
                'label_contract': 'RA-SEP-1', 'cpu_root': str(folder), 'gpu_root': str(obs),
                'cpu_code_commit': CPU_COMMIT, 'gpu_code_commit': GPU_COMMIT, 'upstream_commit': UPSTREAM,
                'cpu_slurm': task['slurm_job'], 'gpu_slurm': read(obs.parent/'COMPLETE.json')['job'],
                'seed': 7, 'ancestry': task['state'], 'cpu_verified_hashes': hashes,
                'cpu_manifest_sha256': sha(folder/'VERIFIED.json'),
                'observation_hashes': {n: sha(obs/n) for n in used}, 'gpu_terminal_manifest_present': anchored,
                'visibility': visibility, 'rationale_grounding': grounding,
                'rationale_scope': 'Observable entity visibility and epistemic limits; not a reconstructed geometric proof',
                'full_physical_state_observable': False,
                'rationale_version': 1}}
        key = tuple(i['sha256'] for i in images)
        if key in pairs:
            require(pairs[key] == summary['label'], 'Contradictory identical visual inputs')
        pairs[key] = summary['label']
        rows.append(row)
    require(len(rows) == 120 and Counter(r['target']['verdict'] for r in rows) == {'feasible':72, 'infeasible':48}, 'Unexpected 120-label set')
    require(len(excluded) == 48, 'Expected all 48 UNKNOWN exclusions')
    path = out/'dataset.jsonl'
    path.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    # Eight fixed samples; both labels, both constructors, two layout variations.
    selected = [r['sample_id'] for r in rows if r['metadata']['layout'] in ('L01','L04') and
        ((r['metadata']['constructor']=='slit' and r['metadata']['variant'] in ('open','.020')) or
         (r['metadata']['constructor']=='cage' and r['metadata']['variant'] in ('.180','.028')))]
    require(len(selected)==8, 'Smoke subset size changed')
    write(out/'smoke_ids.json', selected)
    hidden = Counter(r['target']['verdict'] for r in rows if all(v['entities']['target']['pixels_224']==0 for v in r['metadata']['visibility']['views']))
    audit = {'passed': True, 'engineering_supervision_only': True, 'rows':len(rows),
        'labels':dict(Counter(r['target']['verdict'] for r in rows)), 'excluded':excluded,
        'original_source_scenes':1, 'target_hidden_both_views_by_label': dict(hidden),
        'full_physical_observability_passed':False, 'physical_observability_note':
        'Two RGB views do not certify clearance, all-route separation, hidden target position or dynamics. Rationales explicitly acknowledge this.',
        'rationale_factuality_rule_passed':True, 'contradictory_exact_image_pairs':0,
        'policy_outcomes_used_as_labels':False, 'native_positives_mixed':False,
        'dataset_sha256':sha(path), 'unique_image_pairs':len(pairs), 'smoke_ids':selected,
        'slurm_job':os.environ.get('SLURM_JOB_ID'), 'export_code_commit':os.environ.get('CB_CODE_COMMIT'),
        'scope':'Overfit mechanics only. No random split or cross-source generalization metric.'}
    write(out/'AUDIT.json', audit)
    # Contact sheets include all samples for visual review, separate from inputs.
    for start in range(0, len(rows), 20):
        sheet=Image.new('RGB',(896,10*260),'white'); draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(rows[start:start+20]):
            x=(j%2)*448; y=(j//2)*260
            draw.text((x+2,y+2),r['metadata']['case']+' '+r['target']['verdict'],fill='black')
            for k,i in enumerate(r['student']['images']):
                sheet.paste(Image.open(out/i['path']),(x+k*224,y+28))
        sheet.save(out/f'contact_{start//20:02d}.jpg')
    print(json.dumps({k:v for k,v in audit.items() if k!='excluded'}),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--cpu',type=Path,default=CPU);p.add_argument('--gpu',type=Path,default=GPU);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();export(a.cpu,a.gpu,a.out)
