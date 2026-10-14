"""Audit candidate evidence, ancestry and shortcut readiness without training.

This stage deliberately cannot accept a one-class pool as a feasibility dataset.
Image features are computed on Quest; no external service or learned VLM is used.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
from .inventory import read,sha,write
from .export_candidates import verify_case


def group_components(rows):
    """Label-blind union of source ancestry and exact shared image pairs."""
    parents=list(range(len(rows)))
    def find(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]];i=parents[i]
        return i
    seen={}
    for i,row in enumerate(rows):
        keys=[('source',row['metadata']['source_id']),('images',row['metadata']['image_pair_sha256'])]
        if row['metadata'].get('initial_qpos_sha256'):
            keys.append(('settled_qpos',row['metadata']['initial_qpos_sha256']))
        for kind,key in keys:
            token=(kind,key)
            if token in seen:parents[find(i)]=find(seen[token])
            else:seen[token]=i
    members=defaultdict(list)
    for i,row in enumerate(rows):members[find(i)].append(row['metadata']['source_id'])
    names={k:'ancestry_'+hashlib.sha256('\n'.join(sorted(set(v))).encode()).hexdigest()[:16]
           for k,v in members.items()}
    return [names[find(i)] for i in range(len(rows))]


def validate_public(row):
    if set(row['student'])!={'images','task','constraints'}:
        raise RuntimeError('Unexpected student field: possible label/future leakage')
    if row['training_eligible'] is not False or row['reasoning_target'] is not None:
        raise RuntimeError('This stage accepts unapproved positive candidates only')
    if row['evidence']['kind']!='replayed_safe_success_suffix' or row['evidence']['feasibility']!='feasible':
        raise RuntimeError('Execution failure or unsupported evidence is not a positive')
    images=row['student']['images']
    if len(images)!=2 or {im['camera'] for im in images}!={'external','wrist'}:
        raise RuntimeError('Two original camera views required')
    for im in images:
        rel=Path(im['path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise RuntimeError('Candidate image path escapes the export')


def shortcut_readiness(labels):
    counts=dict(Counter(labels))
    return dict(class_counts=counts,binary_metrics_available=len(counts)==2,
                auroc=None,accuracy=None,
                reason='No fitted shortcut model in the candidate precheck; both independently evidenced classes and a frozen split are required',
                training_or_test_metrics_claimed=False)


def image_statistics(images):
    import numpy as np
    features=[]
    for im in images:
        x=np.asarray(im,dtype=np.float64)/255.
        red=(x[:,:,0]>.4)&(x[:,:,0]>1.4*x[:,:,1])&(x[:,:,0]>1.4*x[:,:,2])
        features.extend(x.mean((0,1)));features.extend(x.std((0,1)))
        features.append(float(red.mean()))
        for block in np.array_split(x,4,axis=0):
            for tile in np.array_split(block,4,axis=1):features.extend(tile.mean((0,1)))
    return features


def main(root,exports):
    import numpy as np
    from PIL import Image
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Pool audit belongs on a CPU allocation')
    source_records=[];rows=[];features=[];duplicate=[];unique={};verified=set()
    for export in exports:
        receipt=read(export/'RECEIPT.json');job=receipt['job']
        state=subprocess.check_output(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode'],text=True).strip()
        if receipt['stage']!='export-candidates' or state!=job+'|COMPLETED|0:0':
            raise RuntimeError('Normal completed candidate export required')
        hashes=read(export/'HASHES.json')
        for path,wanted in hashes.items():
            if sha(export/path)!=wanted:raise RuntimeError('Changed export evidence')
        incoming=[json.loads(line) for line in (export/'positive_candidates.jsonl').read_text().splitlines()]
        if len(incoming)!=read(export/'CANDIDATE_EXPORT.json')['positive_candidates']:
            raise RuntimeError('Incomplete exported manifest')
        source_records.append(dict(root=str(export),job=job,receipt_sha256=sha(export/'RECEIPT.json'),
            manifest_sha256=sha(export/'positive_candidates.jsonl'),hashes_sha256=sha(export/'HASHES.json'),
            rows=len(incoming)))
        for row in incoming:
            validate_public(row);evidence=row['evidence'];folder=Path(evidence['witness_root'])
            summary=read(folder/'SUMMARY.json')
            if not summary['eligible_feasible_witness'] or not summary['success'] or summary['protected_contact_events']!=0:
                raise RuntimeError('No qualifying strict safe witness')
            if folder not in verified:verify_case(folder,summary);verified.add(folder)
            step=evidence['checkpoint_step'];context=folder/('checkpoint_%03d'%step)/'context.json'
            ctx=read(context)
            if (sha(folder/'SUMMARY.json')!=evidence['witness_summary_sha256'] or
                sha(context)!=evidence['checkpoint_context_sha256'] or
                sha(ctx['source_inference'])!=evidence['source_inference_sha256']):
                raise RuntimeError('Broken candidate evidence link')
            if (ctx['initial_success'] or not ctx['safe_history'] or
                evidence['remaining_witness_actions']!=summary['actions']-step or
                evidence['remaining_horizon']!=ctx['remaining_horizon'] or
                not 0<evidence['remaining_witness_actions']<=ctx['remaining_horizon']):
                raise RuntimeError('Invalid suffix or exhausted horizon')
            if row['metadata']['source_id']!=summary['source']['source_id']:
                raise RuntimeError('Source ancestry disagreement')
            original_result=Path(summary['source']['root'])/'RESULT.json'
            if sha(original_result)!=summary['source']['result_sha256']:
                raise RuntimeError('Original rollout result changed')
            row['metadata']['initial_qpos_sha256']=read(original_result)['settled_qpos_sha256']
            images=[]
            for im in row['student']['images']:
                path=export/im['path']
                if sha(path)!=im['sha256']:raise RuntimeError('Image digest mismatch')
                with Image.open(path) as image:images.append(np.asarray(image).copy())
            with np.load(ctx['source_inference'],allow_pickle=False) as original:
                if not (np.array_equal(images[0],original['image']) and
                        np.array_equal(images[1],original['wrist_image']) and
                        row['student']['task']==str(original['prompt'])):
                    raise RuntimeError('Student input differs from actual source observation')
            pixels=hashlib.sha256(images[0].tobytes()+images[1].tobytes()).hexdigest()
            digest=hashlib.sha256(json.dumps(dict(images=pixels,task=row['student']['task'],
                constraints=row['student']['constraints']),sort_keys=True).encode()).hexdigest()
            if digest!=row['metadata']['input_sha256'] or pixels!=row['metadata']['image_pair_sha256']:
                raise RuntimeError('Full input or pixel identity changed')
            if digest in unique:
                duplicate.append(dict(source_export=str(export),sample=row['sample_id'],kept=unique[digest]));continue
            sample='fdpool_%05d'%len(rows);unique[digest]=sample
            row['metadata']['source_export']=str(export)
            row['metadata']['source_sample_id']=row['sample_id'];row['sample_id']=sample
            # Resolve relative paths in the merged audit index; no image content
            # or source name is ever inserted into the student text.
            for im in row['student']['images']:im['path']=str(export/im['path'])
            rows.append(row);features.append(image_statistics(images))
    groups=group_components(rows)
    for row,group in zip(rows,groups):row['metadata']['ancestry_component']=group
    coverage=dict(rows=len(rows),original_states=len({r['metadata']['source_id'] for r in rows}),
        ancestry_components=len(set(groups)),
        task_compositions=len({r['metadata']['task_group'] for r in rows}),
        suite_families=len({r['metadata']['family'] for r in rows}),
        unique_image_pairs=len({r['metadata']['image_pair_sha256'] for r in rows}),
        represented_witnesses=len(verified),
        by_original_source=dict(Counter(r['metadata']['source_id'] for r in rows)),
        by_task=dict(Counter(r['metadata']['task_group'] for r in rows)),
        by_family=dict(Counter(r['metadata']['family'] for r in rows)))
    protocol=read(Path(__file__).with_name('protocol.json'))
    gates=dict(minimum_original_states=coverage['original_states']>=protocol['minimum_original_states'],
        minimum_task_compositions=coverage['task_compositions']>=protocol['minimum_task_compositions'],
        minimum_suite_families=coverage['suite_families']>=protocol['minimum_suite_families'],
        matched_500_positive_500_negative=False,verified_reasoning_targets=False,
        observable_premises=False,frozen_grouped_split=False,visual_text_shortcut_evaluation=False)
    (root/'MERGED_CANDIDATES.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    np.savez_compressed(root/'RGB_FEATURES.npz',features=np.asarray(features),sample_ids=np.asarray([r['sample_id'] for r in rows]))
    write(root/'SOURCE_COVERAGE.json',coverage);write(root/'INHERITED_EXPORTS.json',source_records)
    write(root/'DUPLICATES.json',duplicate)
    write(root/'SHORTCUT_PRECHECK.json',dict(shortcut_readiness([r['evidence']['feasibility'] for r in rows]),
        feature_definition='Each original view: RGB means/std, fixed red-mask fraction, 4x4 pooled RGB; no labels or future data',
        feature_columns=110,feature_rows=len(rows),split='unassigned',
        required_controls=['color and red-area', 'target visibility', 'aperture geometry',
                           'RGB-only', 'task/rules-only', 'relevant-constraint flip', 'irrelevant appearance invariance']))
    write(root/'REASONING_REQUIREMENTS.json',dict(
        finalized_targets=0,api_calls=0,
        existing_evidence=['actual legal action suffix','exact per-action source equality',
                           'zero prohibited contacts','success within remaining horizon'],
        still_required=['observable target and goal grounding','evidence-backed route or separator premises',
                        'explicit per-row safety/geometry assumptions','clause-level evidence pointers',
                        'counterfactual relevance check'],
        prohibition='A correct verdict or a fluent explanation cannot validate its premises. Witness metadata is not student input.'))
    result=dict(passed=True,stage='candidate_precheck',coverage=coverage,gates=gates,
        scientifically_accepted_rows=0,scientific_training_allowed=False,
        binary_shortcut_auroc=None,api_calls=0,slurm_job=os.environ['SLURM_JOB_ID'])
    write(root/'AUDIT_COMPLETE.json',result)
    names=['MERGED_CANDIDATES.jsonl','RGB_FEATURES.npz','SOURCE_COVERAGE.json','INHERITED_EXPORTS.json',
           'DUPLICATES.json','SHORTCUT_PRECHECK.json','REASONING_REQUIREMENTS.json','AUDIT_COMPLETE.json']
    write(root/'HASHES.json',{name:sha(root/name) for name in names})
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--candidate-export',type=Path,action='append',required=True)
    a=p.parse_args();main(a.root,a.candidate_export)
