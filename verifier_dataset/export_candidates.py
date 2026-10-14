"""Export verified positive candidates only; never enable scientific training."""
import argparse
from collections import Counter
import hashlib
import gzip
import json
import os
from pathlib import Path
import subprocess
from .inventory import read, sha, write


def public_fields(saved):
    """Never expose future actions, diagnostic samples, proprioception or IDs."""
    return dict(external_rgb=saved['image'],wrist_rgb=saved['wrist_image'],task=str(saved['prompt']))


def verify_case(folder, summary):
    if read(folder/'SUMMARY.json')!=summary:raise RuntimeError('Shard and per-case summaries disagree')
    for name, expected in read(folder/'HASHES.json').items():
        if sha(folder/name)!=expected:raise RuntimeError('Witness evidence changed: '+str(folder/name))
    equality=read(folder/'EQUALITY.json');tape=read(folder/'ACTION_TAPE.json')
    count=summary['actions']
    if len(equality)!=count or len(tape)!=count or not all(r['equal'] for r in equality):
        raise RuntimeError('Incomplete exact replay proof')
    if [r['step'] for r in tape]!=list(range(1,count+1)):
        raise RuntimeError('Incomplete legal witness tape')
    if not all(len(r['action'])==7 and all(-1<=x<=1 for x in r['action']) for r in tape):
        raise RuntimeError('Illegal exported witness action')
    with gzip.open(folder/'CONTACTS.jsonl.gz','rt') as stream:
        if any(line.strip() for line in stream):raise RuntimeError('Strict safe witness contains prohibited contact events')
    if summary.get('contact_contract_version'):
        from .contact_contract import verify_saved
        verify_saved(folder,summary)


def main(root, replay_root):
    import numpy as np
    from PIL import Image
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Exporter belongs on a compute node')
    receipt=read(replay_root/'RECEIPT.json');job=receipt['job']
    accounting=subprocess.check_output(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode'],text=True)
    lines=[x.strip().split('|') for x in accounting.splitlines() if x.strip()]
    expected=8 if (replay_root/'shards').is_dir() else 1
    if len(lines)!=expected or any(x[1:]!=['COMPLETED','0:0'] for x in lines):
        raise RuntimeError('Every source shard must finish normally before this export')
    write(root/'INHERITED_ACCOUNTING.json',dict(job=job,raw=accounting,receipt_sha256=sha(replay_root/'RECEIPT.json')))
    parents=sorted((replay_root/'shards').iterdir()) if (replay_root/'shards').is_dir() else [replay_root]
    rows=[];duplicates=[];seen={};rejected=[]
    for parent in parents:
        complete=read(parent/'REPLAY_COMPLETE.json')
        if not complete['passed'] or complete['errors']:raise RuntimeError('Unresolved replay error')
        for i,summary in enumerate(complete['rows']):
            folder=parent/('case_%03d'%i)
            if not summary['eligible_feasible_witness']:
                rejected.append(dict(source=summary['source']['root'],reason='No strict safe legal witness'))
                continue
            verify_case(folder,summary)
            source=summary['source'];geometry=read(Path(source['root'])/'geometry.json')
            for step in summary['checkpoints']:
                checkpoint=folder/('checkpoint_%03d'%step);context=read(checkpoint/'context.json')
                if context['initial_success'] or not context['safe_history']:continue
                remaining=context['remaining_horizon']
                if summary['actions']-step>remaining:raise RuntimeError('Witness exceeds remaining budget')
                with np.load(context['source_inference'],allow_pickle=False) as saved:
                    fields=public_fields(saved)
                    images=[fields['external_rgb'].copy(),fields['wrist_rgb'].copy()]
                if any(im.shape!=(224,224,3) or im.dtype!=np.uint8 for im in images):
                    raise RuntimeError('Unexpected original student image format')
                if summary.get('contact_contract_version'):
                    from .contact_contract import DESCRIPTION
                    safety=DESCRIPTION
                else:
                    safety=('No robot or other body with a motion degree of freedom, including movable supports, '
                            'may contact it at the current state or any MuJoCo integration step (contact distance <= 0).')
                constraints=(f'At most {remaining} further commands in the native 7-D action range [-1, 1]. '
                    f'The protected object is {geometry["obstacle"]}. '+safety+
                    ' Complete the stated native task goal. The preceding history is safe.')
                image_pair_hash=hashlib.sha256(images[0].tobytes()+images[1].tobytes()).hexdigest()
                key=hashlib.sha256(json.dumps(dict(images=image_pair_hash,task=fields['task'],constraints=constraints),
                                               sort_keys=True).encode()).hexdigest()
                if key in seen:
                    duplicates.append(dict(kept_sample=seen[key],source=source['root'],step=step,input_sha256=key));continue
                sample='fd_%05d'%len(rows);seen[key]=sample
                image_records=[]
                for camera,im in zip(('external','wrist'),images):
                    rel=Path('images')/sample/(camera+'.png');path=root/rel;path.parent.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(im).save(path)
                    with Image.open(path) as check:
                        if not np.array_equal(np.asarray(check),im):raise RuntimeError('RGB export changed pixels')
                    image_records.append(dict(camera=camera,path=str(rel),sha256=sha(path),width=224,height=224))
                rows.append(dict(schema_version='FD1000-candidate-v1',sample_id=sample,
                    student=dict(images=image_records,task=fields['task'],constraints=constraints),
                    evidence=dict(feasibility='feasible',kind='replayed_safe_success_suffix',witness_root=str(folder),
                        witness_summary_sha256=sha(folder/'SUMMARY.json'),checkpoint_step=step,
                        contact_contract_version=summary.get('contact_contract_version','FD1000-strict-contact-v1'),
                        remaining_witness_actions=summary['actions']-step,remaining_horizon=remaining,
                        checkpoint_context_sha256=sha(checkpoint/'context.json'),source_inference_sha256=sha(context['source_inference'])),
                    metadata=dict(source_id=source['source_id'],task_group=source['task_group'],family=source['family'],
                        source_method=source['method'],source_seed=source['seed'],source_code_commit=source['source_code_commit'],
                        replay_code_commit=summary['code_commit'],slurm_job=summary['slurm_job'],input_sha256=key,
                        image_pair_sha256=image_pair_hash),
                    reasoning_target=None,observability_review='pending',split='unassigned',training_eligible=False))
    (root/'positive_candidates.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    write(root/'DUPLICATES.json',duplicates);write(root/'REJECTED.json',rejected)
    summary=dict(passed=True,positive_candidates=len(rows),duplicates_removed=len(duplicates),
        original_states=len({r['metadata']['source_id'] for r in rows}),
        task_compositions=len({r['metadata']['task_group'] for r in rows}),
        distinct_image_pairs=len({r['metadata']['image_pair_sha256'] for r in rows}),
        candidates_by_original_source=dict(Counter(r['metadata']['source_id'] for r in rows)),
        by_family=dict(Counter(r['metadata']['family'] for r in rows)),
        accepted_dataset_rows=0,scientific_training_allowed=False,negative_examples=0,
        remaining_gates=['matched certified negatives','source coverage','student-observable rationale evidence',
                         'ancestry-grouped split','visual/text shortcut audit'],
        api_calls=0,slurm_job=os.environ['SLURM_JOB_ID'],input_contract='RGB pixels, task and explicit constraints only')
    write(root/'CANDIDATE_EXPORT.json',summary)
    write(root/'HASHES.json',{str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*'))
                            if p.is_file() and 'source' not in p.relative_to(root).parts and p.name!='HASHES.json'
                            and p.suffix not in ('.log','.tar')})
    print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--replay-root',type=Path,required=True)
    a=p.parse_args();main(a.root,a.replay_root)
