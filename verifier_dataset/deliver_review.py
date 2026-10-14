"""Copy immutable, completed-source images for review while a batch is running.

This performs only result-file I/O. It does not fit models, select a final
dataset, accept labels, or treat a partially finished Slurm batch as complete.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
from .inventory import ASSETS,read,write,sha
from .constraint_selection import check_unit


def main(parents,start,stop,output,checkpoint_start=0,checkpoint_stop=3):
    repo=Path.cwd().resolve()
    if repo!=(Path.home()/'crash_bench').resolve():raise RuntimeError('Wrong project checkout')
    output=output.resolve();output.relative_to(repo/'results/verifier_dataset')
    if output.exists() or not 0<=start<stop<=35:raise RuntimeError('New bounded preview window required')
    if not 0<=checkpoint_start<checkpoint_stop<=10:raise RuntimeError('Invalid qualified-checkpoint window')
    records=[];inherited=[];seen=set()
    for parent in parents:
        parent=parent.resolve();parent.relative_to((ASSETS/'verifier_dataset').resolve())
        receipt=read(parent/'RECEIPT.json')
        if receipt['stage']!='constraint-batch':raise RuntimeError('Expected an explicit constraint collection shard')
        inherited.append(dict(root=str(parent),receipt_sha256=sha(parent/'RECEIPT.json'),job=receipt['job'],code_commit=receipt['code_commit']))
        for index in range(start,min(stop,len(read(parent/'CASES.json')))):
            case=parent/('case_%03d'%index)
            if not (case/'COMPLETE.json').exists():raise RuntimeError('Source is not complete yet: '+str(case))
            complete=read(case/'COMPLETE.json')
            if not complete['passed']:raise RuntimeError('Source process failed')
            sid=complete['source']['source_id']
            if sid in seen:raise RuntimeError('Repeated source in preview window')
            seen.add(sid)
            choices=sorted((r for r in complete.get('rows',[]) if r['selected_first_valid_option']),key=lambda r:r['checkpoint'])[checkpoint_start:checkpoint_stop]
            for chosen in choices:
                option=Path(chosen['option']);unit=read(option/'CANDIDATES.json');check_unit(unit)
                records.append((unit,dict(source_case=str(case),source_completion_sha256=sha(case/'COMPLETE.json'))))
    # Do not create a partial output when any requested source is still running.
    output.mkdir(parents=True,exist_ok=False);units=[]
    for i,(original,provenance) in enumerate(records):
        unit=copy.deepcopy(original)
        for row in unit:
            row['metadata'].update(unit_id='preview_%03d'%i,**provenance)
            for im in row['student']['images']:
                src=Path(im['path']);rel=Path('images')/row['metadata']['unit_id']/('assignment_%d'%row['evidence']['assignment'])/(im['camera']+'.png')
                dest=output/rel;dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():shutil.copyfile(src,dest)
                if sha(dest)!=im['sha256']:raise RuntimeError('Transferred image differs')
                im['path']=str(rel)
        units.append(unit)
    write(output/'PREVIEW_UNITS.json',units)
    write(output/'SCOPE.json',dict(stage='completed_source_image_preview',parent_jobs_may_be_running=True,
        parents=inherited,source_case_window=[start,stop],original_sources=len(seen),units=len(units),
        qualified_checkpoint_index_window=[checkpoint_start,checkpoint_stop],final_selection_performed=False,model_predictions=0,
        scientific_training_allowed=False,accepted_rows=0,api_calls=0))
    manifest={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}
    write(output/'DELIVERY.json',manifest)
    print(json.dumps(dict(output=str(output),files=len(manifest),sources=len(seen),units=len(units))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,action='append',required=True)
    p.add_argument('--start',type=int,required=True);p.add_argument('--stop',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--checkpoint-start',type=int,default=0);p.add_argument('--checkpoint-stop',type=int,default=3)
    a=p.parse_args();main(a.parent,a.start,a.stop,a.out,a.checkpoint_start,a.checkpoint_stop)
