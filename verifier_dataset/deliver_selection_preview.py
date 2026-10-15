"""Preview deterministic candidate selection after both collectors finish.

This copies existing files and applies only image-bound visibility exclusions.
It never replaces the independent compute-node evidence or shortcut audit.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
from .inventory import ASSETS,read,sha,write
from .constraint_selection import check_unit,select_units
from .constraint_curation import exclude,reviewed_only,selection_settings


def main(parents,review_path,output):
    repo=Path.cwd().resolve()
    if repo!=(Path.home()/'crash_bench').resolve():raise RuntimeError('Wrong project checkout')
    output=output.resolve();output.relative_to(repo/'results/verifier_dataset')
    if output.exists() or len(set(parents))!=2:raise RuntimeError('New preview and two distinct parents required')
    relative=review_path.resolve().relative_to(repo)
    if relative.parts[:2]!=('verifier_dataset','reviews'):
        raise RuntimeError('Published image-bound review required')
    subprocess.run(['git','ls-files','--error-unmatch',str(relative)],check=True,stdout=subprocess.DEVNULL)
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('Dirty source')
    review=read(review_path);units=[];inheritance=[];seen=set()
    for parent in parents:
        parent=parent.resolve();parent.relative_to((ASSETS/'verifier_dataset').resolve())
        receipt=read(parent/'RECEIPT.json');complete=read(parent/'CONSTRAINT_COMPLETE.json');job=receipt['job']
        if read(parent/'CAMPAIGN.json')['constraint_protocol_sha256']!=sha(repo/'verifier_dataset/constraint_protocol.json'):
            raise RuntimeError('Preview parents must exercise the current frozen construction protocol')
        terminal=subprocess.check_output(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode'],text=True).strip()
        if receipt['stage']!='constraint-batch' or not complete['passed'] or terminal!=job+'|COMPLETED|0:0':
            raise RuntimeError('Both collectors must terminate normally')
        for name,wanted in read(parent/'HASHES.json').items():
            if sha(parent/name)!=wanted:raise RuntimeError('Collection manifest changed')
        inheritance.append(dict(root=str(parent),job=job,receipt_sha256=sha(parent/'RECEIPT.json'),
            completion_sha256=sha(parent/'CONSTRAINT_COMPLETE.json'),terminal=terminal))
        for case in sorted(parent.glob('case_*')):
            c=read(case/'COMPLETE.json');sid=c['source']['source_id']
            if not c['passed'] or sid in seen:raise RuntimeError('Invalid or repeated source')
            seen.add(sid)
            for choice in c.get('rows',[]):
                if choice['selected_first_valid_option']:
                    unit=read(Path(choice['option'])/'CANDIDATES.json');check_unit(unit);units.append(unit)
    frozen={r['source_id']:r for r in read(repo/'verifier_dataset/frozen_sources_67.json')['sources']}
    for unit in units:
        for row in unit:row['split']=frozen[row['metadata']['source_id']]['split']
    protocol=read(repo/'verifier_dataset/constraint_protocol.json')
    target,quotas=selection_settings(review,protocol,sha(repo/'verifier_dataset/constraint_protocol.json'))
    eligible=exclude(units,review)
    if review.get('final_selection'):eligible=reviewed_only(eligible,review)
    selected=select_units(eligible,target=target,quotas=quotas)
    old={}
    for record in review.get('all_reviewed_units',[]):
        key=(record['source_id'],record['checkpoint'])
        if key in old:raise RuntimeError('Repeated prior review')
        old[key]=record
    output.mkdir(parents=True,exist_ok=False);preview=[];selection=[];inherited=0
    for index,original in enumerate(selected):
        unit=copy.deepcopy(original);m=unit[0]['metadata'];key=(m['source_id'],m['checkpoint'])
        images=sorted({im['sha256'] for row in unit for im in row['student']['images']})
        prior=old.get(key)
        if prior is not None and sorted(prior['image_sha256'])!=images:raise RuntimeError('Prior review covers different pixels')
        visible=prior is not None and prior['status']=='premises_visible'
        selection.append(dict(source_id=key[0],checkpoint=key[1],image_sha256=images,
            evidence_root=m['evidence_root'],prior_visible_review=visible))
        if visible:inherited+=1;continue
        for row in unit:
            row['metadata']['unit_id']='preview_%03d'%index
            for im in row['student']['images']:
                rel=Path('images')/row['metadata']['unit_id']/('assignment_%d'%row['evidence']['assignment'])/(im['camera']+'.png')
                dest=output/rel;dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():shutil.copyfile(im['path'],dest)
                if sha(dest)!=im['sha256']:raise RuntimeError('Transferred pixels differ')
                im['path']=str(rel)
        preview.append(unit)
    write(output/'PREVIEW_UNITS.json',preview);write(output/'PRELIMINARY_SELECTION.json',selection)
    write(output/'SCOPE.json',dict(stage='preliminary_selection_image_review',parents=inheritance,
        independent_evidence_audit_still_required=True,shortcut_audit_still_required=True,
        published_review=str(relative),published_review_sha256=sha(review_path),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        candidate_units=len(units),excluded_units=len(units)-len(eligible),proposed_units=len(selected),
        inherited_visible_units=inherited,new_preview_units=len(preview),accepted_rows=0,
        scientific_training_allowed=False,model_predictions=0,api_calls=0))
    write(output/'DELIVERY.json',{str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()})
    print(json.dumps(dict(output=str(output),proposed_units=len(selected),inherited_visible_units=inherited,new_preview_units=len(preview))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,action='append',required=True)
    p.add_argument('--review',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.parent,a.review,a.out)
