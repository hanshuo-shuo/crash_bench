"""Read-only evidence audit and review pack; this never authorizes training."""
from collections import Counter,defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
from .inventory import read,sha,write
from .constraint_evidence import verify_trace,verify_layout,rows as regenerate_rows
from .constraint_selection import check_unit,select_units


def check_hashes(folder):
    for name,wanted in read(folder/'HASHES.json').items():
        if sha(folder/name)!=wanted:raise RuntimeError('Evidence changed: '+str(folder/name))


def parent_accounting(root,parent):
    """Use a terminal Slurm receipt captured before submitting the offline audit."""
    matches=[x for x in read(root/'DEPENDENCY_ACCOUNTING.json') if x['root']==str(parent)]
    if len(matches)!=1:raise RuntimeError('Missing or repeated dependency accounting')
    record=matches[0];receipt=read(parent/'RECEIPT.json');job=receipt['job']
    if record['job']!=job or record['accounting']!=job+'|COMPLETED|0:0':
        raise RuntimeError('Dependency did not terminate normally')
    if record['receipt_sha256']!=sha(parent/'RECEIPT.json') or record['completion_sha256']!=sha(parent/'CONSTRAINT_COMPLETE.json'):
        raise RuntimeError('Dependency receipt or completion changed after submission')
    return record['accounting']


def visibility_check(option,layout):
    import numpy as np
    from PIL import Image
    vis=read(option/'VISIBILITY.json');images=read(option/'IMAGES.json')['assignments']
    if layout.get('camera_scheme')=='balanced_two_overviews_v2':
        from .constraint_visuals import balanced_poses
        camera=read(option/'CAMERA.json');views=camera['rendered_views']
        if camera['camera_scheme']!=layout['camera_scheme'] or camera['native_wrist_used'] or [v['camera'] for v in views]!=['overview_a','overview_b']:
            raise RuntimeError('Recorded overview camera identities differ')
        expected=balanced_poses(layout,views[0]['pose']['fovy'])
        for actual,wanted in zip(views,expected):
            for key in ('position','rotation','focus','distance','fovy','maximum_projected_abs'):
                if not np.allclose(actual['pose'][key],wanted[key],rtol=0,atol=1e-12):
                    raise RuntimeError('Camera pose differs from the label-independent framing rule')
        for group in images:
            if [im['camera'] for im in group]!=['overview_a','overview_b']:raise RuntimeError('Student images are not the declared two overviews')
    for camera in range(2):
        a,b=[option/vis['assignments'][assignment][camera]['segmentation_path'] for assignment in (0,1)]
        with np.load(a,allow_pickle=False) as left,np.load(b,allow_pickle=False) as right:
            if set(left.files)!=set(right.files) or any(not np.array_equal(left[k],right[k]) for k in left.files):
                raise RuntimeError('Color swap changed segmentation or camera geometry')
    for assignment in range(2):
        for image,view in zip(images[assignment],vis['assignments'][assignment]):
            if sha(option/image['path'])!=image['sha256'] or sha(option/view['segmentation_path'])!=view['segmentation_sha256']:
                raise RuntimeError('Visibility/image evidence changed')
            with Image.open(option/image['path']) as im:
                if im.size!=(224,224) or im.mode!='RGB':raise RuntimeError('Unexpected final student image')
    from .constraint_visuals import prescreen
    return prescreen(vis)


def contact_sheets(root,units):
    from PIL import Image,ImageDraw
    records=[];directory=root/'review_sheets';directory.mkdir()
    for page,start in enumerate(range(0,len(units),5)):
        canvas=Image.new('RGB',(920,5*280),'white');draw=ImageDraw.Draw(canvas)
        members=[]
        for local,unit in enumerate(units[start:start+5]):
            i=start+local;row=unit[0];x=0;y=local*280
            pictures=unit[0]['student']['images']+unit[2]['student']['images']
            for view,im in enumerate(pictures):
                with Image.open(im['path']) as pic:canvas.paste(pic,(x+view*224,y))
            meta=row['metadata'];label='%03d %s t%d'%(i,meta['source_id'],meta['checkpoint'])
            draw.text((x,y+228),label,fill='black')
            draw.text((x,y+244),'Check target, both virtual boxes, and green destination.',fill='black')
            draw.text((x,y+258),'Same capture: protect orange / protect purple.',fill='black')
            members.append(dict(unit_index=i,evidence_root=meta['evidence_root'],
                source_id=meta['source_id'],checkpoint=meta['checkpoint'],
                image_sha256=[im['sha256'] for im in pictures]))
        path=directory/('sheet_%02d.png'%page);canvas.save(path)
        records.append(dict(path=str(path.relative_to(root)),sha256=sha(path),units=members))
    write(root/'VISUAL_REVIEW_INDEX.json',records)


def main(root,parents):
    import numpy as np
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Audit belongs on a compute node')
    frozen_path=Path(__file__).with_name('frozen_sources_67.json');frozen=read(frozen_path)
    sources={r['source_id']:r for r in frozen['sources']};units=[];rejected=[];inheritance=[]
    seen_sources=set();trace_checks=[]
    for parent in parents:
        receipt=read(parent/'RECEIPT.json');job=receipt['job']
        if receipt['stage'] not in ('constraint-batch','constraint-prototype'):
            raise RuntimeError('Unexpected constraint collection stage')
        account=parent_accounting(root,parent)
        if account!=job+'|COMPLETED|0:0' or not read(parent/'CONSTRAINT_COMPLETE.json')['passed']:
            raise RuntimeError('Every collection process must finish normally')
        check_hashes(parent);inheritance.append(dict(root=str(parent),receipt_sha256=sha(parent/'RECEIPT.json'),accounting=account))
        for case in sorted(parent.glob('case_*')):
            complete=read(case/'COMPLETE.json');source=complete['source'];sid=source['source_id']
            if sid in seen_sources:raise RuntimeError('Repeated original source inflates collection')
            seen_sources.add(sid)
            if sid not in sources:raise RuntimeError('Source was not assigned before model fitting')
            trace=case/'execution'
            if not (trace/'TRACE_VERIFIED.json').exists():
                rejected.append(dict(source_id=sid,reason=complete.get('reason','No completed trace')));continue
            check_hashes(trace)
            with gzip.open(trace/'SAMPLES.jsonl.gz','rt') as f:samples=[json.loads(line) for line in f]
            actions=read(trace/'ACTIONS.json');geometry=read(trace/'GEOMETRY.json')
            check=verify_trace(samples,actions,geometry)
            if check!=read(trace/'TRACE_VERIFIED.json'):raise RuntimeError('Persisted trace grade changed')
            old=Path(source['root'])/'steps.jsonl'
            if sha(old)!=source['tape_sha256']:raise RuntimeError('Original source tape changed')
            original=[json.loads(line) for line in old.read_text().splitlines()]
            if len(actions)!=len(original):raise RuntimeError('Incomplete source trajectory')
            for action,command in zip(actions,original):
                if action['step']!=command['step'] or action['actual_physics_sha256']!=command['physics_after_sha256'] or action['action']!=np.clip(command['applied'],-1,1).tolist():
                    raise RuntimeError('Independent action/hash comparison failed')
            trace_checks.append(dict(source_id=sid,root=str(trace),hashes_sha256=sha(trace/'HASHES.json'),**check))
            for candidate in complete['rows']:
                if not candidate['selected_first_valid_option']:continue
                option=Path(candidate['option']);check_hashes(option)
                layout=read(option/'LAYOUT.json');proof=verify_layout(layout,samples,actions)
                for key in ('object','geom_name','geom_type','primitive_size','radius','root_name'):
                    if layout['material'][key]!=geometry['material'][key]:
                        raise RuntimeError('Certificate material differs from the recorded trajectory target')
                if proof!=read(option/'VERIFIED.json') or not proof['matched_pair']:
                    raise RuntimeError('Independent constraint certificate/witness failed')
                images=read(option/'IMAGES.json')['assignments'];fresh=regenerate_rows(layout,proof,source,images)
                records=read(option/'CANDIDATES.json')
                for actual,expected in zip(records,fresh):
                    for key in ('student','evidence','teacher'):
                        value=json.loads(json.dumps(actual[key]))
                        if key=='student':
                            for image in value['images']:image['path']=str(Path(image['path']).relative_to(option))
                        if value!=expected[key]:raise RuntimeError('Candidate or reasoning draft does not match its evidence')
                    m=actual['metadata'];split_source=sources[m['source_id']]
                    for key,value in expected['metadata'].items():
                        if m.get(key)!=value:raise RuntimeError('Candidate target, role or source metadata differs from evidence')
                    for key in ('ancestry_component','task_group','family'):
                        if m[key]!=split_source[key]:raise RuntimeError('Frozen source ancestry changed')
                    actual['split']=split_source['split']
                check_unit(records);quality=visibility_check(option,layout)
                if not quality['quantitative_prescreen']:
                    rejected.append(dict(source_id=sid,checkpoint=layout['checkpoint'],evidence_root=str(option),**quality));continue
                for row in records:row['metadata']['visibility_prescreen']=quality
                units.append(records)
    write(root/'INHERITED.json',inheritance);write(root/'TRACE_AUDIT.json',trace_checks)
    write(root/'REJECTED.json',rejected)
    write(root/'ALL_VALID_UNITS.json',units)
    eligible=units
    if (root/'EXCLUSION_REVIEW_SOURCE.json').exists():
        from .constraint_curation import exclude
        provenance=read(root/'EXCLUSION_REVIEW_SOURCE.json');path=root/'source'/provenance['relative_path']
        if sha(path)!=provenance['sha256']:raise RuntimeError('Published exclusion review changed')
        review=read(path);eligible=exclude(units,review);write(root/'EXCLUSION_REVIEW.json',review)
    selected=select_units(eligible,target=250)
    records=[];complete_inputs=set();image_splits={}
    for i,unit in enumerate(selected):
        for r in unit:
            key=json.dumps(dict(images=[im['sha256'] for im in r['student']['images']],
                task=r['student']['task'],constraints=r['student']['constraints']),sort_keys=True)
            if key in complete_inputs:raise RuntimeError('Duplicate complete observations inflate the final dataset')
            complete_inputs.add(key)
            pair=tuple(im['sha256'] for im in r['student']['images'])
            if pair in image_splits and image_splits[pair]!=r['split']:raise RuntimeError('Shared image pair crosses frozen source splits')
            image_splits[pair]=r['split']
            r['metadata']['screened_student_input_sha256']=hashlib.sha256(key.encode()).hexdigest()
            r['sample_id']='fd1000_%04d'%len(records);r['metadata']['unit_id']='unit_%03d'%i;records.append(r)
    coverage=dict(rows=len(records),physical_checkpoint_units=len(selected),original_sources=len({r['metadata']['source_id'] for r in records}),
        ancestry_groups=len({r['metadata']['ancestry_component'] for r in records}),
        native_task_compositions=len({r['metadata']['task_group'] for r in records}),families=len({r['metadata']['family'] for r in records}),
        by_split=dict(Counter(r['split'] for r in records)),by_label=dict(Counter(r['evidence']['feasibility'] for r in records)),
        by_source=dict(Counter(r['metadata']['source_id'] for r in records)),
        by_task=dict(Counter(r['metadata']['task_group'] for r in records)),by_family=dict(Counter(r['metadata']['family'] for r in records)))
    if coverage['original_sources']<30 or coverage['ancestry_groups']<30 or coverage['native_task_compositions']<6 or coverage['families']<3:
        raise RuntimeError('Selected units do not meet the frozen source-coverage requirement')
    contact_sheets(root,selected)
    (root/'PROPOSED_DATASET.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records))
    write(root/'COVERAGE.json',coverage)
    from .constraint_shortcuts import main as shortcuts
    shortcut=shortcuts(root,records)
    write(root/'AUDIT_COMPLETE.json',dict(passed=True,stage='evidence_and_visibility_prescreen',coverage=coverage,
        available_valid_units=len(units),frozen_sources_sha256=sha(frozen_path),
        eligible_after_visual_exclusions=len(eligible),visually_excluded_units=len(units)-len(eligible),
        visual_review_complete=False,shortcut_evaluation_complete=True,shortcut_screen_passed=shortcut['passed'],accepted_training_rows=0,
        scientific_training_allowed=False,api_calls=0,slurm_job=os.environ['SLURM_JOB_ID']))
    write(root/'HASHES.json',{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and 'source' not in p.relative_to(root).parts
        and p.name not in ('HASHES.json','source.tar','source.sha256') and not p.name.startswith('slurm_')})


if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]),list(map(Path,sys.argv[2:])))
