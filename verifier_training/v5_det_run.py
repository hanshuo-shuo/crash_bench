"""V5 detector baseline on Quest (docs/generative_verifier/V5_DET_PROTOCOL.md): GroundingDINO plus P's program, no training.

Stages (openpi environment): preflight (CPU; true points equal to S4's, image hashes, the query list QUERIES.json, the P and
A records), report (CPU; fractions chosen on the training rows, readings D1-D8 and the preregistered reading). The GPU
stage between them is v5_det_detect.py in the AEGIS container. No stage loads test or test_natural.
"""
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from collections import Counter,defaultdict
import numpy as np
from .common import read,write,sha,require
from .v5_contract import validate
from .v5_pos_contract import image_points,anchors,VIEW
from .v5_seeds_contract import ALL_SEEDS,REFERENCE_SEED
from .v5_det_contract import (configuration,caption,floor_point,centre,inside,aegis_box,image_points as det_points,row_program,
    choose_fractions,det_reading,REFERENCE_RULES,SPLITS,S3_FILES,S4_FILES,SEEDS_FILES)
from . import v5_metrics as M
from verifier_dataset.v5_geometry import phrase

TESTS=['verifier_training.tests.test_v5_det_contract','verifier_training.tests.test_v5_pos_contract','verifier_training.tests.test_v5_metrics']


def context(root):
    receipt=read(root/'RECEIPT.json');config=read(root/'config.json')
    require(config==configuration() and sha(root/'config.json')==receipt['configuration_sha256'],'Fixed detector configuration changed')
    require(os.environ['CB_CODE_COMMIT']==receipt['source_commit'],'Frozen detector source differs')
    roots={k:Path(receipt[k+'_root']) for k in ('s3','s4','seeds')}
    for key,names in (('s3',S3_FILES),('s4',S4_FILES),('seeds',SEEDS_FILES)):
        require(set(receipt[key+'_files'])==set(names),'Evidence list differs: '+key)
        for name,digest in receipt[key+'_files'].items():require(sha(roots[key]/name)==digest,'Evidence changed: %s %s'%(key,name))
    base=config['seeds_configuration']
    require(read(roots['seeds']/'config.json')==base and read(roots['s3']/'config.json')==base['s3_configuration']
            and read(roots['s4']/'config.json')==base['s4_configuration'],'Prior configuration differs')
    dataset=Path(receipt['dataset']);require(str(dataset)==config['dataset'],'Unexpected dataset root')
    rows=validate(dataset,include_test=False)
    # A smoke receipt (never written by the launcher) names a few development images; the real run reads every one.
    if 'smoke_images' in receipt:keep=set(receipt['smoke_images']);rows=[r for r in rows if r['metadata']['image_id'] in keep]
    require({r['split'] for r in rows}==set(SPLITS),'Unexpected splits loaded')
    return receipt,config,dataset,roots,rows


def split(rows,name):
    return [r for r in rows if r['split']==name]


def by_image(rows):
    groups={}
    for r in rows:groups.setdefault(r['metadata']['image_id'],[]).append(r)
    return groups


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def prior_records(roots,validation):
    """Validation records of A and P for every seed (713 from S3 and S4, the others from the seed campaign), on the given rows."""
    dirs={REFERENCE_SEED:dict(A=roots['s3']/'A_seed713',P=roots['s4']/'P_seed713')}
    for seed in ALL_SEEDS[1:]:dirs[seed]=dict(A=roots['seeds']/f'A_seed{seed}',P=roots['seeds']/f'P_seed{seed}')
    keep={r['sample_id'] for r in validation}
    return {seed:{arm:[r for r in lines(d/'validation.jsonl') if r['sample_id'] in keep] for arm,d in arms.items()} for seed,arms in dirs.items()}


def check_prior(prior,validation):
    truth={r['sample_id']:(int(r['target']['digit']),r['metadata']['image_id']) for r in validation}
    for seed,arms in prior.items():
        for arm,records in arms.items():
            require(len(records)==len(truth) and {r['sample_id'] for r in records}==set(truth),'Records cover other rows: %s %d'%(arm,seed))
            require(all((r['truth'],r['image_id'])==truth[r['sample_id']] and not r['forced_prefix'] for r in records),'Records differ from the rows')


def preflight(root):
    receipt,config,dataset,roots,rows=context(root);out=root/'preflight';out.mkdir()
    subprocess.run([sys.executable,'-m','unittest',*TESTS],check=True)
    points={};queries=[];splits={}
    for image,members in sorted(by_image(rows).items()):
        points[image]=image_points(dataset,image);parts={r['split'] for r in members};require(len(parts)==1,'Image in two splits: '+image)
        spec=members[0]['student']['images'][0];require(spec['camera']==VIEW and all(r['student']['images'][0]==spec for r in members),'View 1 differs')
        path=dataset/spec['path'];require(sha(path)==spec['sha256'],'Image changed: '+image);splits[image]=parts.pop()
        for entity in sorted({'target','goal'}|{a for r in members for a in anchors(r)}):
            queries.append(dict(query_id='q%05d'%len(queries),image_id=image,split=splits[image],entity=entity,caption=caption(entity),
                path=str(path),sha256=spec['sha256'],width=spec['width'],height=spec['height']))
        require(all(row_program(r,points[image]['points'])==int(r['target']['digit']) for r in members),'Program on the true points disagrees: '+image)
    s4_points=read(roots['s4']/'POINTS.json')
    if 'smoke_images' in receipt:s4_points={k:v for k,v in s4_points.items() if k in points}
    require(json.loads(json.dumps(points))==s4_points,'Projected points differ from the S4 points')
    validation=split(rows,'validation');check_prior(prior_records(roots,validation),validation)
    write(root/'QUERIES.json',queries)
    write(out/'COMPLETE.json',dict(passed=True,rows=len(rows),by_split=dict(Counter(r['split'] for r in rows)),images=len(points),
        images_by_split=dict(Counter(splits.values())),queries=len(queries),queries_by_split=dict(Counter(q['split'] for q in queries)),
        queries_by_entity=dict(Counter('anchor' if q['entity'] not in ('target','goal') else q['entity'] for q in queries)),
        queries_sha256=sha(root/'QUERIES.json'),points_equal_s4=True,program_agrees_with_every_label=True,prior_records_checked=True,
        source_commit=receipt['source_commit'],config_sha256=sha(root/'config.json'),s4_points_sha256=sha(roots['s4']/'POINTS.json'),
        test_rows_loaded=0,training=False,model_predictions=0,packages={k:metadata.version(k) for k in ('numpy',)},
        smoke='smoke_images' in receipt,slurm_job=os.environ['SLURM_JOB_ID']))


def gate(root,receipt):
    done=read(root/'preflight/COMPLETE.json');require(done['passed'] and done['source_commit']==receipt['source_commit'],'Detector preflight gate failed')
    require(sha(root/'QUERIES.json')==done['queries_sha256'] and sha(root/'config.json')==done['config_sha256'],'Preflight evidence changed')
    found=read(root/'detect/COMPLETE.json');require(found['passed'] and found['source_commit']==receipt['source_commit'],'Detection incomplete')
    require(sha(root/'detect/detections.jsonl')==found['detections_sha256'],'Detections changed')
    queries=read(root/'QUERIES.json');records=lines(root/'detect/detections.jsonl')
    require([r['query_id'] for r in records]==[q['query_id'] for q in queries]==sorted({q['query_id'] for q in queries}),'Detections do not match the queries')
    require(all((r['image_id'],r['entity'],r['caption'])==(q['image_id'],q['entity'],q['caption']) for r,q in zip(records,queries)),'Detection fields differ')
    return done,found,records


def boxes(records,choose):
    out=defaultdict(dict)
    for r in records:out[r['image_id']][r['entity']]=choose(r)
    return dict(out)


def verdicts(rows,chosen,static,target,lookup):
    points={i:det_points(b,static,target) for i,b in chosen.items()};out=[]
    for row in rows:
        record=M.row_record(row,'detector','direct',None,None,lookup.get(row['sample_id']))
        record.update(predicted=row_program(row,points[row['metadata']['image_id']]),p_feasible=None,format_valid=True);out.append(record)
    return out


def entity_stats(records,truth,static,target,images):
    """D3 on the given images: hit rate of the chosen box centre, floor-point error and the chosen score, by object kind and name."""
    groups=defaultdict(list)
    for r in records:
        if r['image_id'] not in images:continue
        k=r['entity'];box=r['top']['box'];t=truth[r['image_id']]
        point=floor_point(box,target if k=='target' else static);error=float(np.hypot(point[0]-t['points'][k][0],point[1]-t['points'][k][1]))
        item=dict(hit=inside(centre(box),t['boxes_2d'][k]),error=error,score=r['top']['score'],above=len(r['above']))
        kind='anchor' if k not in ('target','goal') else k;groups[kind].append(item)
        if kind=='anchor':groups['anchor: '+phrase(k)].append(item)
    def stats(items):
        e=[i['error'] for i in items]
        return dict(queries=len(items),hit_rate=float(np.mean([i['hit'] for i in items])),median_error_px=float(np.median(e)),
            within_15px=float(np.mean([x<=15 for x in e])),within_30px=float(np.mean([x<=30 for x in e])),
            median_score=float(np.median([i['score'] for i in items])),none_above_aegis_threshold=float(np.mean([i['above']==0 for i in items])))
    return {k:stats(v) for k,v in sorted(groups.items())}


def decompose(scored,records,truth):
    """D8: wrong rows with at least one object missed versus wrong rows where all four boxes hit their objects."""
    hits={(r['image_id'],r['entity']):inside(centre(r['top']['box']),truth[r['image_id']]['boxes_2d'][r['entity']]) for r in records}
    wrong=[r for r in scored if r['predicted']!=r['truth']];missed=0
    for r in wrong:
        a,b=r['rule'].split(':',1)[1].split('|');missed+=not all(hits[(r['image_id'],k)] for k in (a,b,'target','goal'))
    return dict(wrong_rows=len(wrong),some_object_missed=missed,all_four_hit=len(wrong)-missed)


def report(root):
    """CPU: fractions on the training rows, then every reading on the validation splits, and the preregistered reading."""
    receipt,config,dataset,roots,rows=context(root);done,found,records=gate(root,receipt)
    truth=read(roots['s4']/'POINTS.json');lookup={k:v for part in read(roots['s3']/'LOOKUP.json')['predictions'].values() for k,v in part.items()}
    train=split(rows,'train');validation=split(rows,'validation');natural=split(rows,'validation_natural')
    top=boxes(records,lambda r:r['top']['box']);aegis=boxes(records,aegis_box)
    train_images={r['metadata']['image_id'] for r in train};validation_images={r['metadata']['image_id'] for r in validation}
    fractions=choose_fractions(train,{i:top[i] for i in train_images},truth);fs,ft=fractions['static'],fractions['target']
    scored={name:verdicts(part,top,fs,ft,lookup) for name,part in (('train',train),('validation',validation),('validation_natural',natural))}
    scores={name:M.score(part,'direct') for name,part in scored.items()}
    d=scores['validation']['verdict_accuracy']
    prior=prior_records(roots,validation);check_prior(prior,validation);ids=sorted(r['sample_id'] for r in validation);smoke='smoke_images' in receipt
    keyed={seed:{arm:{r['sample_id']:r for r in rs} for arm,rs in arms.items()} for seed,arms in prior.items()}
    p=np.asarray([[keyed[s]['P'][i]['program']==keyed[s]['P'][i]['truth'] for i in ids] for s in ALL_SEEDS],dtype=float).mean(0)
    a=np.asarray([[keyed[s]['A'][i]['predicted']==keyed[s]['A'][i]['truth'] for i in ids] for s in ALL_SEEDS],dtype=float).mean(0)
    det={r['sample_id']:r for r in scored['validation']};dc=np.asarray([det[i]['predicted']==det[i]['truth'] for i in ids],dtype=float)
    images=[det[i]['image_id'] for i in ids];sources=[det[i]['source_id'] for i in ids];diff=p-dc
    seeds=read(roots['seeds']/'SEEDS_READING.json')['seed_mean']
    require(smoke or (abs(p.mean()-seeds['P_mean'])<1e-12 and abs(a.mean()-seeds['A_mean'])<1e-12),'P or A records differ from the seed reading')
    interval=M.cluster_interval(diff,images);reading=det_reading(d,interval)
    bins=defaultdict(list)
    for i,v,w in zip(ids,p,dc):bins[det[i]['time_bin']].append((v,w))
    by_bin={str(k):dict(rows=len(v),P=float(np.mean([x for x,_ in v])),detector=float(np.mean([y for _,y in v]))) for k,v in sorted(bins.items())}
    aegis_scored=verdicts(validation,aegis,fs,ft,lookup)
    missing=[r for r in records if r['image_id'] in validation_images and not r['above']]
    reference={name:M.score(verdicts(validation,top,s,t,lookup),'direct',bootstrap=False)['verdict_accuracy'] for name,(s,t) in REFERENCE_RULES.items()}
    chosen=set(read(roots['s3']/'SUBSETS.json')['selection']);selection=[r for r in scored['validation'] if r['sample_id'] in chosen]
    require(smoke or len(selection)==config['seeds_configuration']['s3_configuration']['selection']['rows'],'Selection rows missing')
    zero=read(roots['s4']/'zero_shot/SUMMARY.json')['models']
    write(root/'DET_READING.json',dict(passed=True,reading=reading,D1=dict(detector=d,detector_interval=scores['validation']['image_bootstrap_95pct']['verdict_accuracy'],
            P_seed_mean=float(p.mean()),A_seed_mean=float(a.mean()),P_minus_detector=float(diff.mean()),image_bootstrap_95pct=interval,
            source_bootstrap_95pct_reference=M.cluster_interval(diff,sources),rows=len(ids),images=len(set(images))),
        D2_twin_image_all_correct=scores['validation']['twin_image_all_correct'],
        D3_objects=entity_stats(records,truth,fs,ft,validation_images),D4_validation_natural=scores['validation_natural']['verdict_accuracy'],
        D5_aegis_rule=dict(accuracy=M.score(aegis_scored,'direct',bootstrap=False)['verdict_accuracy'],
            rows_unanswered=sum(r['predicted'] is None for r in aegis_scored)/len(aegis_scored),
            queries_without_box=len(missing)/sum(r['image_id'] in validation_images for r in records)),
        D6=dict(reference_rules=reference,calibration=fractions),
        D7_selection_subset=dict(detector=sum(r['predicted']==r['truth'] for r in selection)/len(selection) if selection else None,rows=len(selection),
            zero_shot_pointing={k:v['program_accuracy'] for k,v in zero.items()}),
        D8=dict(errors=decompose(scored['validation'],records,truth),by_time_bin=by_bin,strata=scores['validation']['strata']),
        train_accuracy_at_chosen_fractions=fractions['training_accuracy'],scores=scores,fractions=dict(static=fs,target=ft),
        rules=config['rules'],thresholds=config['thresholds'],detection=dict(queries=found['queries'],mean_query_seconds=found['mean_query_seconds'],
            device=found['device'],repeat_all_scores_identical=found['repeat_all_scores_identical'],packages=found['packages']),
        source_commit=receipt['source_commit'],smoke=smoke,test_rows_loaded=0,training=False,api_calls=0,slurm_job=os.environ.get('SLURM_JOB_ID')))
    print(reading,json.dumps(dict(detector=d,P=float(p.mean()),difference=float(diff.mean()),interval=interval,fractions=[fs,ft])),flush=True)


if __name__=='__main__':
    stage=sys.argv[1];root=Path(sys.argv[2])
    try:
        if stage=='preflight':preflight(root)
        elif stage=='report':report(root)
        else:raise ValueError('Unknown stage')
    except BaseException as error:
        write(root/('STOP_'+stage+'.json'),dict(error=type(error).__name__,message=str(error),slurm_job=os.environ.get('SLURM_JOB_ID')))
        raise
