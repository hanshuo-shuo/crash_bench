"""V5 S4 zero-shot pointing (docs/generative_verifier/V5_S4_PROTOCOL.md): can the base VLMs locate the objects P must write?

Runs in the qwen-vlm environment (transformers 5.x) on two GPUs: the staged Qwen2.5-VL-3B, then the local Qwen3-VL-32B.
Each model sees View 1 of the 192 selection-subset images and is asked for one point per object (pudding, basket, each
anchor named by the two selected rules). A point hits when it falls inside the object's projected 2D box widened by 4
pixels; the program then decides each selection row from the predicted points. No adapter, no API, no test row.
"""
import gc
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
import time
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration,Qwen3VLForConditionalGeneration
from .common import read,write,sha,require
from .v5_contract import validate
from .v5_pos_contract import (configuration,image_points,anchors,entity_phrase,point_prompt,parse_point,program_separated,S3_FILES,SIZE,VIEW)
from .v5_zero_shot_32b import SNAPSHOT,snapshot_manifest
from . import v5_metrics as M

MARGIN=4


def context(root):
    receipt=read(root/'RECEIPT.json');config=read(root/'config.json')
    require(config==configuration() and sha(root/'config.json')==receipt['configuration_sha256'],'Fixed S4 configuration changed')
    require(os.environ['CB_CODE_COMMIT']==receipt['source_commit'],'Frozen S4 source differs')
    s3=Path(receipt['s3_root']);require(set(receipt['s3_files'])==set(S3_FILES),'S3 evidence list differs')
    for name,digest in receipt['s3_files'].items():require(sha(s3/name)==digest,'S3 evidence changed: '+name)
    dataset=Path(receipt['dataset']);assets=Path(receipt['assets'])
    staged=read(assets/'STAGED.json');require(sha(assets/'STAGED.json')==receipt['assets_stage_sha256'],'Model staging manifest changed')
    for name,digest in staged['files'].items():require(sha(assets/'model'/name)==digest,'Staged base model changed')
    chosen=set(read(s3/'SUBSETS.json')['selection'])
    rows=[r for r in validate(dataset,include_test=False) if r['split']=='validation' and r['sample_id'] in chosen]
    require(len(rows)==config['s3_configuration']['selection']['rows'],'Selection rows missing')
    return receipt,config,dataset,assets,staged,rows


def generate(model,processor,image,prompt,max_new):
    messages=[dict(role='user',content=[dict(type='image'),dict(type='text',text=prompt)])]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs=processor(text=[text],images=[image],return_tensors='pt').to(model.device)
    torch.cuda.synchronize();start=time.monotonic()
    with torch.no_grad():
        out=model.generate(**inputs,max_new_tokens=max_new,do_sample=False,temperature=None,top_p=None,top_k=None,repetition_penalty=1.,
            pad_token_id=processor.tokenizer.pad_token_id)
    torch.cuda.synchronize()
    tokens=out[0,inputs['input_ids'].shape[1]:]
    return processor.tokenizer.decode(tokens,skip_special_tokens=True),time.monotonic()-start,inputs['image_grid_thw'].tolist()


def inside(point,box,margin=MARGIN):
    return point is not None and box[0]-margin<=point[0]<=box[2]+margin and box[1]-margin<=point[1]<=box[3]+margin


def run_model(name,loader,folder,scale,alternative,rows,dataset,config,out):
    started=time.monotonic();processor=AutoProcessor.from_pretrained(folder,local_files_only=True)
    model=loader.from_pretrained(folder,local_files_only=True,dtype=torch.bfloat16,device_map='auto',attn_implementation='sdpa');model.eval()
    loaded=time.monotonic()-started;images={};records=[]
    for r in rows:images.setdefault(r['metadata']['image_id'],[]).append(r)
    stream=out/(name.split('/')[-1]+'.jsonl');stream.touch(exist_ok=False);truth={}
    for image_id,members in sorted(images.items()):
        spec=members[0]['student']['images'][0];require(spec['camera']==VIEW,'View 1 is not overview_a')
        path=Path(dataset)/spec['path'];require(sha(path)==spec['sha256'],'Image changed')
        with Image.open(path) as im:image=im.convert('RGB')
        require(image.size==(SIZE,SIZE),'Unexpected image size');truth[image_id]=image_points(dataset,image_id)
        entities=sorted({'target','goal'}|{a for r in members for a in anchors(r)})
        for entity in entities:
            text,seconds,grid=generate(model,processor,image,point_prompt(entity_phrase(entity)),config['zero_shot']['max_new_tokens'])
            raw=parse_point(text);point=[raw[0]*scale,raw[1]*scale] if raw else None;other=[raw[0]*alternative,raw[1]*alternative] if raw else None
            box=truth[image_id]['boxes_2d'][entity];floor=truth[image_id]['points'][entity]
            record=dict(model=name,image_id=image_id,entity=entity,prompt=point_prompt(entity_phrase(entity)),generated=text,raw=raw,point=point,
                hit=inside(point,box),hit_alternative_scale=inside(other,box),box_2d=box,floor_point=floor,
                floor_distance=None if point is None else float(((point[0]-floor[0])**2+(point[1]-floor[1])**2)**.5),
                visible_pixels=truth[image_id]['visibility'].get(entity),seconds=seconds,image_grid=grid)
            records.append(record)
            with stream.open('a') as f:f.write(json.dumps(record)+'\n')
    points={(r['image_id'],r['entity']):r['point'] for r in records};verdicts=[]
    for r in rows:
        a,b=anchors(r);q=[points[(r['metadata']['image_id'],k)] for k in (a,b,'target','goal')]
        program=None if None in q or q[0]==q[1] else (0 if program_separated(*q) else 1)
        verdicts.append(dict(sample_id=r['sample_id'],image_id=r['metadata']['image_id'],source_id=r['metadata']['source_id'],
            truth=int(r['target']['digit']),program=program))
    kinds={'target':['target'],'goal':['goal'],'anchor':[e for e in {x['entity'] for x in records} if e not in ('target','goal')]}
    correct=[v['program']==v['truth'] for v in verdicts]
    summary=dict(model=name,load_seconds=loaded,queries=len(records),parsed=sum(r['raw'] is not None for r in records)/len(records),
        hit_rate={k:sum(r['hit'] for r in records if r['entity'] in v)/max(1,sum(r['entity'] in v for r in records)) for k,v in kinds.items()},
        hit_rate_alternative_scale={k:sum(r['hit_alternative_scale'] for r in records if r['entity'] in v)/max(1,sum(r['entity'] in v for r in records))
            for k,v in kinds.items()},
        median_floor_distance_px={k:_median([r['floor_distance'] for r in records if r['entity'] in v and r['floor_distance'] is not None]) for k,v in kinds.items()},
        program_accuracy=sum(correct)/len(correct),program_unparsed=sum(v['program'] is None for v in verdicts),
        program_feasible_rate=sum(v['program']==1 for v in verdicts)/len(verdicts),
        program_image_bootstrap_95pct=M.cluster_interval([float(c) for c in correct],[v['image_id'] for v in verdicts]),
        coordinate_scale=scale,elapsed_seconds=time.monotonic()-started,
        peak_gpu_gib=[torch.cuda.max_memory_allocated(i)/2**30 for i in range(torch.cuda.device_count())])
    with (out/(name.split('/')[-1]+'_verdicts.jsonl')).open('w') as f:f.writelines(json.dumps(v)+'\n' for v in verdicts)
    del model;gc.collect();torch.cuda.empty_cache();return summary


def _median(values):
    values=sorted(values);return values[len(values)//2] if values else None


def main(root,limit=None):
    receipt,config,dataset,assets,staged,rows=context(root);out=root/'zero_shot';out.mkdir()
    require(torch.cuda.device_count()>=1,'Quest GPU required');torch.manual_seed(1014)
    if limit:keep=sorted({r['metadata']['image_id'] for r in rows})[:limit];rows=[r for r in rows if r['metadata']['image_id'] in keep]
    models=[('Qwen/Qwen2.5-VL-3B-Instruct',Qwen2_5_VLForConditionalGeneration,assets/'model',1.,SIZE/1000),
            ('Qwen/Qwen3-VL-32B-Instruct',Qwen3VLForConditionalGeneration,SNAPSHOT,SIZE/1000,1.)]
    summaries={}
    for name,loader,folder,scale,alternative in models:
        summaries[name]=run_model(name,loader,folder,scale,alternative,rows,dataset,config,out)
        write(out/'SUMMARY.json',dict(passed=False,models=summaries));print(name,json.dumps({k:summaries[name][k] for k in ('hit_rate','program_accuracy')}),flush=True)
    write(out/'SUMMARY.json',dict(passed=True,models=summaries,rows=len(rows),images=len({r['metadata']['image_id'] for r in rows}),limit=limit,
        staged_3b_revision=staged['model_revision'],snapshot_32b=str(SNAPSHOT),snapshot_32b_files=snapshot_manifest(),
        devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        packages={k:metadata.version(k) for k in ('torch','transformers','accelerate','numpy','pillow')},
        source_commit=receipt['source_commit'],test_rows_loaded=0,api_calls=0,slurm_job=os.environ.get('SLURM_JOB_ID')))


if __name__=='__main__':
    root=Path(sys.argv[1])
    try:main(root)
    except BaseException as error:
        write(root/'STOP_zero_shot.json',dict(error=type(error).__name__,message=str(error),slurm_job=os.environ.get('SLURM_JOB_ID')))
        raise
