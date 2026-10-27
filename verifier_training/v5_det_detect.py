"""V5 detector baseline, GPU stage (docs/generative_verifier/V5_DET_PROTOCOL.md), run in the AEGIS container (Python 3.8).

Standalone on purpose: the AEGIS environment has GroundingDINO but not the verifier packages, so this file imports nothing
from them and runs with python -I. It reads the query list written by the CPU preflight, checks every input hash, runs
GroundingDINO's own load_image and predict once per query (box threshold 0, so all 900 candidates come back in model
order) and records the top box, the candidates above the AEGIS threshold and the five best. The first queries run again
at the end and must give the identical top box. Nothing is trained; no API; no test row (the preflight wrote the list).
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

AEGIS_BOX_THRESHOLD=0.35
AEGIS_TEXT_THRESHOLD=0.25
REPEAT_QUERIES=16
BERT_FILES=('config.json','pytorch_model.bin','tokenizer.json','tokenizer_config.json','vocab.txt')


def read(path):
    return json.loads(Path(path).read_text())


def write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def require(condition,message):
    if not condition:raise ValueError(message)


def candidate(index,score,box,phrase=None):
    out=dict(index=int(index),score=float(score),box=[float(x) for x in box])
    if phrase is not None:out['phrase']=phrase
    return out


def detect(model,image,text,predict):
    """All candidates through GroundingDINO's predict; the mask at threshold 0 keeps every box in model order."""
    import torch
    boxes,scores,phrases=predict(model=model,image=image,caption=text,box_threshold=0.0,text_threshold=AEGIS_TEXT_THRESHOLD,device='cuda')
    require(boxes.shape[0]==scores.shape[0]==len(phrases)==900,'Expected all 900 candidates')
    top=int(torch.argmax(scores));best=torch.argsort(scores,descending=True)[:5].tolist()
    above=[i for i in range(len(scores)) if float(scores[i])>AEGIS_BOX_THRESHOLD]
    return dict(top=candidate(top,scores[top],boxes[top],phrases[top]),above=[candidate(i,scores[i],boxes[i],phrases[i]) for i in above],
        best5=[candidate(i,scores[i],boxes[i]) for i in best])


def main(root):
    receipt=read(root/'RECEIPT.json');done=read(root/'preflight/COMPLETE.json')
    require(os.environ['CB_CODE_COMMIT']==receipt['source_commit']==done['source_commit'] and done['passed'],'Preflight gate failed')
    require(sha(root/'QUERIES.json')==done['queries_sha256'],'Query list changed after the preflight')
    assets=receipt['detector_assets']
    require(sha(assets['checkpoint'])==assets['checkpoint_sha256'] and sha(assets['config'])==assets['config_sha256'],'Detector files changed')
    bert=Path(assets['bert']);require(bert.name==assets['bert_revision'] and all((bert/n).is_file() for n in BERT_FILES),'BERT snapshot missing')
    queries=read(root/'QUERIES.json');require(len(queries)==done['queries'] and all(not q['split'].startswith('test') for q in queries),'Query list invalid')
    out=root/'detect';out.mkdir();work=out/'work';work.mkdir();(work/'tmp').mkdir()
    os.environ['TMPDIR']=str(work/'tmp');os.chdir(work)
    # Transformers 4.21 loads bert-base-uncased from a local directory of that name (as the AEGIS runner does).
    (work/'bert-base-uncased').symlink_to(bert,target_is_directory=True)
    import torch
    from groundingdino.util.inference import load_model,load_image,predict
    require(torch.cuda.is_available(),'Quest GPU required');torch.manual_seed(1014)
    started=time.monotonic();model=load_model(assets['config'],assets['checkpoint'],device='cuda');loaded=time.monotonic()-started
    images={};records=[];stream=out/'detections.jsonl';stream.touch(exist_ok=False)
    for q in queries:
        if q['path'] not in images:
            require(sha(q['path'])==q['sha256'],'Image changed: '+q['image_id']);images[q['path']]=load_image(q['path'])
        source,image=images[q['path']];require(source.shape==(q['height'],q['width'],3),'Unexpected image size')
        t=time.monotonic();found=detect(model,image,q['caption'],predict);seconds=time.monotonic()-t
        record=dict(query_id=q['query_id'],image_id=q['image_id'],split=q['split'],entity=q['entity'],caption=q['caption'],seconds=seconds,**found)
        records.append(record)
        with stream.open('a') as f:f.write(json.dumps(record)+'\n')
        if len(records)%500==0:print(len(records),'queries',round(time.monotonic()-started),'s',flush=True)
    repeat=[]
    for q,first in zip(queries[:REPEAT_QUERIES],records):
        again=detect(model,images[q['path']][1],q['caption'],predict)
        repeat.append(dict(query_id=q['query_id'],same_index=again['top']['index']==first['top']['index'],same_box=again['top']['box']==first['top']['box'],
            same_score=again['top']['score']==first['top']['score']))
    require(all(r['same_index'] and r['same_box'] for r in repeat),'Repeated queries gave a different top box')
    import groundingdino,transformers,numpy
    write(out/'COMPLETE.json',dict(passed=True,queries=len(records),images=len(images),detections_sha256=sha(stream),repeat_check=repeat,
        repeat_all_scores_identical=all(r['same_score'] for r in repeat),load_seconds=loaded,elapsed_seconds=time.monotonic()-started,
        mean_query_seconds=sum(r['seconds'] for r in records)/len(records),device=torch.cuda.get_device_name(),
        peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,packages=dict(torch=torch.__version__,transformers=transformers.__version__,
            numpy=numpy.__version__,groundingdino=getattr(groundingdino,'__version__',None)),python=sys.version,
        source_commit=receipt['source_commit'],test_rows_loaded=0,training=False,api_calls=0,slurm_job=os.environ.get('SLURM_JOB_ID')))
    print('detections',len(records),'images',len(images),'seconds',round(time.monotonic()-started),flush=True)


if __name__=='__main__':
    root=Path(sys.argv[1])
    try:main(root)
    except BaseException as error:
        write(root/'STOP_detect.json',dict(error=type(error).__name__,message=str(error),slurm_job=os.environ.get('SLURM_JOB_ID')))
        raise
