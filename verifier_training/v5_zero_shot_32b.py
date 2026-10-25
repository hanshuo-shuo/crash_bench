"""V5 S3 R6 (b): local Qwen3-VL-32B-Instruct zero-shot on the selection subset (docs/generative_verifier/V5_S3_PROTOCOL.md).

Runs in the qwen-vlm environment (transformers 5.x) on two GPUs with device_map=auto. The prompts, images
and greedy decoding match the 3B zero-shot stage; no adapter, no API call, no test row.
"""
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
import time
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from .common import read,write,sha,require
from .v5_contract import campaign,preflight_gate,zero_shot_messages
from . import v5_metrics as M

MODEL_ID='Qwen/Qwen3-VL-32B-Instruct'
SNAPSHOT=Path('/projects/p33100/siosio/huggingface_cache/hub/models--Qwen--Qwen3-VL-32B-Instruct/snapshots/0cfaf48183f594c314753d30a4c4974bc75f3ccb')
PROMPTS=(('direct','direct','Verdict: '),('facts','facts',None))


def snapshot_manifest(folder=SNAPSHOT):
    """Hub-cache blob names (sha256 for weight files) behind every snapshot file; cheap provenance without rehashing 63 GB."""
    return {p.name:dict(blob=os.path.basename(os.path.realpath(p)),bytes=p.stat().st_size) for p in sorted(folder.iterdir())}


def load_images(row,dataset):
    images=[]
    for spec in row['student']['images']:
        path=Path(dataset)/spec['path'];require(sha(path)==spec['sha256'],'Image changed')
        with Image.open(path) as im:images.append(im.convert('RGB'))
    return images


def verdict_position(tokenizer,prefix,tokens,text):
    before=text.strip().rsplit('Verdict: ',1)[0]+'Verdict: '
    hits=[i for i in range(len(tokens)) if (prefix+tokenizer.decode(tokens[:i],skip_special_tokens=True)).lstrip()==before]
    return hits[0] if len(hits)==1 else None


def generate(model,processor,messages,images,kind,prefix,max_new):
    prefix=prefix or ''
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)+prefix
    inputs=processor(text=[text],images=images,return_tensors='pt').to(model.device)
    tokenizer=processor.tokenizer;digits=[tokenizer.encode(d,add_special_tokens=False)[0] for d in ('0','1')]
    stops={tokenizer.convert_tokens_to_ids(t) for t in ('<|im_end|>','<|endoftext|>')}
    torch.cuda.synchronize();start=time.monotonic()
    with torch.no_grad():
        result=model.generate(**inputs,max_new_tokens=max_new,do_sample=False,temperature=None,top_p=None,top_k=None,
            repetition_penalty=1.,use_cache=True,return_dict_in_generate=True,output_logits=True,pad_token_id=tokenizer.pad_token_id)
    torch.cuda.synchronize();seconds=time.monotonic()-start
    tokens=result.sequences[0,inputs['input_ids'].shape[1]:].tolist();count=next((i+1 for i,t in enumerate(tokens) if t in stops),len(tokens))
    generated=prefix+tokenizer.decode(tokens[:count],skip_special_tokens=True);parsed=M.parse_output(generated,kind);p=None
    if parsed['predicted'] is not None:
        at=verdict_position(tokenizer,prefix,tokens[:count],generated)
        if at is not None and tokens[at] in digits:p=float(result.logits[at][0].float()[digits].softmax(0)[1])
    return dict(generated=generated,**parsed,p_feasible=p,generated_token_count=count,stopped=count<len(tokens) or tokens[count-1] in stops,
        prompt_tokens=int(inputs['input_ids'].shape[1]),image_grid=inputs['image_grid_thw'].tolist()),seconds


def main(root,limit=None):
    receipt,config,dataset,rows=campaign(root,splits=('validation',));hashes,subsets,lookup=preflight_gate(root,receipt)
    out=root/'zero_shot_32b';out.mkdir();require(torch.cuda.device_count()>=1,'Quest GPU required')
    chosen=set(subsets['selection']);selection=[r for r in rows if r['sample_id'] in chosen]
    require(len(selection)==config['selection']['rows'],'Selection rows missing')
    if limit:selection=selection[:limit]
    manifest=snapshot_manifest();require(json.loads((SNAPSHOT/'config.json').read_text())['model_type']=='qwen3_vl','Unexpected 32B snapshot')
    torch.manual_seed(1014);started=time.monotonic()
    processor=AutoProcessor.from_pretrained(SNAPSHOT,local_files_only=True)
    model=Qwen3VLForConditionalGeneration.from_pretrained(SNAPSHOT,local_files_only=True,dtype=torch.bfloat16,device_map='auto',
        attn_implementation='sdpa');model.eval();loaded=time.monotonic()-started
    records=[];stream=out/'records.jsonl';stream.touch(exist_ok=False)
    for row in selection:
        images=load_images(row,dataset)
        for mode,kind,prefix in PROMPTS:
            output,seconds=generate(model,processor,zero_shot_messages(row,mode),images,kind,prefix,config['maximum_new_tokens'][kind])
            record=M.row_record(row,'zero_shot_'+mode,kind,prefix,hashes[row['sample_id']],lookup['validation'][row['sample_id']])
            record.update(output,generation_seconds=seconds,gold_rationale_prefix_used=False);records.append(record)
            with stream.open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
    write(out/'SUMMARY.json',dict(passed=True,scores=M.summarize(records),model_id=MODEL_ID,snapshot=str(SNAPSHOT),snapshot_files=manifest,
        adapter=None,rows=len(selection),limit=limit,split='validation selection subset',test_rows_loaded=0,load_seconds=loaded,
        elapsed_seconds=time.monotonic()-started,devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        device_map=sorted({str(v) for v in model.hf_device_map.values()}),
        peak_gpu_gib=[torch.cuda.max_memory_allocated(i)/2**30 for i in range(torch.cuda.device_count())],
        packages={k:metadata.version(k) for k in ('torch','transformers','accelerate','numpy','pillow')},api_calls=0,
        source_commit=receipt['source_commit'],slurm_job=os.environ.get('SLURM_JOB_ID')))


if __name__=='__main__':
    root=Path(sys.argv[1])
    try:main(root)
    except BaseException as error:
        write(root/'STOP_zero_shot_32b.json',dict(error=type(error).__name__,message=str(error),slurm_job=os.environ.get('SLURM_JOB_ID')))
        raise
