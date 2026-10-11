"""Greedy, training-pool shortcut probes; no new data generation or fitting."""
import hashlib
import json
import os
from pathlib import Path
import time
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from .common import *
from .engineering_contract import *


def main(root, assets, variant):
    dataset=root/'dataset';out=root/('diagnostics_'+variant);out.mkdir(exist_ok=False)
    require(sha(dataset/'dataset.jsonl')==DATA_SHA,'Dataset changed')
    audit=read(root/'audit/AUDIT.json');require(audit['passed_for_engineering_only'],'Audit gate failed')
    fit=read(root/variant/'RESULT.json');require(fit['steps']==300 and fit['full_training_launched'],'Full fit required')
    stage=read(assets/'STAGED.json')
    for name,h in stage['files'].items():require(sha(assets/'model'/name)==h,'Model changed')
    rows=[json.loads(l) for l in (dataset/'dataset.jsonl').read_text().splitlines()]
    donors=shuffled_donors(len(rows));write(out/'DONORS.json',{r['sample_id']:rows[d]['sample_id'] for r,d in zip(rows,donors)})
    processor=AutoProcessor.from_pretrained(assets/'model',local_files_only=True,min_pixels=224*224,max_pixels=224*224)
    base=Qwen2_5_VLForConditionalGeneration.from_pretrained(assets/'model',local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    model=PeftModel.from_pretrained(base,root/variant/'adapter',is_trainable=False)
    model.eval();model.requires_grad_(False)
    config=read(root/('execution_'+variant+'.json'))
    torch.manual_seed(config['seed']);torch.backends.cuda.matmul.allow_tf32=True
    provenance={'variant':variant,'code_commit':os.environ['CB_CODE_COMMIT'],'slurm_job':os.environ['SLURM_JOB_ID'],
        'dataset_sha256':DATA_SHA,'fit_provenance_sha256':sha(root/variant/'PROVENANCE.json'),
        'adapter_sha256':sha(root/variant/'adapter/adapter_model.safetensors'),
        'model_revision':stage['model_revision'],'conditions':CONDITIONS,'shuffle_seed':714,
        'scope':'Training pool sensitivity diagnostics only. No held-out or cross-scene generalization.',
        'generation':'greedy; A 12 new tokens, B 160 new tokens; frozen adapter/base; same inference fields across A/B per condition'}
    write(out/'PROVENANCE.json',provenance)
    start=time.monotonic()
    original=read(root/variant/'generations_final.json')
    annotated=[]
    for r,g in zip(rows,original):
        require(r['sample_id']==g['sample_id'],'Output alignment changed')
        annotated.append(dict(g,condition='original',expected_verdict=r['target']['verdict'],
            rationale_assessment=score_rationale(g['generated'],r,'original') if variant=='B' else None))
    write(out/'original.json',annotated)
    for condition in CONDITIONS[1:]:
        outputs=[]
        with (out/(condition+'.jsonl')).open('w') as stream, torch.no_grad():
            for i,r in enumerate(rows):
                donor=rows[donors[i]] if condition=='shuffled_pairs' else r
                changed=diagnostic_row(r,condition,donor)
                messages=diagnostic_messages(changed,variant,condition)
                prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                images=[]
                if condition!='text_only':
                    for im in changed['student']['images']:
                        p=dataset/im['path'];require(sha(p)==im['sha256'],'Image changed')
                        with Image.open(p) as image:images.append(image.convert('RGB'))
                fields=processor(text=[prompt],**({'images':images} if images else {}),return_tensors='pt',padding=False)
                digest=hashlib.sha256()
                for key in sorted(fields):
                    digest.update(key.encode());digest.update(str(fields[key].shape).encode());digest.update(fields[key].numpy().tobytes())
                n=fields['input_ids'].shape[1]
                result=model.generate(**{k:v.to('cuda') for k,v in fields.items()},max_new_tokens=160 if variant=='B' else 12,
                    do_sample=False,use_cache=True,pad_token_id=processor.tokenizer.pad_token_id)
                text=processor.tokenizer.decode(result[0,n:],skip_special_tokens=True).strip()
                pred=parse_verdict(text,variant)
                expected='infeasible' if condition=='horizon_0' else r['target']['verdict']
                record={'sample_id':r['sample_id'],'condition':condition,'donor_id':donor['sample_id'],
                    'input_sha256':digest.hexdigest(),'generated':text,'predicted_verdict':pred,
                    'format_valid':pred is not None,'expected_verdict':expected,'verdict_correct':pred==expected,
                    'donor_verdict':donor['target']['verdict'],'donor_verdict_correct':pred==donor['target']['verdict'],
                    'rationale_assessment':score_rationale(text,donor,condition) if variant=='B' else None}
                outputs.append(record);stream.write(json.dumps(record)+'\n');stream.flush()
                if (i+1)%20==0:print(condition,i+1,'elapsed',time.monotonic()-start,flush=True)
        write(out/(condition+'.json'),outputs)
    write(out/'COMPLETE.json',{'passed':True,'elapsed_seconds':time.monotonic()-start,
        'rows_per_condition':120,'new_simulation':False,'scientific_generalization_claim':False})
