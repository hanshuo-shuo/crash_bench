"""Read-only matched-prefix test of the completed binary adapter; no updates."""
import copy
import json
import os
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor
from .common import read,write,sha,require,MODEL_ID
from .binary_verifier import encode,LABEL_TO_DIGIT
from .binary_run import input_digest
from .engineering_contract import DATA_SHA

def main(parent,out):
    out.mkdir(exist_ok=False)
    receipt=read(parent/'RECEIPT.json');assets=Path(receipt['assets']);dataset=Path(receipt['campaign'])/'dataset'
    require(read(parent/'full/RESULT.json')['passed'],'Full execution incomplete')
    require(sha(dataset/'dataset.jsonl')==DATA_SHA,'Dataset changed')
    rows=[json.loads(x) for x in (dataset/'dataset.jsonl').read_text().splitlines()]
    generated=read(parent/'full/generations_final.json')
    hashes={x['sample_id']:x['input_sha256'] for x in read(parent/'preflight/ENCODINGS.json')}
    staged=read(assets/'STAGED.json');require(staged['model_id']==MODEL_ID,'Unexpected base model')
    for name,digest in staged['files'].items():require(sha(assets/'model'/name)==digest,'Base model asset changed')
    p=AutoProcessor.from_pretrained(assets/'model',local_files_only=True,min_pixels=224*224,max_pixels=224*224)
    torch.manual_seed(713);torch.backends.cuda.matmul.allow_tf32=True
    base=Qwen2_5_VLForConditionalGeneration.from_pretrained(assets/'model',local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    adapter=parent/'full/adapter';digest=sha(adapter/'adapter_model.safetensors')
    model=PeftModel.from_pretrained(base,adapter,is_trainable=False);model.requires_grad_(False);model.eval()
    ids={c:p.tokenizer.encode(v,add_special_tokens=False)[0] for c,v in LABEL_TO_DIGIT.items()}
    names={v:k for k,v in ids.items()};penalty=RepetitionPenaltyLogitsProcessor(1.05);records=[]
    with torch.no_grad():
        for row,g in zip(rows,generated):
            require(row['sample_id']==g['sample_id'] and g['format_valid'],'Invalid output alignment/format')
            own=g['generated'][len('Rationale: '):g['generated'].rfind('\nVerdict: ')]
            conditions={};cache={}
            for condition,body in [('ground_truth_rationale',row['target']['rationale']),('generated_rationale',own)]:
                changed=copy.deepcopy(row);changed['target']['rationale']=body
                item=encode(p,changed,dataset)
                require(input_digest(item)==hashes[row['sample_id']],'Student observation changed')
                original_at=int(item[3].nonzero()[0,1]);n=item[4]
                prefix_ids=(item[0]['input_ids'][0,n:original_at].tolist() if condition=='ground_truth_rationale'
                            else g['native_decision']['prefix_token_ids'])
                key=tuple(prefix_ids)
                if key in cache:conditions[condition]=cache[key];continue
                # Preserve the actual generated prefix tokenization, including any
                # alternate token boundaries. Both conditions use this same full-
                # forward construction, with only prefix tokens allowed to differ.
                suffix=prefix_ids+[ids[row['target']['verdict']],p.tokenizer.convert_tokens_to_ids('<|im_end|>')]
                fields=dict(item[0]);fields['input_ids']=torch.cat([fields['input_ids'][:,:n],torch.tensor([suffix])],dim=1)
                fields['attention_mask']=torch.ones_like(fields['input_ids']);at=n+len(prefix_ids)
                fields={k:v.to('cuda') for k,v in fields.items()}
                logits=model(**fields,use_cache=False).logits[0,at-1].float()
                logp=logits.log_softmax(-1)
                adjusted=penalty(fields['input_ids'][:,:at],logits.unsqueeze(0).clone())[0]
                result={'raw_probabilities':{c:float(logp[i].exp()) for c,i in ids.items()},
                    'raw_log_probabilities':{c:float(logp[i]) for c,i in ids.items()},
                    'raw_argmax_id':int(logits.argmax()),'raw_argmax_label':names.get(int(logits.argmax()),'invalid'),
                    'adjusted_argmax_label':names.get(int(adjusted.argmax()),'invalid'),
                    'rationale_prefix_ids':prefix_ids,
                    'true_class_nll':-float(logp[ids[row['target']['verdict']]]),
                    'binary_normalized':dict(zip(('infeasible','feasible'),logits[[ids['infeasible'],ids['feasible']]].softmax(0).tolist()))}
                cache[key]=result;conditions[condition]=result
            records.append({'sample_id':row['sample_id'],'truth':row['target']['verdict'],
                'same_rationale_text':own==row['target']['rationale'],'free_generated_verdict':g['predicted_verdict'],
                'student_input_sha256':hashes[row['sample_id']],'conditions':conditions})
            if len(records)%20==0:print('matched-prefix rows',len(records),flush=True)
    require(len(records)==120 and sha(adapter/'adapter_model.safetensors')==digest,'Checkpoint changed or data incomplete')
    require(all(not v.requires_grad and v.grad is None for v in model.parameters()),'Unexpected gradients')
    summary={'by_condition':{condition:{'correct':sum(x['conditions'][condition]['raw_argmax_label']==x['truth'] for x in records),
        'by_class':{c:{'n':sum(x['truth']==c for x in records),
            'correct':sum(x['truth']==c and x['conditions'][condition]['raw_argmax_label']==c for x in records),
            'mean_true_class_nll':sum(x['conditions'][condition]['true_class_nll'] for x in records if x['truth']==c)/sum(x['truth']==c for x in records)} for c in ids}}
        for condition in ('ground_truth_rationale','generated_rationale')},
        'changed_rationale_ids':[x['sample_id'] for x in records if not x['same_rationale_text']],
        'gold_correct_own_wrong_ids':[x['sample_id'] for x in records if x['conditions']['ground_truth_rationale']['raw_argmax_label']==x['truth'] and x['conditions']['generated_rationale']['raw_argmax_label']!=x['truth']],
        'both_prefixes_wrong_ids':[x['sample_id'] for x in records if all(x['conditions'][c]['raw_argmax_label']!=x['truth'] for c in x['conditions'])],
        'own_prefix_prefill_matches_free_generated_verdict':sum(x['conditions']['generated_rationale']['adjusted_argmax_label']==x['free_generated_verdict'] for x in records),
        'scope':'Matched full-forward first-token scores. Both prefix conditions use the same forward path; separate from cached free-generation scores. Not a new training run.'}
    write(out/'ROWS.json',records);write(out/'SUMMARY.json',summary)
    write(out/'PROVENANCE.json',{'code_commit':os.environ['CB_CODE_COMMIT'],'job':os.environ['SLURM_JOB_ID'],
        'parent_run':str(parent),'adapter_sha256':digest,'dataset_sha256':DATA_SHA,'seed':713,
        'weights_file_unchanged':True,'optimizer_steps':0,'student_inputs_unchanged':True,
        'model_revision':read(assets/'STAGED.json')['model_revision'],'gpu':torch.cuda.get_device_name()})
    write(out/'COMPLETE.json',{'passed':True,'rows':120,'optimizer_steps':0})

if __name__=='__main__':
    import sys
    main(*map(Path,sys.argv[1:]))
