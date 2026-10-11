"""Controlled B correction: same inputs/architecture, digit targets and equal losses."""
import re
from pathlib import Path
import torch
from PIL import Image
from .common import require, sha, student_messages
from .verdict_diagnostic import token_losses

LABEL_TO_DIGIT={'infeasible':'0','feasible':'1'}
DIGIT_TO_LABEL={v:k for k,v in LABEL_TO_DIGIT.items()}


def target(row):
    return 'Rationale: '+row['target']['rationale']+'\nVerdict: '+LABEL_TO_DIGIT[row['target']['verdict']]


def parse(text):
    m=re.fullmatch(r'Rationale: .+\nVerdict: ([01])',text,flags=re.S)
    return DIGIT_TO_LABEL[m.group(1)] if m else None


def encode(processor,row,dataset,max_length=1536):
    images=[]
    for spec in row['student']['images']:
        path=Path(dataset)/spec['path'];require(sha(path)==spec['sha256'],'Image changed')
        with Image.open(path) as im:images.append(im.convert('RGB'))
    messages=student_messages(row,'B');answer=target(row)
    prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    full=processor.apply_chat_template(messages+[{'role':'assistant','content':answer}],tokenize=False,add_generation_prompt=False)
    require(full.startswith(prompt+answer),'Template changed')
    fields=processor(text=[full],images=images,return_tensors='pt',padding=False)
    prefix=processor(text=[prompt],images=images,return_tensors='pt',padding=False)['input_ids'][0]
    ids=fields['input_ids'][0];n=len(prefix)
    require(len(ids)<=max_length and torch.equal(ids[:n],prefix),'Truncation or prompt boundary mismatch')
    eos_id=processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
    ends=(ids[n:]==eos_id).nonzero();require(len(ends)==1,'Ambiguous assistant end')
    end=n+int(ends[0]);digit=end-1
    digit_ids=processor.tokenizer.encode(LABEL_TO_DIGIT[row['target']['verdict']],add_special_tokens=False)
    require(len(digit_ids)==1 and int(ids[digit])==digit_ids[0],'Verdict must be exactly one standalone binary token')
    before_digit=processor(text=[prompt+answer[:-1]],images=images,return_tensors='pt',padding=False)['input_ids'][0]
    require(torch.equal(ids[:digit],before_digit),'Binary decision boundary is not exact')
    labels=fields['input_ids'].clone();labels[:,:n]=-100;labels[:,end+1:]=-100
    labels[~fields['attention_mask'].bool()]=-100
    vm=torch.zeros_like(labels,dtype=torch.bool);vm[:,digit]=True
    rm=labels.ne(-100)&~vm
    # Native token boundaries around the narrative body, excluding both headers.
    body_start=processor(text=[prompt+'Rationale: '],images=images,return_tensors='pt',padding=False)['input_ids'][0]
    bs=next((i for i,(a,b) in enumerate(zip(body_start,ids)) if a!=b),len(body_start))
    body_end=processor(text=[prompt+'Rationale: '+row['target']['rationale']],images=images,return_tensors='pt',padding=False)['input_ids'][0]
    be=next((i for i,(a,b) in enumerate(zip(body_end,ids)) if a!=b),len(body_end))
    text_mask=torch.zeros_like(vm);text_mask[:,bs:be]=True
    eos_mask=labels.eq(eos_id)
    format_mask=rm&~text_mask&~eos_mask
    require((text_mask&~rm).sum()==0 and text_mask.any(),'Narrative token mask invalid')
    require(torch.equal(rm|vm,labels.ne(-100)) and not (rm&vm).any(),'Assistant partition invalid')
    return fields,labels,rm,vm,n,{'rationale_text':text_mask,'format':format_mask,'eos':eos_mask}


def losses(logits,labels,rm,vm,detail=None):
    valid=labels[:,1:].ne(-100)
    r=rm[:,1:]&valid;v=vm[:,1:]&valid
    require((r|v).equal(valid) and not (r&v).any(),'Loss masks must partition assistant tokens')
    require(torch.all(v.sum(1)==1) and torch.all(r.sum(1)>0),'Exactly one decision token per example required')
    token=token_losses(logits,labels)
    r_per=(token*r).sum(1)/r.sum(1);v_per=(token*v).sum(1)/v.sum(1)
    result={'rationale':r_per.mean(),'verdict':v_per.mean(),
        'total':(.5*r_per+.5*v_per).mean(),'rationale_tokens':int(r.sum()),'verdict_tokens':int(v.sum())}
    if detail is not None:
        for name,mask in detail.items():
            m=mask[:,1:]&valid
            result[name]=token[m].mean() if m.any() else token.sum()*0
            result[name+'_tokens']=int(m.sum())
    return result
