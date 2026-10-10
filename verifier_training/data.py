from pathlib import Path
import torch
from PIL import Image
from .common import require, sha, student_messages, target_text


def encode(processor, row, dataset_root, variant, max_length=1536):
    images=[]
    for item in row['student']['images']:
        path=Path(dataset_root)/item['path']
        require(sha(path)==item['sha256'], 'Exported image changed')
        images.append(Image.open(path).convert('RGB'))
    messages=student_messages(row,variant)
    prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    answer=target_text(row,variant)
    full=processor.apply_chat_template(messages+[{'role':'assistant','content':answer}],tokenize=False,add_generation_prompt=False)
    require(full.startswith(prompt+answer), 'Chat template boundary changed')
    batch=processor(text=[full],images=images,return_tensors='pt',padding=False)
    prefix=processor(text=[prompt],images=images,return_tensors='pt',padding=False)['input_ids'][0]
    ids=batch['input_ids'][0]
    require(len(ids)<=max_length, 'No silent truncation: sample exceeds max_length')
    require(torch.equal(ids[:len(prefix)],prefix), 'Prompt token boundary not exact')
    # Everything before assistant content, including all image tokens, is ignored.
    labels=batch['input_ids'].clone(); labels[:,:len(prefix)]=-100
    attention=batch['attention_mask'].bool();labels[~attention]=-100
    ends=(ids[len(prefix):]==processor.tokenizer.convert_tokens_to_ids('<|im_end|>')).nonzero()
    require(len(ends)==1, 'Expected one assistant end token')
    assistant_end=len(prefix)+int(ends[0])+1
    labels[:,assistant_end:]=-100
    rm=torch.zeros_like(labels,dtype=torch.bool);vm=rm.clone()
    split=answer.index('Verdict: ')+len('Verdict: ')
    verdict_prefix=processor(text=[prompt+answer[:split]],images=images,return_tensors='pt',padding=False)['input_ids'][0]
    # Trailing space can merge with the first verdict token; use the exact common
    # prefix, then verify the decoded supervised span contains the whole label.
    n=0
    for left,right in zip(verdict_prefix.tolist(),ids.tolist()):
        if left!=right:break
        n+=1
    require(n>=len(prefix), 'Invalid verdict span')
    if variant=='B':
        rm[:,len(prefix):n]=True
    vm[:,n:]=labels[:,n:].ne(-100)
    require(row['target']['verdict'] in processor.tokenizer.decode(ids[n:],skip_special_tokens=True), 'Verdict tokens missing')
    require(labels[:,len(prefix):assistant_end].ne(-100).all(), 'Assistant content excluded')
    return batch,labels,rm,vm,len(prefix)


def collate(items,pad_token_id):
    """Right-pad full sequences; pad positions never contribute, even if PAD=EOS."""
    n=max(x[0]['input_ids'].shape[1] for x in items)
    fields={}
    for key in ('input_ids','attention_mask'):
        fill=pad_token_id if key=='input_ids' else 0
        fields[key]=torch.cat([torch.nn.functional.pad(x[0][key],(0,n-x[0][key].shape[1]),value=fill) for x in items])
    for key in ('pixel_values','image_grid_thw'):
        fields[key]=torch.cat([x[0][key] for x in items])
    other=[]
    for index,fill in ((1,-100),(2,False),(3,False)):
        other.append(torch.cat([torch.nn.functional.pad(x[index],(0,n-x[index].shape[1]),value=fill) for x in items]))
    return fields,*other
