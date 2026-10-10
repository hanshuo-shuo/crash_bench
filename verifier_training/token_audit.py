import json
from pathlib import Path
import sys
import torch
from transformers import AutoProcessor
from .common import *
from .data import encode, collate


def main(root,assets):
    processor=AutoProcessor.from_pretrained(assets/'model',local_files_only=True,min_pixels=224*224,max_pixels=224*224)
    rows=[json.loads(l) for l in (root/'dataset.jsonl').read_text().splitlines()]
    stats=[]
    for row in rows:
        for variant in ('A','B'):
            fields,labels,rm,vm,n=encode(processor,row,root,variant)
            require(labels[:,:n].eq(-100).all().item(),'Prompt/image leakage into loss')
            require(int(vm.sum())>0 and (variant=='A' or int(rm.sum())>0),'Missing supervised tokens')
            require(int(fields['image_grid_thw'].shape[0])==2,'Missing image')
            stats.append({'id':row['sample_id'],'variant':variant,'length':labels.numel(),'prompt':n,
                          'rationale':int(rm.sum()),'verdict':int(vm.sum()),
                          'supervised_text':processor.tokenizer.decode(fields['input_ids'][labels.ne(-100)],skip_special_tokens=False)})
    write(root/'TOKEN_AUDIT.json',{'passed':True,'rows':stats,'count':len(stats)})

if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]))
