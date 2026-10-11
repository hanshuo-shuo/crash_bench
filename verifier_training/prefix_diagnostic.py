"""Read-only native cached-generation versus canonical teacher-prefix diagnosis."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers import LogitsProcessor, LogitsProcessorList
from .common import read,write,require,sha,UPSTREAM
from .data import encode
from .engineering_contract import DATA_SHA


class ForcedPrefix(LogitsProcessor):
    """Use native generate/cache; inspect scores before forcing known prefix IDs."""
    def __init__(self,prompt_n,forced,decision_at):
        self.prompt_n=prompt_n;self.forced=forced;self.decision_at=decision_at;self.inspected=[]

    def __call__(self,input_ids,scores):
        step=input_ids.shape[1]-self.prompt_n
        if step>=self.decision_at:self.inspected.append(scores.detach().clone())
        if step<len(self.forced):
            result=torch.full_like(scores,-float('inf'));result[:,self.forced[step]]=0
            return result
        return scores


def cached_candidates(model,fields,prefix,prompt_n,candidates,eos,penalty):
    rationale_ids=prefix[0,prompt_n:].tolist();scores={};first=None;first_processed=None
    for label,tokens in candidates.items():
        forced=ForcedPrefix(prompt_n,rationale_ids+tokens,len(rationale_ids))
        result=model.generate(**fields,max_new_tokens=len(forced.forced)+1,do_sample=False,use_cache=True,
            repetition_penalty=penalty,return_dict_in_generate=True,output_logits=True,
            logits_processor=LogitsProcessorList([forced]))
        require(result.sequences[0,prompt_n:prompt_n+len(forced.forced)].tolist()==forced.forced,'Native forced prefix changed')
        require(len(forced.inspected)==len(tokens)+1,'Missing candidate/EOS scores')
        raw=[result.logits[len(rationale_ids)+i][0].float().log_softmax(-1) for i in range(len(tokens)+1)]
        processed=[v[0].float().log_softmax(-1) for v in forced.inspected]
        nll=[-float(v[t]) for v,t in zip(raw,tokens+[eos])]
        adjusted=[-float(v[t]) for v,t in zip(processed,tokens+[eos])]
        if first is None:first=raw[0];first_processed=processed[0]
        scores[label]={'token_ids':tokens,'token_nll_including_eos':nll,'decisive_nll':nll[0],
            'continuation_nll_sum':sum(nll[1:-1]),'eos_nll':nll[-1],
            'sequence_logp':-sum(nll[:-1]),'sequence_with_eos_logp':-sum(nll),
            'processed_token_nll_including_eos':adjusted,'processed_decisive_nll':adjusted[0],
            'processed_sequence_logp':-sum(adjusted[:-1]),'processed_sequence_with_eos_logp':-sum(adjusted),
            'first_class_token_seen_in_prefix':tokens[0] in prefix[0].tolist()}
        del result,forced
    for name in ('sequence_logp','sequence_with_eos_logp','processed_sequence_logp','processed_sequence_with_eos_logp'):
        probs=torch.tensor([scores[k][name] for k in ('feasible','infeasible')],dtype=torch.float64).softmax(0).tolist()
        for k,v in zip(('feasible','infeasible'),probs):scores[k][name+'_binary_normalized']=v
    return {'candidates':scores,'first_greedy_token_id':int(first.argmax()),
        'processed_first_greedy_token_id':int(first_processed.argmax()),
        'prefix_sha256':hashlib.sha256(prefix.cpu().numpy().tobytes()).hexdigest()}


def main(root,campaign,assets,parent):
    out=root/'prefix';out.mkdir(exist_ok=False)
    require(read(parent/'diagnostic/COMPLETE.json')['passed'],'Parent diagnosis incomplete')
    dataset=campaign/'dataset';require(sha(dataset/'dataset.jsonl')==DATA_SHA,'Dataset changed')
    rows=[json.loads(x) for x in (dataset/'dataset.jsonl').read_text().splitlines()]
    saved=read(campaign/'B/generations_final.json')
    p=AutoProcessor.from_pretrained(assets/'model',local_files_only=True,min_pixels=224*224,max_pixels=224*224)
    torch.manual_seed(713);torch.backends.cuda.matmul.allow_tf32=True
    base=Qwen2_5_VLForConditionalGeneration.from_pretrained(assets/'model',local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    model=PeftModel.from_pretrained(base,campaign/'B/adapter',is_trainable=False);model.requires_grad_(False);model.eval()
    adapter_sha=sha(campaign/'B/adapter/adapter_model.safetensors')
    eos=p.tokenizer.convert_tokens_to_ids('<|im_end|>')
    penalty=model.generation_config.repetition_penalty
    candidates={'feasible':p.tokenizer.encode(' feasible',add_special_tokens=False),
                'infeasible':p.tokenizer.encode(' infeasible',add_special_tokens=False)}
    write(out/'PROVENANCE.json',{'code_commit':os.environ['CB_CODE_COMMIT'],'upstream_commit':UPSTREAM,
        'slurm_job':os.environ['SLURM_JOB_ID'],'seed':713,'gpu':torch.cuda.get_device_name(),
        'dataset_sha256':DATA_SHA,'adapter_sha256':adapter_sha,'parent_diagnostic':str(parent),
        'mode':'Native free generation and forced prefix cache replay. No gradients or optimizer.',
        'candidate_tokens':candidates,'native_generation_config':model.generation_config.to_dict(),
        'repetition_penalty':penalty})
    records=[];start=time.monotonic()
    with (out/'rows.jsonl').open('w') as stream,torch.no_grad():
        for row,old in zip(rows,saved):
            item=encode(p,row,dataset,'B');n=item[4]
            fields={k:(v[:,:n] if k in ('input_ids','attention_mask') else v).to('cuda') for k,v in item[0].items()}
            generated=model.generate(**fields,max_new_tokens=160,do_sample=False,use_cache=True,
                pad_token_id=p.tokenizer.pad_token_id,return_dict_in_generate=True,output_scores=True,output_logits=True)
            sequence=generated.sequences;assistant=sequence[0,n:]
            text=p.tokenizer.decode(assistant,skip_special_tokens=True).strip()
            require('\nVerdict: ' in text,'Generated output lacks verdict')
            body=text.rsplit('\nVerdict: ',1)[0]
            boundary=body+'\nVerdict:'
            indices=[i for i in range(len(assistant)) if p.tokenizer.decode(assistant[:i],skip_special_tokens=True)==boundary]
            require(len(indices)==1,'Cannot identify an exact generated decision boundary')
            d=indices[0];own_prefix=sequence[:,:n+d]
            gt_start=int(item[3].nonzero()[0,1]);gt_prefix=item[0]['input_ids'][:,:gt_start].to('cuda')
            native=generated.scores[d][0].float().log_softmax(-1)
            native_values={k:float(native[v[0]]) for k,v in candidates.items()}
            native_top=int(native.argmax())
            native_raw=generated.logits[d][0].float().log_softmax(-1)
            native_raw_values={k:float(native_raw[v[0]]) for k,v in candidates.items()}
            del generated
            # Whole-prefix prefill isolates sequence-length/cache computation from
            # the autoregressive replay of the exact same canonical prefix IDs.
            packed=dict(fields,input_ids=gt_prefix,attention_mask=torch.ones_like(gt_prefix))
            teacher=model(**packed,use_cache=False).logits[0,-1].float().log_softmax(-1)
            prefill={k:float(teacher[v[0]]) for k,v in candidates.items()};del teacher
            gt=cached_candidates(model,fields,gt_prefix,n,candidates,eos,penalty)
            same=torch.equal(own_prefix,gt_prefix)
            own=gt if same else cached_candidates(model,fields,own_prefix,n,candidates,eos,penalty)
            error=max(abs(-own['candidates'][k]['processed_decisive_nll']-native_values[k]) for k in candidates)
            raw_error=max(abs(-own['candidates'][k]['decisive_nll']-native_raw_values[k]) for k in candidates)
            record={'sample_id':row['sample_id'],'truth':row['target']['verdict'],'generated':text,
                'matches_saved_text':text==old['generated'],'same_rationale_text':body=='Rationale: '+row['target']['rationale'],
                'canonical_and_generated_prefix_ids_identical':same,'canonical_prefix_ids':gt_prefix[0,n:].tolist(),
                'generated_prefix_ids':own_prefix[0,n:].tolist(),'native_decisive_logp':native_values,
                'native_raw_decisive_logp':native_raw_values,'repetition_penalty':penalty,
                'native_greedy_token_id':native_top,'whole_prefix_prefill_decisive_logp':prefill,
                'ground_truth_prefix_cached':gt,'own_prefix_cached':own,'native_replay_logp_max_error':error,
                'native_raw_replay_logp_max_error':raw_error}
            records.append(record);stream.write(json.dumps(record)+'\n');stream.flush()
            require(error<.02 and raw_error<.02,'Cache replay does not reproduce raw and processed native decision probabilities')
            if len(records)%10==0:print('native prefix',len(records),'elapsed',time.monotonic()-start,flush=True)
    write(out/'ROWS.json',records)
    require(sha(campaign/'B/adapter/adapter_model.safetensors')==adapter_sha,'Saved adapter changed')
    require(all(not x.requires_grad and x.grad is None for x in model.parameters()),'Unexpected gradient')
    write(out/'COMPLETE.json',{'passed':True,'rows':120,'optimizer_steps':0,'weights_file_unchanged':True,
        'elapsed_seconds':time.monotonic()-start,'identical_prefix_ids':sum(x['canonical_and_generated_prefix_ids_identical'] for x in records),
        'matches_saved_outputs':sum(x['matches_saved_text'] for x in records)})

if __name__=='__main__':
    import sys
    main(*map(Path,sys.argv[1:]))
