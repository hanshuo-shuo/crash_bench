"""Read-only diagnosis of the saved word-verdict adapter. Never creates an optimizer."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from .common import read, write, require, sha, UPSTREAM
from .data import encode
from .engineering_contract import DATA_SHA, epoch_orders

GROUPS = ('rationale_and_format', 'decisive_class', 'verdict_continuation', 'eos')


def partition(labels, rm, vm, eos_id):
    valid = labels.ne(-100)
    eos = valid & labels.eq(eos_id)
    value = vm & valid & ~eos
    require(value.sum() > 0, 'Missing verdict')
    decisive = torch.zeros_like(value)
    for b in range(len(value)):
        decisive[b, value[b].nonzero()[0, 0]] = True
    masks = dict(zip(GROUPS, (rm & valid, decisive, value & ~decisive, eos)))
    require(torch.equal(sum(m.to(torch.int) for m in masks.values()), valid.to(torch.int)), 'Partition is not exhaustive/disjoint')
    return masks


def token_losses(logits, labels):
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1), ignore_index=-100, reduction='none').reshape_as(labels[:, 1:])


def tensor_digest(tensors):
    h = hashlib.sha256()
    for name, tensor in tensors:
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def gradient_geometry(vectors):
    gram = (vectors.double() @ vectors.double().T).cpu()
    total_sq = float(gram.sum()); norms = gram.diag().clamp_min(0).sqrt()
    return {'norms':dict(zip(GROUPS, norms.tolist())), 'total_norm':max(total_sq, 0)**.5,
        'dot_total_fraction':dict(zip(GROUPS, (gram.sum(1)/max(total_sq, 1e-30)).tolist())),
        'cosine':(gram/(norms[:, None]*norms[None, :]).clamp_min(1e-30)).tolist(),
        'gram':gram.tolist()}


def main(root, campaign, assets):
    out = root/'diagnostic'; out.mkdir(exist_ok=False)
    dataset=campaign/'dataset'; require(sha(dataset/'dataset.jsonl')==DATA_SHA, 'Dataset changed')
    rows=[json.loads(x) for x in (dataset/'dataset.jsonl').read_text().splitlines()]
    require(len(rows)==120 and read(campaign/'audit/AUDIT.json')['passed_for_engineering_only'], 'Data isolation gate failed')
    saved=read(campaign/'B/generations_final.json'); require(len(saved)==120, 'Missing saved outputs')
    stage=read(assets/'STAGED.json')
    for name, digest in stage['files'].items(): require(sha(assets/'model'/name)==digest, 'Model changed')
    adapter=campaign/'B/adapter'; adapter_sha=sha(adapter/'adapter_model.safetensors')
    processor=AutoProcessor.from_pretrained(assets/'model',local_files_only=True,min_pixels=224*224,max_pixels=224*224)
    tok=processor.tokenizer; eos=tok.convert_tokens_to_ids('<|im_end|>')
    strings=['feasible','infeasible','0','1',' feasible',' infeasible',' 0',' 1',
             'Verdict: feasible','Verdict: infeasible','Verdict: 0','Verdict: 1']
    write(out/'TOKENIZATION.json', {s:{'ids':tok.encode(s,add_special_tokens=False),
        'pieces':[tok.decode([i]) for i in tok.encode(s,add_special_tokens=False)]} for s in strings})
    torch.manual_seed(713); torch.backends.cuda.matmul.allow_tf32=True
    base=Qwen2_5_VLForConditionalGeneration.from_pretrained(assets/'model',local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    model=PeftModel.from_pretrained(base,adapter,is_trainable=True); model.eval()
    params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    require(all('lora_' in n and 'visual' not in n for n,p in params), 'Unexpected gradient parameters')
    before=tensor_digest(params)
    write(out/'PROVENANCE.json', {'code_commit':os.environ['CB_CODE_COMMIT'],'upstream_commit':UPSTREAM,
        'slurm':{k:v for k,v in os.environ.items() if k.startswith('SLURM_')}, 'seed':713,
        'campaign':str(campaign),'adapter_sha256':adapter_sha,'dataset_sha256':DATA_SHA,
        'model_revision':stage['model_revision'],'gpu':torch.cuda.get_device_name(),
        'mode':'Read-only: forward/autograd only; no optimizer, no backward into stored weights, no update',
        'probabilities':'Raw language sequence likelihoods and two-candidate normalized likelihoods; not calibrated feasibility probabilities'})
    items=[]; records=[]; started=time.monotonic()
    with (out/'probabilities.jsonl').open('w') as stream, torch.no_grad():
        for row, generated in zip(rows,saved):
            require(row['sample_id']==generated['sample_id'], 'Output alignment changed')
            own=generated['generated'].split('\nVerdict: ')[0]
            require(own.startswith('Rationale: '), 'Invalid generated rationale')
            own=own[len('Rationale: '):]
            cache={}; conditions={}
            for condition, body in [('ground_truth_rationale',row['target']['rationale']),('generated_rationale',own)]:
                if body in cache:
                    conditions[condition]=cache[body]; continue
                candidates={}; encoded={}
                for label in ('feasible','infeasible'):
                    r=copy.deepcopy(row);r['target'].update(verdict=label,rationale=body)
                    item=encode(processor,r,dataset,'B'); encoded[label]=item
                left,right=[encoded[k][0]['input_ids'][0] for k in ('feasible','infeasible')]
                common=next(i for i,(a,b) in enumerate(zip(left,right)) if a!=b)
                for label,item in encoded.items():
                    fields, labels, rm, vm, prefix=item
                    groups=partition(labels,rm,vm,eos)
                    require(int(groups['decisive_class'].nonzero()[0,1])==common, 'First differing class token differs from verdict mask')
                    logits=model(**{k:v.to('cuda') for k,v in fields.items()},use_cache=False).logits
                    losses=token_losses(logits,labels.to('cuda'))[0].cpu()
                    positions=labels[0].ne(-100).nonzero().flatten().tolist()
                    detail=[{'position':i,'id':int(labels[0,i]),'piece':tok.decode([int(labels[0,i])]),
                        'nll':float(losses[i-1]),'group':next(k for k,m in groups.items() if m[0,i])} for i in positions]
                    sums={k:sum(x['nll'] for x in detail if x['group']==k) for k in GROUPS}
                    counts={k:sum(x['group']==k for x in detail) for k in GROUPS}
                    value_ll=-sums['decisive_class']-sums['verdict_continuation']
                    next_logp=logits[0,common-1].float().log_softmax(-1)
                    candidates[label]={'sequence_logp':value_ll,'sequence_probability':math.exp(value_ll),
                        'sequence_with_eos_logp':value_ll-sums['eos'],
                        'decisive_token_probability':math.exp(-sums['decisive_class']),
                        'greedy_first_token_id':int(next_logp.argmax()),
                        'decisive_token_id':int(labels[0,common]),'group_nll_sums':sums,'group_counts':counts,
                        'tokens':detail}
                    if label==row['target']['verdict'] and condition=='ground_truth_rationale': items.append(item)
                    del logits
                for key in ('sequence_logp','sequence_with_eos_logp'):
                    vals=torch.tensor([candidates[k][key] for k in ('feasible','infeasible')],dtype=torch.float64).softmax(0).tolist()
                    for k,v in zip(('feasible','infeasible'),vals): candidates[k][key+'_binary_normalized']=v
                pfx=left[:common].numpy().tobytes()
                value={'prefix_sha256':hashlib.sha256(pfx).hexdigest(),'candidates':candidates}
                cache[body]=value;conditions[condition]=value
            record={'sample_id':row['sample_id'],'truth':row['target']['verdict'],
                'saved_generated_verdict':generated['predicted_verdict'],'rationale_bodies_identical':own==row['target']['rationale'],
                'conditions':conditions}
            records.append(record);stream.write(json.dumps(record)+'\n');stream.flush()
            if len(records)%20==0:print('probabilities',len(records),time.monotonic()-started,flush=True)
    write(out/'PROBABILITIES.json',records)
    # Actual old objective: mean tokens per example, then mean over four examples.
    # Measure all 30 first-epoch batches at the SAVED checkpoint. This is not a
    # reconstruction of gradients at historical checkpoints that were not saved.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads();model.config.use_cache=False;model.train()
    torch.manual_seed(713)
    ps=[p for n,p in params]; size=sum(p.numel() for p in ps)
    aggregate=torch.zeros((4,size),device='cuda',dtype=torch.float32)
    batches=[]
    for bi, order in enumerate(epoch_orders(120,713,1,4)):
        vectors=torch.zeros_like(aggregate); counts=[]; validation=None
        for j,index in enumerate(order):
            fields,labels,rm,vm,_=items[index]
            labels=labels.to('cuda'); groups={k:v.to('cuda') for k,v in partition(labels,rm.to('cuda'),vm.to('cuda'),eos).items()}
            logits=model(**{k:v.to('cuda') for k,v in fields.items()},use_cache=False).logits
            token=token_losses(logits,labels); n=int(labels.ne(-100).sum())
            contributions=[token[m[:,1:]].sum()/n/4 for m in groups.values()]
            reference=None
            if bi==0 and j==0:
                gg=torch.autograd.grad(token.sum()/n/4,ps,retain_graph=True,allow_unused=True)
                reference=torch.cat([(g if g is not None else torch.zeros_like(p)).detach().float().flatten() for g,p in zip(gg,ps)])
                del gg
                summed=torch.zeros_like(reference)
            for gi, loss in enumerate(contributions):
                gg=torch.autograd.grad(loss,ps,retain_graph=gi<3,allow_unused=True)
                flat=torch.cat([(g if g is not None else torch.zeros_like(p)).detach().float().flatten() for g,p in zip(gg,ps)])
                vectors[gi].add_(flat)
                if reference is not None: summed.add_(flat)
                del gg,flat
            if reference is not None:
                validation=float((reference-summed).norm()/reference.norm().clamp_min(1e-30))
                require(validation<.05, 'Gradient decomposition failed BF16 tolerance')
                del reference,summed
            counts.append({'sample_id':rows[index]['sample_id'],'assistant_tokens':n,
                'component_token_weights':{k:int(m.sum())/n for k,m in groups.items()}})
            del logits,token,contributions
        aggregate.add_(vectors/30)
        result={'batch':bi,'sample_weights':counts,**gradient_geometry(vectors),
                'first_example_sum_identity_relative_error':validation}
        batches.append(result)
        with (out/'gradient_batches.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
        print('gradient batch',bi+1, time.monotonic()-started,flush=True)
        del vectors
    write(out/'GRADIENTS.json',{'groups':GROUPS,'batches':batches,'full_pool_mean_gradient':gradient_geometry(aggregate),
        'checkpoint':'Saved final B; not historical trajectory reconstruction',
        'dropout':'Original training mode/dropout .05; diagnostic RNG seed713',
        'objective':'For each example sum assistant token CE / assistant count, then /4 in each recorded batch',
        'note':'Norms are not additive. Dot-total fractions sum to one but may be negative due to cancellation. No optimizer preconditioning/update attribution.'})
    require(tensor_digest(params)==before,'Adapter tensor changed during read-only diagnosis')
    require(sha(adapter/'adapter_model.safetensors')==adapter_sha,'Saved adapter changed')
    require(all(not p.requires_grad and p.grad is None for n,p in model.named_parameters() if 'visual' in n),'Visual gradient detected')
    write(out/'COMPLETE.json',{'passed':True,'rows':120,'prefix_conditions':2,'gradient_batches':30,
        'weights_bitwise_unchanged':True,'optimizer_steps':0,'elapsed_seconds':time.monotonic()-started})


if __name__=='__main__':
    import sys
    main(*map(Path,sys.argv[1:]))
