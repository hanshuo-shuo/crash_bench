"""Official SPR interface preflight on saved images; no environment actions or scores."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
CONTRACT=ROOT/'docs/audits/20260913/external_recovery/spr_contract.json'


def digest(path,algorithm='sha256'):
    h=hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def definitions(path,names,namespace):
    """Load only the named, hash-verified upstream helpers, unchanged."""
    tree=ast.parse(Path(path).read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes}!=set(names):raise ValueError('upstream helper missing')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)


def verify_assets(contract,model):
    records=[]
    for row in contract['model_files']:
        p=model/row['rfilename']
        if p.stat().st_size!=row['size']:raise ValueError('model file size mismatch: '+str(p))
        sha=digest(p)
        if row.get('lfs') and sha!=row['lfs']['sha256']:
            raise ValueError('model weight hash mismatch: '+str(p))
        records.append(dict(path=row['rfilename'],sha256=sha,bytes=p.stat().st_size))
    return records


def prepare(run,contract):
    import numpy as np
    import tensorflow as tf
    from huggingface_hub import snapshot_download
    run.mkdir(parents=True,exist_ok=False)
    write(run/'contract.json',contract)
    asset=Path(contract['asset_root']);code=asset/('code_'+contract['code_revision'][:12])
    code.mkdir(parents=True,exist_ok=True)
    for rel,spec in contract['code_files'].items():
        p=code/Path(rel).name
        if not p.exists():
            url='https://raw.githubusercontent.com/TingjunDai/SPRVLA/'+contract['code_revision']+'/'+rel
            with urllib.request.urlopen(url,timeout=60) as r:p.write_bytes(r.read())
        if digest(p)!=spec['sha256']:raise ValueError('upstream code changed')
    if not (code/'LICENSE').exists():
        url='https://raw.githubusercontent.com/TingjunDai/SPRVLA/'+contract['code_revision']+'/LICENSE'
        with urllib.request.urlopen(url,timeout=60) as r:(code/'LICENSE').write_bytes(r.read())
    model=Path(snapshot_download(contract['model_id'],revision=contract['model_revision'],
        local_dir=str(asset/('model_'+contract['model_revision'][:12])),max_workers=4))
    tokenizer=Path(snapshot_download(contract['tokenizer_id'],revision=contract['tokenizer_revision'],
        allow_patterns=[r['rfilename'] for r in contract['tokenizer_files']],
        local_dir=str(asset/('tokenizer_'+contract['tokenizer_revision'][:12])),max_workers=4))
    model_manifest=verify_assets(contract,model)
    bundle_path=Path(contract['source_bundle'])
    if digest(bundle_path)!=contract['source_bundle_sha256']:raise ValueError('historical bundle changed')
    with bundle_path.open('rb') as f:context=pickle.load(f)
    bundle=context['bundle'];bundle.assert_integrity()
    obs={k[len('observation.'):]:v for k,v in bundle.runtime_state.items() if k.startswith('observation.')}
    helpers={'np':np,'tf':tf}
    definitions(code/'libero_utils.py',('resize_image','get_libero_image','get_libero_wrist_image'),helpers)
    image=helpers['get_libero_image'](obs,256);wrist=helpers['get_libero_wrist_image'](obs,256)
    np.savez_compressed(run/'inputs.npz',image=image,wrist=wrist)
    write(run/'input_identity.json',dict(source_anchor=contract['source_anchor'],
        bundle_id=bundle.bundle_id,bundle_sha256=digest(bundle_path),instruction=context['instruction'],
        input_sha256=digest(run/'inputs.npz'),raw_image_shapes={k:list(obs[k].shape) for k in
            ('agentview_image','robot0_eye_in_hand_image')},environment_steps=0))
    venv=run/'venv'
    python='/projects/p33100/siosio/uv_python/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11'
    subprocess.run([python,'-m','venv',str(venv)],check=True)
    command=[str(venv/'bin/python'),'-m','pip','install','--prefer-binary',
        '--find-links',str(asset/'wheels'),*contract['requirements']]
    with (run/'install.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3000)
    freeze=subprocess.check_output([str(venv/'bin/python'),'-m','pip','freeze'],text=True)
    (run/'environment.txt').write_text(freeze)
    write(run/'prepared.json',dict(status='PREPARED',commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),job=os.environ['SLURM_JOB_ID'],
        model=str(model),tokenizer=str(tokenizer),code=str(code),venv=str(venv),
        model_files=model_manifest,input_sha256=digest(run/'inputs.npz'),
        contract_sha256=digest(CONTRACT),environment_sha256=digest(run/'environment.txt')))
    print('SPR assets and isolated environment prepared; zero simulation actions',flush=True)


def infer(run,contract):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor,AutoConfig,Qwen2Tokenizer
    from vllm import LLM,ModelRegistry
    from vllm.sampling_params import SamplingParams
    from vllm.model_executor.models.registry import _MULTIMODAL_MODELS
    prepared=json.loads((run/'prepared.json').read_text())
    if prepared['contract_sha256']!=digest(CONTRACT):raise ValueError('preflight contract changed')
    if prepared['input_sha256']!=digest(run/'inputs.npz'):raise ValueError('preflight image changed')
    code=Path(prepared['code'])
    for rel,spec in contract['code_files'].items():
        if digest(code/Path(rel).name)!=spec['sha256']:raise ValueError('upstream code changed')
    sys.path.insert(0,str(code))
    from sprvla import SPRVLAForActionReasoning,SPRVLAParser,extract_action_token_lists
    ModelRegistry.register_model('SPRVLAForActionReasoning',SPRVLAForActionReasoning)
    _MULTIMODAL_MODELS['SPRVLAForActionReasoning']=('sprvla','SPRVLAForActionReasoning')
    helpers={'np':np,'torch':torch,'Image':Image,'math':math,'AutoProcessor':AutoProcessor}
    definitions(code/'run_libero_eval_vllm.py',('crop_and_resize_pil','center_crop_image','step'),helpers)
    model_path=prepared['model']
    processor=AutoProcessor.from_pretrained(model_path,trust_remote_code=True,
        torch_dtype='bfloat16',device_map='auto',padding_side='left')
    cfg=AutoConfig.from_pretrained(model_path,trust_remote_code=True)
    # Relocate the upstream parser's Qwen2 tokenizer dependency to its pinned local copy.
    parser=SPRVLAParser(Qwen2Tokenizer.from_pretrained(prepared['tokenizer']),
        getattr(cfg,'norm_stats',{}),getattr(cfg,'n_action_bins',256))
    write(run/'inference_provenance.json',dict(job=os.environ['SLURM_JOB_ID'],
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        gpu=torch.cuda.get_device_name(0),versions={n:importlib.metadata.version(n) for n in
            ('torch','transformers','vllm','numpy')},contract_sha256=digest(CONTRACT),
        environment_steps=0,source_anchor=contract['source_anchor'],
        memory_utilization=.85,enforce_eager=True))
    model=LLM(model=model_path,trust_remote_code=True,tensor_parallel_size=1,
        gpu_memory_utilization=.85,dtype='bfloat16',enforce_eager=True)
    sampling=SamplingParams(max_tokens=contract['max_new_tokens'],temperature=contract['temperature'])
    class Recorder:
        def __init__(self,wrapped):self.wrapped=wrapped;self.text=None
        def generate(self,*args,**kwargs):
            outputs=self.wrapped.generate(*args,**kwargs)
            self.text=outputs[0].outputs[0].text
            return outputs
    recorder=Recorder(model)
    inputs=np.load(run/'inputs.npz',allow_pickle=False)
    identity=json.loads((run/'input_identity.json').read_text())
    records=[]
    for name,instruction in [('normal',identity['instruction']),('rewind','return to initial position')]:
        start=time.monotonic()
        actions,_,trace,count,points=helpers['step'](inputs['image'],inputs['wrist'],
            instruction,recorder,processor,sampling,parser,'libero_spatial_no_noops_modified')
        array=np.asarray(actions,float)
        token_lists=extract_action_token_lists(recorder.text,only_len=7)
        unknown=sum(value is None for tokens in token_lists for value in
                    parser._qwen_tokenizer.convert_tokens_to_ids(tokens))
        valid=(array.ndim==2 and array.shape[1]==7 and len(array)>0 and
               np.isfinite(array).all() and unknown==0 and len(token_lists)==len(array))
        record=dict(mode=name,instruction=instruction,generated_text=recorder.text,
            action_shape=list(array.shape),actions=array.tolist(),finite_action_chunk=bool(valid),
            unknown_action_tokens=unknown,subtask_count=count,
            latency_seconds=time.monotonic()-start)
        write(run/(name+'.json'),record);records.append(record)
        if not valid:raise ValueError('official action parser did not return finite 7-D actions')
        print(name,'valid action chunk',array.shape,flush=True)
    write(run/'complete.json',dict(status='INTERFACE_PREFLIGHT_PASSED',model_queries=len(records),
        environment_steps=0,new_sources=0,recovery_performance_measured=False,
        modes=[r['mode'] for r in records],contract_sha256=digest(CONTRACT)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','infer'],required=True)
    p.add_argument('--run-dir',type=Path,required=True);args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Quest Slurm only')
    contract=json.loads(CONTRACT.read_text())
    if args.stage=='prepare':prepare(args.run_dir,contract)
    else:infer(args.run_dir,contract)


if __name__=='__main__':main()
