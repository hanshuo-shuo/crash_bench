"""Allocated-node independent train-only cutoff and controller-byte verification."""
import argparse
import json
from pathlib import Path
from collections import defaultdict
from common import atomic_json
from preflight import stream_sha


def verify(root):
    import pyarrow.parquet as pq
    plan=json.loads((root/'plan.json').read_text());cfg=plan['configuration']
    for item in cfg['controller_files']:
        if stream_sha(Path(item['path']))!=item['sha256']:raise RuntimeError('Pinned controller/gripper bytes changed')
    audit=Path(cfg['audited_input']);report=json.loads((audit/'INDEPENDENT_AUDIT.json').read_text())
    if not report['passed'] or report['rollouts']!=1200 or stream_sha(audit/'rollouts.parquet')!=report['parquet_sha256']:raise RuntimeError('Complete audited input changed')
    split=json.loads((Path(__file__).parent/'split.json').read_text());train=set(split['train'])
    grouped=defaultdict(lambda:defaultdict(list));ordered=[]
    columns=['scene_id','capability','seed','infer_boundary','disagreement','time_to_crash']
    for r in pq.read_table(audit/'rollouts.parquet',columns=columns).to_pylist():
        if r['scene_id'] not in train or r['capability']!='nominal' or not r['infer_boundary']:continue
        ttc=r['time_to_crash']
        if ttc is None:continue
        if ttc!=-1 and ttc<0:raise RuntimeError('Invalid audited TTC clock')
        grouped[r['scene_id']][r['seed']].append(r['disagreement']);ordered.append(r)
    if len(grouped)!=42 or sum(len(g) for g in grouped.values())!=420:raise RuntimeError('Train42×10 nominal cutoff population differs')
    weighted=[(r['disagreement'],1./len(grouped)/len(grouped[r['scene_id']])/len(grouped[r['scene_id']][r['seed']])) for r in ordered]
    target=sum(w for _,w in weighted)*.9;running=0.;cutoff=None
    for value,w in sorted(weighted):
        running+=w
        if running>=target:cutoff=float(value);break
    if cutoff!=cfg['gate_threshold']:raise RuntimeError('Independent train-only q90 differs from frozen cutoff')
    atomic_json(root/'GATE_INPUT_VERIFIED.json',dict(passed=True,threshold=cutoff,train_states=42,train_rollouts=420,inferences=len(ordered),test_used=False,controller_files=cfg['controller_files'],input_parquet_sha256=report['parquet_sha256']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();verify(a.root)
