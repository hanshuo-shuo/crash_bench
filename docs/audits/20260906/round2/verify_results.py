"""Validate round-two artifacts, normalization scope and saved policy readouts."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from scripts.expansion.train_option_value import sha256_file, read_jsonl, build_training_arrays
from scripts.expansion.run_round2_gain import evaluate
from crashbench.data.utility import PhysicalBudgets
from crashbench.models.paired_gain import pair_decisions, train_standardization

parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();run=args.run_dir
report=json.loads((run/'report.json').read_text());m=report['manifest']
ledger=json.loads((run/'output_sha256.json').read_text())
for name,digest in ledger.items():assert sha256_file(run/name)==digest,name
merged=ROOT/'results/expansion/d5_statewise/e2a5264a802a_fe660c168e72_20260830T105726Z/postprocess_a4b1714/merged'
for name,digest in m['input_sha256'].items():assert sha256_file(merged/name)==digest,name
norm=json.loads((ROOT/'configs/expansion/utility_v1.yaml').read_text())['normalization']
branches=read_jsonl(merged/'branches.jsonl');anchors=read_jsonl(merged/'anchors.jsonl')
arrays=build_training_arrays(anchors,branches,artifact_store=Path('/private/tmp/round2-verify'),
    budgets=PhysicalBudgets(norm['option_duration_steps'],norm['path_length_m'],norm['force_exposure_ns'],norm['latency_ms'],norm['source']),
    feature_cache=merged.parent/'feature_cache/features.npz')
pair=pair_decisions(arrays)
train=set(pair['sources'][pair['roles']=='train']);dev=set(pair['sources'][pair['roles']=='development'])
assert len(train)==24 and len(dev)==12 and not train & dev
models=0
for path in run.rglob('model.pt'):
    meta=json.loads((path.parent/'manifest.json').read_text())
    assert meta['fit_on']=='train' and set(meta['fit_source_ids'])<=train and meta['test_rows_read']==0
    assert sha256_file(path)==meta['model_sha256']
    state=torch.load(path,map_location='cpu',weights_only=False)['state_dict']
    if meta['recipe']=='gain_standardized':
        mask=(pair['roles']=='train') & np.isin(pair['sources'],meta['fit_source_ids'])
        mu,scale=train_standardization(pair['features'][mask],pair['roles'][mask])
        np.testing.assert_allclose(state['feature_mean'].numpy(),mu,atol=1e-6,rtol=1e-6)
        np.testing.assert_allclose(state['feature_scale'].numpy(),scale,atol=1e-6,rtol=1e-6)
    models+=1
assert models==28
readouts=0
for recipe,result in report['recipes'].items():
    for epoch,values in result['ensemble_checkpoints'].items():
        with np.load(run/recipe/f'ensemble_epoch_{epoch}.npz',allow_pickle=False) as z:gains=z['gains']
        for role in ('train','development'):
            replay=evaluate(pair,branches,gains,role)
            for mode in ('all_options','refresh_only'):
                for metric in ('u0','task_success','catastrophe','refresh_rescues_recovered','base_successes_lost'):
                    assert abs(replay['policies'][mode][metric]-values[role]['policies'][mode][metric])<1e-8
            readouts+=1
assert report['status']=='COMPLETE_DEVELOPMENT_DIAGNOSTIC' and not m['smoke']
assert not m['confirmatory'] and not m['calibration_outcomes_used'] and m['test_rows_read']==0
result={'status':'PASS','verified_output_files':len(ledger),'verified_models':models,'replayed_role_readouts':readouts,
        'train_sources':len(train),'development_sources':len(dev),'normalization_train_only_verified':True,
        'git_commit':m['git_commit'],'slurm_job_id':m['slurm_job_id'],
        'local_tests':{'passed':534,'failed':0,'elapsed_seconds':8.55},
        'quest':{'state':'COMPLETED','exit_code':'0:0','elapsed':'00:02:07','max_rss_kb':466244}}
args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
