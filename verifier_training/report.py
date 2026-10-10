"""Package measured curves and gated A/B proposals after both smoke variants pass."""
import json
import argparse
import os
from pathlib import Path
import shutil
import sys
from .common import *


def main(root, out=None):
    out = root if out is None else out
    out.mkdir(parents=True, exist_ok=True)
    require(not (out/'COMPLETE.json').exists(), 'Do not overwrite a completed report')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results={v:read(root/v/'RESULT.json') for v in ('A','B')}
    require(all(r['passed'] for r in results.values()),'Both smoke variants must pass')
    require(all(read(root/v/'PROVENANCE.json').get('input_contract')==INPUT_CONTRACT for v in ('A','B')), 'Incompatible A/B input contracts')
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for v in ('A','B'):
        rows=[json.loads(l) for l in (root/v/'curve.jsonl').read_text().splitlines()]
        for ax,key in zip(axes,('total','rationale','verdict')):
            if v=='A' and key=='rationale':continue
            ax.plot([r['step'] for r in rows],[r[key] for r in rows],label=v,linewidth=1,
                    color={'A':'#1f77b4','B':'#d95f02'}[v])
            ax.set(xlabel='Optimizer step',ylabel='Training token CE',title=key);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Same eight training samples: engineering overfit, no generalization estimate')
    fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=180);plt.close(fig)
    proposals={}
    for v in ('A','B'):
        cfg=read(root/'source/verifier_training/configs/smoke.json')
        cfg.pop('stop_rule')
        cfg.pop('evaluate_every')
        cfg.update(mode='proposed_supervised_pilot_NOT_LAUNCHED',variant=v,training_samples=120,
            original_source_scenes=1,learning_rate=1e-4,lora_dropout=.05,epochs=10,max_steps=300,
            model_revision=read(root/v/'PROVENANCE.json')['model_revision'],
            dataset_sha256=read(root/v/'PROVENANCE.json')['dataset_sha256'],
            input_contract=INPUT_CONTRACT,
            optimizer='AdamW',weight_decay=0.0,learning_rate_schedule='constant',
            resources={'account':'p33100','partition':'gengpu','gpus':1,'gpu_type':'a100',
                       'cpus':4,'memory_gib':48,'wall_minutes_per_arm':60},
            sampling='Shuffle all 120 examples each epoch; same ordering for A and B',
            stopping='Fixed ten epochs; stop on nonfinite loss, infrastructure failure or evidence-integrity failure',
            evaluation='Training fit and grounded generation audit only; no random split accuracy',
            output='binary verdict' if v=='A' else 'grounded observability rationale plus binary verdict',
            approved_for_launch=False,scientific_claim='No cross-source or independent-mechanism generalization')
        cfg['loss'] = ('Mean assistant-token next-token cross entropy for verdict/format/EOS' if v=='A' else
                       'Mean assistant-token next-token cross entropy for every rationale, verdict, format and EOS token')
        proposals[v]=cfg;write(out/f'proposed_{v}.json',cfg)
    write(out/'COMPLETE.json',{'passed':True,'smoke':results,'full_training_launched':False,
        'training_root':str(root),'report_code_commit':os.environ.get('CB_CODE_COMMIT'),
        'remaining_data_requirements':['Multiple independent original source scenes and tasks',
            'Independent physics mechanisms beyond one separator proof family',
            'Both labels within each source/mechanism to break source-label correlation',
            'Matched target visibility and nuisance cues across labels',
            'Label-critical state information observable from the permitted inputs; audit hidden and ambiguous evidence',
            'Externally sourced complete witnesses/certificates and grounded relation/action rationales',
            'Ancestry-grouped held-out sources with no paired variants crossing splits',
            'Unknown/partial evidence retained as such; policy failure never used as infeasibility']})
    dest=Path.home()/'crash_bench/results/verifier_training'/out.name
    dest.mkdir(parents=True,exist_ok=True)
    for name in ('COMPLETE.json','training_curves.png','proposed_A.json','proposed_B.json'):
        shutil.copyfile(out/name,dest/name)
    for v in ('A','B'):
        (dest/v).mkdir(exist_ok=True)
        for name in ('RESULT.json','PROVENANCE.json','GRADIENT_PROBE.json','curve.jsonl','generations_before.json','generations_final.json'):
            shutil.copyfile(root/v/name,dest/v/name)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--out',type=Path)
    a=p.parse_args();main(a.root,a.out)
