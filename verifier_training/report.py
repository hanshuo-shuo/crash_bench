"""Package measured curves and gated A/B proposals after both smoke variants pass."""
import json
from pathlib import Path
import shutil
import sys
from .common import *


def main(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results={v:read(root/v/'RESULT.json') for v in ('A','B')}
    require(all(r['passed'] for r in results.values()),'Both smoke variants must pass')
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for v in ('A','B'):
        rows=[json.loads(l) for l in (root/v/'curve.jsonl').read_text().splitlines()]
        for ax,key in zip(axes,('total','rationale','verdict')):
            if v=='A' and key=='rationale':continue
            ax.plot([r['step'] for r in rows],[r[key] for r in rows],label=v,linewidth=1)
            ax.set(xlabel='Optimizer step',ylabel='Training token CE',title=key);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Same eight training samples: engineering overfit, no generalization estimate')
    fig.tight_layout();fig.savefig(root/'training_curves.png',dpi=180);plt.close(fig)
    proposals={}
    for v in ('A','B'):
        cfg=read(root/'source/verifier_training/configs/smoke.json')
        cfg.update(mode='proposed_supervised_pilot_NOT_LAUNCHED',variant=v,training_samples=120,
            original_source_scenes=1,learning_rate=1e-4,lora_dropout=.05,epochs=10,max_steps=300,
            evaluation='Training fit and grounded generation audit only; no random split accuracy',
            output='binary verdict' if v=='A' else 'grounded observability rationale plus binary verdict',
            approved_for_launch=False,scientific_claim='No cross-source or independent-mechanism generalization')
        proposals[v]=cfg;write(root/f'proposed_{v}.json',cfg)
    write(root/'COMPLETE.json',{'passed':True,'smoke':results,'full_training_launched':False,
        'remaining_data_requirements':['Multiple independent original source scenes and tasks',
            'Independent physics mechanisms beyond one separator proof family',
            'Both labels within each source/mechanism to break source-label correlation',
            'Label-critical state information observable from the permitted inputs; audit hidden and ambiguous evidence',
            'Externally sourced complete witnesses/certificates and grounded relation/action rationales',
            'Ancestry-grouped held-out sources with no paired variants crossing splits',
            'Unknown/partial evidence retained as such; policy failure never used as infeasibility']})
    dest=Path.home()/'crash_bench/results/verifier_training'/root.name
    dest.mkdir(parents=True,exist_ok=True)
    for name in ('COMPLETE.json','training_curves.png','proposed_A.json','proposed_B.json'):
        shutil.copyfile(root/name,dest/name)
    for v in ('A','B'):
        (dest/v).mkdir(exist_ok=True)
        for name in ('RESULT.json','PROVENANCE.json','GRADIENT_PROBE.json','curve.jsonl','generations_before.json','generations_final.json'):
            shutil.copyfile(root/v/name,dest/v/name)

if __name__=='__main__':main(Path(sys.argv[1]))
