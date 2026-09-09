#!/usr/bin/env python3
"""Posthoc closeout only: no fit, no new predictions, no simulator or remote access."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from scripts.expansion.direct_cost_learning import evaluate,paired_means

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output;out.mkdir(parents=True,exist_ok=False)
    inputs={}
    def read(p):
        inputs[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        return json.loads(p.read_text())
    e=ROOT/'docs/audits/20260908/candidate_refresh/verified_evidence'
    r=ROOT/'docs/audits/20260908/direct_cost_learning/run_v1'
    anchors=read(e/'anchors.json');a=read(e/'A.json');b=read(e/'B.json');f=read(r/'freeze.json');m=read(r/'metrics.json')
    assert hashlib.sha256((r/'freeze.json').read_bytes()).hexdigest()==(r/'freeze.sha256').read_text().strip()
    difficult={x['physical_source_id'] for x in anchors if x['age_steps']==3}
    assert len(difficult)==1
    source=next(iter(difficult));idx=next(i for i,x in enumerate(anchors) if x['panel_id']=='b06')
    train=f['models']['metadata'][source]['training_indices']
    target=np.asarray(f['A_target_deltas'])
    assert len(train)==14 and np.all(target[train,2:]==0)
    overrides={}
    for name in ('metadata','observation'):
        altered=copy.deepcopy(f)
        for anchor in anchors:
            if anchor['physical_source_id']==source:altered['choices'][name][anchor['panel_id']]=0
        after=evaluate(anchors,b,altered)['methods'][name]['source_macro']['all']
        overrides[name]={'original':m['methods'][name]['source_macro']['all']['completion_cost_delta'],
                         'posthoc_source_Base_override':after['completion_cost_delta']}
    assert np.isclose(overrides['metadata']['posthoc_source_Base_override'],-.59375)
    assert np.isclose(overrides['observation']['posthoc_source_Base_override'],-.375)
    base=m['methods']['Base']['source_macro']['all'];rule=m['methods']['age_equal1']['source_macro']['all']
    curves={}
    stale={x['panel_id'] for x in anchors if x['condition']=='stale'}
    for option,name in [(0,'Base'),(1,'Refresh')]:
        rows=[x for x in b if x['panel_id'] in stale and x['option']==option]
        curves[name]=[sum(x['task_success'] and x['first_success_step']<=h for x in rows) for h in range(1,201)]
    report={'scope':'posthoc diagnostic, all eight sources retained; no training or deployable rule',
        'difficult_source':source,'training_anchor_count':len(train),'training_success_accident_deltas_all_zero':True,
        'source_override':overrides,'b06_A_completion_delta':paired_means(anchors,a,'A')[idx,0],
        'b06_B_completion_delta':paired_means(anchors,b,'B')[idx,0],
        'age_equal1_relative_completion_saving_percent':-100*rule['completion_cost_delta']/base['completion_cost'],
        'age_equal1_relative_call_saving_percent':-100*rule['inference_calls_delta']/base['inference_calls'],
        'B_stale_deadline_curves_counts_of_36':curves,'input_sha256':inputs}
    (out/'evidence.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for ax in axes:
        for name,y in curves.items():ax.step(range(1,201),np.array(y)/36,where='post',label=name)
        ax.axvline(100,color='grey',ls=':',lw=1);ax.set_xlabel('Post-anchor action deadline');ax.set_ylabel('B stale success fraction');ax.legend()
    axes[0].set(xlim=(0,200),ylim=(0,1),title='Full diagnostic horizon')
    axes[1].set(xlim=(90,110),ylim=(.3,1),title='Deadline neighbourhood (same trajectories)')
    for suffix in ('png','svg'):fig.savefig(out/f'deadline_curve.{suffix}',dpi=170)
    plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k not in ('input_sha256','B_stale_deadline_curves_counts_of_36')},indent=2))
if __name__=='__main__':main()
