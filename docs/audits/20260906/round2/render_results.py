"""Render round-two scientific comparisons from the saved report; no fitting."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--report',type=Path,required=True)
parser.add_argument('--output-prefix',type=Path,required=True)
args=parser.parse_args()
r=json.loads(args.report.read_text())
names={'outcome_interaction':'Outcome: interaction','outcome_per_option':'Outcome: per-option',
       'gain_layernorm':'Paired gain','gain_positive5':'Gain: positive weight 5','gain_standardized':'Gain: standardized'}
order=list(names)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,ax=plt.subplots(2,2,figsize=(13,9.5),layout='constrained')
fig.suptitle('CrashBench round 2: learning when Refresh is useful',fontsize=17,fontweight='bold')
y=np.arange(len(order))
for role,shift,color in [('train',-.17,'#94a3b8'),('development',.17,'#2563eb')]:
 vals=[r['recipes'][k]['ensemble_checkpoints']['500'][role]['gain_diagnostics']['observation_refresh']['auc'] for k in order]
 ax[0,0].barh(y+shift,vals,height=.30,label=role,color=color)
ax[0,0].set(yticks=y,yticklabels=[names[k] for k in order],xlim=(0,1),xlabel='AUROC of true positive Refresh gain',title='A. Ranking at 500 epochs')
ax[0,0].invert_yaxis();ax[0,0].axvline(.5,color='gray',lw=1,ls='--');ax[0,0].legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=2,frameon=False)
for k,color in [('outcome_per_option','#64748b'),('gain_layernorm','#2563eb'),('gain_standardized','#08916b')]:
 tiny=r['tiny_fit'][k]['checkpoints'];epochs=sorted(map(int,tiny))
 vals=[tiny[str(e)]['gain_diagnostics']['observation_refresh']['mse'] for e in epochs]
 ax[0,1].plot(epochs,vals,'o-',label=names[k],color=color)
ax[0,1].set(xscale='log',yscale='log',xlabel='Training epochs',ylabel='Refresh gain MSE (log)',title='B. Tiny fit: 4 selected train sources')
ax[0,1].set_xticks([100,500,2000],labels=['100','500','2000']);ax[0,1].legend()
for i,k in enumerate(order):
 d=r['recipes'][k]['ensemble_checkpoints']['500']['development_paired_vs_DirectQ']['all_options']['u0']
 lo,hi=d['source_paired_bootstrap_95pct']
 ax[1,0].errorbar(d['delta'],i,xerr=np.array([[d['delta']-lo],[hi-d['delta']]]),fmt='o',color='#2563eb',capsize=4)
ax[1,0].set(yticks=y,yticklabels=[names[k] for k in order],xlabel='Source-macro utility difference vs DirectQ',title='C. Development effect and 95% source interval')
ax[1,0].invert_yaxis();ax[1,0].axvline(0,color='gray',ls='--',lw=1)
for i,k in enumerate(order):
 p=r['recipes'][k]['ensemble_checkpoints']['500']['development']['policies']['all_options']
 a,b=p['refresh_rescues_recovered'],p['base_successes_lost']
 ax[1,1].barh(i,a,color='#08916b',label='Refresh rescues' if i==0 else None)
 ax[1,1].barh(i,-b,color='#dc5963',label='Base successes lost' if i==0 else None)
 ax[1,1].text(a+.15,i,str(a),va='center',fontsize=9)
 ax[1,1].text(-b-.15,i,str(b),ha='right',va='center',fontsize=9)
ax[1,1].set(yticks=y,yticklabels=[names[k] for k in order],xlabel='Decision count (11 available Refresh rescues)',title='D. Recovery vs disruption, all options')
ax[1,1].invert_yaxis();ax[1,1].axvline(0,color='gray',lw=1);ax[1,1].legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=2,frameon=False)
fig.supxlabel('Exposed development: 12 sources / 324 decisions. Bootstrap intervals are not adjusted for model selection.',fontsize=10)
args.output_prefix.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(args.output_prefix.with_suffix('.png'),dpi=180)
fig.savefig(args.output_prefix.with_suffix('.svg'))

# Normalize serializer whitespace without changing SVG geometry.
svg = args.output_prefix.with_suffix(".svg")
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
