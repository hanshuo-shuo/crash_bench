"""Render scientific diagnostics from the already frozen analysis tables."""
import argparse,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def render(root):
    curves=read(root/'success_cumulative_curve.csv');sources=read(root/'per_source.csv');compare=read(root/'paired_comparisons.csv')
    methods=['Base','AlwaysDetour','RiskDetour','BenefitGate','RiskOnlyBenefit','A_reference','A_long_reference']
    labels=['Base','Always Detour','Risk → Detour','Benefit gate','Risk-only benefit','A reference (H)','A reference (2H)']
    colors=['#424b57','#da8b31','#a64d54','#26799c','#735a91','#4b926c','#8bae78']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,1,figsize=(10,13),constrained_layout=True)
    for method,label,color in zip(methods,labels,colors):axes[0].plot([int(x['action_step']) for x in curves],[100*float(x[method]) for x in curves],label=label,color=color,lw=1.8,ls='--' if method.startswith('A_') else '-')
    axes[0].axvline(220,color='#888',ls=':',lw=1);axes[0].axvline(440,color='#888',ls=':',lw=1)
    axes[0].set(xlabel='Task actions from episode start',ylabel='Source-macro cumulative success (%)',title='Same continuous trajectory: H=220 and 2H=440');axes[0].legend(ncol=3,fontsize=8)
    ids=sorted({r['source'] for r in sources});index={(r['source'],r['method']):r for r in sources if r['horizon']=='440'}
    yy=np.arange(len(ids));width=.36
    for j,(ref,color) in enumerate([('Base','#26799c'),('RiskDetour','#a64d54')]):
        vals=[100*(float(index[s,'BenefitGate']['success'])-float(index[s,ref]['success'])) for s in ids]
        axes[1].barh(yy+(j-.5)*width,vals,height=width,label='Benefit − '+ref,color=color)
    axes[1].set_yticks(yy,[s[:8] for s in ids]);axes[1].axvline(0,color='#777',lw=.7);axes[1].set(xlabel='2H success difference (percentage points)',ylabel='Physical source (SHA prefix)',title='All 16 sources retained');axes[1].legend(fontsize=8)
    for i,method in enumerate(methods[1:]):
        for j,h in enumerate(('220','440')):
            r=next(r for r in compare if r['horizon']==h and r['method']==method and r['baseline']=='Base' and r['metric']=='success')
            v,lo,hi=[100*float(r[k]) for k in ('delta','lo','hi')]
            axes[2].errorbar(v,i+(j-.5)*.22,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='o',color=('#26799c','#a64d54')[j],capsize=3,label=('H','2H')[j] if i==0 else None)
    axes[2].set_yticks(range(len(methods)-1),labels[1:]);axes[2].axvline(0,color='#777',lw=.7);axes[2].set(xlabel='Success difference vs Base (percentage points)',title='Shared-source bootstrap 95% descriptive intervals');axes[2].legend()
    fig.savefig(root/'diagnostics.png',dpi=180);fig.savefig(root/'diagnostics.svg');plt.close(fig)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);a=p.parse_args();render(a.analysis)
