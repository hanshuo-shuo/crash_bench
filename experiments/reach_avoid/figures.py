"""Standalone scientific figures from saved records, never hand-entered scores."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
COLORS={'feasible':'#167D8D','infeasible':'#C64A47','unknown':'#B69744','safe_completion':'#167D8D','safe_timeout':'#B69744','collision':'#C64A47'}

def read(path):return json.loads(path.read_text())
def finish(fig,path):fig.tight_layout();fig.savefig(path,bbox_inches='tight');plt.close(fig)

def baseline(session,out):
    rows=read(session/'raw/baseline_gpu/COMPLETE.json')['rows']
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),sharey=True)
    for ax,condition in zip(axes,['native','clear']):
        group=[r for r in rows if r['condition']==condition]
        ax.bar([str(r['episode']) for r in group],[r['steps'] for r in group],color=[COLORS[r['outcome']] for r in group])
        for i,r in enumerate(group):ax.text(i,r['steps']+6,r['outcome'].replace('safe_','').replace('_',' ')+'\n'+str(r['steps']),ha='center',fontsize=8)
        ax.set_title('Native bottle' if condition=='native' else 'One fixed clear placement');ax.set_xlabel('Official init index');ax.set_ylim(0,340)
    axes[0].set_ylabel('Execution commands');fig.suptitle('Fixed pi0.5: collision, safe completion, and safe timeout',y=1.04)
    finish(fig,out/'01_baseline.png')
    prompts=read(session/'PROMPT_SUMMARY.json');fig,ax=plt.subplots(figsize=(7,3))
    ax.bar([str(r['episode']) for r in prompts],[r['action_linf'] for r in prompts],color='#6373AC')
    ax.set(xlabel='Clear-placement official init',ylabel='Action chunk maximum absolute change',title='Safety sentence changes native actions; matched duplicates are exact')
    finish(fig,out/'02_prompt_sensitivity.png')

def contact_sheet(session,out):
    cases=['A_slit_open','A_slit_.020','A_slit_.180','A_slit_irrelevant','A_cage_open','A_cage_.020','A_cage_.180','A_cage_irrelevant']
    canvas=Image.new('RGB',(1120,664),'white');draw=ImageDraw.Draw(canvas);font=ImageFont.load_default(size=18)
    for i,case in enumerate(cases):
        source=session/'raw/observe_gpu'/case/'agentview_actual.png';im=Image.open(source).convert('RGB').resize((260,260))
        x=(i%4)*280+10;y=(i//4)*332+56;canvas.paste(im,(x,y));label=read(session/'raw/mechanism_cpu'/case/'summary.json')['label']
        group,family,variant=case.split('_');shown={'open':'open','irrelevant':'off-target'}.get(variant)
        if shown is None:shown=f'{float(variant)*1000:.0f} mm gap'
        draw.text((x,y-50),group+' | '+family.title()+' | '+shown,fill='black',font=font)
        meaning={'feasible':'Feasible: safe witness','infeasible':'Infeasible under RA-SEP-1','unknown':'UNKNOWN'}[label]
        draw.text((x,y-27),meaning,fill=COLORS[label],font=font)
    canvas.save(out/'03_visible_constructions.png')

def boundary(session,out):
    path=session/'CPU_MATRIX_READBACK.json'
    if not path.exists():return
    data=read(path);brackets=data['brackets']
    import collections
    cells=collections.Counter()
    for file in (session/'raw/matrix_cpu').glob('shard_*/*/summary.json'):
        row=read(file)
        if row['variant'] not in ('open','irrelevant'):cells[(float(row['variant'])*1000,row['label'])]+=1
    lows=[b['low']*1000 for b in brackets if b['low'] is not None];highs=[b['high']*1000 for b in brackets if b['high'] is not None]
    fig,ax=plt.subplots(figsize=(9,2.8))
    if lows and highs:
        lo=max(lows);hi=min(highs);ax.axvspan(lo,hi,color='#F0E9D5')
        ax.annotate('',xy=(hi,.68),xytext=(lo,.68),arrowprops=dict(arrowstyle='<->',color='#685C3C'))
        ax.text((hi+lo)/2,.73,f'{hi-lo:.0f} mm sampled unresolved bracket',ha='center')
    for label in ('infeasible','unknown','feasible'):
        values=sorted((gap,n) for (gap,kind),n in cells.items() if kind==label)
        ax.plot([gap for gap,n in values],[1]*len(values),'o',markersize=10,color=COLORS[label],label=label+' ('+','.join(str(n) for gap,n in values)+' cells)')
    ax.set(xlim=(10,190),ylim=(.5,1.15),yticks=[],xticks=sorted({gap for gap,label in cells}),xlabel='Declared aperture parameter (mm)',title='Unknown boundary band retained across all 12 layouts and both constructors')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.28),ncol=3,fontsize=8)
    for side in ('left','right','top'):ax.spines[side].set_visible(False)
    finish(fig,out/'08_unknown_bracket.png')


def readouts(analysis,out):
    result=read(analysis/'READOUT.json');results=result['results'];primary=result['primary_vla_representation']
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),sharey=True)
    for ax,metric in zip(axes,['test','failed_subset']):
        for prefix,label,color in [('layers_','All prefix tokens','#4266A3'),('image_layers_','Image tokens','#A46F36')]:
            values=[results[prefix+'%02d'%i][metric]['auroc'] for i in range(1,19)]
            ax.plot(range(1,19),values,'o-',label=label,color=color,markersize=3)
        for key,label,color in [('own_vision_tower','Own vision tower','#167D8D'),('dino_patch','DINOv2 patch mean','#7B5695')]:
            ax.axhline(results[key][metric]['auroc'],linestyle='--',label=label,color=color)
        ax.axhline(.5,color='gray',linewidth=.6);ax.set(xlabel='Native residual layer (1-based)',ylim=(0,1.03),title='All independently labeled test states' if metric=='test' else 'Among fixed-policy failures')
    axes[0].set_ylabel('Held-out cage AUROC');axes[0].legend(fontsize=7,loc='lower left')
    fig.suptitle('Frozen pi0.5 readouts | primary selected on validation: '+primary,y=1.04)
    finish(fig,out/'04_layer_readouts.png')
    keys=['own_vision_tower','projected_vision',primary,'native_final','dino_patch','rgb_pooled_red','visibility','geometry','knows_fixed']
    keys=list(dict.fromkeys(keys));vals=[results[k]['test']['auroc'] for k in keys]
    intervals=[results[k]['auroc_interval']['interval'] for k in keys]
    for appendix in (False,True):
        fig,ax=plt.subplots(figsize=(9,4.2));ax.barh(range(len(keys)),vals,color='#4266A3',alpha=.8)
        for i,(v,ci) in enumerate(zip(vals,intervals)):
            if appendix and ci:ax.plot(ci,[i,i],color='black');ax.plot(ci,[i,i],'|',color='black')
            ax.text(min(v+.02,.94),i,f'{v:.2f}',va='center',fontsize=8)
        ax.set(yticks=range(len(keys)),yticklabels=keys,xlim=(0,1.03),
            xlabel='AUROC; exploratory 2.5%-97.5% cluster-bootstrap quantiles' if appendix else 'Pooled held-out AUROC (descriptive)',
            title='Appendix: four clusters do not establish nominal coverage' if appendix else 'Constructor transfer; layout-specific comparisons are primary')
        ax.axvline(.5,color='gray',linestyle='--');ax.invert_yaxis()
        finish(fig,out/('10_exploratory_bootstrap.png' if appendix else '05_comparators.png'))
    curves=read(analysis/'LEARNING_CURVES.json');fig,ax=plt.subplots(figsize=(7,3.5))
    for key in [primary,'own_vision_tower','dino_patch','rgb_pooled_red']:
        valid=[p for p in curves[key] if 'test' in p]
        ax.plot([p['groups'] for p in valid],[p['test']['auroc'] for p in valid],'o-',label=key)
    ax.set(xticks=[2,4,6],xlabel='Training scene groups (fixed prefixes)',ylabel='Held-out cage AUROC',ylim=(0,1.03),title='Equal group-count learning curves; validation selects each head');ax.legend(fontsize=8)
    finish(fig,out/'06_learning_curves.png')
    inc=read(analysis/'INCREMENTAL.json')
    if 'not_estimable' not in inc:
        fig,ax=plt.subplots(figsize=(7,3.6));gs=list(inc['raw_groups']);x=np.arange(len(gs))
        for offset,key,label,col in [(-.17,'safe_only','Initial failure score alone','#8E98A5'),(.17,'safe_plus_feasibility','Failure score + feasibility','#167D8D')]:
            vals=[inc['raw_groups'][g][key]['auroc'] for g in gs];ax.bar(x+offset,[v if v is not None else np.nan for v in vals],.34,label=label,color=col)
        ax.set(xticks=x,xticklabels=gs,ylim=(0,1.05),ylabel='Within-group infeasibility AUROC',title='SAFE-style initial-state failure probe\nRetrospective failures; descriptive paired layouts');ax.legend(fontsize=8)
        finish(fig,out/'07_incremental.png')


def layout_figure(session,out):
    path=session/'deliverables/layouts/LAYOUT_COMPARISONS.json'
    if not path.exists():return
    data=read(path);primary=data['primary_representation'];layouts=data['layouts']
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),sharey=True)
    for ax,population,title in zip(axes,('all_known','defined_failures'),('Known initial labels','Retrospective policy failures')):
        for key,label,color,marker in [(primary,'Selected native layer','#4266A3','o'),('own_vision_tower','Own vision tower','#167D8D','s'),
            ('native_final','Final native features','#A46F36','^'),('dino_patch','DINOv2 patches','#7B5695','d')]:
            ax.plot([r['scene_group'] for r in layouts],[r['metrics'][population][key]['auroc'] for r in layouts],marker+'-',label=label,color=color,alpha=.85)
        ax.set(title=title,xlabel='Held-out layout',ylim=(0,1.04));ax.axhline(.5,color='gray',linewidth=.6)
    axes[0].set_ylabel('Within-layout AUROC');axes[0].legend(fontsize=7,loc='lower left')
    fig.suptitle('All four paired layouts; no significance claim',y=1.04)
    finish(fig,out/'09_layout_readouts.png')


def nuisance(supplement,out):
    import csv
    data=read(supplement/'NUISANCE_DIAGNOSTICS.json');names=['nominal_width_only','target_visibility_only','width_plus_target_visibility']
    labels=['Nominal width','Target pixels','Width + target pixels'];fig,axes=plt.subplots(1,2,figsize=(10,3.6),sharey=True)
    populations=['all_known_variants','numeric_gap_variants']
    for ax,population,title in zip(axes,populations,['All known variants','Numeric-gap subset']):
        models=data['populations'][population]['models'];x=np.arange(3)
        for offset,key,label,color in [(-.17,'test','All known test states','#4266A3'),(.17,'failed_subset','Defined policy failures','#167D8D')]:
            ax.bar(x+offset,[models[n][key]['auroc'] for n in names],.34,label=label,color=color)
        ax.set(xticks=x,xticklabels=labels,ylim=(0,1.04),title=title);ax.tick_params(axis='x',labelsize=8);ax.axhline(.5,color='gray',linewidth=.6)
    axes[0].set_ylabel('Test-cage AUROC (descriptive)');axes[0].legend(fontsize=7,loc='lower left')
    fig.suptitle('Supplement: privileged metadata and segmentation diagnostics',y=1.04)
    finish(fig,out/'11_nuisance_diagnostics.png')
    with (supplement/'unknown_predictions.csv').open() as stream:rows=list(csv.DictReader(stream))
    rows=[r for r in rows if r['split']=='test' and r['constructor']=='cage']
    gs=sorted({r['scene_group'] for r in rows});offsets=dict(zip(gs,np.linspace(-.06,.06,len(gs))))
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),sharey=True)
    for ax,population,title in zip(axes,populations,['Fit on all known variants','Fit on numeric-gap subset']):
        for gap,offset,color in [('.034',-.10,'#4266A3'),('.060',.10,'#B69744')]:
            selected=[r for r in rows if r['population']==population and r['variant']==gap]
            ax.scatter([names.index(r['diagnostic'])+offset+offsets[r['scene_group']] for r in selected],
                [float(r['score']) for r in selected],label=f'{float(gap)*1000:.0f} mm UNKNOWN',color=color,s=24,alpha=.8)
        ax.set(xticks=range(3),xticklabels=labels,ylim=(-.03,1.03),title=title);ax.tick_params(axis='x',labelsize=8)
    axes[0].set_ylabel('Infeasibility readout score (unverified)');axes[0].legend(fontsize=7,loc='lower left')
    fig.suptitle('Eight UNKNOWN test-cage states; score distributions, no accuracy',y=1.04)
    finish(fig,out/'12_unknown_scores.png')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('session',type=Path);p.add_argument('out',type=Path);p.add_argument('--analysis',type=Path);p.add_argument('--supplement',type=Path);a=p.parse_args();a.out.mkdir(exist_ok=True,parents=True)
    baseline(a.session,a.out);contact_sheet(a.session,a.out);boundary(a.session,a.out)
    if a.analysis:readouts(a.analysis,a.out)
    layout_figure(a.session,a.out)
    if a.supplement:nuisance(a.supplement,a.out)
