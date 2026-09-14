"""Conference-size figures and compact fresh-source records from completed runs."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'docs/audits/20260913/repeat_value'
COLORS={'A':'#bb6a3b','B':'#237f80','C':'#50578a'}


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def csvrows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))
def one(rows,**selection):
    matches=[r for r in rows if all(str(r.get(k))==str(v) for k,v in selection.items())]
    if len(matches)!=1:raise ValueError('missing or duplicate figure cell: '+str(selection))
    return matches[0]
def save(fig,path):
    for ext in ('png','svg','pdf'):fig.savefig(path.with_suffix('.'+ext),dpi=250)
    plt.close(fig)


def summarize_fresh(run,support,compact,out):
    """Describe the entire fixed population, including A-unselected C opportunities."""
    anchors=read(run/'anchors.json');authoring=read(run/'authoring.json')
    identities=read(run/'source_identities.json');freeze=read(run/'freeze_A.json')
    choice=one(freeze['choices'],method='real_full',selection_horizon=440)['choices']
    rows=[r for r in support if r['horizon']=='440']
    anchor_by_id={a['episode_id']:a for a in anchors}
    by_condition=[]
    for condition in ('glass','offpath','noglass'):
        selected=[r for r in rows if r['condition']==condition]
        if len(selected)!=12:raise ValueError('fresh condition denominator changed')
        for phase in ('A','B','C'):
            by_condition.append(dict(condition=condition,phase=phase,cells=len(selected),
                candidates=sum(anchor_by_id[r['id']]['triggered'] for r in selected),
                Base_success=float(np.mean([float(r[phase+'_Base_success']) for r in selected])),
                Detour_success=float(np.mean([float(r[phase+'_Detour_success']) for r in selected])),
                Base_accident=float(np.mean([float(r[phase+'_Base_accident']) for r in selected])),
                Detour_accident=float(np.mean([float(r[phase+'_Detour_accident']) for r in selected])),
                positive_cells=[r['id'] for r in selected if float(r[phase+'_gain'])>0],
                negative_cells=[r['id'] for r in selected if float(r[phase+'_gain'])<0],
                actual_R_success_cells=[r['id'] for r in selected if anchor_by_id[r['id']]['triggered']
                    and float(r[phase+'_Detour_success'])>0]))
    shared=[dict(episode_id=a['episode_id'],source_id=a['source_id'],condition=a['condition'],
        horizons=a['horizons'],terminal_event=next((e for e in a['events'] if e['reason']),None))
        for a in anchors if not a['triggered']]
    nominal=[dict(source_id=a['source_id'],nominal_success=a['nominal_success'],
        terminal_event=next((e for e in a['events'] if e['reason']),None),
        requested_fraction=a['requested_fraction'],actual_fraction=a['actual_fraction'],
        target_clearance=a['target_clearance']) for a in authoring]
    records=dict(horizon=440,n_physical_sources=len(identities),n_episodes=len(anchors),
        actual_branches={p:len(v) for p,v in compact.items()},
        authoring_nominal_successes=sum(a['nominal_success'] for a in authoring),
        authoring=nominal,shared_prefix_outcomes=shared,conditions=by_condition,
        A_selected_cells=[eid for eid,value in choice.items() if value],
        positive_in_all_three_blocks=[r['id'] for r in rows if all(float(r[p+'_gain'])>0 for p in ('A','B','C'))],
        C_positive_unselected_cells=[r['id'] for r in rows if float(r['C_gain'])>0 and not choice[r['id']]],
        interpretation='All 12 unscreened resets retained; shared prefixes do not represent executed R branches. No C-based selection or source replacement.')
    (out/'fresh_support_summary.json').write_text(json.dumps(records,indent=2)+'\n')
def bars(ax,rows,phases,intervals=True):
    width=.72/len(phases)
    for j,p in enumerate(phases):
        x=np.arange(len(rows))+(j-(len(phases)-1)/2)*width
        means=np.array([float(r[p+'_gain'])*100 for r in rows])
        ax.bar(x,means,width=width*.96,color=COLORS[p],label=p)
        if intervals and all(p+'_gain_lo' in r for r in rows):
            low=np.array([float(r[p+'_gain_lo'])*100 for r in rows])
            high=np.array([float(r[p+'_gain_hi'])*100 for r in rows])
            ax.errorbar(x,means,yerr=[np.maximum(0,means-low),np.maximum(0,high-means)],
                fmt='none',ecolor='#253641',elinewidth=.7,capsize=1.6)
    ax.axhline(0,color='#9ba8ad',linewidth=.7)


def main():
    p=argparse.ArgumentParser();p.add_argument('--fresh-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Scientific figures run on Quest')
    completed=read(args.fresh_run/'complete.json')
    if completed['status']!='COMPLETE':raise ValueError('fresh study incomplete')
    for relative,digest in completed['artifacts'].items():
        if sha(args.fresh_run/relative)!=digest:raise ValueError('completed fresh artifact changed')
    for relative,digest in read(args.fresh_run/'prepare_complete.json')['input_sha256'].items():
        if sha(args.fresh_run/relative)!=digest:raise ValueError('fresh population artifact changed')
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    paths=[OLD/'evidence/overall.csv',OLD/'evidence/curves.csv',
        OLD/'c_detour/analysis/overall.csv',OLD/'c_refresh/analysis/overall.csv',
        args.fresh_run/'analysis/overall.csv',args.fresh_run/'analysis/arm_support.csv']
    ab=csvrows(paths[0]);curves=csvrows(paths[1]);c=csvrows(paths[2])+csvrows(paths[3])
    fresh=csvrows(paths[4]);support=csvrows(paths[5])
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,
        'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
        'legend.fontsize':7.5,'svg.fonttype':'none','pdf.fonttype':42})
    panels=[('detour',440,'Detour\n440'),('candidate_refresh',100,'Refresh\n100'),
            ('candidate_refresh',200,'Refresh\n200'),('selection_retest',100,'Retest\n100')]
    fig,axes=plt.subplots(1,2,figsize=(5.5,2.65),sharey=True)
    for ax,method,title in zip(axes,('real_one','pseudo_one'),('Real intervention','Same-policy pseudo-options')):
        rows=[one(ab,panel=name,method=method,selection_horizon=h,evaluation_horizon=h,subset='all') for name,h,_ in panels]
        bars(ax,rows,('A','B'));ax.set_xticks(range(4),[label for _,_,label in panels]);ax.set_title(title,loc='left')
    axes[0].set_ylabel('Safe success gain (pp)')
    axes[1].legend(frameon=False,title='Execution block',title_fontsize=7.5)
    fig.tight_layout(pad=.7,w_pad=1.0);save(fig,out/'real_and_pseudo')
    fig,axes=plt.subplots(1,2,figsize=(5.5,2.7))
    for ax,name,hs,treatment,title in [(axes[0],'candidate_refresh',(100,200),'stale','Refresh: deadline-sensitive'),
                                     (axes[1],'detour',(220,440),'glass','Detour: later completion')]:
        for h,style in zip(hs,('-','--')):
            rows=[r for r in curves if r['panel']==name and r['selection_horizon']==str(h) and r['subset']=='condition:'+treatment]
            x=[float(r['deadline']) for r in rows]
            if h==hs[0]:ax.plot(x,[100*float(r['B_base_success']) for r in rows],color='#667781',label='Base')
            ax.plot(x,[100*float(r['B_selected_success']) for r in rows],style,
                color=COLORS['B'] if h==hs[0] else COLORS['A'],label=f'A choice at {h}')
        ax.axvline(hs[0],linestyle=':',linewidth=.7,color='#9ba8ad')
        ax.set(xlabel='Action deadline',ylabel='Safe success (%)',ylim=(-2,102),xlim=(0,hs[1]))
        ax.set_title(title,loc='left');ax.legend(frameon=False,loc='lower right',fontsize=7)
    fig.tight_layout(pad=.7,w_pad=1);save(fig,out/'deadline_curves')
    panels=[('candidate_refresh',100,'Refresh\n100'),('candidate_refresh',200,'Refresh\n200'),
            ('detour',220,'Detour\n220'),('detour',440,'Detour\n440')]
    fig,axes=plt.subplots(1,2,figsize=(5.5,2.75),sharey=True)
    for ax,method,title in zip(axes,('real_one','real_full'),('One execution per arm','All repeats per arm')):
        rows=[]
        for name,h,_ in panels:
            sel=dict(panel=name,method=method,selection_horizon=h,evaluation_horizon=h,subset='all')
            a,b=one(ab,**sel),one(c,**sel)
            if any(abs(float(a[k])-float(b[k]))>1e-12 for k in ('A_gain','B_gain')):
                raise ValueError('original forecasts changed')
            rows.append(dict(a,**{k:v for k,v in b.items() if k.startswith('C_')}))
        bars(ax,rows,('A','B','C'));ax.set_xticks(range(4),[label for _,_,label in panels]);ax.set_title(title,loc='left')
    axes[0].set_ylabel('Safe success gain (pp)');axes[1].legend(frameon=False,ncol=3,loc='upper right')
    fig.tight_layout(pad=.7,w_pad=1);save(fig,out/'abc_value')
    fig,axes=plt.subplots(1,2,figsize=(5.5,2.8))
    methods=('real_full','real_one','pseudo_one','pseudo_one_swapped')
    rows=[one(fresh,method=m,selection_horizon=440,evaluation_horizon=440,subset='all') for m in methods]
    bars(axes[0],rows,('A','B','C'),intervals=False)
    axes[0].set_xticks(range(4),['Two\nrepeats','One\nrepeat','Pseudo','Pseudo\nswapped'])
    axes[0].set_title('Fresh sources: locked A choices',loc='left')
    axes[0].set_ylabel('Safe success gain (pp)');axes[0].legend(frameon=False,ncol=3)
    rows=sorted([r for r in support if r['horizon']=='440' and r['condition']=='glass'],key=lambda r:r['source_id'])
    for phase,marker in [('A','o'),('B','s'),('C','^')]:
        axes[1].plot(range(12),[100*float(r[phase+'_gain']) for r in rows],
            color=COLORS[phase],marker=marker,markersize=3.5,linewidth=.8,label=phase)
    axes[1].axhline(0,color='#9ba8ad',linewidth=.7)
    axes[1].set_xticks(range(12),[r['source_id'] for r in rows],rotation=90,fontsize=7)
    axes[1].set_title('All on-path sources',loc='left');axes[1].set_ylabel('Detour minus Base (pp)')
    fig.tight_layout(pad=.7,w_pad=1);save(fig,out/'fresh_value')
    # Lossless terminal/provenance fields, with raw event/physics sequences left on Quest.
    compact={}
    for phase in ('A','B','C'):
        path=args.fresh_run/(phase+'.json');paths.append(path)
        compact[phase]=[]
        for r in read(path):
            small={k:v for k,v in r.items() if k not in ('events','physics')}
            small['terminal_event']=next((e for e in r['events'] if e['reason']),None)
            compact[phase].append(small)
    (out/'fresh_terminal_records.json').write_text(json.dumps(compact,indent=2)+'\n')
    summarize_fresh(args.fresh_run,support,compact,out)
    paths.extend(args.fresh_run/rel for rel in ('anchors.json','authoring.json','source_identities.json','freeze_A.json'))
    meta=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        job=os.environ['SLURM_JOB_ID'],operation='fixed-data presentation and terminal extraction; no refit or simulation',
        input_sha256={str(p):sha(p) for p in paths})
    (out/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps({p.name:sha(p) for p in out.iterdir() if p.is_file()},indent=2)+'\n')
    print('Conference figures and compact fresh records:',out)


if __name__=='__main__':main()
