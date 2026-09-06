"""Render paper-audit diagnostics from preserved records, without fitting."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
neutral=json.loads((HERE/'evidence/neutral_report.json').read_text())
obs=json.loads((HERE/'evidence/observation_replay_report.json').read_text())
old=json.loads((HERE/'existing_evidence.json').read_text())
run=ROOT/'results/paper_review/20c2bd98b0a3_obs_5629071'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
fig.suptitle('CrashBench paper audit: matched starts are not enough',fontsize=17,fontweight='bold')
values=[];labels=[]
for row in neutral['anchors']:
 labels.append(f'Task {row["task_id"]}: '+row['condition'].replace('_control','').replace('_',' '))
 values.append([2 if row['outcomes'][k]['catastrophe'] else 0 if row['outcomes'][k]['task_success'] else 1 for k in ['base_a','base_b','refresh_a','refresh_b']])
axes[0,0].imshow(values,cmap=ListedColormap(['#95cfac','#f6d58c','#e3959c']),vmin=0,vmax=2,aspect='auto')
for i,row in enumerate(neutral['anchors']):
 for j,k in enumerate(['base_a','base_b','refresh_a','refresh_b']):
  v=values[i][j];axes[0,0].text(j,i,['Success','Safe NC','Crash'][v]+f'\n{row["outcomes"][k]["steps"]} steps',ha='center',va='center',fontsize=9)
axes[0,0].set(xticks=range(4),xticklabels=['Base A','Base B','Refresh A','Refresh B'],yticks=range(6),yticklabels=labels,title='A. Repeated options from the same saved anchor')
for row,col in zip(obs['anchors'],['#2563eb','#9b59b6']):
 t=row['task_id'];diff=row['observation_differences'];x=[r['step'] for r in diff]
 y=[max(r['fields']['full_image']['max_abs'],r['fields']['wrist_image']['max_abs']) for r in diff]
 axes[0,1].step(x,y,where='post',label=f'Task {t}',color=col)
axes[0,1].axvline(10,color='gray',ls='--',lw=1,label='First action divergence')
axes[0,1].annotate('Initial drift: 1 intensity level',xy=(2,1),xytext=(4,32),arrowprops={'arrowstyle':'->','color':'#444'},fontsize=9)
axes[0,1].set(xlabel='Step after anchor (zero-based)',ylabel='Max camera difference (0–255 scale)',title='B. Image differences precede action divergence');axes[0,1].legend()
for task,col in [(0,'#2563eb'),(2,'#9b59b6')]:
 a=json.loads((run/f'task{task}_live_a.json').read_text())
 for mode,style in [('live_b','-'),('replay_a','--')]:
  b=json.loads((run/f'task{task}_{mode}.json').read_text());d=[max(abs(np.asarray(x['state'])-y['state'])) for x,y in zip(a,b)]
  axes[1,0].plot(range(len(d)),d,style,color=col,label=f'Task {task}: '+('live repeat' if mode=='live_b' else 'input replay'))
axes[1,0].set(xlabel='Step after anchor (zero-based)',ylabel='Flat-state max difference (mixed units)',title='C. Input replay removes 35-step state divergence');axes[1,0].legend()
models=[('DirectQ','DirectQ'),('outcome_interaction_500_all','Outcome, all options'),('gain_standardized_500_refresh_only','Standardized gain, Refresh only')]
for i,(key,label) in enumerate(models):
 c=old['policies'][key]['subgroup_counters'];st=c['stale']['refresh_rescue'];ct=c['fresh_control']['refresh_rescue']+c['matched_buffer_control']['refresh_rescue']
 axes[1,1].barh(i,st,color='#399765',label='Stale states' if i==0 else None)
 axes[1,1].barh(i,ct,left=st,color='#dda947',label='Neutral-control states' if i==0 else None)
 axes[1,1].text(st/2,i,str(st),ha='center',va='center');
 if ct:axes[1,1].text(st+ct/2,i,str(ct),ha='center',va='center')
axes[1,1].set(yticks=range(3),yticklabels=[x[1] for x in models],xlabel='Recorded Refresh rescues in exposed development',title='D. Two of the six reported rescues are controls');axes[1,1].invert_yaxis();axes[1,1].legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=2,frameon=False)
fig.supxlabel('Engineering probe: 2 sources / 6 anchors. Development accounting: 12 sources / 324 decisions. No new method-confirmation claim.',fontsize=10)
fig.savefig(HERE/'audit_figure.png',dpi=180);fig.savefig(HERE/'audit_figure.svg')
p=HERE/'audit_figure.svg';p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
