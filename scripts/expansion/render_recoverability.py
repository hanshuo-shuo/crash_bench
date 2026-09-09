"""Static mechanism figures from completed direct controller instrumentation."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def render(root):
 c=json.loads((root/'config.json').read_text());records=json.loads((root/'records.json').read_text());children=json.loads((root/'children.json').read_text());out=root/'analysis'
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
 for ax,parent in zip(axes.flat,c['parents']):
  for delay,color in zip(c['offsets'],['#236e96','#d68729','#348568']):
   rr=next((r for r in records if r['parent']==parent and r['offset']==delay and r['option']==1 and r['repeat']==0),None)
   if rr is None:continue
   child=next(x for x in children if x['parent']==parent and x['offset']==delay);physics=rr['physics'];eef=np.array([p['eef'] for p in physics]);bowl=np.array([p['bowl'] for p in physics]);term=rr['suffix_horizons']['440']
   ax.plot(eef[:,0],eef[:,1],color=color,alpha=.65,label=f'+{delay}: {term["reason"]}')
   ax.plot(bowl[:,0],bowl[:,1],color=color,ls=':',lw=1.5);ax.scatter(*eef[0,:2],color=color,s=18)
  if any(x['parent']==parent for x in children):
   ch=next(x for x in children if x['parent']==parent);glass=ch['glass'];ax.add_patch(Circle(glass['pos'][:2],glass['size'][0],facecolor='#b3c7d3',edgecolor='#577281',alpha=.7));ax.scatter(*ch['bowl'][:2],marker='o',color='black',s=25);ax.scatter(*ch['plate'][:2],marker='s',color='black',s=25)
  ax.set(title=parent,xlabel='World x (m)',ylabel='World y (m)',aspect='equal');ax.legend(fontsize=8)
 fig.suptitle('Recorded controller-view trajectories: solid EEF, dotted bowl; first repeat')
 fig.savefig(out/'physical_paths.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(11,8),constrained_layout=True);names=[]
 colors=['#467d9b','#daa150','#5a997b','#8b739a'];groups=['Approach (0–5)','Grasp/lift (6–8)','Carry/place/release (9–11)','Resumed Base']
 for i,child in enumerate(children):
  r=next(r for r in records if r['episode_id']==child['episode_id'] and r['option']==1 and r['repeat']==0);counts=[0]*4;caps=[]
  for p in r['physics']:
   group=3 if not p['controller'] else 0 if p['stage_before']<6 else 1 if p['stage_before']<9 else 2;counts[group]+=1
   if p.get('cap_exit') and p.get('in_leg',-1)+1>=140:caps.append(p['stage_before'])
  left=0
  for j,value in enumerate(counts):ax.barh(i,value,left=left,color=colors[j],label=groups[j] if i==0 else None);left+=value
  ax.text(left+3,i,f'{r["suffix_horizons"]["440"]["reason"]}; caps {caps}',va='center',fontsize=8);names.append(child['episode_id'])
 ax.set_yticks(range(len(names)),names);ax.invert_yaxis();ax.set(xlabel='Actions after recovery decision',title='Directly logged recovery stages, first repeat; unchanged controller');ax.set_xlim(0,600);ax.legend(ncol=2,fontsize=8)
 fig.savefig(out/'controller_stages.png',dpi=160);plt.close(fig)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();render(a.run)
