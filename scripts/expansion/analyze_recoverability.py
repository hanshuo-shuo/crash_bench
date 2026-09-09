"""Analyze the frozen tiny mechanism probe without fitting or selecting variants."""
import argparse,csv,json
from pathlib import Path
import numpy as np

def table(path,rows):
 with path.open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def analyze(root):
 read=lambda n:json.loads((root/n).read_text());complete=read('complete.json');assert complete['status']=='COMPLETE'
 children=read('children.json');records=read('records.json');config=read('config.json');missing=read('unavailable.json');out=root/'analysis';out.mkdir(exist_ok=False)
 lookup={(c['parent'],c['offset']):c for c in children};rs=[];stages=[];cells=[]
 assert len(lookup)==len(children)
 assert len(records)==complete['scored_branches']
 assert len({(r['episode_id'],r['repeat'],r['option']) for r in records})==len(records)
 for r in records:
  for h,term in r['suffix_horizons'].items():rs.append({k:r[k] for k in ('episode_id','parent','offset','repeat','option')}|{'suffix_horizon':h}|term)
  if r['option']==1:
   by={}
   for x in r['physics']:
    if not x['controller']:continue
    i=x['stage_before'];s=by.setdefault(i,{'start':x['step'],'end':x['step'],'cap_exit':False,'min_error':x['target_error'],'kind':x['leg_kind']});s['end']=x['step'];s['cap_exit']|=x['cap_exit'] and x.get('in_leg',-1)+1>=140
    if x['target_error'] is not None:s['min_error']=min(s['min_error'],x['target_error'])
   for i,s in by.items():stages.append({'parent':r['parent'],'offset':r['offset'],'repeat':r['repeat'],'stage':i,'duration':s['end']-s['start']+1}|s)
 for parent in config['parents']:
  for delay in config['offsets']:
   c=lookup.get((parent,delay));rr=[r for r in records if r['parent']==parent and r['offset']==delay]
   row={'parent':parent,'offset':delay,'available':c is not None,'unavailable_reason':next((x['reason'] for x in missing if x['parent']==parent and x['offset']==delay),'')}
   for h in (220,440):
    for o in (0,1):
     v=[r['suffix_horizons'][str(h)] for r in rr if r['option']==o]
     row[f'o{o}_success_{h}']=float(np.mean([x['success'] for x in v])) if v else None
     row[f'o{o}_accident_{h}']=float(np.mean([x['accident'] for x in v])) if v else None
    row[f'delta_success_{h}']=row[f'o1_success_{h}']-row[f'o0_success_{h}'] if c else None
   row['absolute_440_R_success']=float(np.mean([r['horizons']['440']['success'] for r in rr if r['option']==1])) if c else None
   row['R_success_suffix_steps']=json.dumps([r['suffix_horizons']['440']['success_step'] for r in rr if r['option']==1])
   row['R_reasons']=json.dumps([r['suffix_horizons']['440']['reason'] for r in rr if r['option']==1])
   row['R_cap_exits']=json.dumps([sum(bool(x['cap_exit']) and x.get('in_leg',-1)+1>=140 for x in r['physics']) for r in rr if r['option']==1])
   row['eef_bowl_xy']=float(np.linalg.norm(np.array(c['robot_state'][:2])-np.array(c['bowl'][:2]))) if c else None
   row['eef_glass_xy']=float(np.linalg.norm(np.array(c['robot_state'][:2])-np.array(c['glass']['pos'][:2]))) if c else None
   cells.append(row)
 curves=[];mechanism=[];costs=[]
 for c in children:
  for option in (0,1):
   rr=[r for r in records if r['episode_id']==c['episode_id'] and r['option']==option]
   assert len(rr)==2
   for h in range(441):
    terminals=[next((e for e in r['events'] if c['anchor_step']<e['step']<=c['anchor_step']+h and e['reason']),None) for r in rr]
    curves.append({'parent':c['parent'],'offset':c['offset'],'option':option,'suffix_actions':h,'success':sum(bool(e and e['reason']=='success') for e in terminals)/len(rr),'accident':sum(bool(e and e['reason']=='accident') for e in terminals)/len(rr)})
   for r in rr:
    costs.append({'parent':r['parent'],'offset':r['offset'],'option':option,'repeat':r['repeat'],'branch_elapsed_seconds':r['branch_elapsed_seconds'],'charged_calls':r['suffix_horizons']['440']['calls'],'executed_actions':r['suffix_horizons']['440']['steps'],'controller_actions':r['suffix_horizons']['440']['controller_steps']})
    if option==1 and r['physics']:
     b=np.array([p['bowl'] for p in r['physics']]);mechanism.append({'parent':r['parent'],'offset':r['offset'],'repeat':r['repeat'],'bowl_max_lift_m':float((b[:,2]-b[0,2]).max()),'bowl_max_xy_motion_m':float(np.linalg.norm(b[:,:2]-b[0,:2],axis=1).max()),'last_logged_stage':r['physics'][-1]['stage_before'],'resumed_base_step':r['controller_resume_base_step']})
 table(out/'cumulative_curves.csv',curves);table(out/'costs.csv',costs)
 if mechanism:table(out/'mechanism.csv',mechanism)
 table(out/'cells.csv',cells);table(out/'terminals.csv',rs)
 if stages:table(out/'stages.csv',stages)
 findings={'complete':complete,'unavailable':missing,'new_R_success_cells':[],'new_task_rescue_cells':[],'lost_positive_cells':[],'limitations':['four selected exposed parents','two repeats per child','fixed-controller failure is not universal irrecoverability','Base progression changes multiple physical variables','uniform suffix budget differs from historical absolute deadline']}
 for parent in config['parents']:
  base=next(c for c in cells if c['parent']==parent and c['offset']==0)
  for r in [c for c in cells if c['parent']==parent and c['offset']>0 and c['available']]:
   if base['o1_success_440']==0 and r['o1_success_440']>0:
    findings['new_R_success_cells'].append([parent,r['offset']])
    if r['delta_success_440']>0:findings['new_task_rescue_cells'].append([parent,r['offset']])
   if base['o1_success_440']>r['o1_success_440']:findings['lost_positive_cells'].append([parent,r['offset']])
 (out/'findings.json').write_text(json.dumps(findings,indent=2)+'\n');print(json.dumps(findings,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();analyze(args.run)
