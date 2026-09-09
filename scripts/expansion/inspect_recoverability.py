"""Read-only forward-kinematic reconstruction of existing fixed-controller traces."""
import argparse,csv,gzip,hashlib,json,pickle,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from crashbench.recovery import DetourComplete

def read_trace(path):
 with gzip.open(path,'rb') as f:
  while True:
   try:yield pickle.load(f)
   except EOFError:return

def controller(obs,glass,config):
 d=config['detour'];bowl=obs['akita_black_bowl_1_pos'];plate=obs['plate_1_pos'];g={'pos':glass['pos'],'size':[glass['size'][0],glass['size'][0],glass['size'][1]]}
 return DetourComplete(g,bowl,plate,side=d['side'],lane_margin=d['lane_margin'],transit_z=float(bowl[2]+d['lift_offset']),descend_off=d['descend_offset'],leg_cap=d['leg_cap'],target_name='akita_black_bowl_1',orientation_target=None,path_aligned=True,grasp_xy_offset=d['grasp_xy_offset'],departure_clearance=d['departure_clearance'])

def main():
 import mujoco
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
 config=json.loads((args.run/'config.json').read_text());anchors=json.loads((args.run/'anchors.json').read_text());rows=[];details=[]
 for e in anchors:
  if e['condition']!='glass' or e['role']!='fitting':continue
  with (args.run/e['episode_id']/'bundle.pkl').open('rb') as f:ctx=pickle.load(f)
  bundle=ctx['bundle'];bundle.assert_integrity();obs={k[len('observation.'):]:v.copy() for k,v in bundle.runtime_state.items() if k.startswith('observation.')}
  model=mujoco.MjModel.from_xml_string(bundle.model_xml);data=mujoco.MjData(model)
  def geometry(flat):
   assert len(flat)==1+model.nq+model.nv
   data.time=float(flat[0]);data.qpos[:]=flat[1:1+model.nq];data.qvel[:]=flat[1+model.nq:];mujoco.mj_forward(model,data)
   return {'robot0_eef_pos':data.site('gripper0_grip_site').xpos.copy(),'akita_black_bowl_1_pos':data.body('akita_black_bowl_1_main').xpos.copy(),'plate_1_pos':data.body('plate_1_main').xpos.copy()}
  g=geometry(bundle.flat_state)
  for key in g:
   if not np.allclose(g[key],obs[key],atol=1e-6):raise ValueError('FK mismatch at anchor: '+e['episode_id']+key)
  glass=ctx['glasses'][0];ctrl=controller(obs,glass,config);ctrl.engage(obs);waypoints=[np.array(x[1]) for x in ctrl.legs if x[0]=='move']
  initial={'episode':e['episode_id'],'source':e['source'],'anchor_step':e['anchor_step'],'eef_x':float(obs['robot0_eef_pos'][0]),'eef_y':float(obs['robot0_eef_pos'][1]),'eef_z':float(obs['robot0_eef_pos'][2]),'eef_glass_xy':float(np.linalg.norm(obs['robot0_eef_pos'][:2]-np.array(glass['pos'][:2]))),'eef_bowl_xy':float(np.linalg.norm(obs['robot0_eef_pos'][:2]-obs['akita_black_bowl_1_pos'][:2])),'bowl_plate_xy':float(np.linalg.norm(obs['akita_black_bowl_1_pos'][:2]-obs['plate_1_pos'][:2])),'glass_radius':glass['size'][0],'glass_half_height':glass['size'][1],'min_waypoint_x':float(min(x[0] for x in waypoints)),'max_waypoint_y':float(max(x[1] for x in waypoints))}
  stages=[];mismatch=0;maxerror=0.;previous=None
  for r in read_trace(args.run/e['episode_id']/'r0_o1.pkl.gz'):
   if not r['event']['controller_steps']:continue
   obs=geometry(r['state_before']);i=ctrl.i;leg=ctrl.legs[i];distance=float(np.linalg.norm(np.asarray(leg[1])-obs['robot0_eef_pos'])) if leg[0]=='move' else None
   action=ctrl.step(obs);error=float(np.max(np.abs(action-r['action'])));maxerror=max(maxerror,error);mismatch+=int(error>1e-4)
   if previous!=i:stages.append({'stage':i,'kind':leg[0],'start':r['event']['step'],'end':r['event']['step'],'minimum_target_error':distance,'exit_by_cap':False});previous=i
   stages[-1]['end']=r['event']['step']
   if distance is not None:stages[-1]['minimum_target_error']=min(stages[-1]['minimum_target_error'],distance)
   if ctrl.i!=i:stages[-1]['exit_by_cap']=leg[0]=='move' and distance>=ctrl.tol
  terminal=json.loads((args.run/e['episode_id']/'r0_o1.json').read_text())['horizons']['440']
  rows.append(initial|{'outcome':terminal['reason'],'steps':terminal['steps'],'reconstruction_action_mismatches':mismatch,'max_action_error':maxerror,'final_stage':ctrl.i,'cap_exits':sum(s['exit_by_cap'] for s in stages)})
  details.append({'episode':e['episode_id'],'stages':stages,'valid_reconstruction':mismatch==0})
 with (args.output/'physical_table.csv').open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 (args.output/'stage_details.json').write_text(json.dumps(details,indent=2)+'\n');print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
