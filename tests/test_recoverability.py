import copy,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.expansion.run_recoverability import suffix

def test_suffix_budget_is_separate_from_old_episode_clock():
 events=[{'step':15,'reason':None,'calls':1},{'step':450,'reason':'success','calls':1}]
 assert suffix(events,20,440)['success_step']==430
 assert suffix(events,20,220)['success']==0
 assert suffix(events,20,440)['calls']==1

def test_frozen_probe_bounds_and_source_membership():
 c=json.loads(Path('configs/recoverability/probe_v1.json').read_text())
 assert c['parents']==['e18','e21','e27','e06'] and len(set(c['parents']))==4
 assert len(c['parents'])*len(c['offsets'])*len(c['repeats'])*2==48
 assert max(c['offsets'])==10 and c['learning'] is False and c['confirmation'] is False

@pytest.mark.parametrize('option',[0,1])
def test_instrumentation_leaves_executed_actions_unchanged(tmp_path,monkeypatch,option):
 from scripts.expansion import run_detour_benefit as run
 import crashbench.branching.state as state
 obs={'robot0_eef_pos':np.array([-.1,0.,1.2]),'akita_black_bowl_1_pos':np.array([0.,.2,.9]),'plate_1_pos':np.array([.1,.2,.9]),'robot0_gripper_qpos':np.array([.04,-.04])}
 ctx={'bundle':object(),'glasses':[],'pending_action':np.zeros(7),'prefix_events':[],'proposal_latency':0.,'policy_input':{},'instruction':'test'}
 monkeypatch.setattr(state,'restore_exact_state',lambda *a:copy.deepcopy(obs))
 monkeypatch.setattr(run,'load',lambda p:copy.deepcopy(ctx));monkeypatch.setattr(run,'file_sha256',lambda p:'hash');monkeypatch.setattr(run,'make_crash',lambda *a:None)
 monkeypatch.setattr(run,'measured_act',lambda *a:(np.ones(7)*.01,{},0.))
 executed=[]
 def step(env,o,action,crash,glasses,index,calls,control,stream,*extra):
  executed.append(action.copy());new=copy.deepcopy(o);new['robot0_eef_pos']+=np.asarray(action[:3])*.001
  return new,{'step':index+1,'reason':None,'calls':calls,'controller_steps':int(control)}
 monkeypatch.setattr(run,'step',step)
 config=json.loads(Path('configs/detour_benefit/development_v1.json').read_text());config['long_H']=3
 row={'episode_id':'x','source':'s','condition':'glass','role':'fitting','placement':{'on_path_glass':{'type':'cylinder','pos':[0,.1,.96],'size':[.03,.06]}}};anchor={'bundle_sha256':'hash','bundle_id':'b','anchor_step':0}
 results=[]
 for audit in (False,True):
  folder=tmp_path/str(audit);folder.mkdir();executed.clear()
  record=run.branch(None,None,row,anchor,config,folder,0,option,audit_controller=audit);results.append([x.copy() for x in executed])
  if audit:assert len(record['physics'])==3
 np.testing.assert_array_equal(results[0],results[1])

def test_unavailable_and_fixed_budget_analysis(tmp_path):
 from scripts.expansion.analyze_recoverability import analyze
 def write(name,obj):(tmp_path/name).write_text(json.dumps(obj))
 write('complete.json',{'status':'COMPLETE','scored_branches':4});write('config.json',{'parents':['p'],'offsets':[0,3]})
 write('children.json',[{'parent':'p','offset':0,'episode_id':'p0','anchor_step':0,'robot_state':[0,0,1,0,0,0,0,0],'bowl':[0,1,0],'glass':{'pos':[0,.5,0]}}]);write('unavailable.json',[{'parent':'p','offset':3,'reason':'prefix_accident'}])
 records=[]
 for repeat in (0,1):
  for option in (0,1):
   records.append({'episode_id':'p0','parent':'p','offset':0,'repeat':repeat,'option':option,'branch_elapsed_seconds':1.,'events':[{'step':20,'reason':'success' if option else 'accident'}],'suffix_horizons':{str(h):{'calls':1,'steps':20,'controller_steps':20 if option else 0,'success':option,'accident':1-option,'success_step':20 if option else None,'reason':'success' if option else 'accident'} for h in (220,440)},'horizons':{'440':{'success':option}},'physics':[]})
 write('records.json',records);analyze(tmp_path)
 findings=json.loads((tmp_path/'analysis/findings.json').read_text());assert len(findings['unavailable'])==1 and findings['new_R_success_cells']==[]
