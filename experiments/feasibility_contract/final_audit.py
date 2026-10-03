"""Forward-only final pose audit and split-aware delivery; no rollout actions."""
import json
import os
from pathlib import Path
import sys
import numpy as np
from run import write, sha, ARRAYS, clean
from contract import bounds, predicate

def audit_run(directory,out):
    verified=json.loads((directory/'VERIFIED.json').read_text())
    for name,value in verified['hashes'].items():
        if sha(directory/name)!=value:raise RuntimeError('Input changed: '+str(directory/name))
    state=json.loads((directory/'manifest.json').read_text())['state']
    snap=json.loads((directory/'final_restore.json').read_text())
    recorded=json.loads((directory/'final_poses.json').read_text())
    from robosuite.utils.binding_utils import MjSim
    sim=MjSim.from_xml_string((directory/'model.xml').read_text())
    sim.set_state_from_flattened(np.asarray(snap['sim_state']))
    for field in ARRAYS:np.asarray(getattr(sim.data,field))[:]=np.asarray(snap['arrays'][field])
    sim.forward()
    m,d=sim.model,sim.data
    target=m.body_name2id(state['target']+'_main');site=m.site_name2id(state['goal_site'])
    expected=recorded['synchronized_poses']
    p=np.asarray(d.body_xpos[target]).copy();s=np.asarray(d.site_xpos[site]).copy()
    R=np.asarray(d.site_xmat[site]).reshape(3,3).copy()
    if not np.allclose(p,expected['target_pos'],rtol=0,atol=1e-12) or not np.allclose(s,expected['site_pos'],rtol=0,atol=1e-12):
        raise RuntimeError('Final restored forward pose differs from execution-time synchronized copy')
    success=predicate(p,s,R,expected['site_size'])
    if success!=recorded['synchronized_success']:raise RuntimeError('Final independent predicate differs')
    lo,hi=bounds(s,R,expected['site_size'])
    clearances=np.r_[p-np.asarray(lo),np.asarray(hi)-p]
    payload=dict(input_directory=str(directory),input_verified_sha256=sha(directory/'VERIFIED.json'),
        final_state_sha256=sha(directory/'final_restore.json'),audit_job=os.environ['SLURM_JOB_ID'],
        audit_code_commit=os.environ['CB_CODE_COMMIT'],time=float(d.time),state=state,
        native_success=recorded['native_success'],synchronized_success=success,
        target_pos=p,target_quaternion_wxyz=d.body_xquat[target],site_pos=s,site_rotation=R,
        site_bounds=[lo,hi],face_clearance_order=['lower_x','lower_y','lower_z','upper_x','upper_y','upper_z'],
        face_clearances_m=clearances,minimum_face_clearance_m=float(min(clearances)),
        native_cached_to_synchronized_target_delta_m=p-np.asarray(recorded['native_poses']['target_pos']),
        bodies=[dict(id=i,name=m.body_id2name(i),position=d.body_xpos[i],quaternion_wxyz=d.body_xquat[i],rotation=d.body_xmat[i]) for i in range(m.nbody)],
        sites=[dict(id=i,name=m.site_id2name(i),position=d.site_xpos[i],rotation=d.site_xmat[i]) for i in range(m.nsite)],
        contacts=[dict(geom_pair=[int(c.geom1),int(c.geom2)],distance_m=float(c.dist)) for c in d.contact[:d.ncon] if c.dist<=0],
        scope='All bodies/sites synchronized by forward-only reconstruction of saved final state; executing simulations and source results unchanged')
    write(out,payload)
    summary=json.loads((directory/'summary.json').read_text())
    return dict(run_id=directory.name,split=state['split'],state=state['id'],layout_group=state['layout_group'],
        task_success=summary['task_success'],contact_safe=summary['contact_safe'],
        native_proxy_safe=summary['native_proxy_safe'],synchronized_success=success,
        actions=summary['suffix_actions'],prefix_actions=summary['prefix_actions'],samples=summary['samples'],
        seconds=summary['wall_seconds'],minimum_face_clearance_m=float(min(clearances)),
        final_pose_artifact=str(out),source=summary['code_commit'],job=summary['job'],input_directory=str(directory))

def main(root,core,smoke):
    if not json.loads((core/'COMPLETE.json').read_text())['passed']:raise RuntimeError('Core pilot incomplete')
    decisions=json.loads((core/'decisions.json').read_text())
    files=[]
    for decision in decisions:
        for variant in ['center','side']:
            files.append(Path(decision['evidence_root'])/'runs'/(decision['state']+'_'+variant))
    smoke_dirs=sorted((smoke/'runs').iterdir())
    (root/'synchronized_final_poses').mkdir()
    rows=[]
    for directory in files+smoke_dirs:
        rows.append(audit_run(directory,root/'synchronized_final_poses'/(directory.name+'.json')))
    groups={}
    for split in sorted({d['split'] for d in decisions}):
        ds=[d for d in decisions if d['split']==split]
        groups[split]=dict(states=len(ds),layout_groups=len({d['layout_group'] for d in ds}),
             labels={label:sum(d['label']==label for d in ds) for label in ['feasible','infeasible','unknown']},
             expert300_covered=sum(d['expert_300']!='unknown' for d in ds),
             pool300_covered=sum(d['pool_total300']!='unknown' for d in ds),
             pool600_covered=sum(d['pool_total600']!='unknown' for d in ds))
    summary=dict(core_root=str(core),smoke_root=str(smoke),groups=groups,
        forward_audits=len(rows),new_action_steps_in_this_job=0,api_calls=0,
        all_synchronized_final_poses_verified=True,
        completed_execution_action_cost=sum(r['actions']+r['prefix_actions'] for r in rows),
        completed_execution_wall_seconds=sum(r['seconds'] for r in rows),
        distinct_core_states=len(decisions),distinct_core_layouts=len({d['layout_group'] for d in decisions}),
        heldout_layouts=sum(g['layout_groups'] for s,g in groups.items() if s.endswith('holdout')),
        unresolved_requirement='No nontrivial infeasible case with an individually legal goal is certified.',
        job=os.environ['SLURM_JOB_ID'],code_commit=os.environ['CB_CODE_COMMIT'])
    write(root/'FINAL_AUDIT.json',summary);write(root/'execution_audit.json',rows)
    write(root/'decisions.json',decisions)
    write(root/'sanity_decisions.json',json.loads((smoke/'decisions.json').read_text()))
    lines=['# Bounded feasibility witness pilot — verified result','',
      'This completes the infrastructure/witness pilot. It does **not** complete a feasible/infeasible reachability benchmark.', '',
      '| Split | States | Independent layout groups | Feasible | Infeasible | Unknown |',
      '|---|---:|---:|---:|---:|---:|']
    for split,g in groups.items():lines.append('|%s|%d|%d|%d|%d|%d|'%(split,g['states'],g['layout_groups'],g['labels']['feasible'],g['labels']['infeasible'],g['labels']['unknown']))
    lines+=['','The two natural checkpoints share the previously exposed Object5 layout. They are not independent held-out layouts.',
      'The five new layouts were held out from development of this pilot, not from prior model pretraining or the historical benchmark.', '',
      '| Run | Native success | Contact safe | Synchronized success | Actions | Final minimum face clearance (mm) |',
      '|---|---|---|---|---:|---:|']
    for r in rows:lines.append('|%s|%s|%s|%s|%d|%.6f|'%(r['run_id'],r['task_success'],r['contact_safe'],r['synchronized_success'],r['actions'],r['minimum_face_clearance_m']*1000))
    lines+=['', 'Native goal success is point containment, not stable placement or grasp confirmation. Contact safety and the original displacement proxy are separate metrics in execution_audit.json.',
      'Each final pose file contains every body and site position/orientation from the same forward-synchronized saved state, plus all six containment clearances. No mixed-time geometry is used.',
      '', '## Interpretation and next decision','',
      'The core pilot has no certified infeasible labels. Failure of either or both fixed references remains unknown. Zero determinate errors here is evidence-rule conformance, not an independent accuracy estimate for a new predictor.',
      'The development smoke alone has two definition-based negative sanity controls, one witnessed positive setting and one unknown near-boundary setting. They are excluded from core-pilot coverage and cannot establish physical unreachability.',
      'A possible next synthetic slice would require an explicitly declared separating keep-out curtain, a receiver guaranteed to stay on the opposite side, and a signed-side crossing monitor that rejects discrete tunneling. Initial and goal states could then be individually legal while all allowed paths cross the curtain. A gate-open pair still needs a real legal success witness.',
      'That proposal changes the safety contract and receiver constraints. It is not an impossibility result for unchanged native SafeLIBERO. It was not implemented or submitted here. The next decision is whether such a synthetic topological slice fits the intended contribution; otherwise retain native-state negatives as unknown.',
      '', 'No training, model/API call, controller improvement, threshold search or sample expansion was performed. Failed startup and prefix-gate roots remain recoverable.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(root/'COMPLETE.json',dict(passed=True,stage='final_audit',job=os.environ['SLURM_JOB_ID'],summary=summary))

if __name__=='__main__':
    root,core,smoke=map(Path,sys.argv[1:4])
    try:main(root,core,smoke)
    except BaseException as e:
        write(root/'STOP.json',dict(reason=str(e),type=type(e).__name__,job=os.environ.get('SLURM_JOB_ID')))
        raise
