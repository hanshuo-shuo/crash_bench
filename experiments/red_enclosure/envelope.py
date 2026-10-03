"""One CPU replay of the saved successful original task; no enclosure rollout.

Whole collision meshes (convex hulls) and conservative primitive boxes are clipped
to the proposed wall height. This is design evidence with a 5 mm clearance, not
a claim of continuous swept-volume certification from sampled states alone.
"""
import gzip,json,os,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
BASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE/'experiments/feasibility_contract'))
from run import configure,create_env,Audit,numeric_state,state_digest,write,sha
from contract import SMOKE
from geometry import object_points
import itertools
from envelope_trace import recorded_rows

ORIGINAL=Path('/projects/p33100/siosio/crashbench_safelibero/feasibility_contract/20261003T001208Z_smoke_0cbf26b9a554/runs/object_I_t2_e0_center')
MARGIN=.005
THICK=.015

def local_geometry(m,g):
    kind=int(m.geom_type[g]);size=np.asarray(m.geom_size[g])
    if kind==7:
        mesh=int(m.geom_dataid[g]);a=int(m.mesh_vertadr[mesh]);n=int(m.mesh_vertnum[mesh]);v=np.asarray(m.mesh_vert[a:a+n],float)
    else:
        if kind==2:half=np.repeat(size[0],3)
        elif kind==3:half=np.array([size[0],size[0],size[0]+size[1]])
        elif kind==5:half=np.array([size[0],size[0],size[1]])
        elif kind in [4,6]:half=size
        else:raise ValueError('Unsupported actor geom '+str(kind))
        v=np.asarray(list(itertools.product(*[[-x,x] for x in half])))
    hull=ConvexHull(v)
    edges=set()
    for face in hull.simplices:
        for a,b in itertools.combinations(face,2):edges.add(tuple(sorted((int(a),int(b)))))
    return v,np.asarray(sorted(edges),int)

def clip_bounds(v,edges,zlo,zhi):
    if v[:,2].max()<zlo or v[:,2].min()>zhi:return None
    kept=[v[(v[:,2]>=zlo)&(v[:,2]<=zhi)]]
    a,b=v[edges[:,0]],v[edges[:,1]];dz=b[:,2]-a[:,2]
    for z in [zlo,zhi]:
        mask=(dz!=0)&(np.minimum(a[:,2],b[:,2])<=z)&(np.maximum(a[:,2],b[:,2])>=z)
        aa,bb=a[mask],b[mask]
        if len(aa):kept.append(aa+(bb-aa)*((z-aa[:,2])/(bb[:,2]-aa[:,2]))[:,None])
    points=np.concatenate(kept)
    return np.r_[points.min(0),points.max(0)]

def collide(a,b):return bool(np.all(a[3:]>=b[:3]) and np.all(a[:3]<=b[3:]))

def main(root):
    configure(root);directory=root/'replay';directory.mkdir()
    state=SMOKE[0];env,obs=create_env(state,directory);audit=Audit(env,state,directory)
    expected=recorded_rows([json.loads(x) for x in (ORIGINAL/'steps.jsonl').read_text().splitlines()])
    if state_digest(numeric_state(env))!=expected[0]['physics_sha256']:raise RuntimeError('Original initial state differs')
    summary=json.loads((ORIGINAL/'summary.json').read_text())
    if len(expected)-1!=summary['suffix_actions']:raise RuntimeError('Original action count mismatch')
    if not(summary['task_success'] and summary['contact_safe']):raise RuntimeError('Original not a safe success')
    write(root/'source_trace.json',dict(original=str(ORIGINAL),source_summary=summary,
          expected_actions=len(expected)-1,hashes={name:sha(ORIGINAL/name) for name in ['summary.json','steps.jsonl','physics.npz','model.xml','initial_restore.json']}))
    m=env.sim.model
    actors=sorted(g for g in audit.robot|audit.target if m.geom_contype[g] or m.geom_conaffinity[g])
    local={g:local_geometry(m,g) for g in actors}
    initial_sample=audit.synchronized();d=audit.sync.data
    goal=(initial_sample['lower'],initial_sample['upper'])
    initial_target=np.concatenate([local[g][0]@np.asarray(d.geom_xmat[g]).reshape(3,3).T+d.geom_xpos[g] for g in audit.target])
    # One deterministic height from initial target top, not a wall-hit sweep.
    roof=float(np.ceil((initial_target[:,2].max()+.010)*1000)/1000)
    bottom=-.015
    samples=[];all_bounds=[];initial_all=[];steps=[];transforms=[]
    other={}
    for g in range(m.ngeom):
        if (m.geom_contype[g] or m.geom_conaffinity[g]) and int(m.geom_type[g])!=0:
            other[g]=local_geometry(m,g)
            v,e=other[g];w=v@np.asarray(d.geom_xmat[g]).reshape(3,3).T+d.geom_xpos[g]
            initial_all.append(dict(id=g,name=m.geom_id2name(g),body=m.body_id2name(int(m.geom_bodyid[g])),bounds=np.r_[w.min(0),w.max(0)]))
    def record():
        dd=audit.sync.data
        frame=[];poses=[]
        for g in actors:
            v,e=local[g];R=np.asarray(dd.geom_xmat[g]).reshape(3,3);p=np.asarray(dd.geom_xpos[g]);w=v@R.T+p
            full=np.r_[w.min(0),w.max(0)];frame.append(full);poses.append(np.r_[p,R.flatten()])
            clipped=clip_bounds(w,e,bottom-THICK,roof+THICK)
            if clipped is not None:samples.append(np.r_[audit.action_step,audit.substep,g,clipped])
        all_bounds.append(frame);transforms.append(poses);steps.append([audit.action_step,audit.substep,float(dd.time)])
    record();original=audit.original_step
    def observed():
        v=original();audit.substep+=1;audit.total_integrations+=1
        audit.sample(env.sim.data,'integration_cached');audit.synchronized();record();return v
    env.sim.step=observed
    protected_contacts=0
    for item in expected[1:]:
        audit.action_step=item['step'];audit.substep=0;audit.stage='suffix';audit.reset_meter()
        obs,_,done,_=env.step(item['action'])
        got=state_digest(numeric_state(env))
        if got!=item['physics_sha256']:raise RuntimeError('Original physics replay differs at '+str(item['step']))
        audit.endpoint(obs,None,item['action'],done)
        protected_contacts+=audit.contact_count
    native=bool(env.check_success());safe=audit.contact_count==0
    # Record every original reference checkpoint match; these are not new samples.
    np.savez_compressed(root/'envelope.npz',actor_ids=np.asarray(actors),steps=np.asarray(steps),
        full_bounds=np.asarray(all_bounds),transforms=np.asarray(transforms),height_slice_bounds=np.asarray(samples))
    slabs=np.asarray(samples);rects=slabs[:,3:]
    region=np.r_[initial_target.min(0)[:2],initial_target.max(0)[:2]]
    selected=np.zeros(len(rects),bool)
    # Conservative connected envelope near the target: expanding an AABB may merge
    # extra components, which can only make the proposed opening more conservative.
    for iteration in range(100):
        mask=(rects[:,3]>=region[0]-2*MARGIN)&(rects[:,0]<=region[2]+2*MARGIN)&(rects[:,4]>=region[1]-2*MARGIN)&(rects[:,1]<=region[3]+2*MARGIN)
        union=selected|mask
        new=np.r_[rects[union,:2].min(0),rects[union,3:5].max(0)]
        if np.array_equal(union,selected):break
        selected=union;region=new
    else:raise RuntimeError('Envelope component did not converge')
    lower=np.r_[region[:2]-MARGIN,bottom];upper=np.r_[region[2:]+MARGIN,roof]
    walls=[]
    for axis in range(3):
        for side in [0,1]:
            lo=lower-THICK;hi=upper+THICK
            if side==0:hi[axis]=lower[axis]
            else:lo[axis]=upper[axis]
            walls.append(dict(name='red_%s_%s'%('xyz'[axis],'high' if side else 'low'),bounds=np.r_[lo,hi]))
    openwalls=[w for w in walls if w['name']!='red_z_high']
    conflicts=[]
    for wall in openwalls:
        b=np.asarray(wall['bounds'])
        # Clipped bounds are conservative enclosures of full collision-hull slices.
        mask=np.all(rects[:,3:]>=b[:3],axis=1)&np.all(rects[:,:3]<=b[3:],axis=1)
        for r in slabs[mask][:10]:conflicts.append(dict(wall=wall['name'],step=int(r[0]),substep=int(r[1]),geom=int(r[2])))
    initial_conflicts=[]
    for item in initial_all:
        for wall in walls:
            if collide(np.asarray(item['bounds']),np.asarray(wall['bounds'])):
                initial_conflicts.append(dict(geom=item['id'],name=item['name'],body=item['body'],wall=wall['name'],bounds=item['bounds']))
    audit.synchronized()
    goal_disjoint=any(goal[1][i]<lower[i] or goal[0][i]>upper[i] for i in range(3))
    result=dict(original_success_replayed=native,exact_checkpoint_matches=len(expected)-1,
       integrations=audit.total_integrations,actors=[dict(id=g,name=m.geom_id2name(g),vertices=len(local[g][0])) for g in actors],
       design_rule='roof = ceil(initial target collision top + 10 mm, to 1 mm); 5 mm envelope clearance; 15 mm wall thickness',
       roof=roof,bottom=bottom,initial_target_bounds=[initial_target.min(0),initial_target.max(0)],
       interior_lower=lower,interior_upper=upper,interior_size=upper-lower,geometry=walls,
       sampled_open_path_conflicts=conflicts,initial_closed_geometry_conflicts=initial_conflicts,
       goal_disjoint=goal_disjoint,protected_contact_records=protected_contacts,
       compatible_candidate=bool(native and protected_contacts==0 and not conflicts and not initial_conflicts and goal_disjoint),
       scope='Full convex collision meshes / conservative primitives at every integration; conservative height clipping and component envelope, 5 mm margin. Still requires actual restored safe witness and certificate; not a claim of exact continuous robot swept volume.',
       no_new_policy_or_reference_selection=True,code_commit=os.environ['CB_CODE_COMMIT'],job=os.environ['SLURM_JOB_ID'])
    write(root/'GEOMETRY_DESIGN.json',result);write(root/'initial_collision_bounds.json',initial_all)
    audit.close();env.close();write(root/'COMPLETE.json',dict(passed=True,compatible_candidate=result['compatible_candidate']))
    print(json.dumps({k:v for k,v in result.items() if k not in ['actors','geometry']},default=lambda x:x.tolist(),indent=2))

if __name__=='__main__':
    root=Path(sys.argv[1])
    try:main(root)
    except BaseException as e:write(root/'STOP.json',dict(error=type(e).__name__,reason=str(e)));raise
