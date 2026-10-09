"""Read-only signed collision-geometry distances using separate MuJoCo data."""
import hashlib
import numpy as np
from itertools import product

ARRAYS = ['qpos', 'qvel', 'act', 'ctrl', 'qacc', 'qacc_warmstart', 'qfrc_applied',
          'xfrc_applied', 'mocap_pos', 'mocap_quat', 'userdata']

SIGNS=np.asarray(list(product([-1.,1.],repeat=3)))
EDGES=np.asarray([(i,j) for i in range(8) for j in range(i+1,8)
                  if np.count_nonzero(SIGNS[i]!=SIGNS[j])==1])


def box_distance(ca,ra,sa,cb,rb,sb):
    """Exact OBB separation; overlap signed by minimum translational SAT depth.

    Separated convex polytope closest features are vertex-face or edge-edge.
    Enumerate all vertices/projections and12×12 edge pairs, including every
    segment boundary candidate. No collision margins or live model edits.
    """
    ca,cb=np.asarray(ca),np.asarray(cb);ra=np.asarray(ra).reshape(3,3);rb=np.asarray(rb).reshape(3,3)
    sa,sb=np.asarray(sa),np.asarray(sb);delta=cb-ca
    axes=np.concatenate((ra.T,rb.T,np.cross(ra.T[:,None,:],rb.T[None,:,:]).reshape(-1,3)))
    lengths=np.linalg.norm(axes,axis=1);axes=axes[lengths>1e-12]/lengths[lengths>1e-12,None]
    gaps=np.abs(axes@delta)-np.abs(axes@ra)@sa-np.abs(axes@rb)@sb
    index=int(np.argmax(gaps));gap=float(gaps[index])
    va=ca+(SIGNS*sa)@ra.T;vb=cb+(SIGNS*sb)@rb.T
    if gap<=0:
        axis=axes[index]*(-1 if np.dot(delta,axes[index])<0 else 1)
        # Support points accompany the signed SAT depth; not a unique contact point.
        pa=ca+ra@(np.sign(ra.T@axis)*sa);pb=cb-rb@(np.sign(rb.T@axis)*sb)
        return gap,np.concatenate((pa,pb))
    closest_b=cb+np.clip((va-cb)@rb,-sb,sb)@rb.T
    closest_a=ca+np.clip((vb-ca)@ra,-sa,sa)@ra.T
    pas=[va,closest_a];pbs=[closest_b,vb]
    p=va[EDGES[:,0]][:,None,:];u=(va[EDGES[:,1]]-va[EDGES[:,0]])[:,None,:]
    q=vb[EDGES[:,0]][None,:,:];v=(vb[EDGES[:,1]]-vb[EDGES[:,0]])[None,:,:]
    w=p-q;a=np.sum(u*u,axis=-1);b=np.sum(u*v,axis=-1);c=np.sum(v*v,axis=-1)
    d=np.sum(u*w,axis=-1);e=np.sum(v*w,axis=-1);denom=a*c-b*b
    valid=denom>1e-18
    s=np.divide(b*e-c*d,denom,out=np.zeros_like(denom),where=valid)
    t=np.divide(a*e-b*d,denom,out=np.zeros_like(denom),where=valid)
    inside=valid&(s>=0)&(s<=1)&(t>=0)&(t<=1)
    candidates=[(np.zeros_like(b),np.clip(e/c,0,1)),(np.ones_like(b),np.clip((e+b)/c,0,1)),
                (np.clip(-d/a,0,1),np.zeros_like(b)),(np.clip((b-d)/a,0,1),np.ones_like(b)),(s,t)]
    for i,(ss,tt) in enumerate(candidates):
        pa=np.broadcast_to(p,p.shape[:1]+q.shape[1:])+ss[...,None]*u
        pb=np.broadcast_to(q,p.shape[:1]+q.shape[1:])+tt[...,None]*v
        if i==4:pa=pa[inside];pb=pb[inside]
        pas.append(pa.reshape(-1,3));pbs.append(pb.reshape(-1,3))
    pa=np.concatenate(pas);pb=np.concatenate(pbs);squared=np.sum((pa-pb)**2,axis=1)
    index=int(np.argmin(squared))
    return float(np.sqrt(squared[index])),np.concatenate((pa[index],pb[index]))


def physical_hash(env):
    h = hashlib.sha256()
    h.update(np.asarray([env.sim.data.time], dtype='<f8').tobytes())
    for k in ARRAYS:
        h.update(np.asarray(getattr(env.sim.data, k)).tobytes())
    h.update(np.asarray(env.sim.model.body_pos).tobytes())
    h.update(np.asarray(env.sim.model.body_quat).tobytes())
    return h.hexdigest()


class Distance:
    def __init__(self, env, obstacle, maximum):
        import mujoco
        self.mujoco, self.env, self.maximum = mujoco, env, maximum
        self.model = env.sim.model._model
        self.shadow = mujoco.MjData(self.model)
        self.native_max_abs = 0.; self.witness = None
        m = env.sim.model
        root = m.body_name2id(env.robots[0].robot_model.root_body)
        self.robot, self.protected = [], []
        for geom in range(m.ngeom):
            if not (m.geom_contype[geom] or m.geom_conaffinity[geom]):
                continue
            body = int(m.geom_bodyid[geom]); ancestor = body
            while ancestor:
                if ancestor == root:
                    self.robot.append(geom); break
                ancestor = int(m.body_parentid[ancestor])
            gn, bn = m.geom_id2name(geom) or '', m.body_id2name(body) or ''
            if gn.startswith(obstacle+'_') or bn.startswith(obstacle+'_'):
                self.protected.append(geom)
        if not self.robot or not self.protected or not hasattr(mujoco, 'mj_geomDistance'):
            raise RuntimeError('Signed geometry distance unavailable')

    def details(self):
        m = self.env.sim.model
        return dict(definition='Signed collision-geometry separation; independent of official displacement label',
                    box_box='exact OBB Euclidean separation; negative minimum-translation SAT depth for overlapping boxes; replaces verified MuJoCo3.2.3 false-negative box distance',
                    approximation='MuJoCo native mesh collision representation, including convex-hull approximation; not a physical no-contact certificate',
                    robot=[dict(id=g,name=m.geom_id2name(g),type=int(m.geom_type[g]),contype=int(m.geom_contype[g]),conaffinity=int(m.geom_conaffinity[g])) for g in self.robot],
                    protected=[dict(id=g,name=m.geom_id2name(g),type=int(m.geom_type[g]),contype=int(m.geom_contype[g]),conaffinity=int(m.geom_conaffinity[g])) for g in self.protected])

    def read(self):
        before = physical_hash(self.env)
        live = self.env.sim.data
        for name in ARRAYS:
            getattr(self.shadow, name)[:] = np.asarray(getattr(live, name))
        self.shadow.time = live.time
        self.mujoco.mj_forward(self.model, self.shadow)
        minimum, pair = self.maximum, None
        for a in self.robot:
            for b in self.protected:
                lower=float(np.linalg.norm(self.shadow.geom_xpos[a]-self.shadow.geom_xpos[b])-
                            self.model.geom_rbound[a]-self.model.geom_rbound[b])
                if lower>max(0.,minimum):continue
                if int(self.model.geom_type[a])==6 and int(self.model.geom_type[b])==6:
                    value,_=box_distance(self.shadow.geom_xpos[a],self.shadow.geom_xmat[a],self.model.geom_size[a],
                                         self.shadow.geom_xpos[b],self.shadow.geom_xmat[b],self.model.geom_size[b])
                else:
                    value=float(self.mujoco.mj_geomDistance(self.model,self.shadow,a,b,self.maximum,None))
                if value < minimum:
                    minimum, pair = value, [a, b]
        if pair is None or not np.isfinite(minimum):
            raise RuntimeError('Geometry distance saturated or unsupported')
        points = np.zeros(6); native_points = np.zeros(6)
        legacy=float(self.mujoco.mj_geomDistance(self.model,self.shadow,*pair,self.maximum,points))
        native = float(self.mujoco.mj_geomDistance(self.model,live._data,*pair,self.maximum,native_points))
        error=abs(legacy-native);self.native_max_abs=max(self.native_max_abs,error)
        method='mujoco_native_convex_collision'
        if all(int(self.model.geom_type[g])==6 for g in pair):
            a,b=pair
            _,points=box_distance(self.shadow.geom_xpos[a],self.shadow.geom_xmat[a],self.model.geom_size[a],
                                  self.shadow.geom_xpos[b],self.shadow.geom_xmat[b],self.model.geom_size[b])
            method='exact_obb_euclidean_separation_or_signed_sat_depth'
        self.witness=dict(geom_pair=pair,points_world=points.tolist(),native_distance_m=native,
                          measured_distance_m=minimum,legacy_shadow_distance_m=legacy,
                          shadow_native_legacy_max_abs=error,method=method,
                          robot_position_world=np.asarray(self.shadow.geom_xpos[pair[0]]).tolist(),
                          protected_position_world=np.asarray(self.shadow.geom_xpos[pair[1]]).tolist())
        if before != physical_hash(self.env):
            raise RuntimeError('Read-only geometry changed live simulation')
        return minimum, pair
