"""Propose one matched enclosure from a verified Long-task action witness.

This is a read-only design pass, not a labeler. Modified-scene replay, executed
fixture checks, image grounding and shortcut controls are separate gates.
"""
import itertools
import math
from .inventory import write

TARGET='chocolate_pudding_1'
GOAL='living_room_table_plate_right_region'
MARGIN=.005
THICK=.012


def walls(lower,upper):
    result=[]
    for axis in range(3):
        for side in (0,1):
            lo=[float(x)-THICK for x in lower];hi=[float(x)+THICK for x in upper]
            if side:lo[axis]=float(upper[axis])
            else:hi[axis]=float(lower[axis])
            result.append(dict(name='fd_wall_%d_%d'%(axis,side),lower=lo,upper=hi))
    return result


def local_shape(model,g):
    import numpy as np
    from scipy.spatial import ConvexHull
    kind=int(model.geom_type[g]);size=np.asarray(model.geom_size[g])
    if kind==7:
        mesh=int(model.geom_dataid[g]);start=int(model.mesh_vertadr[mesh]);count=int(model.mesh_vertnum[mesh])
        vertices=np.asarray(model.mesh_vert[start:start+count],dtype=float)
    else:
        if kind==2:half=np.repeat(size[0],3)
        elif kind==3:half=np.array([size[0],size[0],size[0]+size[1]])
        elif kind==5:half=np.array([size[0],size[0],size[1]])
        elif kind in (4,6):half=size
        else:raise RuntimeError('Unsupported moving collision primitive')
        vertices=np.asarray(list(itertools.product(*[[-x,x] for x in half])))
    hull=ConvexHull(vertices);edges=set()
    for face in hull.simplices:
        for a,b in itertools.combinations(face,2):edges.add(tuple(sorted((int(a),int(b)))))
    return vertices,np.asarray(sorted(edges),dtype=int)


def height_slice(vertices,edges,lower,upper):
    import numpy as np
    if vertices[:,2].max()<lower or vertices[:,2].min()>upper:return None
    points=[vertices[(vertices[:,2]>=lower)&(vertices[:,2]<=upper)]]
    a,b=vertices[edges[:,0]],vertices[edges[:,1]];delta=b[:,2]-a[:,2]
    for z in (lower,upper):
        keep=(delta!=0)&(np.minimum(a[:,2],b[:,2])<=z)&(np.maximum(a[:,2],b[:,2])>=z)
        aa,bb=a[keep],b[keep]
        if len(aa):points.append(aa+(bb-aa)*((z-aa[:,2])/(bb[:,2]-aa[:,2]))[:,None])
    all_points=np.concatenate(points)
    return np.r_[all_points.min(0),all_points.max(0)]


class Design:
    def __init__(self,env,folder,geometry):
        import numpy as np
        self.env=env;self.folder=folder;self.m=env.sim.model;d=env.sim.data
        goals=env.env.parsed_problem['goal_state']
        if ['on',TARGET,GOAL] not in [[str(x).lower() if i==0 else x for i,x in enumerate(g)] for g in goals]:
            raise RuntimeError('Design is restricted to the audited fixed-site Long pudding goal')
        self.root=int(env.env.obj_body_id[TARGET]);self.target=[]
        for g in range(self.m.ngeom):
            if not(self.m.geom_contype[g] or self.m.geom_conaffinity[g]):continue
            b=int(self.m.geom_bodyid[g])
            while b and b!=self.root:b=int(self.m.body_parentid[b])
            if b==self.root:self.target.append(g)
        self.actors=sorted(set(geometry['robot_geoms'])|set(self.target))
        self.shapes={g:local_shape(self.m,g) for g in self.actors}
        def world(g):return self.shapes[g][0]@np.asarray(d.geom_xmat[g]).reshape(3,3).T+np.asarray(d.geom_xpos[g])
        target=np.concatenate([world(g) for g in self.target])
        self.initial_target=[target.min(0).tolist(),target.max(0).tolist()]
        self.roof=math.ceil((float(target[:,2].max())+.015)*1000)/1000
        self.bottom=float(target[:,2].min())-.020
        self.root_position=np.asarray(d.body_xpos[self.root]).copy()
        site=self.m.site_name2id(GOAL)
        b=int(self.m.site_bodyid[site]);fixed=True
        while b:
            fixed &= int(self.m.body_dofnum[b])==0;b=int(self.m.body_parentid[b])
        if not fixed:raise RuntimeError('Goal site is not world-fixed')
        size=np.asarray(env.env.object_sites_dict[GOAL].size);rotation=np.asarray(d.site_xmat[site]).reshape(3,3)
        # Pinned SiteObject.under uses R @ (p-s), strict x/y bounds and this z band.
        corners=np.asarray(list(itertools.product([-size[0],size[0]],[-size[1],size[1]],
                                                  [size[2]-.005,size[2]+.10])))
        goal_points=corners@rotation+np.asarray(d.site_xpos[site])
        self.goal=[goal_points.min(0).tolist(),goal_points.max(0).tolist()]
        self.site=dict(name=GOAL,world_fixed=fixed,position=np.asarray(d.site_xpos[site]).tolist(),
                       rotation=rotation.tolist(),size=size.tolist(),necessary_goal_bounds=self.goal)
        self.samples=[];self.initial_moving=[];self.total=0
        for g in range(self.m.ngeom):
            if not(self.m.geom_contype[g] or self.m.geom_conaffinity[g]):continue
            b=int(self.m.geom_bodyid[g]);moving=False
            while b:
                moving |= int(self.m.body_dofnum[b])>0;b=int(self.m.body_parentid[b])
            if not moving:continue
            v,_=local_shape(self.m,g);v=v@np.asarray(d.geom_xmat[g]).reshape(3,3).T+np.asarray(d.geom_xpos[g])
            self.initial_moving.append(dict(id=g,name=self.m.geom_id2name(g),bounds=np.r_[v.min(0),v.max(0)].tolist()))

    def sample(self,action,substep):
        import numpy as np
        d=self.env.sim.data;self.total+=1
        for g in self.actors:
            local,edges=self.shapes[g]
            v=local@np.asarray(d.geom_xmat[g]).reshape(3,3).T+np.asarray(d.geom_xpos[g])
            bounds=height_slice(v,edges,self.bottom-THICK,self.roof+THICK)
            if bounds is not None:self.samples.append([action,substep,g]+bounds.tolist())

    def finish(self,summary):
        import numpy as np
        samples=np.asarray(self.samples);rects=samples[:,3:]
        # Version 2 keeps a local cell and opens every face crossed by the
        # existing full-geometry witness. Version 1's connected envelope could
        # include the destination and unrelated objects; its failed result stays
        # preserved. No policy score or classifier threshold chooses these gates.
        region=np.r_[self.initial_target[0][:2],self.initial_target[1][:2]]
        lower=np.r_[region[:2]-MARGIN,self.bottom];upper=np.r_[region[2:]+MARGIN,self.roof]
        closed=walls(lower,upper);opened=[];gates=[]
        for wall in closed:
            keep=np.all(rects[:,3:]>=wall['lower'],axis=1)&np.all(rects[:,:3]<=wall['upper'],axis=1)
            if keep.any():gates.append(dict(wall=wall['name'],sample_count=int(keep.sum())))
            else:opened.append(wall)
        conflicts=[]
        for item in self.initial_moving:
            a=np.asarray(item['bounds'])
            for wall in closed:
                if np.all(a[3:]>=wall['lower']) and np.all(a[:3]<=wall['upper']):
                    conflicts.append(dict(geom=item['name'],wall=wall['name']))
        open_conflicts=[]
        for wall in opened:
            keep=np.all(rects[:,3:]>=wall['lower'],axis=1)&np.all(rects[:,:3]<=wall['upper'],axis=1)
            if keep.any():open_conflicts.append(dict(wall=wall['name'],sample_count=int(keep.sum())))
        disjoint=any(self.goal[1][i]<lower[i] or self.goal[0][i]>upper[i] for i in range(3))
        result=dict(passed=True,compatible_proposal=bool(summary['eligible_feasible_witness'] and not conflicts and not open_conflicts and disjoint),
            initial_target_bounds=self.initial_target,initial_root=self.root_position.tolist(),
            interior_lower=lower.tolist(),interior_upper=upper.tolist(),closed=closed,opened=opened,
            goal_site=self.site,goal_disjoint=disjoint,initial_moving_geometry_conflicts=conflicts,
            open_path_envelope_conflicts=open_conflicts,recorded_samples=self.total,
            design_version=2,open_gate_faces=gates,retained_wall_count=len(opened),
            source=summary['source'],negative_certified=False,positive_modified_scene_verified=False,
            label='unknown',rule='Target top +15 mm roof; target bottom -20 mm floor; 5 mm target collision envelope clearance; 12 mm walls; open all faces intersected by conservative native actor path bounds',
            scope='Read-only conservative envelope design from one original witness. It requires actual modified-scene validation, explicit swept guard and input observability before any training label.')
        np.savez_compressed(self.folder/'DESIGN_ENVELOPE.npz',samples=samples)
        write(self.folder/'FIXTURE_PROPOSAL.json',result)
