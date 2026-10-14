"""Read-only initial geometry facts for screening certificate applicability.

These facts do not establish a negative label. In particular a bowl root may
lie in a hollow cavity: its visual bounding radius is not material geometry.
"""
import math


def box_material_radius(half_sizes,root_local):
    if len(half_sizes)!=3 or len(root_local)!=3 or not all(math.isfinite(float(x)) for x in list(half_sizes)+list(root_local)):
        raise ValueError('Invalid rigid collision box')
    return max(0.,min(float(s)-abs(float(p)) for s,p in zip(half_sizes,root_local)))


def initial_geometry(env):
    import numpy as np
    m,d=env.sim.model,env.sim.data
    goals=env.env.parsed_problem['goal_state'];names=sorted({x for goal in goals for x in goal[1:] if isinstance(x,str)})
    objects=[]
    for name in names:
        if name not in env.env.obj_body_id:continue
        root=int(env.env.obj_body_id[name]);boxes=[];other=[]
        for g in range(m.ngeom):
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
            body=int(m.geom_bodyid[g]);b=body;rigid=True
            while b and b!=root:
                rigid &= int(m.body_dofnum[b])==0;b=int(m.body_parentid[b])
            if b!=root or not rigid:continue
            if int(m.geom_type[g])!=6:
                other.append(dict(id=g,name=m.geom_id2name(g),type=int(m.geom_type[g])));continue
            pos=np.asarray(d.body_xpos[root]);rotation=np.asarray(d.geom_xmat[g]).reshape(3,3)
            local=rotation.T@(pos-np.asarray(d.geom_xpos[g]));half=np.asarray(m.geom_size[g])
            boxes.append(dict(id=g,name=m.geom_id2name(g),half_sizes=half.tolist(),root_local=local.tolist(),
                              root_material_ball_radius=box_material_radius(half,local)))
        objects.append(dict(name=name,root_body_id=root,root_world=np.asarray(d.body_xpos[root]).tolist(),
            collision_boxes=boxes,other_collision_geometry=other,
            box_only_material_radius=max([b['root_material_ball_radius'] for b in boxes]+[0.])))
    return dict(native_goals=goals,goal_objects=objects,certificate_label='unknown',
        missing_for_negative=['executed matched fixture geometry','admissible goal set disjoint from separator',
                              'all six faces independently covered','initial validity and explicit swept-motion guard'],
        scope='Initial read-only geometry diagnostic; visual bounds, policy failure and radius alone cannot certify infeasibility')
