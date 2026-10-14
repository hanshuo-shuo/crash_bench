"""Geometry for task-contrast pairs, with an explicit fixed destination region.

The native goal is conjoined with the displayed fixed green region. This is a
derived dataset task, not an unchanged SafeLIBERO score. A material point away
from an object's root needs a goal bound enlarged for every possible rotation.
"""
import itertools
import math
from .fixture_design import local_shape,walls
from .contact_contract import descendants
from experiments.reach_avoid.certificate import certify

CONTRACT='FD1000-MARKED-PAIR-v1'


def sphere_radius(kind,size):
    if kind in (2,3):return float(size[0])
    if kind==5:return float(min(size[:2]))
    if kind in (4,6):return float(min(size))
    return 0.


def expanded_point_goal(root_goal,offset):
    if not all(math.isfinite(float(x)) for x in root_goal[0]+root_goal[1]+offset):
        raise ValueError('Nonfinite material-point goal')
    if any(a>=b for a,b in zip(*root_goal)):
        raise ValueError('Degenerate fixed destination')
    radius=math.sqrt(sum(float(v)**2 for v in offset))
    return [[x-radius for x in root_goal[0]],[x+radius for x in root_goal[1]]]


def point_certificate(boxes,lower,upper,material,root_goal,initial_safe,static=True):
    goal=expanded_point_goal(root_goal,material['center_in_root'])
    proof=certify(boxes,lower,upper,material['initial_world_center'],goal,
        material['radius'],initial_safe,static,material['rigid_primitive_verified'],True)
    return dict(contract=CONTRACT,label='conditionally_infeasible' if proof['label']=='infeasible' else 'unknown',
        material_ball=material,required_root_goal=root_goal,necessary_material_point_goal=goal,
        orientation_bound='The material center is at a fixed root-frame offset; each world coordinate differs from the root by at most the offset norm for every rotation.',
        separator=proof,
        scope='Explicit native/synchronized swept material-center guard and fixed displayed final root region; no unrestricted physical impossibility claim')


def material_sphere(env,name):
    import numpy as np
    m,d=env.sim.model,env.sim.data;root=int(env.env.obj_body_id[name]);choices=[]
    for g in sorted(descendants(m,{root})):
        body=int(m.geom_bodyid[g]);rigid=True
        while body!=root:
            rigid &= int(m.body_dofnum[body])==0;body=int(m.body_parentid[body])
        if not rigid:continue
        kind=int(m.geom_type[g]);size=np.asarray(m.geom_size[g])
        radius=sphere_radius(kind,size)
        if radius<=2e-6:continue
        center=np.asarray(d.geom_xpos[g]);rotation=np.asarray(d.body_xmat[root]).reshape(3,3)
        offset=rotation.T@(center-np.asarray(d.body_xpos[root]))
        choices.append(dict(object=name,geom_name=m.geom_id2name(g),geom_type=kind,
            primitive_size=size.tolist(),radius=radius,center_in_root=offset.tolist(),
            initial_world_center=center.tolist(),root_name=m.body_id2name(root),rigid_primitive_verified=True))
    if not choices:raise RuntimeError('No verified material sphere in an actual rigid collision primitive')
    return sorted(choices,key=lambda r:(-r['radius'],r['geom_name']))[0]


def body_bounds(env,name):
    import numpy as np
    m,d=env.sim.model,env.sim.data;root=int(env.env.obj_body_id[name]);points=[]
    for g in sorted(descendants(m,{root})):
        vertices,_=local_shape(m,g)
        points.append(vertices@np.asarray(d.geom_xmat[g]).reshape(3,3).T+np.asarray(d.geom_xpos[g]))
    if not points:raise RuntimeError('No collision geometry for task object')
    points=np.concatenate(points)
    return [points.min(0).tolist(),points.max(0).tolist()]


def destination(env,goal):
    """Declare a fixed region from the current receiver, never a future witness."""
    import numpy as np
    m,d=env.sim.model,env.sim.data;predicate,target,receiver=goal
    if predicate not in ('in','on'):raise RuntimeError('Unsupported derived goal predicate')
    if receiver in env.env.object_sites_dict:
        site=m.site_name2id(receiver);p=np.asarray(d.site_xpos[site]);R=np.asarray(d.site_xmat[site]).reshape(3,3)
        size=np.asarray(env.env.object_sites_dict[receiver].size)
        if predicate=='in':
            extent=np.abs(R@size);lo=p-extent;hi=p+extent;lo[2]-=.01
        else:
            corners=np.asarray(list(itertools.product([-size[0],size[0]],[-size[1],size[1]],
                [size[2]-.005,size[2]+.10])))
            points=corners@R+p;lo=points.min(0);hi=points.max(0)
        # A common 2 cm destination allowance; success still requires the native
        # predicate. The fixed box is an explicit extra task condition, not a
        # claim that a movable receiver is static.
        bounds=[(lo-.02).tolist(),(hi+.02).tolist()]
    elif receiver in env.env.obj_body_id and predicate=='on':
        p=np.asarray(d.body_xpos[int(env.env.obj_body_id[receiver])])
        bounds=[(p+[-.08,-.08,-.01]).tolist(),(p+[.08,.08,.25]).tolist()]
    else:raise RuntimeError('Unsupported receiver for fixed destination construction')
    return dict(native_goal=goal,fixed_root_region=bounds,
        region_definition='Declared from initial receiver geometry, held fixed throughout this derived task',
        receiver_motion_forbidden=False,source_of_region='Initial scene only; no future positions or actions')


def inside(point,bounds):return all(a<x<b for a,x,b in zip(bounds[0],point,bounds[1]))


def disjoint(a,b):return any(a[1][i]<b[0][i] or b[1][i]<a[0][i] for i in range(3))


def proposals(env,primary,protected_names,goal,max_candidates=3,required_enclosed=None):
    import numpy as np
    m=env.sim.model;free=body_bounds(env,primary);region=destination(env,goal)
    eligible=[];rejected=[]
    native_targets={g[1] for g in env.env.parsed_problem['goal_state'] if len(g)>=2}
    for name in sorted(env.env.objects_dict,key=lambda n:(n not in native_targets,n)):
        if required_enclosed is not None and name!=required_enclosed:continue
        if 'obstacle' in name or 'box_base' in name or 'box_small_base' in name:continue
        if name==primary or name==goal[2] or name in protected_names or name not in env.env.obj_body_id:continue
        root=int(env.env.obj_body_id[name])
        if int(m.body_dofnum[root])!=6:continue
        try:
            material=material_sphere(env,name);bounds=body_bounds(env,name)
            lower=(np.asarray(bounds[0])+[-.015,-.015,-.020]).tolist()
            upper=(np.asarray(bounds[1])+[.015,.015,.015]).tolist()
            if not disjoint(free,[lower,upper]):raise RuntimeError('Primary object is not outside proposed enclosure')
            boxes=walls(lower,upper)
            # Initial contact safety is NOT inferred by this design function.
            hypothetical=point_certificate(boxes,lower,upper,material,region['fixed_root_region'],True)
            if hypothetical['label']!='conditionally_infeasible':raise RuntimeError('No separated material-point goal')
            eligible.append(dict(primary=primary,enclosed=name,primary_goal=goal,
                enclosed_goal=[goal[0],name,goal[2]],destination=region,boxes=boxes,
                lower=lower,upper=upper,primary_bounds=free,enclosed_bounds=bounds,
                material=material,design_only=True,actual_initial_safety_checked=False,label='unknown'))
        except RuntimeError as error:rejected.append(dict(object=name,reason=str(error)))
    return eligible[:max_candidates],rejected
