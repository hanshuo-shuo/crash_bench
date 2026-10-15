"""Initial-geometry-only designs for both sides of a closed separator.

The positive guard either contains neither endpoint or contains both. The
negative guard separates them in the outward or inward direction. Actual
positive admissibility still requires the complete persisted source suffix.
"""
import hashlib
import math
from .paired_geometry import expanded_point_goal,disjoint

EXIT='start_inside_goal_outside'
ENTRY='start_outside_goal_inside'
LEGACY_VERSION='balanced_start_goal_sides_v3'
VERSION='factorial_endpoint_sides_v4'
BOTH='both_inside'
NEITHER='both_outside'


def cell(mode,positive_side):
    if mode not in (EXIT,ENTRY) or positive_side not in (BOTH,NEITHER):raise ValueError('Unknown endpoint factors')
    return mode+'__'+positive_side


def factors_for(source,step):
    if step<0 or step%5:raise ValueError('Expected the fixed early checkpoint schedule')
    seed=hashlib.sha256((source+'|factorial-endpoints-v4').encode()).digest()
    index=step//5;direction=(index%2)^(seed[0]%2)
    side=((index//2+index)%2)^(seed[1]%2)
    mode=(EXIT,ENTRY)[direction];positive=(BOTH,NEITHER)[side]
    return dict(construction_mode=mode,positive_side=positive,construction_cell=cell(mode,positive))


def selection_quotas(version=VERSION):
    if version==LEGACY_VERSION:return {s:{EXIT:n,ENTRY:n} for s,n in (('train',75),('validation',25),('test',25))}
    if version!=VERSION:raise ValueError('Unknown construction version')
    # Exact direction and positive-side margins; four cells differ by at most one.
    return {s:{cell(EXIT,BOTH):a,cell(EXIT,NEITHER):b,cell(ENTRY,BOTH):b,cell(ENTRY,NEITHER):a}
            for s,a,b in (('train',38,37),('validation',13,12),('test',12,13))}


def mode_for(source,step):
    if step<0 or step%5:raise ValueError('Expected the fixed early checkpoint schedule')
    bit=hashlib.sha256((source+'|side-balance-v3').encode()).digest()[0]%2
    return (EXIT,ENTRY)[(step//5+bit)%2]


def contains_bounds(outer,inner):
    return all(a<c<d<b for a,b,c,d in zip(outer[0],outer[1],inner[0],inner[1]))


def entry_regions(target,root_goal,material,source,step,horizontal=False):
    """No future endpoint, controller action, label query or model prediction."""
    goal=expanded_point_goal(root_goal,material['center_in_root'],material.get('root_offset_norm_upper_bound'))
    gaps=[]
    for axis in ((0,1) if horizontal else range(3)):
        gaps.extend([(goal[0][axis]-target[1][axis],axis,1),
                     (target[0][axis]-goal[1][axis],axis,-1)])
    gap,axis,direction=max(gaps)
    if gap<=2e-6:raise RuntimeError('No whole-target / necessary-goal separation for inward certificate')
    # Fixed prior workspace allowance, never fitted to the future trajectory.
    margin=.12
    b=[[min(target[0][j],goal[0][j])-margin for j in range(3)],
       [max(target[1][j],goal[1][j])+margin for j in range(3)]]
    size=[hi-lo for lo,hi in zip(*b)];mid=[(lo+hi)/2 for lo,hi in zip(*goal)]
    if horizontal:mid[2]=(b[0][2]+b[1][2])/2
    seed=hashlib.sha256((source+'|'+str(step)+'|entry-region-v3').encode()).digest()
    for j in range(3):
        if j!=axis and not (horizontal and j==2) and abs(mid[j]-(b[0][j]+b[1][j])/2)<.02:
            mid[j]+=.035*(1 if seed[j]%2 else -1)
    a=[[mid[j]-size[j]/2 for j in range(3)],[mid[j]+size[j]/2 for j in range(3)]]
    if horizontal:a[0][2]=b[0][2];a[1][2]=b[1][2]
    if direction==1:
        a[0][axis]=(target[1][axis]+goal[0][axis])/2;a[1][axis]=a[0][axis]+size[axis]
    else:
        a[1][axis]=(target[0][axis]+goal[1][axis])/2;a[0][axis]=a[1][axis]-size[axis]
    if not (contains_bounds(a,goal) and contains_bounds(b,goal) and contains_bounds(b,target) and disjoint(a,target)):
        raise RuntimeError('Inward initial geometry does not establish the declared endpoint sides')
    return dict(a=a,b=b,initial_geometry_allowance_metres=margin,
        separation_axis=axis,separation_direction=direction,
        necessary_material_goal=goal,region_overlap_allowed=True,
        geometry_source='Current target/receiver geometry and fixed 12 cm allowance only; no future actions or witness endpoint')


def exit_regions(target,root_goal,material,source,step,horizontal=False):
    """Use exactly the same dimensions as entry mode, removing a size-mode cue."""
    shared=entry_regions(target,root_goal,material,source,step,horizontal=horizontal)
    envelope=shared['b'];goal=shared['necessary_material_goal']
    size=[hi-lo for lo,hi in zip(*envelope)]
    axis=shared['separation_axis'];direction=shared['separation_direction']
    midpoint=[(lo+hi)/2 for lo,hi in zip(*target)]
    if horizontal:midpoint[2]=(envelope[0][2]+envelope[1][2])/2
    seed=hashlib.sha256((source+'|'+str(step)+'|exit-region-v3').encode()).digest()
    for j in range(3):
        if j!=axis and not (horizontal and j==2) and abs(midpoint[j]-(envelope[0][j]+envelope[1][j])/2)<.02:
            midpoint[j]+=.035*(1 if seed[j]%2 else -1)
    a=[[midpoint[j]-size[j]/2 for j in range(3)],[midpoint[j]+size[j]/2 for j in range(3)]]
    if horizontal:a[0][2]=envelope[0][2];a[1][2]=envelope[1][2]
    if direction==1:
        a[1][axis]=(target[1][axis]+goal[0][axis])/2;a[0][axis]=a[1][axis]-size[axis]
    else:
        a[0][axis]=(target[0][axis]+goal[1][axis])/2;a[1][axis]=a[0][axis]+size[axis]
    tangent=min((j for j in (0,1) if j!=axis),key=lambda j:size[j])
    signs=(1,-1) if seed[3]%2 else (-1,1);result=[]
    for sign in signs:
        b=[[value for value in corner] for corner in envelope]
        for corner in b:corner[tangent]+=sign*(size[tangent]+.03)
        if not (contains_bounds(a,target) and disjoint(a,goal) and disjoint(b,envelope)):
            raise RuntimeError('Outward initial geometry does not establish the declared endpoint sides')
        result.append(dict(a=a,b=b,initial_geometry_allowance_metres=.12,
            separation_axis=axis,separation_direction=direction,necessary_material_goal=goal,
            region_overlap_allowed=False,shared_initial_envelope=envelope,
            outside_region_axis=tangent,outside_region_sign=sign,outside_envelope_gap_metres=.03,
            geometry_source='Same initial target/goal envelope dimensions as inward mode; outside region lies 3 cm beyond that envelope; no future actions'))
    return result


def factorial_regions(target,root_goal,material,source,step):
    factors=factors_for(source,step)
    inward=entry_regions(target,root_goal,material,source,step,horizontal=True)
    outward=exit_regions(target,root_goal,material,source,step,horizontal=True)
    candidates=outward if factors['positive_side']==NEITHER else outward[:1]
    result=[]
    for option in candidates:
        a=inward['a'] if factors['construction_mode']==ENTRY else option['a']
        b=inward['b'] if factors['positive_side']==BOTH else option['b']
        design=dict(option,a=a,b=b,**factors)
        design.update(shared_initial_envelope=inward['b'],same_vertical_bounds=True,
            region_overlap_allowed=not disjoint(a,b),
            geometry_source='Independent preassigned separator direction and positive-region side; horizontal separation; congruent regions with identical vertical bounds; current target/goal envelope plus fixed 12 cm allowance; no future actions')
        if factors['positive_side']==BOTH:
            for key in ('outside_region_axis','outside_region_sign','outside_envelope_gap_metres'):design.pop(key,None)
        result.append(design)
    return result


def verify_sides(layout):
    mode=layout.get('construction_mode')
    if mode is None:return None
    a,b,target,goal=[layout[k] for k in ('a','b','target','goal')]
    version=layout.get('construction_version')
    if version not in (LEGACY_VERSION,VERSION):raise RuntimeError('Unknown endpoint-side construction')
    side=layout.get('positive_side',BOTH if mode==ENTRY else NEITHER)
    if version==VERSION:
        if layout.get('positive_side') not in (BOTH,NEITHER):raise RuntimeError('Missing positive-region factor')
        if layout.get('construction_cell')!=cell(mode,side):raise RuntimeError('Endpoint factor cell differs')
        if layout.get('separation_axis') not in (0,1):raise RuntimeError('V4 requires a horizontal separator')
    checks=dict(equal_region_dimensions=all(math.isclose(ahi-alo,bhi-blo,rel_tol=0,abs_tol=1e-12)
        for alo,ahi,blo,bhi in zip(a[0],a[1],b[0],b[1])))
    if mode==EXIT:
        checks.update(target_inside_negative=contains_bounds(a,target),goal_outside_negative=disjoint(a,goal))
    elif mode==ENTRY:
        checks.update(target_outside_negative=disjoint(a,target),goal_inside_negative=contains_bounds(a,goal))
    else:raise RuntimeError('Unknown endpoint-side mode')
    if side==BOTH:checks.update(target_inside_positive=contains_bounds(b,target),goal_inside_positive=contains_bounds(b,goal))
    elif side==NEITHER:checks.update(target_outside_positive=disjoint(b,target),goal_outside_positive=disjoint(b,goal))
    else:raise RuntimeError('Unknown positive-region side')
    if version==VERSION:checks['common_vertical_interval']=all(a[i][2]==b[i][2] for i in (0,1))
    if not all(checks.values()):raise RuntimeError('Declared endpoint sides disagree with geometry')
    return checks
