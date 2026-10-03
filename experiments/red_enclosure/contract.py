"""RE-1: static enclosure and piecewise-linear material-point safety contract.

The exclusion is conditional on this explicitly added digital safety rule. It is
not a claim that MuJoCo soft contacts implement continuous rigid-body collision.
All coordinates are metres. Pure stdlib functions are independently testable.
"""
import math
import xml.etree.ElementTree as ET

CENTER = [-0.17716081610142778, -0.3942955030243276]
LOWER = [CENTER[0]-.11, CENTER[1]-.11, -.015]
UPPER = [CENTER[0]+.11, CENTER[1]+.11, .18]
THICKNESS = .015
EPS = 1e-9
STATE = dict(id='milk_e0_enclosure', suite='safelibero_object', level='I',
             task=2, episode=0, target='milk_1', goal='basket_1',
             goal_site='basket_1_contain_region', seed=7, budget=300,
             split='exposed_mechanism_calibration', layout_group='milk_e0')

def inside(p, lower, upper):
    return all(lower[i] < p[i] < upper[i] for i in range(3))

def segment_box(a, b, lower, upper):
    """Closed AABB slab intersection, including tangency and zero motion.

    A 1 nm outward numerical tolerance makes this conservative. A jumped-through
    wall is detected even when both sampled endpoints are outside its volume.
    """
    if not all(math.isfinite(x) for x in list(a)+list(b)+list(lower)+list(upper)):
        raise ValueError('Nonfinite geometry')
    enter, leave = 0., 1.
    for i in range(3):
        lo, hi = lower[i]-EPS, upper[i]+EPS
        delta = b[i]-a[i]
        if delta == 0:
            if a[i] < lo or a[i] > hi: return False
        else:
            t0, t1 = (lo-a[i])/delta, (hi-a[i])/delta
            enter, leave = max(enter, min(t0,t1)), min(leave, max(t0,t1))
            if enter > leave: return False
    return True

def walls(sealed):
    out=[]
    for axis in range(3):
        for side in [0,1]:
            if axis==2 and side==1 and not sealed: continue
            lo=[x-THICKNESS for x in LOWER]
            hi=[x+THICKNESS for x in UPPER]
            if side==0: hi[axis]=LOWER[axis]
            else: lo[axis]=UPPER[axis]
            out.append(dict(name='red_%s_%s'%('xyz'[axis], 'high' if side else 'low'),
                            lower=lo, upper=hi))
    return out

def modify_xml(xml, sealed):
    root=ET.fromstring(xml);world=root.find('worldbody')
    if root.findall('.//*[@name="red_x_low"]'): raise ValueError('Already modified')
    for wall in walls(sealed):
        lo,hi=wall['lower'],wall['upper']
        # Direct children of worldbody cannot move; there is no body/freejoint.
        ET.SubElement(world,'geom', dict(name=wall['name'],type='box',
              pos=' '.join(format((lo[i]+hi[i])/2,'.17g') for i in range(3)),
              size=' '.join(format((hi[i]-lo[i])/2,'.17g') for i in range(3)),
              rgba='.85 .025 .035 1', contype='1',conaffinity='1',group='1',
              friction='1 .005 .0001',solref='.02 1'))
    return ET.tostring(root,encoding='unicode')

def enclosure_names():
    """Exact allowlist avoids the existing red_coffee_mug obstacle family."""
    return {w['name'] for w in walls(True)}

def certify(boxes, material_point_inside_target, initial_target, fixed_goal,
            initial_safe, static_colliders):
    """Check six exact boundary-covering colliders, not rollout failure."""
    expected=walls(True)
    actual={x['name']:x for x in boxes}
    geometry_ok=set(actual)=={x['name'] for x in expected}
    for w in expected:
        if w['name'] not in actual:continue
        geometry_ok &= all(abs(actual[w['name']][key][i]-w[key][i])<1e-12
                           for key in ['lower','upper'] for i in range(3))
    goal_disjoint=any(fixed_goal[1][i] < LOWER[i] or fixed_goal[0][i] > UPPER[i]
                      for i in range(3))
    conditions=dict(six_faces_covered=bool(geometry_ok),
        world_fixed_collision_boxes=bool(static_colliders),
        target_root_is_material_point=bool(material_point_inside_target),
        target_root_initially_inside=inside(initial_target,LOWER,UPPER),
        initial_contact_and_geometry_safe=bool(initial_safe),
        frozen_goal_disjoint_from_interior=goal_disjoint)
    return dict(label='infeasible' if all(conditions.values()) else 'unknown',
       contract='RE-1', conditions=conditions,
       proof='Every allowed adjacent-state target-root segment avoids the actual red boxes. '
             'Their union covers all six faces of the bounded initial interior. Thus the '
             'connected piecewise-linear root path remains inside; success requires the '
             'root in a disjoint fixed-world region. Contradiction.',
       scope='Conditional digital safety theorem; not a contact-only physics theorem; '
             'not a planner-failure inference. Same geometry-derived rule in both arms.')
