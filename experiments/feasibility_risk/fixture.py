"""Deterministic fixtures and independent conditional exclusion, stdlib only."""
import math
import xml.etree.ElementTree as ET
THICK=.015
MARGIN=.005
EPS=1e-9

def walls(lower,upper):
    out=[]
    for axis in range(3):
        for side in (0,1):
            lo=[v-THICK for v in lower];hi=[v+THICK for v in upper]
            if side:lo[axis]=upper[axis]
            else:hi[axis]=lower[axis]
            out.append(dict(name='red_%s_%s'%('xyz'[axis],'high' if side else 'low'),lower=lo,upper=hi))
    return out

def variants(lower,upper,parking_near_x):
    closed=walls(lower,upper)
    opened=[w for w in closed if w['name']!='red_z_high']
    lid=next(w for w in closed if w['name']=='red_z_high')
    shift=parking_near_x-lid['lower'][0]
    parked=dict(name='red_parked_lid',lower=[lid['lower'][0]+shift]+lid['lower'][1:],upper=[lid['upper'][0]+shift]+lid['upper'][1:])
    return dict(open=opened,sealed=closed,parked=opened+[parked])

def modify_xml(xml,boxes):
    root=ET.fromstring(xml);world=root.find('worldbody')
    for box in boxes:
        lo,hi=box['lower'],box['upper']
        if not all(math.isfinite(v) for v in lo+hi) or any(a>=b for a,b in zip(lo,hi)):raise ValueError('Invalid box')
        if root.find('.//*[@name="%s"]'%box['name']) is not None:raise ValueError('Duplicate fixture')
        ET.SubElement(world,'geom',dict(name=box['name'],type='box',pos=' '.join(format((a+b)/2,'.17g') for a,b in zip(lo,hi)),size=' '.join(format((b-a)/2,'.17g') for a,b in zip(lo,hi)),rgba='.85 .025 .035 1',contype='1',conaffinity='1',group='1',friction='1 .005 .0001',solref='.02 1'))
    return ET.tostring(root,encoding='unicode')

def certificate(actual,lower,upper,material,initial,goal,initial_safe,static):
    expected=walls(lower,upper);byname={w['name']:w for w in actual}
    covered=all(w['name'] in byname and all(abs(byname[w['name']][k][i]-w[k][i])<1e-12 for k in ('lower','upper') for i in range(3)) for w in expected)
    conditions=dict(six_faces_covered=covered,world_fixed_collision_boxes=bool(static),target_root_is_material_point=bool(material),target_root_initially_inside=all(a<x<b for a,x,b in zip(lower,initial,upper)),initial_contact_and_geometry_safe=bool(initial_safe),frozen_goal_disjoint_from_interior=any(goal[1][i]<lower[i] or goal[0][i]>upper[i] for i in range(3)))
    return dict(label='infeasible' if all(conditions.values()) else 'unknown',contract='RE-1',conditions=conditions,proof='A continuous piecewise-linear material-root path from strict interior to disjoint fixed goal crosses a covered boundary face, forbidden by the integration-segment rule.',scope='Conditional digital contract theorem; no controller failure or pure MuJoCo impossibility inference.')

def decide(safe_success,cert):
    if safe_success and cert['label']=='infeasible':raise RuntimeError('Contradictory evidence')
    return 'feasible' if safe_success else cert['label']
