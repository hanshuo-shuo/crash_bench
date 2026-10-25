"""V5 S4 contract: B writes positions, a program decides (docs/generative_verifier/V5_S4_PROTOCOL.md).

Arm P keeps S3's inputs and training settings; only B's target changes. Its facts are the View 1 pixel positions
of the floor points under the two anchors, the pudding and the basket, from which a program decides whether the
line separates the pudding from the basket. The positions come from the stored world geometry projected with the
stored render cameras; no new rendering. A is the S3 A run, read from its records.
"""
import json
import re
from pathlib import Path
from .common import require
from .v5_contract import configuration as s3_configuration,DATASET,TRAIN_LOOKUP
from verifier_dataset.v5_geometry import phrase

VERSION='V5-object-P-v1'
VIEW='overview_a'
SIZE=448
FLOOR_Z=0.
FORCED='Line:'
FIELDS=('line','line_points','target_point','goal_point','separated')
POINT=r'\[(-?\d+), (-?\d+)\]'
STRICT=(r'Line: (.+)\nLine points: '+POINT+' '+POINT+r'\nTarget point: '+POINT+r'\nGoal point: '+POINT+
        r'\nSeparated: (yes|no)\nVerdict: [01]')
TARGET_PHRASE='the chocolate pudding'
S3_FILES=('RECEIPT.json','config.json','preflight/COMPLETE.json','preflight/ENCODINGS.json','ORDERS.json','SUBSETS.json','LOOKUP.json',
    'S3_READING.json','A_seed713/RESULT.json','A_seed713/validation.jsonl','B_seed713/RESULT.json')
GOAL_PHRASE='the basket'


def project(point,view):
    """World point to View pixels (x right, y down) for a stored render camera; the student images are flipped left-right."""
    import numpy as np
    R=np.asarray(view['rotation'],dtype=float).reshape(3,3);c=(np.asarray(point,dtype=float)-np.asarray(view['position'],dtype=float))@R
    require(c[2]<0,'Point behind the camera');t=np.tan(np.radians(view['fovy'])/2)
    return float((1-c[0]/(-c[2])/t)/2*SIZE),float((1-c[1]/(-c[2])/t)/2*SIZE)


def image_points(dataset,image_id):
    """Floor points (z = 0) under the pudding's material point, the basket centre and every anchor centre, in View 1 pixels."""
    folder=Path(dataset)/'units'/image_id
    camera=json.loads((folder/'CAMERA.json').read_text());geometry=json.loads((folder/'RENDER_GEOMETRY.json').read_text())
    view=camera['views'][0];centre=lambda b:[(b[0][0]+b[1][0])/2,(b[0][1]+b[1][1])/2,FLOOR_Z]
    points=dict(target=project([geometry['material'][0],geometry['material'][1],FLOOR_Z],view),goal=project(centre(geometry['goal_box']),view),
        **{k:project(centre(b),view) for k,b in geometry['anchor_boxes'].items()})
    corners=lambda b:[[x,y,z] for x in (b[0][0],b[1][0]) for y in (b[0][1],b[1][1]) for z in (b[0][2],b[1][2])]
    boxes={k:b for k,b in dict(target=geometry['target_box'],goal=geometry['goal_box'],**geometry['anchor_boxes']).items()}
    extents={}
    for k,b in boxes.items():
        uv=[project(c,view) for c in corners(b)];extents[k]=[min(u for u,v in uv),min(v for u,v in uv),max(u for u,v in uv),max(v for u,v in uv)]
    return dict(points=points,boxes_2d=extents,visibility=geometry['visibility'][VIEW])


def side(a,b,p):
    value=(b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0]);return (value>0)-(value<0)


def program_separated(a,b,target,goal):
    """Separated only when the pudding and the basket lie strictly on opposite sides of the line through a and b."""
    return side(a,b,target)*side(a,b,goal)<0


def anchors(row):
    return row['metadata']['rule'].split(':',1)[1].split('|')


def facts(row,points):
    a,b=anchors(row);r=lambda k:[int(round(x)) for x in points[k]]
    pts=dict(a=r(a),b=r(b),target=r('target'),goal=r('goal'))
    for q in pts.values():require(all(0<=x<SIZE for x in q),'Point outside View 1')
    separated=program_separated(pts['a'],pts['b'],pts['target'],pts['goal'])
    require(('yes' if separated else 'no')==row['target']['facts']['separated'],'Program on the true points disagrees with the label')
    fmt=lambda q:'[%d, %d]'%tuple(q)
    return dict(line=row['target']['facts']['line'],line_points=fmt(pts['a'])+' '+fmt(pts['b']),target_point=fmt(pts['target']),
        goal_point=fmt(pts['goal']),separated=row['target']['facts']['separated']),pts


def answer(f,digit):
    return ('Line: %s\nLine points: %s\nTarget point: %s\nGoal point: %s\nSeparated: %s\nVerdict: %s'
            %(f['line'],f['line_points'],f['target_point'],f['goal_point'],f['separated'],digit))


def parse(text):
    """Points, the written Separated, the written verdict, and the program verdict from the points."""
    text=text.strip()
    found=re.findall(r'^Verdict: ([01])[ \t]*$',text,flags=re.M);terminal=re.search(r'(?:^|\n)Verdict: ([01])\s*$',text)
    predicted=int(found[0]) if len(found)==1 and terminal else None
    grab=lambda pattern:re.findall(pattern,text,flags=re.M)
    line=grab(r'^Line points: '+POINT+' '+POINT+r'[ \t]*$');target=grab(r'^Target point: '+POINT+r'[ \t]*$');goal=grab(r'^Goal point: '+POINT+r'[ \t]*$')
    sep=grab(r'^Separated: (yes|no)[ \t]*$')
    pts=None
    if len(line)==1 and len(target)==1 and len(goal)==1:
        l=[int(x) for x in line[0]];pts=dict(a=l[:2],b=l[2:],target=[int(x) for x in target[0]],goal=[int(x) for x in goal[0]])
    program=None
    if pts and pts['a']!=pts['b']:program=0 if program_separated(pts['a'],pts['b'],pts['target'],pts['goal']) else 1
    return dict(predicted=predicted,program=program,points=pts,separated=sep[0] if len(sep)==1 else None,
        format_valid=bool(predicted is not None and re.fullmatch(STRICT,text)))


def readings():
    return {'program':'Program verdict from the generated points (primary)','model':'The generated Verdict line (secondary)'}


def configuration():
    base=s3_configuration()
    return dict(version=VERSION,s3_configuration=base,arm='P',seeds=list(base['seeds']),view=VIEW,floor_z=FLOOR_Z,
        target='Line / Line points / Target point / Goal point / Separated / Verdict; points are View 1 pixels [x, y] of the floor point '
            'under each anchor centre, the pudding material point and the basket centre',
        program='Separated when the pudding and basket points lie strictly on opposite sides of the line through the two line points; '
            'verdict 0 when separated, else 1; missing or unparsable points count as wrong',
        maximum_new_tokens=128,primary_reading='program',selection=dict(base['selection'],value='program verdict accuracy'),
        training_fit=dict(base['training_fit']),evaluation_updates=list(base['evaluation_updates']),optimizer_updates=base['optimizer_updates'],
        unchanged_from_s3=['model_id','lora_rank','lora_alpha','lora_dropout','learning_rate','optimizer','weight_decay','gradient_clip',
            'micro_batch_size','effective_batch_size','epochs','training_rows','optimizer_updates','evaluation_updates','max_length',
            'image_pixels','objective','selection','training_fit','bootstrap'],
        comparison='Paired against the S3 A records on the full validation split (same rows, same inference inputs)',
        zero_shot=dict(models=['Qwen/Qwen2.5-VL-3B-Instruct','Qwen/Qwen3-VL-32B-Instruct'],images='the 192 selection-subset images, View 1 only',
            coordinates={'Qwen/Qwen2.5-VL-3B-Instruct':'absolute pixels of the 448 input','Qwen/Qwen3-VL-32B-Instruct':'relative 0-1000, scaled by 0.448'},
            hit='point inside the projected 2D box of the object, widened by 4 pixels',max_new_tokens=48),
        rules=dict(positions_ground='P program accuracy >= 0.65, P - A >= 0.10 and the image-bootstrap 95% interval of P - A above 0',
            positions_fail='P program accuracy <= 0.60',unclear='Any other result'),
        lookup_bound=TRAIN_LOOKUP,test_usage='No stage loads test or test_natural',hyperparameter_search=False,api_calls=0)


def s4_reading(p,a,interval):
    if p>=.65 and p-a>=.10 and interval[0]>0:return 'positions_ground'
    if p<=.60:return 'positions_fail'
    return 'unclear'


def point_prompt(name):
    return ('Point to %s in the image. Answer with JSON only: {"point_2d": [x, y]}.'%name)


def entity_phrase(key):
    return {'target':TARGET_PHRASE,'goal':GOAL_PHRASE}.get(key) or 'the '+phrase(key)


def parse_point(text):
    m=re.search(r'"?point_2d"?\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]',text) or re.search(r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]',text)
    return [float(m.group(1)),float(m.group(2))] if m else None
