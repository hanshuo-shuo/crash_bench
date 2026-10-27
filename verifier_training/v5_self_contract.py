"""V5 S5 contract: arm S writes P's positions, then the side computation, then its own verdict (docs/generative_verifier/V5_S5_PROTOCOL.md).

The first four lines are P's target. The model then writes the line vector, the two offsets from the first anchor, the
two cross products as two multiplications and a subtraction, the two signs, Separated and Verdict. The reading is the
model's own Verdict; the program on its points (S4's) is only a diagnostic. Inputs and training follow S4's P.
"""
import re
from .common import require
from .v5_pos_contract import configuration as s4_configuration,facts as p_facts,parse as p_parse,program_separated,FORCED
from .v5_metrics import REPLICATES,SEED

VERSION='V5-object-S-v1'
ARM='S'
SEED_RUN=713
STEPS=('vector','target_offset','goal_offset','target_products','target_cross','goal_products','goal_cross','target_side','goal_side',
    'separated','verdict')
S3_FILES=('RECEIPT.json','config.json','preflight/COMPLETE.json','preflight/ENCODINGS.json','SUBSETS.json','LOOKUP.json','S3_READING.json',
    'A_seed713/RESULT.json','A_seed713/validation.jsonl')
S4_FILES=('RECEIPT.json','config.json','preflight/COMPLETE.json','preflight/ENCODINGS.json','POINTS.json','ORDERS.json','S4_READING.json',
    'P_seed713/RESULT.json','P_seed713/validation.jsonl')
INT=r'(-?\d+)'
WRAPPED=r'\(?(-?\d+)\)?'
PAIR=r'\['+INT+', '+INT+r'\]'
CROSS=INT+r'\*'+WRAPPED+' - '+WRAPPED+r'\*'+WRAPPED+' = '+INT+' - '+WRAPPED+' = '+INT
SIDE=r'(positive|negative|zero)'
STRICT=(r'Line: (.+)\nLine points: \[-?\d+, -?\d+\] \[-?\d+, -?\d+\]\nTarget point: \[-?\d+, -?\d+\]\nGoal point: \[-?\d+, -?\d+\]\n'
        r'Line vector: \[-?\d+, -?\d+\]\nTarget offset: \[-?\d+, -?\d+\]\nGoal offset: \[-?\d+, -?\d+\]\n'
        r'Target cross: '+CROSS+r'\nGoal cross: '+CROSS+r'\nTarget side: '+SIDE+r'\nGoal side: '+SIDE+r'\nSeparated: (yes|no)\nVerdict: [01]')


def wrap(v):
    return '(%d)'%v if v<0 else '%d'%v


def word(v):
    return 'positive' if v>0 else 'negative' if v<0 else 'zero'


def cross(vector,offset):
    """The two products and their difference: vector x times offset y, minus vector y times offset x."""
    first=vector[0]*offset[1];second=vector[1]*offset[0];return first,second,first-second


def chain(pts):
    a,b,t,g=pts['a'],pts['b'],pts['target'],pts['goal']
    v=[b[0]-a[0],b[1]-a[1]];u=[t[0]-a[0],t[1]-a[1]];w=[g[0]-a[0],g[1]-a[1]];tc=cross(v,u);gc=cross(v,w)
    separated=(tc[2]>0 and gc[2]<0) or (tc[2]<0 and gc[2]>0)
    require(separated==program_separated(a,b,t,g),'Computation chain disagrees with the S4 program')
    return dict(vector=v,target_offset=u,goal_offset=w,target_cross=tc,goal_cross=gc,separated='yes' if separated else 'no')


def cross_text(vector,offset,products):
    return '%d*%s - %s*%s = %d - %s = %d'%(vector[0],wrap(offset[1]),wrap(vector[1]),wrap(offset[0]),products[0],wrap(products[1]),products[2])


def answer(f,pts,digit):
    c=chain(pts);require(c['separated']==f['separated'],'Chain disagrees with the label')
    pair=lambda q:'[%d, %d]'%tuple(q)
    return ('Line: %s\nLine points: %s\nTarget point: %s\nGoal point: %s\nLine vector: %s\nTarget offset: %s\nGoal offset: %s\n'
            'Target cross: %s\nGoal cross: %s\nTarget side: %s\nGoal side: %s\nSeparated: %s\nVerdict: %s'
            %(f['line'],f['line_points'],f['target_point'],f['goal_point'],pair(c['vector']),pair(c['target_offset']),pair(c['goal_offset']),
              cross_text(c['vector'],c['target_offset'],c['target_cross']),cross_text(c['vector'],c['goal_offset'],c['goal_cross']),
              word(c['target_cross'][2]),word(c['goal_cross'][2]),c['separated'],digit))


def facts(row,points):
    """P's facts and points from the true geometry, the S target text, and a check that no cross product is zero."""
    f,pts=p_facts(row,points);c=chain(pts);require(c['target_cross'][2]!=0 and c['goal_cross'][2]!=0,'A true point lies on the line')
    return f,pts,answer(f,pts,row['target']['digit'])


def one(pattern,text):
    found=re.findall(pattern,text,flags=re.M);return found[0] if len(found)==1 else None


def parse(text):
    """P's fields (points, program verdict on them, written Separated and Verdict) plus every written step and its check.

    Each step is checked against the model's own earlier numbers, so one slip is counted once; steps that cannot be read
    count as wrong."""
    out=p_parse(text);text=text.strip();ints=lambda m:None if m is None else [int(x) for x in (m if isinstance(m,tuple) else (m,))]
    w=dict(vector=ints(one(r'^Line vector: '+PAIR+r'[ \t]*$',text)),target_offset=ints(one(r'^Target offset: '+PAIR+r'[ \t]*$',text)),
        goal_offset=ints(one(r'^Goal offset: '+PAIR+r'[ \t]*$',text)),target_cross=ints(one(r'^Target cross: '+CROSS+r'[ \t]*$',text)),
        goal_cross=ints(one(r'^Goal cross: '+CROSS+r'[ \t]*$',text)),target_side=one(r'^Target side: '+SIDE+r'[ \t]*$',text),
        goal_side=one(r'^Goal side: '+SIDE+r'[ \t]*$',text))
    pts=out['points'];ok={}
    ok['vector']=bool(pts and w['vector']==[pts['b'][0]-pts['a'][0],pts['b'][1]-pts['a'][1]])
    for k in ('target','goal'):
        ok[k+'_offset']=bool(pts and w[k+'_offset']==[pts[k][0]-pts['a'][0],pts[k][1]-pts['a'][1]])
        c=w[k+'_cross'];v=w['vector'];o=w[k+'_offset']
        # Products: the factors copy the written vector and offset, and each multiplication is right.
        ok[k+'_products']=bool(c and v and o and c[:4]==[v[0],o[1],v[1],o[0]] and c[4]==c[0]*c[1] and c[5]==c[2]*c[3])
        ok[k+'_cross']=bool(c and c[6]==c[4]-c[5])
        ok[k+'_side']=bool(c and w[k+'_side']==word(c[6]))
    sides=(w['target_side'],w['goal_side'])
    ok['separated']=bool(None not in sides and out['separated']==('yes' if set(sides)=={'positive','negative'} else 'no'))
    ok['verdict']=bool(out['separated'] is not None and out['predicted'] is not None and out['predicted']==(0 if out['separated']=='yes' else 1))
    out.update(written=w,steps=ok,all_steps=all(ok[s] for s in STEPS),execution_agreement=out['predicted'] is not None and out['predicted']==out['program'],
        format_valid=bool(out['predicted'] is not None and re.fullmatch(STRICT,text)))
    return out


def configuration():
    s4=s4_configuration();base=s4['s3_configuration']
    return dict(version=VERSION,s4_configuration=s4,s3_configuration=base,arm=ARM,seeds=[SEED_RUN],view=s4['view'],floor_z=s4['floor_z'],
        target='P\'s four lines, then Line vector, Target offset, Goal offset (from the first line point), Target cross and Goal cross '
            '(two products and a difference), Target side and Goal side (positive, negative or zero), Separated, Verdict',
        computation='cross = vector x * offset y - vector y * offset x; separated when one side is positive and the other negative; '
            'verdict 0 when separated, else 1; negative numbers after * or - are written in parentheses',
        maximum_new_tokens=256,primary_reading='model',selection=dict(base['selection'],value='own Verdict accuracy (free generation)'),
        training_fit=dict(base['training_fit']),evaluation_updates=list(base['evaluation_updates']),optimizer_updates=base['optimizer_updates'],
        order='The S4 P order for seed 713, unchanged',unchanged_from_s4=list(s4['unchanged_from_s3'])+['order','inputs','points'],
        comparison='Paired against the S3 A and S4 P records (seed 713) on the full validation split; same rows and inference inputs',
        steps=list(STEPS),step_check='Each written step is checked against the model\'s own earlier numbers; unreadable steps count as wrong',
        rules=dict(self_computes='Own Verdict accuracy S >= 0.85 and the image-bootstrap 95% interval of S - A above 0',
            partial='0.65 <= S < 0.85 and the interval of S - A above 0',no_self_compute='S <= 0.60',unclear='Any other result'),
        bootstrap=dict(unit='image',replicates=REPLICATES,seed=SEED),test_usage='No stage loads test or test_natural',
        hyperparameter_search=False,api_calls=0,forced_prefix_boundary=FORCED)


def s5_reading(s,interval):
    if s>=.85 and interval[0]>0:return 'self_computes'
    if .65<=s<.85 and interval[0]>0:return 'partial'
    if s<=.60:return 'no_self_compute'
    return 'unclear'
