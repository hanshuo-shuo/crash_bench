"""V5 detector baseline (docs/generative_verifier/V5_DET_PROTOCOL.md): an off-the-shelf detector plus P's program, no training.

GroundingDINO SwinT-OGC, the weights AEGIS uses, looks for each object of a row on View 1 with the object's rule phrase.
The top-scoring box becomes a floor point at a fixed fraction of its height: one fraction for the static objects (anchors,
basket) and one for the pudding, both chosen on the training rows by program accuracy. S4's program then decides each
row. The AEGIS rule (first box above 0.35) is a diagnostic. The comparison is P, averaged over its three seeds.
"""
from .common import require
from .v5_contract import DATASET,TRAIN_LOOKUP
from .v5_pos_contract import program_separated,anchors,entity_phrase,SIZE,VIEW
from .v5_seeds_contract import configuration as seeds_configuration,ALL_SEEDS
from .v5_metrics import REPLICATES,SEED

VERSION='V5-object-det-v1'
DETECTOR=dict(name='GroundingDINO SwinT-OGC',package='groundingdino-py 0.4.0',checkpoint='groundingdino_swint_ogc.pth',
    checkpoint_sha256='3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799',config='GroundingDINO_SwinT_OGC.py',
    text_encoder='bert-base-uncased',bert_revision='86b5e0934494bd15c9632b12f734a8a67f723594',
    preprocessing='groundingdino.util.inference.load_image (short side 800)',scoring='groundingdino.util.inference.predict: max sigmoid over caption tokens')
STATIC_FRACTIONS=(.5,.6,.7,.8,.9,1.)
TARGET_FRACTIONS=(.5,.75,1.,1.25,1.5,2.,2.5,3.)
REFERENCE_RULES={'centre':(.5,.5),'bottom':(1.,1.)}
AEGIS_BOX_THRESHOLD=.35
AEGIS_TEXT_THRESHOLD=.25
MARGIN=4
REPEAT_QUERIES=16
THRESHOLDS=dict(close=.85,needed=.75,margin=.10)
SPLITS=('train','validation','validation_natural')
S3_FILES=('RECEIPT.json','config.json','SUBSETS.json','LOOKUP.json','A_seed713/RESULT.json','A_seed713/validation.jsonl')
S4_FILES=('RECEIPT.json','config.json','POINTS.json','S4_READING.json','zero_shot/SUMMARY.json','P_seed713/RESULT.json','P_seed713/validation.jsonl')
SEEDS_FILES=('RECEIPT.json','config.json','SEEDS_READING.json',*('%s_seed%d/%s'%(a,s,f) for a in ('A','P') for s in (714,715)
    for f in ('RESULT.json','validation.jsonl')))
ZERO_SHOT_REFERENCE={'Qwen/Qwen2.5-VL-3B-Instruct':'S4 R6 selection-subset program accuracy','Qwen/Qwen3-VL-32B-Instruct':'same'}


def configuration():
    return dict(version=VERSION,seeds_configuration=seeds_configuration(),dataset=str(DATASET),detector=dict(DETECTOR),view=VIEW,image_size=SIZE,
        splits=list(SPLITS),queries='One per image and object: the pudding, the basket and every anchor of the image\'s rules',
        caption='The rule phrase without its article (v5_pos_contract.entity_phrase); one wording, no prompt search',
        box=dict(primary='The highest-scoring of the 900 candidate boxes',
            aegis='The first candidate (in model order) scoring above %s, as AEGIS get_point_cloud; none counts as a missing point'%AEGIS_BOX_THRESHOLD,
            aegis_box_threshold=AEGIS_BOX_THRESHOLD,aegis_text_threshold=AEGIS_TEXT_THRESHOLD),
        point='x = box centre; y = box top + fraction x box height, in View 1 pixels',
        fractions=dict(static=list(STATIC_FRACTIONS),target=list(TARGET_FRACTIONS),
            selection='Highest program accuracy on the training rows; ties by the lower median floor-point error over all training image '
                'points, then the smaller static and target fractions'),
        reference_rules={k:list(v) for k,v in REFERENCE_RULES.items()},program='v5_pos_contract.program_separated unchanged; a missing point '
            'or coincident anchor points count as wrong',truth='S4 POINTS.json, recomputed by the preflight',
        hit='Box centre inside the projected 2D box of the object widened by %d pixels (S4 R6)'%MARGIN,
        comparison=dict(P='Row mean over seeds %s of the P program verdict (S4 and seed-campaign records)'%list(ALL_SEEDS),
            A='Row mean over the same seeds, reference only',difference='Mean over validation rows of (P row mean - detector correctness)',
            bootstrap=dict(unit='image',replicates=REPLICATES,seed=SEED,reference='source clusters, reference only')),
        repeat_check='The first %d queries run again after all; the top box index and coordinates must be identical'%REPEAT_QUERIES,
        rules=dict(detector_close='Detector program accuracy D >= 0.85 on the validation split',
            training_needed='D <= 0.75 and the image-bootstrap 95% interval of P - D at or above 0.10',between='Any other result'),
        thresholds=dict(THRESHOLDS),lookup_bound=TRAIN_LOOKUP,test_usage='No stage loads test or test_natural',training=False,api_calls=0)


def caption(entity):
    text=entity_phrase(entity);require(text.startswith('the '),'Unexpected phrase');return text[4:]


def floor_point(box,fraction):
    """[cx, cy, w, h] normalised box to the floor point in View 1 pixels: the centre column, a fraction down from the top."""
    cx,cy,w,h=box;return [cx*SIZE,(cy-h/2+fraction*h)*SIZE]


def centre(box):
    return [box[0]*SIZE,box[1]*SIZE]


def inside(point,box,margin=MARGIN):
    return point is not None and box[0]-margin<=point[0]<=box[2]+margin and box[1]-margin<=point[1]<=box[3]+margin


def aegis_box(record):
    """AEGIS's choice: the first candidate above the box threshold in model order, or None."""
    above=record['above']
    require([c['index'] for c in above]==sorted(c['index'] for c in above),'Candidates out of model order')
    require(all(c['score']>AEGIS_BOX_THRESHOLD for c in above),'Candidate below the AEGIS threshold')
    return above[0]['box'] if above else None


def image_points(boxes,static,target):
    """Floor points of one image from its chosen boxes (entity -> box or None)."""
    return {k:(None if b is None else floor_point(b,target if k=='target' else static)) for k,b in boxes.items()}


def row_program(row,points):
    a,b=anchors(row);q=[points.get(k) for k in (a,b,'target','goal')]
    if any(p is None for p in q) or q[0]==q[1]:return None
    return 0 if program_separated(*q) else 1


def median_error(boxes_by_image,truth,static,target):
    import numpy as np
    d=[float(np.hypot(p[0]-truth[i]['points'][k][0],p[1]-truth[i]['points'][k][1])) for i,boxes in boxes_by_image.items()
       for k,p in image_points(boxes,static,target).items() if p is not None]
    return float(np.median(d)) if d else float('inf')


def accuracy(rows,boxes_by_image,static,target):
    points={i:image_points(b,static,target) for i,b in boxes_by_image.items()}
    return sum(row_program(r,points[r['metadata']['image_id']])==int(r['target']['digit']) for r in rows)/len(rows)


def choose_fractions(rows,boxes_by_image,truth):
    """The (static, target) pair with the best training accuracy; ties by median floor error, then the smaller fractions."""
    require(rows and all(r['split']=='train' for r in rows),'Fractions are chosen on training rows only')
    table=[dict(static=s,target=t,accuracy=accuracy(rows,boxes_by_image,s,t),median_error_px=median_error(boxes_by_image,truth,s,t))
           for s in STATIC_FRACTIONS for t in TARGET_FRACTIONS]
    best=min(table,key=lambda c:(-c['accuracy'],c['median_error_px'],c['static'],c['target']))
    return dict(static=best['static'],target=best['target'],training_accuracy=best['accuracy'],table=table)


def det_reading(d,interval,t=THRESHOLDS):
    if d>=t['close']:return 'detector_close'
    if d<=t['needed'] and interval[0]>=t['margin']:return 'training_needed'
    return 'between'
