"""V5 seed campaign: arms A and P with seeds 714 and 715, validation only (docs/generative_verifier/V5_SEEDS_PROTOCOL.md).

A keeps the frozen S3 settings and P the frozen S4 settings; both reuse their training code (v5_run.train_arm,
v5_pos_run.train_p). Seed 713 is read from the finished S3 A and S4 P records. For each new seed one
v5_contract.schedule order serves both arms, as it did for 713. The selection and training-fit subsets and the
lookup come from S3, the true points from S4; the preflight recomputes each and requires equality. The reading
averages each row over the three seeds and bootstraps images.
"""
from .common import require
from .v5_contract import configuration as s3_configuration,schedule,check_schedule,DATASET,TRAIN_LOOKUP
from .v5_pos_contract import configuration as s4_configuration
from .v5_metrics import cluster_interval,REPLICATES,SEED

VERSION='V5-object-seeds-v1'
ARMS=('A','P')
SEEDS=(714,715)
REFERENCE_SEED=713
ALL_SEEDS=(REFERENCE_SEED,)+SEEDS
THRESHOLDS=dict(p_floor=.65,p_fail=.60,delta=.10)
S3_FILES=('RECEIPT.json','config.json','preflight/COMPLETE.json','preflight/ENCODINGS.json','ORDERS.json','SUBSETS.json','LOOKUP.json',
    'S3_READING.json','A_seed713/RESULT.json','A_seed713/validation.jsonl')
S4_FILES=('RECEIPT.json','config.json','preflight/COMPLETE.json','preflight/ENCODINGS.json','POINTS.json','ORDERS.json','S4_READING.json',
    'P_seed713/RESULT.json','P_seed713/validation.jsonl')
PAIRED_FIELDS=('truth','image_id','source_id','student_input_sha256')


def configuration():
    s3=s3_configuration();s4=s4_configuration()
    return dict(version=VERSION,s3_configuration=s3,s4_configuration=s4,arms=list(ARMS),seeds=list(SEEDS),reference_seed=REFERENCE_SEED,
        all_seeds=list(ALL_SEEDS),dataset=str(DATASET),
        arm_settings={'A':'S3 arm A unchanged: target Verdict: 0/1, selection by free-generation verdict accuracy',
            'P':'S4 arm P unchanged: positions then a program verdict, selection by program verdict accuracy'},
        reference='Seed 713 is the S3 A run and the S4 P run, read from their records; neither is retrained',
        order='v5_contract.schedule for each new seed (one epoch, 448 updates of 4 rows); A and P share the order of a seed; the '
            'preflight requires the same function to reproduce the S3 orders and the S4 P order for 713',
        subsets='S3 SUBSETS.json (selection 384 validation rows, training fit 256 rows), recomputed and required equal',
        lookup='S3 LOOKUP.json (rule x progress), recomputed and required equal',
        points='S4 POINTS.json, recomputed with v5_pos_contract.image_points and required equal',
        inputs='Every row input digest of A and P equal to the S3 preflight input_sha256',
        aggregation=dict(row='For each validation row and arm, mean correctness over the three seeds (A verdict, P program verdict)',
            difference='Mean over rows of (P row mean - A row mean)',
            bootstrap=dict(unit='image',replicates=REPLICATES,seed=SEED,reference='source clusters (4 validation sources), reference only',
                note='Seeds are fixed; seed spread is reported per seed')),
        rules=dict(stable='Every seed P >= 0.65 and P - A >= 0.10, and the image-bootstrap 95% interval of the seed-mean P - A above 0',
            unstable='At least one seed P <= 0.60',unclear='Any other result',
            per_seed='The S4 reading for each seed alone (positions_ground, positions_fail, unclear) is reported, not used'),
        thresholds=dict(THRESHOLDS),lookup_bound=TRAIN_LOOKUP,
        test_usage='No stage loads test or test_natural; the one test evaluation needs the user and its own code',
        hyperparameter_search=False,api_calls=0)


def orders(train_ids,config):
    """One-epoch orders for the new seeds; P gets the facts-kind order that v5_contract.schedule gives B, as S4 did for 713."""
    plan=schedule(train_ids,dict(config['s3_configuration'],seeds=list(config['seeds'])))
    return {seed:{'A':arms['A'],'P':arms['B']} for seed,arms in plan.items()}


def check_orders(plan,train_ids,config):
    base=config['s3_configuration'];require(list(plan)==[str(s) for s in config['seeds']],'Seeds of the orders differ')
    sequences=[]
    for seed,arms in plan.items():
        require(set(arms)==set(ARMS),'Arms of the orders differ')
        check_schedule(arms['A'],train_ids,'A',base);check_schedule(arms['P'],train_ids,'B',base)
        a,p=([s for b in arms[k] for s,_ in b] for k in ARMS);require(a==p,'A and P of one seed differ in sample order');sequences.append(tuple(a))
    require(len(set(sequences))==len(sequences),'Two seeds share one sample order')


def program_view(records):
    """P records read by the program verdict (the S4 primary reading), as v5_pos_run.views does."""
    return [dict(r,predicted=r['program'],p_feasible=None,reading='program') for r in records]


def seed_average(a_by_seed,p_by_seed):
    """Row means of correctness over seeds for A and P, their difference, image-bootstrap intervals and sources for reference."""
    import numpy as np
    seeds=sorted(a_by_seed);require(seeds==sorted(p_by_seed) and len(seeds)>=2,'A and P need the same seeds')
    keyed={}
    for arm,by in (('A',a_by_seed),('P',p_by_seed)):
        for seed in seeds:
            rs={r['sample_id']:r for r in by[seed]};require(len(rs)==len(by[seed]),'Repeated sample id');keyed[(arm,seed)]=rs
    first=keyed[('A',seeds[0])];ids=sorted(first)
    for rs in keyed.values():
        require(sorted(rs)==ids,'Seeds or arms cover different rows')
        for s in ids:
            require(all(rs[s][f]==first[s][f] for f in PAIRED_FIELDS),'Rows differ across seeds or arms')
            require(not rs[s].get('forced_prefix'),'Seed average needs free generation')
    correct=lambda arm:np.asarray([[keyed[(arm,seed)][s]['predicted']==first[s]['truth'] for s in ids] for seed in seeds],dtype=float)
    a=correct('A');p=correct('P');am=a.mean(0);pm=p.mean(0);d=pm-am
    images=[first[s]['image_id'] for s in ids];sources=[first[s]['source_id'] for s in ids]
    same=lambda arm:float(np.mean([len({keyed[(arm,seed)][s]['predicted'] for seed in seeds})==1 for s in ids]))
    by_seed=lambda m:{str(k):float(v) for k,v in zip(seeds,m.mean(1))}
    return dict(seeds=seeds,rows=len(ids),images=len(set(images)),sources=len(set(sources)),A_by_seed=by_seed(a),P_by_seed=by_seed(p),
        difference_by_seed=by_seed(p-a),A_mean=float(am.mean()),P_mean=float(pm.mean()),difference=float(d.mean()),
        image_bootstrap_95pct=dict(difference=cluster_interval(d,images),A=cluster_interval(am,images),P=cluster_interval(pm,images)),
        source_bootstrap_95pct_reference=dict(difference=cluster_interval(d,sources),A=cluster_interval(am,sources),P=cluster_interval(pm,sources)),
        rows_correct_every_seed=dict(A=int((a.min(0)==1).sum()),P=int((p.min(0)==1).sum())),
        rows_wrong_every_seed=dict(A=int((a.max(0)==0).sum()),P=int((p.max(0)==0).sum())),
        same_prediction_every_seed=dict(A=same('A'),P=same('P')),replicates=REPLICATES,bootstrap_seed=SEED,
        definitions=dict(difference='mean over rows of (P row mean - A row mean), equal to the mean of the per-seed differences',
            same_prediction_every_seed='rows whose prediction (None counted as a value) is identical for all seeds'))


def seeds_reading(p_by_seed,delta_by_seed,interval,t=THRESHOLDS):
    """Preregistered reading over the three seeds' P (program) accuracies, their P - A, and the seed-mean P - A interval."""
    require(sorted(p_by_seed)==sorted(delta_by_seed)==sorted(str(s) for s in ALL_SEEDS),'Reading needs all three seeds')
    if min(p_by_seed.values())>=t['p_floor'] and min(delta_by_seed.values())>=t['delta'] and interval[0]>0:return 'stable'
    if min(p_by_seed.values())<=t['p_fail']:return 'unstable'
    return 'unclear'
