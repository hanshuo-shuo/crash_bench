"""Pre-model source splits and complete-unit sampling for FD1000."""
from collections import Counter,defaultdict
import hashlib
import json

EXPOSED={
    'safelibero_object/I/task2/episode6',
    'safelibero_spatial/I/task0/episode0',
    'safelibero_goal/I/task0/episode1',
    'safelibero_long/I/task2/episode1',
    'safelibero_long/I/task3/episode1',
}


def order(value):return hashlib.sha256(('fd1000/1014/'+value).encode()).hexdigest()


def freeze_sources(rows):
    sources={};components=defaultdict(list)
    for row in rows:
        m=row['metadata'];source=m['source_id']
        value={k:m[k] for k in ('source_id','task_group','family','ancestry_component')}
        if source in sources and sources[source]!=value:raise RuntimeError('Source ancestry changed')
        sources[source]=value
    for source,value in sources.items():components[value['ancestry_component']].append(source)
    strata=defaultdict(list);assignment={}
    for group,members in components.items():
        if set(members)&EXPOSED:assignment[group]='train';continue
        key=tuple(sorted({sources[s]['task_group'] for s in members}));strata[key].append(group)
    for members in strata.values():
        members=sorted(members,key=order);n=len(members)
        held=max(1,round(n*.2)) if n>=3 else 0
        for i,group in enumerate(members):
            assignment[group]='validation' if i<held else 'test' if i<2*held else 'train'
    return dict(version='FD1000-source-split-v1',seed=1014,
        rule='Ancestry grouped, stratified by native task; fixed before any verifier or shortcut readout fit',
        exposed_sources_forced_to_train=sorted(EXPOSED),
        test_usage='Locked for the final model comparison; validation is used for dataset shortcut screening',
        sources=[dict(sources[s],split=assignment[sources[s]['ancestry_component']]) for s in sorted(sources)],
        counts=dict(Counter(assignment.values())))


def check_unit(rows):
    if len(rows)!=4:raise RuntimeError('A complete contrast unit must contain four rows')
    if Counter(r['evidence']['feasibility'] for r in rows)!={'feasible':2,'conditionally_infeasible':2}:
        raise RuntimeError('Unit labels are not counterbalanced')
    for key in ('source_id','checkpoint','ancestry_component'):
        if len({r['metadata'][key] for r in rows})!=1:raise RuntimeError('Unit provenance differs')
    for key in ('construction_mode','construction_version','positive_side','construction_cell'):
        if len({r['metadata'].get(key) for r in rows})!=1:raise RuntimeError('Unit construction differs')
    images=defaultdict(Counter);texts=defaultdict(Counter);full={}
    for row in rows:
        if set(row['student'])!={'images','task','constraints'}:raise RuntimeError('Privileged student field')
        y=row['evidence']['feasibility'];student=row['student']
        ik=tuple(im['sha256'] for im in student['images'])
        tk=(student['task'],student['constraints'])
        images[ik][y]+=1;texts[tk][y]+=1
        if (ik,tk) in full:raise RuntimeError('Duplicate complete student input in a contrast unit')
        full[(ik,tk)]=y
    if len(images)!=2 or len(texts)!=2 or any(c!={'feasible':1,'conditionally_infeasible':1} for c in list(images.values())+list(texts.values())):
        raise RuntimeError('Image-only or text-only label balance failed')
    if len({tuple(r['metadata']['goal']) for r in rows})!=1:
        raise RuntimeError('Selected object/category or task changed across the unit')
    return True


def selection_cell(row):
    m=row['metadata']
    if 'construction_cell' in m:
        from .constraint_sides import cell
        if m['construction_cell']!=cell(m['construction_mode'],m['positive_side']):raise RuntimeError('Invalid factor-cell metadata')
        return m['construction_cell']
    return m.get('construction_mode')


def select_units(units,target=250,maximum_per_source=10,quotas=None):
    grouped=defaultdict(list)
    limits={(split,mode):n for split,values in (quotas or {}).items() for mode,n in values.items()}
    if limits and (sum(limits.values())!=target or any(type(n) is not int or n<=0 for n in limits.values())):
        raise RuntimeError('Invalid predeclared split/mode quotas')
    for unit in units:
        check_unit(unit);m=unit[0]['metadata'];grouped[m['source_id']].append(unit)
    for source,pool in grouped.items():
        checkpoints=[u[0]['metadata']['checkpoint'] for u in pool]
        if len(set(checkpoints))!=len(checkpoints):raise RuntimeError('Multiple layouts inflate the same source checkpoint')
        pool.sort(key=lambda u:u[0]['metadata']['checkpoint'])
    selected=[];counts=Counter();source_counts=Counter()
    for index in range(maximum_per_source):
        ordered=sorted(grouped,key=lambda s:(source_counts[s],order(s))) if limits else sorted(grouped,key=order)
        for source in ordered:
            if index<len(grouped[source]):
                unit=grouped[source][index]
                if limits:
                    key=(unit[0]['split'],selection_cell(unit[0]))
                    if key not in limits:raise RuntimeError('Unknown split/construction mode')
                    if counts[key]>=limits[key]:continue
                    counts[key]+=1
                selected.append(unit)
                source_counts[source]+=1
            if len(selected)==target:return selected
    raise RuntimeError('Insufficient complete observable units: '+str(len(selected)))
