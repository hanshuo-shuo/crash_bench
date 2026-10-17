"""Package the 896 V4 rows as development data (user decision 2026-10-16) with checked structured facts.

Run on a CPU compute node. Student text stays exactly as audited; only images move
beneath dataset/images and each row gains geometry-checked facts. Nothing here
fits a model or reads model predictions on the locked test split.
"""
from collections import Counter,defaultdict
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import sys
from .common import read,write,sha,require
from .fd1000_contract import FACT_KEYS,DEVELOPMENT_FILES,facts_text,rule_verdict,validate
from .fd1000_metrics import auroc

AUDIT_ROOT=Path('/projects/p33100/siosio/crashbench_safelibero/verifier_dataset/20261015T042328Z_audit-constraints_75be81f8f066')
PROPOSED_SHA256='f463a315d9f253e8bcabb3055158f8fa21969a026bb041fb8da9a6f7c7661bbd'
REVIEW='verifier_dataset/reviews/fd896_v4_final_visual.json'
SHORTCUT_THRESHOLD=.75
STRATIFIED_READOUTS=('color_statistics_rule_interactions_linear','rgb_color_rule_interactions_linear')
USER_DECISION=dict(date='2026-10-16',
    statement='The V4 review need not be strict ("as long as it works"). The 896-row V4 set is development data for verifier '
        'training and evaluation; its failed shortcut gate stays disclosed and is reported as a baseline, not a training precondition.',
    recorded_in=['AGENTS.md','docs/CURRENT.md','verifier_dataset/CURRENT.md','docs/generative_verifier/PLAN.md'])


def metadata_facts(row):
    """Facts implied by the construction factors (PLAN.md section 4.0)."""
    from verifier_dataset.constraint_sides import EXIT,ENTRY,BOTH,NEITHER
    m=row['metadata'];protected=row['evidence']['protected_boundary']
    if protected=='a':
        sides={EXIT:('yes','no'),ENTRY:('no','yes')}
        require(m['construction_mode'] in sides,'Unknown separator construction');target,goal=sides[m['construction_mode']]
    else:
        require(protected=='b' and m['positive_side'] in (BOTH,NEITHER),'Unknown positive-region side')
        target=goal='yes' if m['positive_side']==BOTH else 'no'
    return dict(protected=m['appearance_assignment'][protected],target_inside=target,goal_inside=goal,
                separated='yes' if target!=goal else 'no')


def relation(box,bounds):
    from verifier_dataset.constraint_sides import contains_bounds
    from verifier_dataset.paired_geometry import disjoint
    if contains_bounds(box,bounds):return 'yes'
    require(disjoint(box,bounds),'Endpoint is neither strictly inside nor disjoint from the protected box')
    return 'no'


def geometry_facts(layout,protected,color):
    """Facts recomputed from LAYOUT.json: target box, fixed destination and necessary material goal."""
    box=layout[protected];target=relation(box,layout['target'])
    goals={relation(box,layout['goal']),relation(box,layout['necessary_material_goal'])}
    require(len(goals)==1,'Destination and necessary material goal disagree');goal=goals.pop()
    return dict(protected=color,target_inside=target,goal_inside=goal,separated='yes' if target!=goal else 'no')


def check_evidence(row,layout,verified):
    """The label must follow from the certificate (negative) or the safe witness (positive)."""
    from verifier_dataset.constraint_sides import EXIT,ENTRY,verify_sides
    m=row['metadata'];e=row['evidence']
    require(verified['contract']==e['contract']==layout['contract'] and verified['checkpoint']==m['checkpoint']==layout['checkpoint'],
            'Evidence belongs to another checkpoint or contract')
    require(verified['matched_pair'] is True and verify_sides(layout)==verified['endpoint_side_checks'],'Recomputed endpoint sides differ')
    if e['protected_boundary']=='a':
        require(e['feasibility']=='conditionally_infeasible' and verified['conditional_negative'] is True,'Negative lacks a certificate')
        contract={EXIT:'RA-SEP-1',ENTRY:'FD1000-INWARD-SEP-1'}[m['construction_mode']]
        for name in ('native','synchronized'):
            c=verified['certificates'][name];s=c['separator']
            require(c['label']=='conditionally_infeasible' and s['label']=='infeasible' and s['contract']==contract and
                    all(s['conditions'].values()) and all(s['covered_faces']),'Closed-separator certificate incomplete')
            require(c.get('direction')==('inward' if m['construction_mode']==ENTRY else None),'Certificate direction differs')
        return dict(kind='closed_separator_certificate',contract=contract,histories=['native','synchronized'])
    require(e['feasibility']=='feasible' and verified['positive_witness'] is True,'Positive lacks a safe witness')
    require(0<verified['witness_commands']<=layout['remaining_horizon'],'Witness exceeds the remaining horizon')
    return dict(kind='safe_source_suffix',witness_commands=verified['witness_commands'],remaining_horizon=layout['remaining_horizon'])


def stratified_shortcut(rows,predictions):
    """Validation-only strata of the fixed shortcut readouts (PLAN.md section 2.2)."""
    from verifier_dataset.constraint_sides import EXIT
    by_id={r['sample_id']:r for r in rows};ids=predictions['sample_ids']
    require(all(by_id[s]['split']=='validation' for s in ids) and len(ids)==sum(r['split']=='validation' for r in rows),
            'Shortcut predictions are not exactly the validation split')
    result={}
    for name in STRATIFIED_READOUTS:
        score=predictions['predictions'][name];strata=defaultdict(lambda:([],[]))
        for sid,p in zip(ids,score):
            r=by_id[sid];key=('EXIT' if r['metadata']['construction_mode']==EXIT else 'ENTRY')+(' x checkpoint 0' if r['metadata']['checkpoint']==0 else ' x checkpoint >0')
            strata[key][0].append(int(r['target']['verdict']=='feasible'));strata[key][1].append(p)
        result[name]=dict(all=dict(rows=len(ids),auroc=auroc([int(by_id[s]['target']['verdict']=='feasible') for s in ids],score)),
            **{k:dict(rows=len(v[0]),auroc=auroc(*v)) for k,v in sorted(strata.items())})
    return result


def sample_units(units):
    from verifier_dataset.constraint_selection import selection_cell
    cells=defaultdict(list)
    for name,unit in units.items():
        if unit[0]['split']=='train':cells[selection_cell(unit[0])].append(name)
    require(len(cells)==4,'Sample pack needs all four construction cells')
    order=lambda n:hashlib.sha256(('v4-samples/1014/'+n).encode()).hexdigest()
    return [n for cell in sorted(cells) for n in sorted(cells[cell],key=order)[:2]]


CELL_NAMES={'start_inside_goal_outside':'EXIT：分隔盒包住目标 / separator encloses the target',
    'start_outside_goal_inside':'ENTRY：分隔盒包住终点 / separator encloses the destination',
    'both_inside':'BOTH：另一个盒子同时包住两者 / the other box encloses both',
    'both_outside':'NEITHER：另一个盒子都不包 / the other box encloses neither'}


def samples_html(units,names,evidence):
    """32-row look-through pack; not a review gate."""
    e=html.escape;parts=[]
    for name in names:
        unit=sorted(units[name],key=lambda r:(r['evidence']['assignment'],r['evidence']['protected_boundary']));m=unit[0]['metadata']
        parts.append(f'<section><h2>{e(name)} · {e(m["source_id"])} · checkpoint {m["checkpoint"]}</h2>'
            f'<p class="cell">{e(CELL_NAMES[m["construction_mode"]])}<br>{e(CELL_NAMES[m["positive_side"]])}</p>'
            f'<p><b>Task</b> {e(unit[0]["student"]["task"])}</p>')
        for assignment in (0,1):
            rows=[r for r in unit if r['evidence']['assignment']==assignment];colors=rows[0]['metadata']['appearance_assignment']
            parts.append(f'<h3>Assignment {assignment}: box a = {e(colors["a"])}, box b = {e(colors["b"])} (a is the separator)</h3><div class="views">'+
                ''.join(f'<figure><img src="{e(im["path"])}" alt="{e(im["camera"])}"><figcaption>{e(im["camera"])}</figcaption></figure>' for im in rows[0]['student']['images'])+
                '</div><div class="rules">')
            for r in rows:
                proof=evidence[r['sample_id']]
                why=(f'Certificate {proof["contract"]} in both histories (closed six-face separator)' if proof['kind']=='closed_separator_certificate'
                     else f'Safe source suffix: {proof["witness_commands"]} of {proof["remaining_horizon"]} commands, never crossing the protected box')
                parts.append(f'<div class="rule {"pos" if r["target"]["verdict"]=="feasible" else "neg"}"><p><b>Rule</b> protects '
                    f'{e(r["target"]["facts"]["protected"])} (box {e(r["evidence"]["protected_boundary"])}) → label <b>{e(r["target"]["verdict"])}</b></p>'
                    f'<pre>{e(facts_text(r["target"]["facts"]))}\nVerdict: {int(r["target"]["verdict"]=="feasible")}</pre>'
                    f'<p class="proof">证据 / evidence (hidden from the model): {e(why)}</p></div>')
            parts.append('</div>')
        parts.append(f'<details><summary>Full constraint text (model-visible; only the protected color changes)</summary><pre class="wrap">{e(unit[0]["student"]["constraints"])}</pre></details></section>')
    style=('body{font:15px/1.5 system-ui,sans-serif;margin:24px auto;max-width:1100px;padding:0 16px;background:#fff;color:#1d1d1f}'
        'section{border-top:2px solid #ddd;padding:12px 0}.views{display:flex;gap:8px;flex-wrap:wrap}'
        'figure{margin:0}img{width:320px;max-width:100%;height:auto;border:1px solid #ccc}'
        '.rules{display:flex;gap:12px;flex-wrap:wrap}.rule{flex:1 1 300px;border-radius:6px;padding:4px 12px}'
        '.pos{background:#eef7ee}.neg{background:#fbeeee}pre{background:#f5f5f5;padding:8px}.wrap{white-space:pre-wrap}'
        '.cell,.proof{color:#555}@media (prefers-color-scheme:dark){body{background:#1d1d1f;color:#eee}'
        '.pos{background:#203520}.neg{background:#3a2020}pre{background:#2a2a2a}.cell,.proof{color:#aaa}}')
    intro=('<h1>V4 结构化事实样例 / structured-fact samples</h1><p>8 个训练集四行单元（每种构造 2 个），共 32 行。'
        '模型只看到两张图、任务文本和约束文本；单元 ID、证据、几何和标签来源都不进入输入。'
        'This is a look-through pack for the user, not a review gate.</p>')
    return f'<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V4 Fact Samples</title><style>{style}</style></head><body>{intro}{"".join(parts)}</body></html>\n'


def main(root):
    require(os.environ.get('SLURM_JOB_ID'),'Development export belongs on a compute node')
    from verifier_dataset.constraint_audit import check_hashes
    from verifier_dataset.finalize_constraints import review_matches
    receipt=read(root/'RECEIPT.json');source=root/'source'
    require(sha(AUDIT_ROOT/'PROPOSED_DATASET.jsonl')==PROPOSED_SHA256,'Proposed V4 rows changed')
    check_hashes(AUDIT_ROOT)
    audit=read(AUDIT_ROOT/'AUDIT_COMPLETE.json');shortcut=read(AUDIT_ROOT/'SHORTCUT_AUDIT.json')
    require(audit['passed'] and not shortcut['passed'] and shortcut['locked_test_scored'] is False,'Unexpected audit state')
    rows=[json.loads(line) for line in (AUDIT_ROOT/'PROPOSED_DATASET.jsonl').read_text().splitlines()]
    review=read(source/REVIEW);review_matches(review,rows,PROPOSED_SHA256)
    for trace in read(AUDIT_ROOT/'TRACE_AUDIT.json'):
        folder=Path(trace['root']);require(sha(folder/'HASHES.json')==trace['hashes_sha256'],'Audited raw trace manifest changed');check_hashes(folder)
    for folder in sorted({Path(r['metadata']['evidence_root']) for r in rows}):check_hashes(folder)
    out=root/'dataset';out.mkdir();evidence={};units=defaultdict(list)
    for row in rows:
        m=row['metadata'];e=row['evidence'];folder=Path(m['evidence_root'])
        layout=read(folder/'LAYOUT.json');verified=read(folder/'VERIFIED.json')
        facts=metadata_facts(row)
        require(geometry_facts(layout,e['protected_boundary'],facts['protected'])==facts,'Construction facts disagree with LAYOUT.json')
        require(rule_verdict(facts['target_inside'],facts['goal_inside'])==int(e['feasibility']=='feasible'),'Facts do not imply the label')
        evidence[row['sample_id']]=check_evidence(row,layout,verified)
        require(row['student']['constraints'].count('Only the %s boundary is protected'%facts['protected'])==1,'Rule text names another color')
        listed=read(folder/'IMAGES.json')['assignments'][e['assignment']]
        require([(i['camera'],i['sha256']) for i in listed]==[(i['camera'],i['sha256']) for i in row['student']['images']],'Image manifest differs')
        for image in row['student']['images']:
            original=Path(image['path']);require(original.parent==folder/('assignment_%d'%e['assignment']),'Image outside its assignment')
            rel=Path('images')/m['unit_id']/original.parent.name/(image['camera']+'.png');dest=out/rel
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():require(sha(dest)==image['sha256'],'Packaged image collision')
            else:
                shutil.copyfile(original,dest);require(sha(dest)==image['sha256'],'Copied image differs')
            image['path']=str(rel)
        label='feasible' if e['feasibility']=='feasible' else 'infeasible'
        row.update(schema_version='V4-dev-v1',acceptance_mode='development',training_eligible=True,observability_review='passed',
            target=dict(verdict=label,facts=facts,rationale=facts_text(facts),scope='conditional_on_stated_contract'),
            reasoning_target=dict(facts=facts,verdict=int(label=='feasible')))
        m.update(fact_version='v4_structured_facts_v1',fact_checks=dict(construction_metadata=True,layout_geometry=True,
            evidence=evidence[row['sample_id']]),visual_review_sha256=sha(source/REVIEW),
            student_text_unchanged_from_proposed=True,raw_candidate_draft_preserved_in_teacher=True)
        units[m['unit_id']].append(row)
    for name,unit in units.items():
        colors={r['evidence']['assignment']:r['metadata']['appearance_assignment'] for r in unit}
        require(colors[0]=={'a':colors[1]['b'],'b':colors[1]['a']},'Second assignment must swap the box colors: '+name)
    (out/'dataset.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    write(out/'EVIDENCE_AUDIT.json',audit);write(out/'SHORTCUT_AUDIT.json',shortcut);write(out/'VISUAL_REVIEW.json',review)
    for name in ('SHORTCUT_VALIDATION_PREDICTIONS.json','SHORTCUT_FEATURES.npz'):shutil.copyfile(AUDIT_ROOT/name,out/name)
    shutil.copyfile(AUDIT_ROOT/'source/verifier_dataset/frozen_sources_67.json',out/'FROZEN_SOURCES.json')
    require(sha(out/'FROZEN_SOURCES.json')==audit['frozen_sources_sha256'],'Frozen source split differs')
    readouts={k:v['auroc'] for k,v in shortcut['results'].items() if 'interactions' in k}
    acceptance=dict(passed=True,acceptance_mode='development',development_training_allowed=True,scientific_training_allowed=False,
        user_decision=USER_DECISION,
        shortcut_gate=dict(passed=False,threshold_validation_auroc=SHORTCUT_THRESHOLD,interaction_readout_auroc=readouts,
            maximum=max(readouts.values()),reported_as_baseline=True,locked_test_scored=False,
            stratified_validation_only=stratified_shortcut(rows,read(out/'SHORTCUT_VALIDATION_PREDICTIONS.json')),
            interpretation='The readout has no target or goal features; within a four-row unit any additive image+rule score has AUROC 0.5, '
                'so the signal reflects protected-box appearance or placement. EXIT x checkpoint 0 holds only 8 units (evidence, not proof).'),
        locked_test=dict(rows=sum(r['split']=='test' for r in rows),model_predictions=0,scored=False,
            usage='Scored once per selected checkpoint by evaluate-test after all runs, readings and metrics are frozen'),
        student_input='Unchanged V4 images, task and constraint text; metadata, evidence, geometry and labels stay out of prompts',
        visual_review='Codex per-unit premise-visibility review (fd896_v4_final_visual.json), not an independent human review',
        facts='Derived from construction factors and asserted equal to LAYOUT.json geometry; every label checked against VERIFIED.json',
        prepared_dataset_sha256=PROPOSED_SHA256,audit_root=str(AUDIT_ROOT),
        **{key:sha(out/name) for name,key in DEVELOPMENT_FILES.items()},
        source_commit=receipt['source_commit'],api_calls=0,slurm_job=os.environ['SLURM_JOB_ID'])
    write(out/'DEV_ACCEPTANCE.json',acceptance)
    validated,_=validate(out,mode='development')
    names=sample_units(units);(out/'SAMPLES.html').write_text(samples_html(units,names,evidence))
    write(out/'SAMPLES.json',dict(units=names,rows=4*len(names),purpose='User look-through; not a review gate'))
    write(out/'MANIFEST.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    write(root/'COMPLETE.json',dict(passed=True,dataset=str(out),records=len(validated),
        by_split=dict(Counter(r['split'] for r in validated)),cells=dict(Counter(r['metadata']['construction_cell'] for r in validated)),
        dev_acceptance_sha256=sha(out/'DEV_ACCEPTANCE.json'),manifest_sha256=sha(out/'MANIFEST.json'),
        sample_units=names,training_started=False,api_calls=0,slurm_job=os.environ['SLURM_JOB_ID']))


if __name__=='__main__':
    root=Path(sys.argv[1])
    try:main(root)
    except BaseException as error:
        write(root/'STOP.json',dict(error=type(error).__name__,message=str(error),slurm_job=os.environ.get('SLURM_JOB_ID')));raise
