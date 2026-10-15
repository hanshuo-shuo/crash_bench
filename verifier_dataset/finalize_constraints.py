"""Package reviewed rule pairs only after independent evidence and shortcut gates."""
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
from .inventory import read,sha,write
from .constraint_audit import check_hashes


def final_rationale(row):
    """Explain visible constraints; do not pretend a hidden trajectory was input."""
    colors=row['metadata']['appearance_assignment'];protected=row['evidence']['protected_boundary']
    from .constraint_sides import ENTRY,BOTH,NEITHER
    if protected=='b' and 'positive_side' in row['metadata']:
        side=row['metadata']['positive_side'];color=colors['b']
        if side==BOTH:
            return (f'The white-outlined target and the green destination are inside the protected {color} boundary. '
                'Motion between them does not require crossing its closed faces. The other stated safety constraints still apply.')
        if side!=NEITHER:raise RuntimeError('Unknown positive-region side')
        movement='enter the destination enclosure' if row['metadata']['construction_mode']==ENTRY else 'leave the target enclosure'
        return (f'The white-outlined target and the green destination are outside the protected {color} boundary. '
            f'The {colors["a"]} boundary may be crossed to {movement}. The other stated safety constraints still apply.')
    if row['metadata'].get('construction_mode')==ENTRY:
        color=colors[protected]
        if protected=='a':
            return (f'The white-outlined target is outside the {color} boundary, while the green destination is inside. '
                f'The rule protects {color}. Every route to that destination enters through a face of the closed boundary, which the stated material-point rule forbids.')
        if protected!='b':raise RuntimeError('Unexpected protected region')
        return (f'The white-outlined target and the green destination are inside the protected {color} boundary. '
            'Motion between them does not require crossing its closed faces. The other stated safety constraints still apply.')
    if protected=='a':
        return (f'The white-outlined target is inside the {colors["a"]} boundary, while the green destination is outside. '
            f'The rule protects {colors["a"]}. Every route out crosses a face of this closed boundary, which the stated material-point rule forbids.')
    if protected!='b':raise RuntimeError('Unexpected protected region')
    return (f'The white-outlined target is inside the {colors["a"]} boundary, which this rule allows it to cross. '
        f'The protected {colors["b"]} boundary is separate from the target and the green destination. '
        'The enclosing region therefore permits an exit toward the goal; the other stated safety constraints still apply.')


def review_matches(review,rows,prepared_digest):
    if not review.get('complete') or review.get('unresolved_units') or review.get('prepared_dataset_sha256')!=prepared_digest:
        raise RuntimeError('Visual review is incomplete or bound to another dataset')
    units=defaultdict(list)
    for row in rows:units[row['metadata']['unit_id']].append(row)
    if set(units)!=set(review['reviewed_units']):raise RuntimeError('Visual review omits or adds units')
    for name,unit in units.items():
        verdict=review['reviewed_units'][name]
        if verdict['status']!='passed':raise RuntimeError('Unresolved visible premises')
        wanted=sorted({im['sha256'] for row in unit for im in row['student']['images']})
        if sorted(verdict['image_sha256'])!=wanted:raise RuntimeError('Review refers to different pixels')
        if not all(verdict.get(k) is True for k in ('target_identifiable','target_boundary_relation_visible','destination_relation_visible','both_rule_colors_identifiable')):
            raise RuntimeError('Rationale premises have not all passed visual review')
    return True


def main(root):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Final data acceptance belongs on a compute node')
    inherited=read(root/'FINALIZE_SOURCE.json');prepared=Path(inherited['prepared'])
    if sha(prepared/'HASHES.json')!=inherited['prepared_hashes_sha256']:raise RuntimeError('Prepared evidence manifest changed')
    check_hashes(prepared)
    audit=read(prepared/'AUDIT_COMPLETE.json');shortcut=read(prepared/'SHORTCUT_AUDIT.json')
    if not audit['passed'] or not shortcut['passed'] or shortcut['locked_test_scored']:
        raise RuntimeError('Evidence/shortcut audit failed or locked test was used during screening')
    review_path=root/'source'/inherited['review_relative_path']
    if sha(review_path)!=inherited['review_sha256']:raise RuntimeError('Published visual review changed')
    review=read(review_path);source_path=prepared/'PROPOSED_DATASET.jsonl'
    rows=[json.loads(line) for line in source_path.read_text().splitlines()]
    review_matches(review,rows,sha(source_path))
    # Rehash all persisted raw evidence used by the independent audit. This is
    # an integrity check, not a claim that simulation was executed a third time.
    for trace in read(prepared/'TRACE_AUDIT.json'):
        folder=Path(trace['root'])
        if sha(folder/'HASHES.json')!=trace['hashes_sha256']:raise RuntimeError('Audited raw trace manifest changed')
        check_hashes(folder)
    evidence_roots={Path(r['metadata']['evidence_root']) for r in rows}
    for folder in evidence_roots:check_hashes(folder)
    out=root/'dataset';out.mkdir()
    for row in rows:
        unit=row['metadata']['unit_id'];assignment=row['evidence']['assignment']
        trace_geometry=read(Path(row['metadata']['trace_root'])/'GEOMETRY.json')
        actors=set(trace_geometry['actors'])
        selected=trace_geometry['goal_definition']['goal'][1]
        target_geoms=trace_geometry['contact_scope']['targets'].get(selected,[])
        if not target_geoms or not set(target_geoms)<=actors or not set(trace_geometry['contact_scope']['robot'])<=actors:
            raise RuntimeError('Source witness does not cover the visible selected-target contact rule')
        old_scope='The robot and native task targets must avoid contact with '
        if row['student']['constraints'].count(old_scope)!=1:raise RuntimeError('Unexpected source contact wording')
        # The derived atomic task selects one target. Source replay checked all
        # original native targets as well, which is stronger positive evidence.
        # The negative separator is sufficient regardless of other actors.
        row['student']['constraints']=row['student']['constraints'].replace(old_scope,
            'The robot and the selected object must avoid contact with ',1)
        for image in row['student']['images']:
            original=Path(image['path'])
            if sha(original)!=image['sha256']:raise RuntimeError('Reviewed image changed before packaging')
            rel=Path('images')/unit/('assignment_%d'%assignment)/(image['camera']+'.png');dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():
                if sha(dest)!=image['sha256']:raise RuntimeError('Packaged image collision')
            else:shutil.copyfile(original,dest)
            image['path']=str(rel)
        label=row['evidence']['feasibility']
        if label not in ('feasible','conditionally_infeasible'):raise RuntimeError('Unknown cannot be packaged as a binary label')
        rationale=final_rationale(row)
        row.update(schema_version='FD1000-reviewed-v1',training_eligible=True,observability_review='passed',
            target=dict(verdict='feasible' if label=='feasible' else 'infeasible',rationale=rationale,scope='conditional_on_stated_contract'),
            reasoning_target=dict(rationale=rationale,verdict=int(label=='feasible')))
        row['metadata'].update(rationale_version='constraint_observable_clauses_v1',
            rationale_scope='Visible/explicit constraint logic. The independent trajectory supplies positive existential evidence; no hidden trajectory is presented as a student observation.',
            effective_contact_scope='Robot and current selected target; inherited positive witness additionally checks all original source-task targets',
            visual_review_sha256=inherited['review_sha256'],raw_candidate_draft_preserved_in_teacher=True)
    (out/'dataset.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    shortcut['final_input_scope_clarification']=dict(
        change='Replace the opaque native-task-target phrase with the current selected object; positive evidence checks a superset of actors and the negative guard proof does not depend on other actors',
        pixels_and_protected_color_unchanged=True,joint_readout_features_unchanged=True,
        text_only_chance_still_guaranteed_by_exact_pair_balance=True,
        raw_screening_artifact=str(prepared/'SHORTCUT_AUDIT.json'),raw_screening_sha256=sha(prepared/'SHORTCUT_AUDIT.json'))
    write(out/'EVIDENCE_AUDIT.json',audit);write(out/'SHORTCUT_AUDIT.json',shortcut);write(out/'VISUAL_REVIEW.json',review)
    shutil.copyfile(prepared/'source/verifier_dataset/frozen_sources_67.json',out/'FROZEN_SOURCES.json')
    if sha(out/'FROZEN_SOURCES.json')!=audit['frozen_sources_sha256']:raise RuntimeError('Frozen source split differs')
    acceptance=dict(passed=True,scientific_training_allowed=True,
        dataset_sha256=sha(out/'dataset.jsonl'),evidence_audit_sha256=sha(out/'EVIDENCE_AUDIT.json'),
        shortcut_audit_sha256=sha(out/'SHORTCUT_AUDIT.json'),visual_review_sha256=sha(out/'VISUAL_REVIEW.json'),
        frozen_sources_sha256=sha(out/'FROZEN_SOURCES.json'),prepared_dataset_sha256=sha(source_path),
        scope='Controlled source-held-out virtual-rule feasibility; one separator mechanism, not an unchanged benchmark or proof of generative necessity',
        api_calls=0,slurm_job=os.environ['SLURM_JOB_ID'])
    write(out/'ACCEPTANCE.json',acceptance)
    from verifier_training.fd1000_contract import validate
    validated,_=validate(out)
    write(out/'MANIFEST.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    write(root/'COMPLETE.json',dict(passed=True,dataset=str(out),records=len(validated),
        acceptance_sha256=sha(out/'ACCEPTANCE.json'),manifest_sha256=sha(out/'MANIFEST.json'),training_started=False,api_calls=0))


if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
