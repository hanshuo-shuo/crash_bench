"""Supplementary case review: verdict errors as well as format/visibility errors.

Read completed immutable campaign records; never rerun or alter training.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
from .common import read,write,sha,require
from .engineering_contract import CONDITIONS,DATA_SHA

MANUAL=Path(__file__).parent/'manual_reviews/full120_no_image.json'


def manual_review(record,variant,condition):
    decision=read(MANUAL)
    if (variant==decision['variant'] and condition==decision['condition'] and
        record.get('input_sha256')==decision['input_sha256'] and
        hashlib.sha256(record['generated'].encode()).hexdigest()==decision['generated_sha256']):
        require(decision['dataset_sha256']==DATA_SHA,'Manual review dataset mismatch')
        return decision['rationale_status']
    return None


def issues(record,variant,condition):
    tags=[]
    if condition=='text_only':tags.append('case_verdict_not_identifiable_from_identical_text_only_inputs')
    if not record['format_valid']:tags.append('invalid_response_format')
    if not record['verdict_correct']:
        tags.append('wrong_verdict_for_evaluated_label')
        if condition=='horizon_0':tags.append('violates_zero_action_contract_on_recorded_initial_state')
        elif condition=='shuffled_pairs':tags.append('recipient_label_mismatch_is_not_a_donor_label_error')
    if variant=='B':
        assessment=record['rationale_assessment']
        status=assessment['status']
        if status!='supported_complete':tags.append('rationale_'+status)
        if assessment.get('contradictions'):tags.append('visibility_contradicts_input_evidence')
        if assessment.get('unsupported'):tags.append('visibility_assertions_without_images')
        if assessment.get('unparsed'):tags.append('unparsed_prose_needs_manual_review')
    return tags


def main(root,out):
    require(read(root/'COMPLETE.json')['passed'],'Campaign must complete before case review')
    require(sha(root/'dataset/dataset.jsonl')==DATA_SHA,'Dataset changed')
    out.mkdir(parents=True,exist_ok=False)
    cases=[];groups=defaultdict(list);summary={}
    for variant in ('A','B'):
        summary[variant]={}
        for condition in CONDITIONS:
            records=read(root/('diagnostics_'+variant)/(condition+'.json'))
            count=Counter()
            for r in records:
                tags=issues(r,variant,condition)
                manual=manual_review(r,variant,condition)
                if manual:
                    tags=[t for t in tags if t not in ('rationale_needs_manual_review','unparsed_prose_needs_manual_review')]
                count.update(tags)
                case={'variant':variant,'condition':condition,'sample_id':r['sample_id'],
                      'expected_verdict':r['expected_verdict'],'predicted_verdict':r['predicted_verdict'],
                      'verdict_correct':r['verdict_correct'],'format_valid':r['format_valid'],
                      'issues':tags,'generated':r['generated'],
                      'rationale_status':r['rationale_assessment']['status'] if variant=='B' else 'not_applicable',
                      'manual_rationale_status':manual,
                      'causal_verdict_justification_established':False if variant=='B' else None}
                cases.append(case)
                if variant=='B':groups[r['generated']].append({'condition':condition,'sample_id':r['sample_id'],
                    'rationale_status':case['rationale_status'],'manual_rationale_status':manual,
                    'verdict_correct':r['verdict_correct'],'issues':tags})
            summary[variant][condition]={'n':len(records),'issue_counts':dict(count),
                'verdict_errors':sum(not r['verdict_correct'] for r in records)}
    write(out/'ALL_CASES.json',cases)
    write(out/'FLAGGED_CASES.json',[r for r in cases if r['issues']])
    write(out/'B_UNIQUE_OUTPUTS.json',[{'generated':text,'occurrences':uses} for text,uses in groups.items()])
    write(out/'MANUAL_REVIEW.json',read(MANUAL))
    write(out/'SUMMARY.json',summary)
    with (out/'ALL_CASES.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(cases[0]));writer.writeheader()
        writer.writerows(dict(c,issues='; '.join(c['issues'])) for c in cases)
    write(out/'PROVENANCE.json',{'training_campaign':str(root),'training_commit':read(root/'CAMPAIGN.json')['code_commit'],
        'review_code_commit':os.environ['CB_CODE_COMMIT'],'slurm_job':os.environ['SLURM_JOB_ID'],
        'source_summary_sha256':sha(root/'SUMMARY.json'),'dataset_sha256':DATA_SHA,
        'manual_review_sha256':sha(MANUAL),
        'interpretation':'Finite-grammar visibility grounding is distinct from physical causal justification. Shuffled recipient-label errors are sensitivity measurements; donor labels remain the relevant visual-state labels. Every wrong zero-horizon verdict is explicitly listed.',
        'training_or_simulator_actions':0})
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();main(a.root,a.out)
