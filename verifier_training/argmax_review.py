"""Supplemental decision-token audit: distinguish token argmax from sequence MAP."""
import csv
import json
from pathlib import Path
import re
import shutil
from verifier_training.common import read, write, sha, require
from verifier_training.engineering_contract import score_rationale, DATA_SHA


def decision_rows(native, teacher, candidate_tokens):
    require(len(native)==len(teacher),'Different diagnostic row counts')
    names={tokens[0]:label for label,tokens in candidate_tokens.items()}
    result=[]
    for x,t in zip(native,teacher):
        require(x['sample_id']==t['sample_id'],'Diagnostic alignment changed')
        raw_id=x['own_prefix_cached']['first_greedy_token_id']
        adjusted_id=x['native_greedy_token_id']
        require(adjusted_id==x['own_prefix_cached']['processed_first_greedy_token_id'],'Processed replay differs')
        scores=x['own_prefix_cached']['candidates']
        sequence=max(scores,key=lambda c:scores[c]['sequence_logp'])
        old_id=t['conditions']['ground_truth_rationale']['candidates'][x['truth']]['greedy_first_token_id']
        result.append({'sample_id':x['sample_id'],'truth':x['truth'],
            'raw_argmax_token_id':raw_id,'raw_argmax_label':names.get(raw_id,'invalid'),
            'adjusted_argmax_token_id':adjusted_id,'adjusted_argmax_label':names.get(adjusted_id,'invalid'),
            'raw_complete_sequence_choice':sequence,'teacher_first_argmax_label':names.get(old_id,'invalid'),
            'exact_raw_class_tie':x['native_raw_decisive_logp']['feasible']==x['native_raw_decisive_logp']['infeasible'],
            'same_prefix_token_ids':x['canonical_and_generated_prefix_ids_identical'],
            'raw_true_class_nll':-x['native_raw_decisive_logp'][x['truth']],
            'adjusted_true_class_nll':-x['native_decisive_logp'][x['truth']],
            'native_replay_error':x['native_replay_logp_max_error'],
            'matches_saved_generation':x['matches_saved_text']})
    return result


def decision_summary(rows):
    rules=('raw_argmax_label','adjusted_argmax_label','raw_complete_sequence_choice','teacher_first_argmax_label')
    by_rule={rule:{'correct':sum(r[rule]==r['truth'] for r in rows),
        'by_class':{c:{'n':sum(r['truth']==c for r in rows),
                      'correct':sum(r['truth']==c and r[rule]==c for r in rows)} for c in ('feasible','infeasible')}} for rule in rules}
    changes={c:{'correct_to_wrong':sum(r['truth']==c and r['raw_argmax_label']==c and r['adjusted_argmax_label']!=c for r in rows),
                'wrong_to_correct':sum(r['truth']==c and r['raw_argmax_label']!=c and r['adjusted_argmax_label']==c for r in rows)} for c in ('feasible','infeasible')}
    return {'decision_rules':by_rule,'penalty_transitions':changes,
        'class_ties':[r['sample_id'] for r in rows if r['exact_raw_class_tie']],
        'forward_path_argmax_disagreements':[r['sample_id'] for r in rows if r['teacher_first_argmax_label']!=r['raw_argmax_label']],
        'tie_rule':'Native torch argmax uses stored vocabulary token IDs; do not resolve class ties using Python dictionary order.',
        'scope':'Raw argmax is conditional on the observed fixed rationale prefix. It is not a separately measured penalty-free full-generation run.'}


def audit_fit(run,campaign,case_csv=None):
    require(sha(campaign/'dataset/dataset.jsonl')==DATA_SHA,'Dataset changed')
    rows=[json.loads(x) for x in (campaign/'dataset/dataset.jsonl').read_text().splitlines()]
    rows_by_id={x['sample_id']:x for x in rows}
    truth={x['sample_id']:x['target']['verdict'] for x in rows}
    hidden={x['sample_id']:all(v['entities']['target']['pixels_224']==0 for v in x['metadata']['visibility']['views']) for x in rows}
    require(sum(hidden.values())==32,'Original observability strata changed')
    original=read(run/'full/generations_final.json');zero=read(run/'full/generations_horizon_0.json')
    require(len(original)==len(zero)==120,'Incomplete full fit outputs')
    require({x['sample_id'] for x in original}==set(truth),'Sample set differs')
    result={};case_rows=[]
    for condition,records in [('original',original),('zero_command',zero)]:
        expected={k:('infeasible' if condition=='zero_command' else v) for k,v in truth.items()}
        for x in records:
            m=re.fullmatch(r'Rationale: .+\nVerdict: ([01])',x['generated'],flags=re.S)
            pred={'0':'infeasible','1':'feasible'}[m.group(1)] if m else None
            row=rows_by_id[x['sample_id']];wanted=expected[x['sample_id']]
            text='Rationale: '+row['target']['rationale']+'\nVerdict: '+{'infeasible':'0','feasible':'1'}[wanted]
            grade=score_rationale(x['generated'],row,'original' if condition=='original' else 'horizon_0')
            require(x['predicted_verdict']==pred and x['format_valid']==bool(m),'Stored parse differs from independent parse')
            require(x['verdict_correct']==(pred==wanted) and x['exact_match']==(x['generated']==text),'Stored correctness differs')
            require(x['rationale_assessment']==grade,'Stored rationale grade differs')
            decision=x['native_decision']
            case_rows.append({'sample_id':x['sample_id'],'condition':condition,'source_label':truth[x['sample_id']],
                'expected_verdict':wanted,'predicted_verdict':pred,'verdict_correct':pred==wanted,
                'rationale_status':grade['status'],'exact_full_output':x['generated']==text,'format_valid':bool(m),
                'milk_hidden_both':hidden[x['sample_id']],
                'native_raw_true_class_nll':decision['true_raw_class_nll'] if decision else None,
                'native_processed_true_class_nll':decision['true_class_nll'] if decision else None,
                'contradictions':' | '.join(grade['contradictions']),'unsupported':' | '.join(grade['unsupported']),
                'unparsed':' | '.join(grade['unparsed']),'generated':x['generated']})
        result[condition]={'n':len(records),'verdict_correct':sum(x['predicted_verdict']==expected[x['sample_id']] for x in records),
            'by_class':{c:{'n':sum(v==c for v in expected.values()),
                'correct':sum(expected[x['sample_id']]==c and x['predicted_verdict']==c for x in records)} for c in ('feasible','infeasible')},
            'rationale_supported_complete':sum(x['rationale_assessment']['status']=='supported_complete' for x in records),
            'invalid_format':sum(not x['format_valid'] for x in records),
            'exact_full_outputs':sum(x['exact_match'] for x in records),
            'unsupported_or_contradictory_rationale_ids':[x['sample_id'] for x in records if x['rationale_assessment']['status']!='supported_complete']}
        result[condition]['observability_strata']={name:{'n':sum(hidden[i]==flag for i in truth),
            'verdict_correct':sum(hidden[x['sample_id']]==flag and x['predicted_verdict']==expected[x['sample_id']] for x in records),
            'rationale_supported':sum(hidden[x['sample_id']]==flag and x['rationale_assessment']['status']=='supported_complete' for x in records)}
            for name,flag in [('milk_hidden_both',True),('milk_present_any',False)]}
    original_by_id={x['sample_id']:x for x in original};zero_by_id={x['sample_id']:x for x in zero}
    positive=[i for i,v in truth.items() if v=='feasible']
    result['positive_source_cases']=len(positive)
    result['original_correct_feasible_predictions']=sum(original_by_id[i]['predicted_verdict']=='feasible' for i in positive)
    result['zero_correct_among_positive_source_cases']=sum(zero_by_id[i]['predicted_verdict']=='infeasible' for i in positive)
    result['actual_feasible_to_infeasible_flips_among_positive_source_cases']=sum(original_by_id[i]['predicted_verdict']=='feasible' and zero_by_id[i]['predicted_verdict']=='infeasible' for i in positive)
    result['all_predictions_unchanged_under_zero']=sum(original_by_id[i]['predicted_verdict']==zero_by_id[i]['predicted_verdict'] for i in truth)
    if case_csv is not None:
        with Path(case_csv).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(case_rows[0]));writer.writeheader();writer.writerows(case_rows)
    return result


def main(run,out):
    out.mkdir(exist_ok=False)
    receipt=read(run/'RECEIPT.json');parent=Path(receipt['diagnostic']);prefix=Path(receipt['prefix_diagnostic']);campaign=Path(receipt['campaign'])
    require(read(parent/'diagnostic/COMPLETE.json')['passed'] and read(prefix/'prefix/COMPLETE.json')['passed'],'Diagnostics incomplete')
    paths={'native_rows':prefix/'prefix/ROWS.json','teacher_rows':parent/'diagnostic/PROBABILITIES.json',
           'native_provenance':prefix/'prefix/PROVENANCE.json','diagnostic_summary':run/'diagnostic_summary/SUMMARY.json'}
    d=decision_rows(read(paths['native_rows']),read(paths['teacher_rows']),read(paths['native_provenance'])['candidate_tokens'])
    require(len(d)==120 and all(x['same_prefix_token_ids'] and x['native_replay_error']==0 and x['matches_saved_generation'] for x in d),'Native diagnostic integrity failed')
    summary=decision_summary(d);write(out/'DECISION_TOKEN_SUMMARY.json',summary)
    with (out/'DECISION_TOKEN_AUDIT.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(d[0]));writer.writeheader();writer.writerows(d)
    full=audit_fit(run,campaign,out/'ALL_OUTPUTS_AUDIT.csv') if (run/'full/RESULT.json').exists() else None
    write(out/'FIT_READBACK.json',full)
    paths['smoke_result']=run/'smoke/RESULT.json'
    if full:paths['full_result']=run/'full/RESULT.json'
    write(out/'PROVENANCE.json',{'parent_run':str(run),'files':{k:{'path':str(p),'sha256':sha(p)} for k,p in paths.items()},
        'purpose':'Correctly separate native first-token argmax, complete-label sequence scoring and processed generation; independently recount fit outputs',
        'new_training':False,'new_simulation':False})
    paragraphs=['# Decision-token and full-output readback','',
        'This supplements the immutable primary run. All scores concern the same120 training examples from one original scene.',
        '', '| Decision rule, same observed rationale prefix | Feasible | Infeasible | Total |', '|---|---:|---:|---:|']
    for key,title in [('teacher_first_argmax_label','Raw full-forward token argmax'),('raw_argmax_label','Raw native cached token argmax'),('raw_complete_sequence_choice','Raw complete-verdict sequence score'),('adjusted_argmax_label','Native penalty-adjusted token argmax')]:
        s=summary['decision_rules'][key];paragraphs.append(f"| {title} | {s['by_class']['feasible']['correct']}/72 | {s['by_class']['infeasible']['correct']}/48 | {s['correct']}/120 |")
    paragraphs += ['',f"There are {len(summary['class_ties'])} exact raw class-logit ties. Native argmax uses vocabulary order (304 for ` in`,42203 for ` feasible`). Complete-verdict scoring includes the negative word's continuation tokens; it is a different decision rule. Earlier102/120 sequence scores must not be described as native raw greedy accuracy.",
        f"Penalty transitions by true class: {summary['penalty_transitions']}",
        f"Full-forward/cached raw argmax disagreements: {summary['forward_path_argmax_disagreements']}. Both have the same aggregate accuracy; no rationale-prefix token drift occurs on any of the120 cases.",
        '', 'Raw conditional argmax is not a separate penalty-free full-generation experiment. Rationale-prefix token IDs are held fixed for this diagnostic. Probabilities are language likelihoods, not calibrated physical feasibility probabilities.']
    if full:
        f=full['original'];z=full['zero_command']
        paragraphs += ['',f"Corrected fixed fit: {f['verdict_correct']}/120 verdicts, {f['exact_full_outputs']}/120 exact full outputs, {f['rationale_supported_complete']}/120 complete observability templates.",
            f"Zero commands: {z['verdict_correct']}/120 infeasible; actual feasible-to-infeasible flips among the72 positive-source cases: {full['actual_feasible_to_infeasible_flips_among_positive_source_cases']}.",
            'Rationale correctness here means factual visibility/caveat agreement with the audited target. These are not case-specific physical proofs. The frozen model architecture and the one-source observability limitations remain unchanged.']
    conditioning=out.parent/'conditioning/SUMMARY.json'
    if conditioning.exists():
        c=read(conditioning)
        paragraphs += ['',f"Matched full-forward prefix scores: {c['by_condition']['ground_truth_rationale']['correct']}/120 with the gold rationale; {c['by_condition']['generated_rationale']['correct']}/120 with the actual generated rationale token IDs.",
            f"Gold-correct / generated-prefix-wrong cases: {c['gold_correct_own_wrong_ids']}.",
            f"Wrong under both prefixes: {c['both_prefixes_wrong_ids']}.",
            'This is a read-only conditioning intervention with fixed student observations and fixed weights, evaluated on the same forward path. The final free-generation metric remains the separately recorded native generation result.']
    (out/'REPORT.md').write_text('\n'.join(paragraphs)+'\n')
    write(out/'COMPLETE.json',{'passed':True,'diagnostic_rows':120,'fit_reviewed':full is not None,'new_training':False})

if __name__=='__main__':
    import sys
    main(*map(Path,sys.argv[1:]))
