"""Quest-side aggregation/plots for a single controlled engineering correction."""
import csv
import json
import math
from pathlib import Path
import shutil
import statistics
from .common import read,write,require,sha


def table(path,rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def summarize_diagnostics(root):
    receipt=read(root/'RECEIPT.json');parent=Path(receipt['diagnostic']);prefix=Path(receipt['prefix_diagnostic'])
    old=read(parent/'diagnostic/PROBABILITIES.json');new=read(prefix/'prefix/ROWS.json')
    require(len(old)==len(new)==120,'Incomplete diagnosis')
    output=root/'diagnostic_summary';output.mkdir(exist_ok=False)
    by_mode={};flat=[]
    for mode in ('teacher_full_sequence','whole_prefix_prefill','canonical_cached','generated_cached','canonical_cached_penalized','generated_cached_penalized'):
        records=[]
        for a,b in zip(old,new):
            require(a['sample_id']==b['sample_id'],'Diagnostic alignment changed')
            if mode=='teacher_full_sequence':
                candidates=a['conditions']['ground_truth_rationale']['candidates']
                scores={k:v['sequence_logp'] for k,v in candidates.items()}
                decisive=candidates[a['truth']]['group_nll_sums']['decisive_class']
                eos=candidates[a['truth']]['group_nll_sums']['eos']
                continuation=candidates[a['truth']]['group_nll_sums']['verdict_continuation']
            elif mode=='whole_prefix_prefill':
                scores=b['whole_prefix_prefill_decisive_logp'];decisive=-scores[a['truth']];eos=None;continuation=None
            else:
                candidates=b['ground_truth_prefix_cached' if mode.startswith('canonical_') else 'own_prefix_cached']['candidates']
                penalized=mode.endswith('_penalized')
                scores={k:v['processed_sequence_logp' if penalized else 'sequence_logp'] for k,v in candidates.items()}
                truth=candidates[a['truth']]
                losses=truth['processed_token_nll_including_eos' if penalized else 'token_nll_including_eos']
                decisive=losses[0];eos=losses[-1];continuation=sum(losses[1:-1])
            m=max(scores.values());denom=sum(math.exp(v-m) for v in scores.values())
            probs={k:math.exp(v-m)/denom for k,v in scores.items()}
            rec={'sample_id':a['sample_id'],'mode':mode,'truth':a['truth'],'predicted':max(scores,key=scores.get),
                'p_feasible_two_candidate':probs['feasible'],'p_infeasible_two_candidate':probs['infeasible'],
                'raw_feasible_logp':scores['feasible'],'raw_infeasible_logp':scores['infeasible'],
                'true_decisive_nll':decisive,'true_continuation_nll_sum':continuation,'true_eos_nll':eos,
                'prefix_ids_identical':b['canonical_and_generated_prefix_ids_identical']}
            records.append(rec);flat.append(rec)
        by_mode[mode]={'correct':sum(x['predicted']==x['truth'] for x in records),
            'by_class':{c:{'n':sum(x['truth']==c for x in records),
                'correct':sum(x['truth']==c and x['predicted']==c for x in records),
                'mean_decisive_nll':statistics.mean(x['true_decisive_nll'] for x in records if x['truth']==c),
                'mean_continuation_nll_sum':statistics.mean(x['true_continuation_nll_sum'] for x in records if x['truth']==c) if mode!='whole_prefix_prefill' else None,
                'mean_eos_nll':statistics.mean(x['true_eos_nll'] for x in records if x['truth']==c) if mode!='whole_prefix_prefill' else None,
                'mean_true_two_candidate_probability':statistics.mean(x['p_'+c+'_two_candidate'] for x in records if x['truth']==c)} for c in ('feasible','infeasible')}}
    gradients=read(parent/'diagnostic/GRADIENTS.json')
    weights={k:statistics.mean(x['component_token_weights'][k] for b in gradients['batches'] for x in b['sample_weights']) for k in gradients['groups']}
    first_probe=read(Path(receipt['campaign'])/'B/GRADIENT_PROBE.json')
    first_count=sum(v['tokens'] for v in first_probe.values())
    initial={k:dict(v,old_objective_weight=v['tokens']/first_count,
        weighted_per_example_gradient_norm=v['lora_gradient_norm']*v['tokens']/first_count,
        weighted_first_microbatch_gradient_norm=v['lora_gradient_norm']*v['tokens']/first_count/4)
        for k,v in first_probe.items()}
    tokenization=read(parent/'diagnostic/TOKENIZATION.json')
    differences=[]
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(Path(receipt['assets'])/'model',local_files_only=True)
    for x in new:
        a=x['canonical_prefix_ids'];b=x['generated_prefix_ids']
        if a!=b:
            index=next((i for i,(left,right) in enumerate(zip(a,b)) if left!=right),min(len(a),len(b)))
            differences.append({'sample_id':x['sample_id'],'first_difference':index,
                'canonical_ids':a,'generated_ids':b,'canonical_pieces':[tokenizer.decode([v]) for v in a],
                'generated_pieces':[tokenizer.decode([v]) for v in b]})
    write(output/'PREFIX_TOKEN_DIFFERENCES.json',differences)
    summary={'modes':by_mode,'tokenization':tokenization,'same_rationale_text':sum(x['same_rationale_text'] for x in new),
        'same_prefix_token_ids':sum(x['canonical_and_generated_prefix_ids_identical'] for x in new),
        'native_reproduces_saved_outputs':sum(x['matches_saved_text'] for x in new),
        'cache_replay_max_logp_error':max(x['native_replay_logp_max_error'] for x in new),
        'native_raw_replay_max_logp_error':max(x['native_raw_replay_logp_max_error'] for x in new),
        'native_generation_config':read(prefix/'prefix/PROVENANCE.json')['native_generation_config'],
        'class_token_previously_seen_counts':{c:sum(x['own_prefix_cached']['candidates'][c]['first_class_token_seen_in_prefix'] for x in new) for c in ('feasible','infeasible')},
        'old_objective_mean_token_weights':weights,'weighted_gradient':gradients['full_pool_mean_gradient'],
        'gradient_median_batch_norm':{k:statistics.median(b['norms'][k] for b in gradients['batches']) for k in gradients['groups']},
        'gradient_sum_identity_relative_error':gradients['batches'][0]['first_example_sum_identity_relative_error'],
        'historical_first_training_example_gradient':initial,
        'historical_gradient_note':'Rescaled actual saved first-example probes. Initial verdict group includes EOS. Norms are not additive; initial total-direction shares were not saved.',
        'diagnostic_complete_sha256':receipt['diagnostic_complete_sha256'],'prefix_complete_sha256':receipt['prefix_complete_sha256']}
    write(output/'SUMMARY.json',summary);table(output/'ALL120_PROBABILITIES.csv',flat)


def deliver(root):
    dest=Path.home()/'crash_bench/results/verifier_training'/root.name;dest.mkdir(parents=True,exist_ok=True)
    for source in root.rglob('*'):
        rel=source.relative_to(root)
        if source.is_file() and not any(x in rel.parts for x in ('source','adapter')) and source.name!='source.tar':
            target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    files=[x for x in sorted(dest.rglob('*')) if x.is_file() and x.name!='DELIVERY.sha256']
    (dest/'DELIVERY.sha256').write_text(''.join(sha(x)+'  '+str(x.relative_to(dest))+'\n' for x in files))


def main(root):
    receipt=read(root/'RECEIPT.json');campaign=Path(receipt['campaign']);out=root/'report';out.mkdir(exist_ok=False)
    full=read(root/'full/RESULT.json');smoke=read(root/'smoke/RESULT.json')
    require(full['passed'] and smoke['passed'],'Missing completed experiments')
    diagnostic=read(root/'diagnostic_summary/SUMMARY.json')
    records=[]
    for condition,name in [('original','generations_final.json'),('horizon_0','generations_horizon_0.json')]:
        for x in read(root/'full'/name):
            records.append({k:x[k] for k in ('sample_id','condition','generated','predicted_verdict','expected_verdict','verdict_correct','format_valid','exact_match')})
    table(out/'GENERATED_OUTPUTS.csv',records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    curve=[json.loads(x) for x in (root/'full/curve.jsonl').read_text().splitlines()]
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for ax,key in zip(axes.flatten(),('total','rationale','verdict','rationale_text','format','eos')):
        ax.plot([x['step'] for x in curve],[max(x[key],1e-8) for x in curve],lw=1,alpha=.65,label='training batches')
        checks=[(0,read(root/'full/loss_before.json')['means'][key])]
        checks.extend((int(p.stem.rsplit('_',1)[1]),read(p)['means'][key]) for p in sorted((root/'full').glob('loss_step_*.json')))
        ax.plot([x[0] for x in checks],[max(x[1],1e-8) for x in checks],'o-',ms=3,label='all120, eval mode')
        ax.set_yscale('log');ax.set_title(key.replace('_',' '));ax.set_xlabel('optimizer step');ax.set_ylabel('cross entropy');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Binary B: fixed120 training-pool engineering fit (one source; no generalization)')
    fig.savefig(out/'training_curves.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.8),constrained_layout=True)
    for label,path,parse_digits in [('Original A',campaign/'A',False),('Original B',campaign/'B',False),('Binary B',root/'full',True)]:
        xs=[];ys=[]
        for step,name in [(0,'generations_before.json'),(150,'generations_step_0150.json'),(300,'generations_final.json')]:
            g=read(path/name);xs.append(step);ys.append(sum(v['verdict_correct'] for v in g)/120)
        ax.plot(xs,ys,'o-',label=label)
    ax.set(xlabel='optimizer step',ylabel='training verdict accuracy',ylim=(-.03,1.03),title='Same120 examples; engineering fit only')
    ax.grid(alpha=.25);ax.legend();fig.savefig(out/'training_accuracy.png',dpi=170);plt.close(fig)
    final_loss=read(root/'full/loss_step_0300.json')
    summary={'diagnostic':diagnostic,'smoke':smoke,'fixed_full_fit':full,'final_loss_by_class':final_loss['by_class'],
        'parent_A':read(campaign/'A/RESULT.json'),'parent_B':read(campaign/'B/RESULT.json'),
        'provenance':receipt,'scope':'Engineering only: same120 training cases from one source. No new scientific result.'}
    write(out/'SUMMARY.json',summary)
    lines=['# Controlled binary-verdict engineering experiment','',
        'All results are on the same120 training examples from one original source scene. They do not measure generalization.',
        '',f"Balanced smoke: {smoke['final_score']['exact_full_output']}/8 exact outputs after {smoke['steps']} updates.",
        f"Fixed120 fit: {full['final_score']['verdict_correct']}/120 verdicts; {full['final_score']['exact_full_output']}/120 exact full outputs.",
        f"Class counts: {full['final_score']['by_class']}",
        f"Grounded observability templates: {full['final_score']['rationale_supported_complete']}/120. This is visibility/caveat correctness, not a verified physical justification of the verdict.",
        f"Zero-command: {full['zero_command_score']['verdict_correct']}/120 infeasible; flips among original feasible cases: {full['zero_flips_among_original_feasible']}/72.",
        '', 'The only supervised changes are 0=infeasible/1=feasible and two equally weighted, separately normalized loss components. The narrative component includes format and EOS; the verdict component is exactly the single digit. Every assistant token stays supervised.',
        'The backbone, vision/projector freeze, input observations, full-fit LR1e-4/dropout.05/rank16/alpha32/effective batch4/seed713/300 updates and complete-epoch order are unchanged. The full fit starts from the same fresh base, not from the smoke adapter.',
        '', 'See SUMMARY.json for exact class-specific decisive-token, narrative, format and EOS losses; GENERATED_OUTPUTS.csv for every full/zero output; diagnostic_summary/ALL120_PROBABILITIES.csv for all four scoring modes. Raw diagnostics and gradient attribution are preserved in the two preceding run roots.',
        '', '![Loss curves](training_curves.png)','![Training verdict accuracy](training_accuracy.png)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(root/'COMPLETE.json',{'passed':True,'scientific_generalization_claim':False,'one_fixed_full_fit':True,
        'full_exact_fit_criterion':full['exact_fit_criterion'],'report_rows':len(records),'new_simulation':False})
    for path in (Path(receipt['diagnostic']),Path(receipt['prefix_diagnostic']),root):deliver(path)

if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
