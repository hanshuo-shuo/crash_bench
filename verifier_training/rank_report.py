"""Quest-only plots and independently recounted paired rank16/rank64 results."""
import json
import os
from pathlib import Path
from .common import read, write, sha, require
from .argmax_review import audit_fit
from .binary_report import table, deliver
from .rank_contract import validate_parent, config_difference


def main(root):
    receipt=read(root/'RECEIPT.json');parent=validate_parent(receipt)
    campaign=Path(receipt['campaign']);old_review=Path(receipt['rank16_review'])
    require(sha(old_review/'conditioning/SUMMARY.json')==receipt['baseline_conditioning_sha256'],
            'Baseline conditioning changed')
    require(read(root/'smoke/RESULT.json')['passed'],'Rank64 smoke failed')
    require(read(root/'conditioning/COMPLETE.json')['passed'],'Rank64 conditioning incomplete')
    out=root/'report';out.mkdir(exist_ok=False)
    runs={'rank16':parent,'rank64':root};scores={};curves={};all_outputs={}
    for label,run in runs.items():
        result=read(run/'full/RESULT.json')
        require(result['passed'] and result['steps']==300,'Full fixed execution incomplete')
        require(result['vision_and_projector_bitwise_unchanged'],'Visual freeze failed')
        audit=audit_fit(run,campaign,out/(label+'_OUTPUTS_AUDIT.csv'))
        curves[label]=[json.loads(line) for line in (run/'full/curve.jsonl').read_text().splitlines()]
        require([v['step'] for v in curves[label]]==list(range(1,301)),'Incomplete training curve')
        conditioning=read((old_review if label=='rank16' else root)/'conditioning/SUMMARY.json')
        scores[label]={'audit':audit,'teacher_losses':read(run/'full/loss_step_0300.json'),
            'training_result':result,'conditioning':conditioning,
            'trainable_parameters':read(run/'full/PROVENANCE.json')['trainable_parameters'],
            'smoke':read(run/'smoke/RESULT.json')}
        all_outputs[label]={condition:read(run/'full'/filename) for condition,filename in
            [('original','generations_final.json'),('zero_command','generations_horizon_0.json')]}
    require(read(root/'full/TRAINING_ORDER.json')==read(parent/'full/TRAINING_ORDER.json'),'Training order changed')
    require(read(root/'preflight/ENCODINGS.json')==read(parent/'preflight/ENCODINGS.json'),'Encoding records changed')
    changes={};paired=[]
    for condition in ('original','zero_command'):
        changes[condition]={key:{'recovered':[],'regressed':[]} for key in ('verdict','rationale','full_output')}
        for a,b in zip(all_outputs['rank16'][condition],all_outputs['rank64'][condition]):
            require(a['sample_id']==b['sample_id'] and a['input_sha256']==b['input_sha256'],
                    'Paired student input changed')
            flags=lambda x:{'verdict':x['verdict_correct'],
                'rationale':x['rationale_assessment']['status']=='supported_complete','full_output':x['exact_match']}
            af,bf=flags(a),flags(b)
            for metric in af:
                if af[metric]!=bf[metric]:
                    changes[condition][metric]['recovered' if bf[metric] else 'regressed'].append(a['sample_id'])
            paired.append({'sample_id':a['sample_id'],'condition':condition,'expected_verdict':a['expected_verdict'],
                'rank16_verdict':a['predicted_verdict'],'rank64_verdict':b['predicted_verdict'],
                'rank16_rationale_correct':af['rationale'],'rank64_rationale_correct':bf['rationale'],
                'rank16_full_correct':af['full_output'],'rank64_full_correct':bf['full_output'],
                'rank16_generated':a['generated'],'rank64_generated':b['generated']})
    table(out/'PAIRED_OUTPUTS.csv',paired)
    differences={stage:config_difference(read(parent/(stage+'_config.json')),read(root/(stage+'_config.json')))
                 for stage in ('smoke','full')}
    summary={'results':scores,'paired_changes':changes,'config_differences':differences,
        'same_training_order':True,'same_inputs_targets':True,'one_new_full_fit':True,
        'scientific_generalization_claim':False,'provenance':receipt,
        'interpretation_limit':'Single-seed recipe comparison. Same alpha/r and seed do not equalize initialization or optimization dynamics. One source; visibility rationales are not physical proofs.'}
    write(out/'SUMMARY.json',summary)
    plot(runs,curves,out)
    metric_rows=[]
    for label in runs:
        s=scores[label];a=s['audit'];o=a['original'];z=a['zero_command']
        metric_rows.append({'rank':label,'feasible_correct':o['by_class']['feasible']['correct'],
            'infeasible_correct':o['by_class']['infeasible']['correct'],'verdict_correct':o['verdict_correct'],
            'rationale_correct':o['rationale_supported_complete'],'full_output_correct':o['exact_full_outputs'],
            'gold_prefix_verdict_correct':s['conditioning']['by_condition']['ground_truth_rationale']['correct'],
            'zero_command_correct':z['verdict_correct'],
            'actual_zero_flips_positive_source':a['actual_feasible_to_infeasible_flips_among_positive_source_cases'],
            'zero_unchanged':a['all_predictions_unchanged_under_zero'],
            'teacher_verdict_loss':s['teacher_losses']['means']['verdict'],
            'teacher_rationale_loss':s['teacher_losses']['means']['rationale'],
            'trainable_parameters':s['trainable_parameters']})
    table(out/'METRICS.csv',metric_rows)
    lines=['# Rank64 vs rank16 SFT engineering comparison','',
        'Same120 training samples from one original source. No held-out or generalization claim.',
        '', '| Metric | Rank16 | Rank64 |','|---|---:|---:|']
    for key in metric_rows[0]:
        if key!='rank':lines.append(f"| {key} | {metric_rows[0][key]} | {metric_rows[1][key]} |")
    lines += ['', 'Only rank16/alpha32 -> rank64/alpha128 changes the training configuration. Both keep alpha/r=2, identical252 target modules, fresh initialization, binary0/1 output, equally weighted normalized component losses, frozen vision/projector, LR1e-4/dropout.05, seed713, batch4 and300 updates. The rank64 smoke uses the same balanced8 and the original separate smoke recipe.',
        '',f"Rank64 smoke passed at{scores['rank64']['smoke']['steps']} updates. Rank64 full exact-fit criterion: {scores['rank64']['training_result']['exact_fit_criterion']}.",
        '',f"Paired original-output changes: {changes['original']}",
        '', 'Rationale correctness checks the audited visibility statements and caveats, not a complete physical proof. Gold-prefix scoring is a read-only conditioning intervention, not a free-generation score. Zero-command accuracy includes already-infeasible predictions; actual transitions are counted separately.',
        '', 'This single-seed fixed-budget comparison tests a training recipe. Different adapter shapes change random initialization and dropout RNG consumption. Rank, scaling and optimizer dynamics are not causally separated. More training accuracy cannot resolve hidden geometry or establish cross-source generalization.',
        '', 'See SUMMARY.json for class-specific token losses, hidden-target strata, gradient probes, resource use and prefix results. PAIRED_OUTPUTS.csv contains every original and zero-command output from both ranks. Both OUTPUTS_AUDIT.csv files were independently parsed and grounded against the frozen dataset.',
        '', '![Loss curves](training_curves.png)','![Generated training accuracy](training_accuracy.png)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(out/'PROVENANCE.json',{'source_commit':receipt['source_commit'],'slurm_job':os.environ['SLURM_JOB_ID'],
          'run_root':str(root),'baseline_root':str(parent),'new_training_in_report':False})
    validate_parent(receipt)
    write(root/'COMPLETE.json',{'passed':True,'one_new_full_fit':True,'report_rows':len(paired),
        'baseline_unchanged':True,'full_exact_fit_criterion':scores['rank64']['training_result']['exact_fit_criterion'],
        'scientific_generalization_claim':False,'new_simulation':False})
    deliver(root)


def plot(runs,curves,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'rank16':'#277da1','rank64':'#d65f26'}
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for ax,key in zip(axes.flat,('total','rationale','verdict','rationale_text','format','eos')):
        for label,run in runs.items():
            curve=curves[label]
            ax.plot([x['step'] for x in curve],[max(x[key],1e-8) for x in curve],
                    color=colors[label],alpha=.18,lw=.7)
            checks=[(0,read(run/'full/loss_before.json')['means'][key])]
            checks.extend((int(p.stem.rsplit('_',1)[1]),read(p)['means'][key])
                          for p in sorted((run/'full').glob('loss_step_*.json')))
            ax.plot([x[0] for x in checks],[max(x[1],1e-8) for x in checks],
                    'o-',color=colors[label],label=label,ms=3,lw=1.4)
        ax.set(xlabel='optimizer step',ylabel='cross entropy (log scale)',title=key.replace('_',' '),yscale='log')
        ax.grid(alpha=.2)
    axes[0,0].legend()
    fig.suptitle('Same120 training cases: rank16 vs rank64 (faint: train batches; solid: full-pool eval)')
    fig.savefig(out/'training_curves.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for ax,metric in zip(axes,('verdict','rationale','full_output')):
        for label,run in runs.items():
            values=[]
            for name in ('generations_before.json','generations_step_0150.json','generations_final.json'):
                g=read(run/'full'/name)
                values.append(sum((x['verdict_correct'] if metric=='verdict' else x['exact_match']
                    if metric=='full_output' else x['rationale_assessment']['status']=='supported_complete') for x in g)/120)
            ax.plot([0,150,300],values,'o-',color=colors[label],label=label)
        ax.set(xlabel='optimizer step',ylabel='fraction of120',ylim=(-.03,1.03),title=metric.replace('_',' '))
        ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Free-generated training outputs; engineering fit only, one source')
    fig.savefig(out/'training_accuracy.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
