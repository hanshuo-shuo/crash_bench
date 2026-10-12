"""Audit and plot a single fixed lower-LR run; never performs model updates."""
import json
import os
from pathlib import Path
from .common import read,write,require,sha
from .engineering_contract import DATA_SHA
from .rank_contract import validate_parent,lr_difference,LOW_LR_PROFILE
from .rank_smoke_report import audit
from .binary_report import table,deliver


def summarize(run,stage,rows,label,out):
    path=run/stage;result=read(path/'RESULT.json')
    require(result['vision_and_projector_bitwise_unchanged'],'Visual freeze failed')
    original=audit(rows,read(path/'generations_final.json'))
    zero=audit(rows,read(path/'generations_horizon_0.json'),True)
    records=[dict(arm=label,stage=stage,**x) for x in original+zero]
    table(out/(stage+'_'+label+'_OUTPUTS.csv'),records)
    checkpoints=[]
    for p in sorted(path.glob('generations_step_*.json')):
        values=audit(rows,read(p))
        checkpoints.append({'arm':label,'stage':stage,'step':int(p.stem.rsplit('_',1)[1]),
            **{k:sum(x[k] for x in values) for k in ('verdict_correct','rationale_correct','full_output_correct','format_valid')}})
    curve=[json.loads(x) for x in (path/'curve.jsonl').read_text().splitlines()]
    require([x['step'] for x in curve]==list(range(1,result['steps']+1)),'Incomplete curve')
    summary={'passed':result['passed'],'exact_fit_criterion':result['exact_fit_criterion'],'steps':result['steps'],
        'n':len(rows),'final':{k:sum(x[k] for x in original) for k in ('verdict_correct','rationale_correct','full_output_correct','format_valid')},
        'by_class':{c:{'n':sum(x['source_label']==c for x in original),
                       'correct':sum(x['source_label']==c and x['verdict_correct'] for x in original)} for c in ('feasible','infeasible')},
        'zero_command_correct':sum(x['verdict_correct'] for x in zero),
        'zero_actual_positive_source_flips':sum(a['source_label']=='feasible' and a['predicted']=='feasible' and b['predicted']=='infeasible' for a,b in zip(original,zero)),
        'zero_unchanged':sum(a['predicted']==b['predicted'] for a,b in zip(original,zero)),
        'teacher_final':read(path/f"loss_step_{result['steps']:04d}.json"),
        'result':result,'first140_clipped_steps':sum(x['gradient_norm']>1 for x in curve[:140]),
        'max_preclip_gradient_norm':max(x['gradient_norm'] for x in curve),
        'errors':[x for x in original if not x['full_output_correct']],
        'hidden_milk':{'n':sum(x['milk_hidden_both'] for x in original),
                       'verdict_correct':sum(x['milk_hidden_both'] and x['verdict_correct'] for x in original),
                       'rationale_correct':sum(x['milk_hidden_both'] and x['rationale_correct'] for x in original)}}
    return summary,records,checkpoints,curve


def main(root,stage):
    receipt=read(root/'RECEIPT.json');base16=validate_parent(receipt)
    require(receipt['comparison_profile']==LOW_LR_PROFILE,'Not the fixed lower-LR run')
    base64=Path(receipt['rank64_reference'])
    passed=read(root/'smoke/RESULT.json')['passed']
    if stage=='smoke-report':
        require(not passed and read(root/'smoke/RESULT.json')['steps']==240,'Not a completed bounded smoke failure')
        require(not (root/'full_SUBMISSION.json').exists(),'Full unexpectedly launched')
    else:
        require(stage=='report' and passed,'Smoke gate did not pass')
        require(read(root/'conditioning/COMPLETE.json')['passed'],'Conditioning diagnosis incomplete')
    dataset=Path(receipt['campaign'])/'dataset';require(sha(dataset/'dataset.jsonl')==DATA_SHA,'Dataset changed')
    rows=[json.loads(x) for x in (dataset/'dataset.jsonl').read_text().splitlines()]
    ids=read(root/'preflight/COMPLETE.json')['smoke_ids'];eight=[r for r in rows if r['sample_id'] in ids]
    out=root/'report';out.mkdir(exist_ok=False)
    summary={'config_differences':{s:lr_difference(read(base64/(s+'_config.json')),read(root/(s+'_config.json'))) for s in ('smoke','full')},
        'smoke_passed':passed,'full120_launched':stage=='report','provenance':receipt,
        'scientific_generalization_claim':False,'results':{}}
    all_outputs=[];all_checkpoints=[]
    stages=['smoke','full'] if stage=='report' else ['smoke']
    for name in stages:
        use=eight if name=='smoke' else rows
        runs={'rank64_original_lr':base64,'rank64_quarter_lr':root} if name=='smoke' else {'rank16_original_lr':base16,'rank64_quarter_lr':root}
        summaries={};curves={};cps=[]
        for label,run in runs.items():
            s,outputs,checkpoints,curve=summarize(run,name,use,label,out)
            summaries[label]=s;curves[label]=curve;all_outputs+=outputs;all_checkpoints+=checkpoints;cps+=checkpoints
        summary['results'][name]=summaries
        reference=next(iter(runs.values()))
        for filename in ('generations_final.json','generations_horizon_0.json'):
            a=read(reference/name/filename);b=read(root/name/filename)
            require([(x['sample_id'],x['input_sha256']) for x in a]==[(x['sample_id'],x['input_sha256']) for x in b],'Student input changed')
        if name=='full':
            require(read(root/'full/TRAINING_ORDER.json')==read(base16/'full/TRAINING_ORDER.json'),'Full order changed')
            require(read(root/'full/RESULT.json')['steps']==300,'Full step budget changed')
            old_review=Path(receipt['rank16_review'])
            require(sha(old_review/'conditioning/SUMMARY.json')==receipt['baseline_conditioning_sha256'],'Baseline conditioning changed')
            summary['conditioning']={'rank16_original_lr':read(old_review/'conditioning/SUMMARY.json'),
                                     'rank64_quarter_lr':read(root/'conditioning/SUMMARY.json')}
        make_plots(runs,name,curves,cps,out,len(use))
    summary['smoke_initial_loss_equal']=read(base64/'smoke/loss_before.json')==read(root/'smoke/loss_before.json')
    summary['smoke_initial_gradient_probe_equal']=read(base64/'smoke/GRADIENT_PROBE.json')==read(root/'smoke/GRADIENT_PROBE.json')
    table(out/'ALL_GENERATED_OUTPUTS.csv',all_outputs);table(out/'GENERATION_CHECKPOINTS.csv',all_checkpoints)
    write(out/'SUMMARY.json',summary)
    lines=['# Fixed rank64 lower-learning-rate engineering comparison','',
        'Only learning rate differs from the prior rank64 configuration: smoke2e-4 ->5e-5; conditional full1e-4 ->2.5e-5. Rank64/alpha128, observations, losses, frozen vision/projector, seed713 and all stage-specific settings stay fixed.',
        '',f'Smoke passed: {passed}. Full120 launched: {stage=="report"}.',
        '', 'The smoke is the controlled same-rank LR comparison. The prior rank64 smoke failed, so there is no high-LR rank64 full120 result. Any full comparison against rank16 changes both rank and LR and cannot isolate either effect. No additional high-LR full job is run.',
        '', 'All results are engineering training-pool measurements from one source scene. Rationale correctness means audited visibility/caveat agreement, not a physical proof. Zero-budget transitions are counted separately from already-infeasible predictions.']
    for name in stages:
        lines+=['',f'## {name}','', '| Arm | Steps | Feasible | Infeasible | Rationale | Exact full output | Verdict CE | Zero correct | Actual zero flips |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for label,s in summary['results'][name].items():
            c=s['by_class'];f=s['final']
            lines.append(f"| {label} | {s['steps']} | {c['feasible']['correct']}/{c['feasible']['n']} | {c['infeasible']['correct']}/{c['infeasible']['n']} | {f['rationale_correct']}/{s['n']} | {f['full_output_correct']}/{s['n']} | {s['teacher_final']['means']['verdict']:.8g} | {s['zero_command_correct']}/{s['n']} | {s['zero_actual_positive_source_flips']} |")
        lines += ['',f'![Losses]({name}_training_curves.png)',f'![Generated outputs]({name}_training_accuracy.png)']
    lines+=['','See SUMMARY.json for class-specific decision, rationale, format and EOS losses, real-model gradients, generation errors, visibility strata and (if full fit ran) gold/own-prefix scores. ALL_GENERATED_OUTPUTS.csv includes independently parsed and grounded original/zero outputs for both comparisons. The failed prior run and all baseline checkpoints remain unchanged.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    validate_parent(receipt)
    write(root/'COMPLETE.json',{'passed':True,'report_complete':True,'smoke_passed':passed,'full120_launched':stage=='report',
        'report_rows':len(all_outputs),'scientific_generalization_claim':False,'new_simulation':False,
        'report_source_commit':os.environ['CB_CODE_COMMIT'],'report_job':os.environ['SLURM_JOB_ID']})
    deliver(root)


def make_plots(runs,stage,curves,checkpoints,out,n):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(zip(runs,('#277da1','#d65f26')))
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for ax,key in zip(axes.flat,('total','rationale','verdict','rationale_text','format','eos')):
        for label,run in runs.items():
            curve=curves[label];ax.plot([x['step'] for x in curve],[max(x[key],1e-8) for x in curve],color=colors[label],alpha=.16,lw=.7)
            checks=[(0,read(run/stage/'loss_before.json')['means'][key])]
            checks.extend((int(p.stem.rsplit('_',1)[1]),read(p)['means'][key]) for p in sorted((run/stage).glob('loss_step_*.json')))
            ax.plot([x[0] for x in checks],[max(x[1],1e-8) for x in checks],'o-',color=colors[label],label=label,ms=3,lw=1.3)
        ax.set(xlabel='optimizer step',ylabel='cross entropy',title=key.replace('_',' '),yscale='log');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle(f'{stage}: same{n} engineering training cases (faint: minibatch; solid: full-pool eval)')
    fig.savefig(out/(stage+'_training_curves.png'),dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for ax,key in zip(axes,('verdict_correct','rationale_correct','full_output_correct')):
        for label in runs:
            c=[x for x in checkpoints if x['arm']==label]
            ax.plot([x['step'] for x in c],[x[key]/n for x in c],'o-',color=colors[label],label=label)
        ax.set(xlabel='optimizer step',ylabel=f'fraction of{n}',title=key.replace('_',' '),ylim=(-.03,1.03));ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Free-generated training outputs; single-source engineering only')
    fig.savefig(out/(stage+'_training_accuracy.png'),dpi=170);plt.close(fig)


if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]),sys.argv[2])
