"""Read-only delivery of a failed bounded rank64 smoke; never launches training."""
import json
import os
from pathlib import Path
import re
from .common import read, write, require, sha
from .engineering_contract import DATA_SHA, score_rationale
from .binary_report import table, deliver
from .rank_contract import validate_parent, config_difference


def audit(rows, records, zero=False):
    require(len(rows)==len(records) and [x['sample_id'] for x in rows]==[x['sample_id'] for x in records],
            'Smoke output alignment changed')
    result=[]
    for row,x in zip(rows,records):
        m=re.fullmatch(r'Rationale: .+\nVerdict: ([01])',x['generated'],flags=re.S)
        pred={'0':'infeasible','1':'feasible'}[m.group(1)] if m else None
        expected='infeasible' if zero else row['target']['verdict']
        target='Rationale: '+row['target']['rationale']+'\nVerdict: '+{'infeasible':'0','feasible':'1'}[expected]
        grade=score_rationale(x['generated'],row,'horizon_0' if zero else 'original')
        require(x['predicted_verdict']==pred and x['format_valid']==bool(m),'Stored parse differs')
        require(x['verdict_correct']==(pred==expected) and x['exact_match']==(x['generated']==target),'Stored correctness differs')
        require(x['rationale_assessment']==grade,'Stored grounding differs')
        result.append({'sample_id':row['sample_id'],'source_label':row['target']['verdict'],
            'condition':'zero_command' if zero else 'original','expected':expected,'predicted':pred,
            'verdict_correct':pred==expected,'rationale_correct':grade['status']=='supported_complete',
            'format_valid':bool(m),'full_output_correct':x['generated']==target,
            'contradictions':' | '.join(grade['contradictions']),'generated':x['generated'],
            'milk_hidden_both':all(v['entities']['target']['pixels_224']==0 for v in row['metadata']['visibility']['views'])})
    return result


def main(parent,outroot):
    receipt=read(parent/'RECEIPT.json');baseline=validate_parent(receipt)
    failed=read(parent/'smoke/RESULT.json')
    require(failed['passed'] is False and failed['steps']==240,'Expected completed bounded smoke failure')
    require(read(parent/'STOP_smoke.json')['message']=='Bounded smoke failed; no full fit permitted','Different failure')
    require(not (parent/'full').exists() and not (parent/'full_SUBMISSION.json').exists(),'Full training unexpectedly launched')
    dataset=Path(receipt['campaign'])/'dataset';require(sha(dataset/'dataset.jsonl')==DATA_SHA,'Dataset changed')
    ids=read(parent/'preflight/COMPLETE.json')['smoke_ids']
    rows=[json.loads(s) for s in (dataset/'dataset.jsonl').read_text().splitlines() if json.loads(s)['sample_id'] in ids]
    require(len(rows)==8 and sum(x['target']['verdict']=='feasible' for x in rows)==4,'Not balanced eight')
    out=outroot/'report';out.mkdir(exist_ok=False)
    runs={'rank16':baseline,'rank64':parent};summaries={};all_rows=[];checkpoints=[];curves={}
    for label,run in runs.items():
        stage=run/'smoke';res=read(stage/'RESULT.json')
        require(res['vision_and_projector_bitwise_unchanged'],'Frozen visual weights changed')
        original=audit(rows,read(stage/'generations_final.json'))
        zero=audit(rows,read(stage/'generations_horizon_0.json'),True)
        for x in original+zero:all_rows.append(dict(rank=label,**x))
        curve=[json.loads(s) for s in (stage/'curve.jsonl').read_text().splitlines()];curves[label]=curve
        require([x['step'] for x in curve]==list(range(1,res['steps']+1)),'Incomplete curve')
        cp=[]
        for p in sorted(stage.glob('generations_step_*.json')):
            step=int(p.stem.rsplit('_',1)[1]);a=audit(rows,read(p))
            record={'rank':label,'step':step,'verdict_correct':sum(x['verdict_correct'] for x in a),
                'rationale_correct':sum(x['rationale_correct'] for x in a),
                'full_output_correct':sum(x['full_output_correct'] for x in a),
                'format_valid':sum(x['format_valid'] for x in a)}
            checkpoints.append(record);cp.append(record)
        summaries[label]={'steps':res['steps'],'passed':res['passed'],'best_checkpoint_full_outputs':max(x['full_output_correct'] for x in cp),
            'final_verdict_correct':sum(x['verdict_correct'] for x in original),
            'feasible_correct':sum(x['source_label']=='feasible' and x['verdict_correct'] for x in original),
            'infeasible_correct':sum(x['source_label']=='infeasible' and x['verdict_correct'] for x in original),
            'final_rationale_correct':sum(x['rationale_correct'] for x in original),
            'final_full_correct':sum(x['full_output_correct'] for x in original),
            'final_format_valid':sum(x['format_valid'] for x in original),
            'zero_command_correct':sum(x['verdict_correct'] for x in zero),
            'actual_zero_flips_positive_source':sum(a['source_label']=='feasible' and a['predicted']=='feasible' and z['predicted']=='infeasible' for a,z in zip(original,zero)),
            'zero_predictions_unchanged':sum(a['predicted']==z['predicted'] for a,z in zip(original,zero)),
            'teacher_final':read(stage/f"loss_step_{res['steps']:04d}.json"),
            'native_generated_scores':res['final_score'],'zero_scores':res['zero_command_score'],
            'gradient_probe':res['gradient_probe'],'peak_gpu_gib':res['peak_gpu_gib'],
            'initial_loss':res['before'],'final_loss':res['after'],
            'first140_gradient_clipped_steps':sum(x['gradient_norm']>1 for x in curve[:140]),
            'max_preclip_gradient_norm':max(x['gradient_norm'] for x in curve),
            'trainable_parameters':read(stage/'PROVENANCE.json')['trainable_parameters']}
    require(summaries['rank16']['initial_loss']==summaries['rank64']['initial_loss'],'Initial base-model losses differ')
    for filename in ('generations_final.json','generations_horizon_0.json'):
        a=read(baseline/'smoke'/filename);b=read(parent/'smoke'/filename)
        require([(x['sample_id'],x['input_sha256']) for x in a]==[(x['sample_id'],x['input_sha256']) for x in b],'Paired student inputs changed')
    summary={'results':summaries,'config_differences':{s:config_difference(read(baseline/(s+'_config.json')),read(parent/(s+'_config.json'))) for s in ('smoke','full')},
        'same_initial_base_losses':True,'same_student_inputs':True,'full120_launched':False,
        'reason_full_fit_not_launched':'Rank64 failed the predeclared balanced8 generation and component-loss gate at240 steps',
        'training_root':str(parent),'baseline_root':str(baseline),'source_commit':receipt['source_commit'],
        'scope':'Bounded engineering smoke only, one source, single seed. Does not establish rank64 capacity limits or generalization.'}
    write(out/'SUMMARY.json',summary);table(out/'GENERATED_OUTPUTS.csv',all_rows);table(out/'GENERATION_CHECKPOINTS.csv',checkpoints)
    make_plots(runs,curves,checkpoints,out)
    lines=['# Rank64 SFT comparison: bounded smoke failed','',
        'Rank64/alpha128 failed the unchanged8-sample smoke gate after240 updates. The120-example full fit was not launched. Rank16/alpha32 had passed the same gate at140 updates.',
        '', '| Metric | Rank16 | Rank64 |','|---|---:|---:|']
    for key in ('steps','passed','best_checkpoint_full_outputs','feasible_correct','infeasible_correct',
                'final_rationale_correct','final_full_correct','final_format_valid','zero_command_correct',
                'actual_zero_flips_positive_source','zero_predictions_unchanged','trainable_parameters',
                'first140_gradient_clipped_steps','peak_gpu_gib'):
        lines.append(f"| {key} | {summaries['rank16'][key]} | {summaries['rank64'][key]} |")
    lines += ['', 'The same fixed smoke recipe uses LR2e-4/dropout0, batch4, seed713 and the original balanced eight. Only rank and proportional alpha differ; alpha/r stays2. All120 encoded records passed comparison,41 preflight unit tests passed and240 encoding checks passed. Initial full-pool base-model losses match exactly. Both loss components have nonzero LoRA gradients and visual/projector parameter hashes are unchanged.',
        '', 'The rank64 final model predicts feasible for all eight original and all eight zero-command cases. It produces two false milk-visibility statements. At steps140 and220 it temporarily fails to produce a valid verdict, with repetitive output at140. The final generation format recovers, but classification does not. The best evaluated rank64 checkpoint has6/8 exact full outputs; no checkpoint passes the8/8 gate.',
        '', 'The final rank64 full-eight teacher decision-token loss is1.703861, versus its initial1.343727. Narrative/format/EOS component loss falls2.937793 to0.007438. The low final minibatch loss is not the full-eight loss and must not be reported as successful fitting. Native generated decision-token NLL is class-specific in SUMMARY.json, separate from EOS.',
        '', 'This result rejects the claim that simply increasing rank at the unchanged recipe solves the current engineering failure. It does not establish that higher rank lacks capacity. The oscillation and transient generation collapse make optimization sensitivity a plausible explanation, not a proven cause. Keeping alpha/r fixed does not equalize rank-dependent update dynamics. Different parameter shapes also change initialization and RNG consumption.',
        '', 'The full fit is gated off, not an infrastructure failure: the GPU job intentionally exits1 after saving its240-step checkpoint, all outputs, losses and the failed gate. No extra training, full fine-tuning, simulator generation or scientific-result changes were performed. These eight examples still come from one source scene. Visibility-template agreement is not a physical feasibility proof.',
        '', '![Loss curves](training_curves.png)','![Generation checkpoints](training_accuracy.png)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    validate_parent(receipt)
    write(outroot/'COMPLETE.json',{'passed':True,'report_complete':True,'smoke_passed':False,'full120_launched':False,
        'training_root':str(parent),'report_rows':len(all_rows),'scientific_generalization_claim':False})
    deliver(parent);deliver(outroot)


def make_plots(runs,curves,checkpoints,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'rank16':'#277da1','rank64':'#d65f26'}
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for ax,key in zip(axes.flat,('total','rationale','verdict','rationale_text','format','eos')):
        for label,run in runs.items():
            c=curves[label];ax.plot([x['step'] for x in c],[max(x[key],1e-8) for x in c],color=colors[label],alpha=.18,lw=.7)
            measured=[(0,read(run/'smoke/loss_before.json')['means'][key])]
            measured.extend((int(p.stem.rsplit('_',1)[1]),read(p)['means'][key]) for p in sorted((run/'smoke').glob('loss_step_*.json')))
            ax.plot([x[0] for x in measured],[max(x[1],1e-8) for x in measured],'o-',color=colors[label],label=label,ms=3,lw=1.4)
        ax.set(xlabel='optimizer step',ylabel='cross entropy',title=key.replace('_',' '),yscale='log');ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('Balanced8 engineering smoke: faint=train batches, solid=full-eight evaluation')
    fig.savefig(out/'training_curves.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for ax,key in zip(axes,('verdict_correct','rationale_correct','full_output_correct')):
        for label in runs:
            c=[x for x in checkpoints if x['rank']==label]
            ax.plot([x['step'] for x in c],[x[key] for x in c],'o-',color=colors[label],label=label)
        ax.set(xlabel='optimizer step',ylabel='correct /8',ylim=(-.3,8.3),title=key.replace('_',' '));ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Free-generated smoke outputs; rank16 stops on pass at140, rank64 fails at240')
    fig.savefig(out/'training_accuracy.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    import sys
    main(*map(Path,sys.argv[1:]))
