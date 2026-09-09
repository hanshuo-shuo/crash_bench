#!/usr/bin/env python3
"""Source-equal, paired, descriptive analysis; never fits or changes frozen gates."""
import argparse,csv,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from crashbench.detour_benefit import METHODS,HORIZONS,episode_outcome,validate_phase,readout
from scripts.expansion.hash_tree_manifest import file_sha256

def table(path,rows):
    with path.open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
def analyze(root,bootstrap=5000):
    read=lambda n:json.loads((root/n).read_text())
    assert read('complete.json')['status']=='COMPLETE'
    anchors,a,b,frozen=read('anchors.json'),read('A.json'),read('B.json'),read('freeze.json')
    assert file_sha256(root/'freeze.json')==(root/'freeze.sha256').read_text().strip()
    for filename,key in [('A.json','A_sha256'),('anchors.json','anchors_sha256'),('models.pkl','models_sha256')]:assert file_sha256(root/filename)==frozen[key]
    validate_phase(a,anchors,'A');validate_phase(b,anchors,'B')
    for e in anchors:
        assert file_sha256(root/e['episode_id']/'prefix.pkl.gz')==e['prefix_sha256']
        if e['triggered']:assert file_sha256(root/e['episode_id']/'bundle.pkl')==e['bundle_sha256']
    for r in a+b:assert file_sha256(root/r['episode_id']/f"r{r['repeat']}_o{r['option']}.pkl.gz")==r['trace_sha256']
    out=root/'analysis';out.mkdir(exist_ok=False)
    sources=sorted({e['source'] for e in anchors});rng=np.random.default_rng(2027);resamples=rng.integers(0,len(sources),(bootstrap,len(sources)))
    cells=[];source_rows=[];overall=[];vectors={};normal_vectors={}
    for h in HORIZONS:
        for method in METHODS:
            chosen=frozen['choices'][method];by=defaultdict(list);normal=defaultdict(list)
            for e in anchors:
                option=chosen[e['episode_id']];base=episode_outcome(e,b,0,h);got=episode_outcome(e,b,option,h)
                harm=new=0.
                if e['triggered'] and option:
                    lookup={(r['repeat'],r['option']):r['horizons'][str(h)] for r in b if r['episode_id']==e['episode_id']}
                    harm=np.mean([lookup[r,0]['success'] and not lookup[r,1]['success'] for r in (2,3)])
                    new=np.mean([lookup[r,1]['accident'] and not lookup[r,0]['accident'] for r in (2,3)])
                v=dict(success=got['success'],accident=got['accident'],delta_success=got['success']-base['success'],delta_accident=got['accident']-base['accident'],success_loss=float(harm),new_accident=float(new),calls=got['calls'],controller_steps=got['controller_steps'],completion_cost=got['completion_cost'],intervention=float(option and e['triggered']),no_trigger=float(not e['triggered']))
                cells.append(dict(horizon=h,method=method,episode=e['episode_id'],source=e['source'],role=e['role'],condition=e['condition'],**v));by[e['source']].append(v)
                if e['condition']!='glass':normal[e['source']].append(v)
            sr=[]
            for source in sources:
                v={k:float(np.mean([r[k] for r in by[source]])) for k in by[source][0]};source_rows.append(dict(horizon=h,method=method,source=source,**v));sr.append(v)
            vectors[h,method]={k:np.array([r[k] for r in sr]) for k in sr[0]}
            result=dict(horizon=h,method=method,**{k:float(v.mean()) for k,v in vectors[h,method].items()})
            for k in ('delta_success','delta_accident'):
                ci=np.percentile(vectors[h,method][k][resamples].mean(1),[2.5,97.5]);result[k+'_lo'],result[k+'_hi']=ci
            result['normal_success_loss']=float(np.mean([np.mean([r['success_loss'] for r in normal[s]]) for s in sources]))
            overall.append(result)
    comparisons=[]
    for h in HORIZONS:
        for method in METHODS:
            for baseline in ('Base','RiskDetour','DirectQ_sanity'):
                for metric in ('success','accident'):
                    diff=vectors[h,method][metric]-vectors[h,baseline][metric];ci=np.percentile(diff[resamples].mean(1),[2.5,97.5]);comparisons.append(dict(horizon=h,method=method,baseline=baseline,metric=metric,delta=float(diff.mean()),lo=float(ci[0]),hi=float(ci[1]),positive_sources=int((diff>0).sum()),negative_sources=int((diff<0).sum())))
    diagnostics=[]
    for e in anchors:
        for h in HORIZONS:
            va=[episode_outcome(e,phase,1,h)['success']-episode_outcome(e,phase,0,h)['success'] for phase in (a,b)]
            row=dict(episode=e['episode_id'],source=e['source'],role=e['role'],condition=e['condition'],horizon=h,triggered=e['triggered'],A_delta_success=va[0],B_delta_success=va[1])
            for phase_name,phase in [('A',a),('B',b)]:
                for o in (0,1):
                    rs=[r for r in phase if r['episode_id']==e['episode_id'] and r['option']==o]
                    row[f'{phase_name}_{o}_terminal_disagreement']=len({(r['horizons'][str(h)]['success'],r['horizons'][str(h)]['accident']) for r in rs})>1
                    row[f'{phase_name}_{o}_success_steps']=json.dumps([r['horizons'][str(h)]['success_step'] for r in rs])
            diagnostics.append(row)
    # Exact success cumulative curves from these same continuous B trajectories.
    curves=[]
    index=defaultdict(list)
    for r in b:index[r['episode_id'],r['option']].append(r)
    for h in range(441):
        row={'action_step':h}
        for method in METHODS:
            by=defaultdict(list)
            for e in anchors:
                rs=index[e['episode_id'],frozen['choices'][method][e['episode_id']]] if e['triggered'] else [{'events':e['events']}]
                by[e['source']].append(float(np.mean([readout(r['events'],h)['success'] for r in rs])))
            row[method]=float(np.mean([np.mean(v) for v in by.values()]))
        curves.append(row)
    table(out/'overall.csv',overall);table(out/'paired_comparisons.csv',comparisons);table(out/'per_source.csv',source_rows);table(out/'all_episode_conditions.csv',cells);table(out/'repeat_diagnostics.csv',diagnostics);table(out/'success_cumulative_curve.csv',curves)
    cost={'prefixes':len(anchors),'scored_branches':len(a)+len(b),'triggered_episodes':sum(e['triggered'] for e in anchors),'actual_prefix_calls':sum(e['prefix_calls'] for e in anchors),'actual_branch_new_calls':sum(sum(v.get('calls',0) for v in r['events'])-next(e['prefix_calls'] for e in anchors if e['episode_id']==r['episode_id']) for r in a+b),'prefix_wall_seconds':sum(e['prefix_elapsed_seconds'] for e in anchors),'branch_wall_seconds':sum(r['branch_elapsed_seconds'] for r in a+b),'candidate_risk_calls':sum(e['risk_call_count'] for e in anchors),'candidate_risk_seconds':sum(e['risk_elapsed_seconds'] for e in anchors),'fit_seconds':frozen['info']['fit_elapsed_seconds'],'slurm_provenance':read('provenance.json')}
    gate440=next(r for r in overall if r['horizon']==440 and r['method']=='BenefitGate');reference440=next(r for r in overall if r['horizon']==440 and r['method']=='A_long_reference');risk440=next(r for r in overall if r['horizon']==440 and r['method']=='RiskDetour')
    decision='DEVELOPMENT_SUPPORTS_REVIEW_FOR_CONDITIONAL_VALIDATION' if gate440['delta_success']>0 and gate440['success']>risk440['success'] and reference440['delta_success']>0 else 'NO_AUTOMATIC_VALIDATION_EXPLAIN_SUPPORT_AND_VARIABILITY'
    (out/'metrics.json').write_text(json.dumps({'cost':cost,'decision':decision,'frozen_info':frozen['info'],'bootstrap':{'unit':'source','replicates':bootstrap,'seed':2027,'descriptive_only':True}},indent=2)+'\n')
    lines=['# 固定 Detour：收益门控开发结果','',f'完成 {len(anchors)} 条前缀、{len(a)+len(b)} 个 scored branches；统计单位为16个物理来源。测试重复B在模型和阈值冻结后执行。','', '| H | 方法 | 成功 | 相对Base成功差 | 事故差 | 原成功损失 | 新增事故 | 干预率 |','|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in overall:lines.append(f"|{r['horizon']}|{r['method']}|{r['success']:.2%}|{r['delta_success']:+.2%}|{r['delta_accident']:+.2%}|{r['success_loss']:.2%}|{r['new_accident']:.2%}|{r['intervention']:.2%}|")
    lines += ['',f'机器判定：`{decision}`。该判定只组织下一步讨论，不自动提交新来源验证。', '', '同输入两臂DirectQ与benefit Ridge采用相同线性配方，已合并为代数sanity check，不能称两种算法。强RiskDetour是在共同候选机会上的比较，不是完整from-reset Risk→BestFixed。', '', 'H=220与2H=440取自同一连续轨迹，前缀和干预均计入预算。完整成功累计曲线见success_cumulative_curve.csv；不重新挑有利截止。未触发和提前终局全部保留在分母。', '', '来源共同重采样描述性区间、配对比较见paired_comparisons.csv；逐来源差见per_source.csv；所有来源条件见all_episode_conditions.csv；A/B兑现和同选项波动见repeat_diagnostics.csv。A/B不是来源隔离，greedy策略的种子变化不构成新随机策略分布。', '', '零事故和校准零margin不构成安全保证；小样本和区间宽时保持未决。单状态两次重复不能认证期望因果效应。正常对照保持和新增事故独立报告。', '', '实际全执行成本见metrics.json；调用成本为实测计数，wall time为本次研究执行成本，未包装成真实机器人加速。完整bundle、输入和轨迹保留在运行目录。']
    (out/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'decision':decision,'cost':cost},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();analyze(args.run)
