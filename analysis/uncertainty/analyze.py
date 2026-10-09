"""Frozen Task2 analysis, guarded by a complete1200-case independent audit."""
import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from collections import Counter,defaultdict
from common import atomic_json,config,scene_id,sha
from stat_rules import (collision_target,first_five_eligible,grouped_weights,
    pre_event_action_mean,confident_failure_indicator,failure_window_fraction,weighted_quantile)
import cluster_stats
import platt

PROXIES={'disagreement':1.,'churn':1.,'act_norm':1.,'min_dist':-1.}
ARMS=['nominal','aegis']


def write_csv(path,rows):
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)


def finite(value):return value is not None and math.isfinite(value)


def load(audited):
    report=json.loads((audited/'INDEPENDENT_AUDIT.json').read_text())
    if not report['passed'] or report['rollouts']!=1200 or report['states']!=60 or not report['all_diagnostic_sample_tensors_independently_reduced']:
        raise RuntimeError('Complete independent1200-case audit required; never fit partial input')
    import pyarrow.parquet as pq
    if sha(audited/'rollouts.parquet')!=report['parquet_sha256']:raise RuntimeError('Audited Parquet changed')
    results=json.loads((audited/'rollouts.json').read_text());grouped=defaultdict(list)
    for row in pq.read_table(audited/'rollouts.parquet').to_pylist():grouped[(row['scene_id'],row['capability'],row['seed'])].append(row)
    runs=[];seen=set()
    for result in results:
        spec=result['spec'];scene=scene_id(spec['scene']);key=(scene,spec['method'],result['seed'])
        if key in seen or result['status']!='complete':raise RuntimeError('Duplicate/incomplete rollout')
        seen.add(key);rows=grouped.pop(key)
        if [r['step'] for r in rows]!=list(range(1,result['actions']+1)):raise RuntimeError('Incomplete ordered action clock')
        runs.append(dict(id=spec['name'],scene=scene,task=scene.rsplit('/',1)[0],arm=spec['method'],repeat=spec['repeat'],seed=result['seed'],
            C=result['collision_step'],L=result['actions'],success=int(result['success']),safe_success=int(result['outcome']=='safe_success'),
            outcome=result['outcome'],rows=rows,inferences=[r for r in rows if r['infer_boundary']]))
    split=json.loads((Path(__file__).parent/'split.json').read_text());train=set(split['train']);test=set(split['test'])
    if grouped or len(runs)!=1200 or train&test or {r['scene'] for r in runs}!=train|test:raise RuntimeError('Frozen input/split coverage differs')
    for run in runs:run['split']='train' if run['scene'] in train else 'test'
    return runs,split,report


class Analysis:
    def __init__(self,runs,split,output):
        self.runs=runs;self.output=output;self.cfg=config()['task2'];self.models={};self.fit_failures=[]
        self.states=sorted(split['test']);self.all_states=sorted(split['train']+split['test'])
        tasks=lambda ss:[s.rsplit('/',1)[0] for s in ss]
        self.mult=cluster_stats.draws(self.states,tasks(self.states),self.cfg['bootstrap_replicates'],self.cfg['bootstrap_seed'])
        self.all_mult=cluster_stats.draws(self.all_states,tasks(self.all_states),self.cfg['bootstrap_replicates'],self.cfg['bootstrap_seed'])

    def weights(self,records):return grouped_weights([r['scene'] for r in records],[r['id'] for r in records])

    def estimate(self,records,values,raw=False,all_states=False,weights=None):
        if not records:return dict(value=None,low=None,high=None,defined=0,undefined=self.cfg['bootstrap_replicates'])
        states,mult=(self.all_states,self.all_mult) if all_states else (self.states,self.mult)
        weights=weights if weights is not None else ([1.]*len(records) if raw else self.weights(records))
        point,ci=cluster_stats.mean(values,[r['scene'] for r in records],weights,states,mult)
        return dict(value=point,**ci)

    def fit_model(self,key,records):
        if any(r['split']!='train' for r in records):raise RuntimeError('Test rows reached fit')
        if not records:self.models[key]=dict(status='no_eligible_train_rows',rows=0);return None
        try:model=platt.fit([r['x'] for r in records],[r['y'] for r in records],self.weights(records),self.cfg['platt'])
        except Exception as error:
            self.fit_failures.append(dict(key=key,error=str(error)));atomic_json(self.output/'FIT_FAILURE.json',self.fit_failures);raise
        self.models[key]=model;return model

    def evaluate(self,records,model):
        if not records or model is None:
            return {k:dict(value=None,ci=dict(low=None,high=None,defined=0,undefined=self.cfg['bootstrap_replicates'])) for k in ['AUROC','BA','Brier','calibrated_AUROC']}
        predictions=platt.predict(model,[r['x'] for r in records])
        metrics=cluster_stats.classification([r['y'] for r in records],[r['x'] for r in records],predictions,
            [r['scene'] for r in records],self.weights(records),self.states,self.mult)
        metrics['calibrated_AUROC']=cluster_stats.classification([r['y'] for r in records],predictions,predictions,
            [r['scene'] for r in records],self.weights(records),self.states,self.mult)['AUROC']
        return metrics

    def metrics_rows(self,records,model,identity):
        rows=[]
        for arm in ['pooled']+ARMS:
            selected=[r for r in records if r['split']=='test' and (arm=='pooled' or r['arm']==arm)]
            metrics=self.evaluate(selected,model)
            for metric,estimate in metrics.items():
                rows.append(dict(**identity,arm=arm,metric=metric,value=estimate['value'],**estimate['ci'],rows=len(selected),
                    states=len({r['scene'] for r in selected}),rollouts=len({r['id'] for r in selected}),positive=sum(r['y'] for r in selected),
                    negative=sum(1-r['y'] for r in selected),calibration='fixed_pooled_train_1D_Platt',AUROC_score='calibrated_probability' if metric=='calibrated_AUROC' else 'predefined_raw_direction'))
        return rows

    def collision(self):
        table=[];exclusions=[]
        for horizon in self.cfg['horizons_actions']:
            for proxy,direction in PROXIES.items():
                records=[];counts=Counter()
                for run in self.runs:
                    for row in run['inferences']:
                        label=collision_target(row['step'],run['C'],run['L'],bool(run['safe_success']),horizon)
                        reason=None
                        if run['C'] is not None and run['C']<row['step']:reason='post_crash'
                        elif label is None:reason='short_event_free_incomplete_censored'
                        elif not finite(row[proxy]):reason='missing_proxy'
                        counts[(run['split'],run['arm'],reason or 'eligible')]+=1
                        if reason is None:records.append(dict(id=run['id'],scene=run['scene'],split=run['split'],arm=run['arm'],x=direction*row[proxy],y=label))
                key='collision/'+proxy+'/h'+str(horizon);model=self.fit_model(key,[r for r in records if r['split']=='train'])
                table+=self.metrics_rows(records,model,dict(analysis='collision',proxy=proxy,horizon_actions=horizon,unit='preaction_at_risk_infer'))
                exclusions += [dict(proxy=proxy,horizon_actions=horizon,split=s,arm=a,reason=r,inferences=n) for (s,a,r),n in sorted(counts.items())]
                print(json.dumps(dict(stage='collision',proxy=proxy,horizon=horizon,eligible=len(records))),flush=True)
        write_csv(self.output/'collision_metrics.csv',table);write_csv(self.output/'collision_eligibility.csv',exclusions)

    def completion(self):
        table=[];prediction_rows=[];eligibility=[];curves=[];paired_model=None;paired_records=[]
        for stage in ['first_infer','first_five_landmark']:
            for proxy,direction in PROXIES.items():
                records=[]
                for run in self.runs:
                    reason=None;selected=run['inferences'][:1 if stage=='first_infer' else self.cfg['first_inferences']]
                    if stage=='first_five_landmark' and not first_five_eligible(run['inferences'],run['C'],run['L']):reason='did_not_reach_preaction21_landmark'
                    values=[r[proxy] for r in selected if finite(r[proxy])]
                    if not values:reason=reason or 'no_finite_proxy'
                    eligibility.append(dict(stage=stage,proxy=proxy,id=run['id'],scene=run['scene'],arm=run['arm'],split=run['split'],eligible=reason is None,reason=reason))
                    if reason is None:records.append(dict(id=run['id'],scene=run['scene'],arm=run['arm'],repeat=run['repeat'],split=run['split'],x=-direction*sum(values)/len(values),y=run['safe_success']))
                model=self.fit_model('completion/'+stage+'/'+proxy,[r for r in records if r['split']=='train'])
                table+=self.metrics_rows(records,model,dict(analysis='safe_success',stage=stage,proxy=proxy,unit='single_rollout'))
                if model is not None:
                    for record,p in zip(records,platt.predict(model,[r['x'] for r in records])):
                        record['probability']=float(p);prediction_rows.append(dict(stage=stage,proxy=proxy,**record))
                    for arm in ['pooled']+ARMS:
                        test=[r for r in records if r['split']=='test' and (arm=='pooled' or r['arm']==arm)];weights=self.weights(test)
                        for b in range(self.cfg['calibration_bins']):
                            lower=b/self.cfg['calibration_bins'];upper=(b+1)/self.cfg['calibration_bins']
                            indices=[i for i,r in enumerate(test) if lower<=r['probability'] and (r['probability']<upper or b==self.cfg['calibration_bins']-1)]
                            selected=[test[i] for i in indices];w=[weights[i] for i in indices]
                            observed=self.estimate(selected,[r['y'] for r in selected],weights=w)
                            predicted=self.estimate(selected,[r['probability'] for r in selected],weights=w)
                            curves.append(dict(stage=stage,proxy=proxy,arm=arm,bin_lower=lower,bin_upper=upper,rows=len(selected),states=len({r['scene'] for r in selected}),observed=observed['value'],observed_low=observed['low'],observed_high=observed['high'],undefined=observed['undefined'],predicted=predicted['value']))
                if stage=='first_five_landmark' and proxy=='disagreement':paired_model=model;paired_records=records
        write_csv(self.output/'completion_metrics.csv',table);write_csv(self.output/'completion_predictions.csv',prediction_rows)
        write_csv(self.output/'completion_eligibility.csv',eligibility);write_csv(self.output/'calibration_curve.csv',curves)
        self.paired(paired_records,paired_model)

    def paired(self,records,model):
        import numpy as np
        from scipy.stats import rankdata
        paired=[];by_state=defaultdict(dict)
        for r in records:
            if r['split']=='test' and 'probability' in r:by_state[r['scene']][(r['repeat'],r['arm'])]=r
        for scene in self.states:
            available=by_state[scene];seeds=[s for s in range(10) if (s,'nominal') in available and (s,'aegis') in available]
            paired.append(dict(scene=scene,task=scene.rsplit('/',1)[0],paired_seeds=len(seeds),
                delta_c=sum(available[(s,'aegis')]['probability']-available[(s,'nominal')]['probability'] for s in seeds)/len(seeds) if seeds else None,
                delta_safe_success=sum(available[(s,'aegis')]['y']-available[(s,'nominal')]['y'] for s in seeds)/len(seeds) if seeds else None))
        valid=[r for r in paired if r['paired_seeds']];indices=[self.states.index(r['scene']) for r in valid]
        x=np.asarray([r['delta_c'] for r in valid]);y=np.asarray([r['delta_safe_success'] for r in valid])
        def correlation(a,b,rank):
            if len(a)<2 or np.ptp(a)==0 or np.ptp(b)==0:return None
            if rank:a=rankdata(a,method='average');b=rankdata(b,method='average')
            return float(np.corrcoef(a,b)[0,1])
        results=[]
        for name,rank in [('Spearman',True),('Pearson',False)]:
            reps=[]
            for counts in self.mult[:,indices]:
                draw=np.repeat(np.arange(len(valid)),counts)
                c=correlation(x[draw],y[draw],rank);reps.append(float('nan') if c is None else c)
            results.append(dict(metric=name,value=correlation(x,y,rank),**cluster_stats.interval(reps),states=len(valid),paired_seeds=sum(r['paired_seeds'] for r in valid),interpretation='conditional_post_treatment_association_not_causal'))
        write_csv(self.output/'paired_state_deltas.csv',paired);write_csv(self.output/'paired_delta_correlations.csv',results)

    def thresholds(self):
        records=[]
        for run in self.runs:
            if run['split']!='train' or run['arm']!='nominal':continue
            records += [dict(id=run['id'],scene=run['scene'],value=row['disagreement']) for row in run['inferences'] if run['C'] is None or row['step']<=run['C']]
        weights=self.weights(records)
        thresholds={name:weighted_quantile([r['value'] for r in records],weights,self.cfg[key]) for name,key in [('supplement_q10','low_proxy_train_quantile'),('gate_q90','gate_train_quantile')]}
        atomic_json(self.output/'TRAIN_THRESHOLDS.json',dict(**thresholds,train_states=len({r['scene'] for r in records}),train_rollouts=len({r['id'] for r in records}),inferences=len(records),population='train_nominal_preaction_at_risk_infer',weighted_quantile_rule='first_sorted_value_at_or_above_cumulative_target',test_used=False))
        return thresholds

    def confident_failure(self):
        references={};reference_rows=[];cases=[];summary=[]
        tasks=sorted({r['task'] for r in self.runs})
        for task in tasks:
            for arm in ARMS:
                for k in self.cfg['failure_windows_actions']:
                    eligible=[];all_success=[r for r in self.runs if r['split']=='train' and r['task']==task and r['arm']==arm and r['safe_success']]
                    for run in all_success:
                        window=pre_event_action_mean(run['rows'],run['L'],k)
                        if window['eligible']:eligible.append(dict(id=run['id'],scene=run['scene'],mean=window['mean']))
                    median=weighted_quantile([r['mean'] for r in eligible],self.weights(eligible),.5) if eligible else None
                    references[(task,arm,k)]=median
                    reference_rows.append(dict(task=task,arm=arm,K_actions=k,reference_median=median,total_train_safe_success=len(all_success),eligible_rollouts=len(eligible),eligible_states=len({r['scene'] for r in eligible}),excluded=len(all_success)-len(eligible)))
        for run in self.runs:
            if run['split']!='test' or run['outcome']!='crash':continue
            for k in self.cfg['failure_windows_actions']:
                window=pre_event_action_mean(run['rows'],run['C'],k);reference=references[(run['task'],run['arm'],k)]
                reason=window['reason'] if not window['eligible'] else ('no_train_safe_success_reference' if reference is None else None)
                cases.append(dict(id=run['id'],scene=run['scene'],task=run['task'],arm=run['arm'],K_actions=k,C=run['C'],mean_disagreement=window['mean'],reference_median=reference,
                    eligible=reason is None,reason=reason,low_mean=confident_failure_indicator(window['mean'],reference) if reason is None else None,window_start=window['start'],window_end=window['end']))
        for arm in ARMS:
            for k in self.cfg['failure_windows_actions']:
                for task in ['pooled']+tasks:
                    all_cases=[r for r in cases if r['arm']==arm and r['K_actions']==k and (task=='pooled' or r['task']==task)]
                    eligible=[r for r in all_cases if r['eligible']]
                    for weighting in ['raw_rollout_proportion','secondary_equal_state_rollout']:
                        estimate=self.estimate(eligible,[r['low_mean'] for r in eligible],raw=weighting=='raw_rollout_proportion')
                        summary.append(dict(task=task,arm=arm,K_actions=k,weighting=weighting,**estimate,total_test_crash=len(all_cases),eligible=len(eligible),n_low=sum(r['low_mean'] for r in eligible),
                            excluded_incomplete_window=sum(r['reason']=='incomplete_window' for r in all_cases),excluded_missing_reference=sum(r['reason']=='no_train_safe_success_reference' for r in all_cases),excluded_other=sum(r['reason'] not in [None,'incomplete_window','no_train_safe_success_reference'] for r in all_cases)))
        write_csv(self.output/'confident_failure_reference.csv',reference_rows);write_csv(self.output/'confident_failure_cases.csv',cases);write_csv(self.output/'confident_failure.csv',summary)

    def supplement(self,cutoff):
        cases=[];summary=[]
        for run in self.runs:
            if run['split']!='test' or run['safe_success']:continue
            at_risk=[r for r in run['inferences'] if run['C'] is None or r['step']<=run['C']]
            for k in self.cfg['failure_windows_actions']:
                fraction=failure_window_fraction(at_risk,run['C'] or run['L'],k,cutoff)
                cases.append(dict(id=run['id'],scene=run['scene'],arm=run['arm'],outcome=run['outcome'],K_actions=k,**fraction))
        for arm in ARMS:
            for outcome in ['failed_pooled','crash','safe_incomplete']:
                for k in self.cfg['failure_windows_actions']:
                    total=[r for r in cases if r['arm']==arm and r['K_actions']==k and (outcome=='failed_pooled' or r['outcome']==outcome)]
                    eligible=[r for r in total if r['fraction'] is not None]
                    summary.append(dict(arm=arm,outcome=outcome,K_actions=k,cutoff=cutoff,**self.estimate(eligible,[r['fraction'] for r in eligible]),total_failed=len(total),eligible=len(eligible),excluded=len(total)-len(eligible),boundaries=sum(r['boundaries'] for r in eligible),truncated_windows=sum(r['partial'] for r in total)))
        write_csv(self.output/'low_boundary_cases.csv',cases);write_csv(self.output/'low_boundary_fraction.csv',summary)

    def temporal(self):
        rows=[]
        for population in ['test_primary','all_states_descriptive']:
            for arm in ARMS:
                for step in range(1,301,5):
                    candidates=[]
                    for run in self.runs:
                        if run['arm']!=arm or (population=='test_primary' and run['split']!='test') or (run['C'] is not None and run['C']<step):continue
                        if run['L']>=step:candidates.append((run,run['rows'][step-1]))
                    for proxy in PROXIES:
                        selected=[dict(id=run['id'],scene=run['scene'],value=r[proxy]) for run,r in candidates if finite(r[proxy])]
                        estimate=self.estimate(selected,[r['value'] for r in selected],all_states=population=='all_states_descriptive')
                        rows.append(dict(population=population,arm=arm,step=step,time_seconds=(step-1)/20.,proxy=proxy,**estimate,available_states=len({r['scene'] for r in selected}),available_rollouts=len(selected)))
        write_csv(self.output/'temporal_curves.csv',rows)

    def outcomes(self):
        rows=[]
        for population in ['test_primary','all_states_descriptive']:
            for arm in ARMS:
                runs=[r for r in self.runs if r['arm']==arm and (population!='test_primary' or r['split']=='test')]
                for name in ['CAR','crash_rate','TSR','safe_success']:
                    values=[int(r['C'] is None) if name=='CAR' else int(r['C'] is not None) if name=='crash_rate' else r['success'] if name=='TSR' else r['safe_success'] for r in runs]
                    rows.append(dict(population=population,arm=arm,metric=name,**self.estimate(runs,values,all_states=population=='all_states_descriptive'),rollouts=len(runs),states=len({r['scene'] for r in runs})))
        write_csv(self.output/'outcomes.csv',rows)

    def run(self):
        import numpy as np
        np.save(self.output/'test_bootstrap_multiplicities.npy',self.mult,allow_pickle=False)
        self.collision();self.completion();thresholds=self.thresholds();self.confident_failure();self.supplement(thresholds['supplement_q10']);self.temporal();self.outcomes()
        atomic_json(self.output/'PLATT_MODELS.json',self.models)


def main(audited,output,smoke=False):
    runs,split,report=load(audited);output.mkdir(exist_ok=False)
    if smoke:
        subset=config()['task2']['smoke'];test=set(subset['test_states'])
        if len(test)!=2 or not test<=set(split['test']) or subset['repeats']!=[0,1]:raise RuntimeError('Frozen two-scene two-seed smoke required')
        runs=[r for r in runs if r['split']=='train' or (r['scene'] in test and r['repeat'] in subset['repeats'])]
        split=dict(split,test=sorted(test))
    here=Path(__file__).parent
    import numpy as np,scipy,pyarrow
    atomic_json(output/'PROVENANCE.json',dict(input=str(audited),input_audit_sha256=sha(audited/'INDEPENDENT_AUDIT.json'),input_parquet_sha256=report['parquet_sha256'],collection_commit=report['code_commit'],
        analysis_code_commit=os.environ.get('CB_CODE_COMMIT'),slurm_job=os.environ.get('SLURM_JOB_ID'),started_unix=time.time(),software=dict(numpy=np.__version__,scipy=scipy.__version__,pyarrow=pyarrow.__version__),
        collection_source_plan_sha256=sha(Path(report['collection'])/'source/analysis/uncertainty/STATS_PLAN.md'),analysis_plan_sha256=sha(here/'STATS_PLAN.md'),split_sha256=sha(here/'split.json'),config_sha256=sha(here/'config.yaml'),
        analysis_source_sha256={p.name:sha(p) for p in [here/'analyze.py',here/'stat_rules.py',here/'cluster_stats.py',here/'platt.py']},test_states=split['test'],train_states=split['train'],
        guard='all1200 independently audited; train-only fits/cutoffs; fixed grouped weights multiplied by stratified state multiplicities',raw_evidence_location=report['collection'],raw_local_delivery='see separate DELIVERY_STATUS.json; not required for compact audited input fit'))
    Analysis(runs,split,output).run()
    atomic_json(output/'ANALYSIS_COMPLETE.json',dict(passed=True,stage='smoke' if smoke else 'full',audited_input_rollouts=1200,analyzed_rollouts=len(runs),states=len(split['train'])+len(split['test']),test_states=len(split['test']),test_rollouts=sum(r['split']=='test' for r in runs),train_states=42,files={p.name:sha(p) for p in output.iterdir() if p.is_file()},api_calls=0,new_rollouts=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('audited',type=Path);p.add_argument('output',type=Path);p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.audited.resolve(),a.output.resolve(),a.smoke)
