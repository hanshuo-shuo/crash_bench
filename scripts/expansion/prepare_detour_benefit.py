"""Freeze a provenance-based source panel without selecting recovery outcomes."""
import csv, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 cap=ROOT/'results/counterfactual_router/full_d4751330395e_20260814T152231Z'
 risk=ROOT/'results/counterfactual_router/fresh_online_2eab4a4a53dc_20260817T080708Z/router.json'
 meta=json.loads((cap/'decision_metadata.json').read_text())
 used={m['source_state_sha256'] for m in meta if m['split'] in ('train','calibration')}
 assert len(used)==12
 manifests=sorted((ROOT/'results/glass_recovery_v2').glob('*/frontier_placements/placements.json'))
 sources={}
 for p in manifests:
  for row in json.loads(p.read_text())['placements']:
   sources.setdefault(row['source_state_sha256'],[]).append((str(p.relative_to(ROOT)),row))
 assert used<=sources.keys()
 # All twelve sources previously used by the frozen risk recipe remain fitting-side.
 # Four calibration sources are picked by identity hash only, with no terminal reads.
 candidates=sorted(set(sources)-used,key=lambda s:hashlib.sha256(('detour-benefit-v1|2027|'+s).encode()).hexdigest())
 selected=[(s,'fitting') for s in sorted(used)]+[(s,'calibration') for s in candidates[:4]]
 forbidden={r['source_state_sha256'] for r in json.loads((ROOT/'results/expansion/governance/split_manifest_v1_1.json').read_text())['assignments'] if r['role'].startswith('confirmatory')}
 assert not ({s for s,_ in selected}&forbidden)
 panel=[]
 for s,role in selected:
  path,row=min(sources[s],key=lambda x:(x[0],x[1]['placement_id']))
  for cond in ('glass','offpath','noglass'):
   panel.append({'episode_id':f'e{len(panel):02d}','source':s,'role':role,'condition':cond,'placement_manifest':path,'placement':row})
 r=json.loads(risk.read_text())
 config={'schema_version':1,'kind':'detour_benefit_development','seed':2027,'H':220,'long_H':440,'settle_steps':10,'candidate_last_action':219,
 'source_selection':'risk-fit/calibration-used 12 stay fitting; remaining identities sha256 ordered; lexicographic placement per source; no recovery outcome selection',
 'conditions':['glass','offpath','noglass'],'panel':panel,'max_prefixes':48,'max_branches':384,'repeats':4,'A_repeats':[0,1],'B_repeats':[2,3],
 'policy_seed_override':{'0':None,'1':None,'2':20270902,'3':20270903},'seed_semantics':'restore same full bundle; A same RNG; B overrides torch policy RNG only; greedy decoding remains fixed',
 'option_order':'(episode_index+repeat)%2 alternates B/R order','risk_path':str(risk.relative_to(ROOT)),
 'risk_threshold':r['calibration']['binary_risk_retreat_frontier']['target_0.4']['threshold'],
 'risk_threshold_provenance':'historical target0.4 calibrated on 7 sources now entirely fitting-side; frozen before new prefixes',
 'checkpoint':'openvla/openvla-7b-finetuned-libero-spatial','checkpoint_revision':'962318cec55ac10993ff0f5f43eda9a270b4c873',
 'detour':json.loads((cap/'capture_manifest.json').read_text())['protocol']['detour'],
 'model':{'ridge_alpha':1.0,'pca_components':4,'minimum_fitting_anchors':5,'rank_failure':'all_base_with_insufficient_support_flag','input':'current_hidden_robot_state_nominal_action','history':False},
 'gate':{'calibration_sources':4,'objective':'H success primary; separately report long_H; no cutoff reselection','constraints':'delta_accident<=0 and normal-control success_loss<=0 at both H and 2H','tie_break':['lower_accident','lower_intervention','higher_threshold'],'allow_all_base':True},
 'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [*manifests,risk,risk.with_suffix('.npz'),cap/'decision_metadata.json',ROOT/'results/expansion/governance/split_manifest_v1_1.json']},
 'conditional_validation_auto_submit':False}
 out=ROOT/'configs/detour_benefit/development_v1.json';out.write_text(json.dumps(config,indent=2)+'\n')
 print(f'{len(selected)} sources, {len(panel)} episodes; threshold={config["risk_threshold"]}')
if __name__=='__main__':main()
