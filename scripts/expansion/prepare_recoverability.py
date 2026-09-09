"""Freeze four-parent, three-delay probe before any new outcomes."""
import hashlib,json,pickle,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 run=ROOT/'results/detour_benefit/a89f755da200_job5791132';c=json.loads((run/'config.json').read_text());es=json.loads((run/'anchors.json').read_text());a=json.loads((run/'A.json').read_text());vectors={};eligible=[]
 for e in es:
  if e['role']!='fitting' or e['condition']!='glass':continue
  with (run/e['episode_id']/'bundle.pkl').open('rb') as f:ctx=pickle.load(f)
  o={k[12:]:v for k,v in ctx['bundle'].runtime_state.items() if k.startswith('observation.')};g=ctx['glasses'][0]
  vectors[e['episode_id']]=np.r_[o['robot0_eef_pos']-o['akita_black_bowl_1_pos'],np.array(g['pos'])-o['akita_black_bowl_1_pos'],o['plate_1_pos']-o['akita_black_bowl_1_pos'],g['size']]
  if all(r['horizons']['440']['success']==0 for r in a if r['episode_id']==e['episode_id'] and r['option']==1):eligible.append(e['episode_id'])
 selected=['e18','e21'];matches=[]
 for positive in selected[:]:
  dist,negative=min((float(np.linalg.norm(vectors[positive]-vectors[n])),n) for n in eligible if n not in selected)
  selected.append(negative);matches.append({'positive':positive,'negative':negative,'distance_m':dist})
 config={'kind':'recoverability_physical_probe_v1','parent_run':str(run.relative_to(ROOT)),'parents':selected,'matches':matches,'selection':'positives e18/e21; greedy distinct nearest failed fitting glass in metre-valued relative EEF/glass/plate-to-bowl coordinates plus glass radius/height, A failure labels only','offsets':[0,3,10],'suffix_horizons':[220,440],'repeats':[0,1],'max_scored_branches':48,'max_prefix_continuations':4,'max_added_prefix_actions_per_parent':10,'seed_semantics':'each child exact bundle restored identically for both options and both repeats; no policy sampling change','option_order':'parent_index+offset+repeat parity','controller_unchanged':True,'learning':False,'confirmation':False,'physical_hypothesis':'a bounded Base approach changes the state and may enter or leave the fixed-controller completion region; distinguish from mere residual-budget censoring','unavailable_child':'retain prefix terminal, do not branch after accident/success or replace parent','input_sha256':{str((run/n).relative_to(ROOT)):sha(run/n) for n in ['A.json','anchors.json','config.json','checkpoint_hashes.json']},'parent_bundle_sha256':{e:sha(run/e/'bundle.pkl') for e in selected}}
 p=ROOT/'configs/recoverability/probe_v1.json';p.write_text(json.dumps(config,indent=2)+'\n');print(json.dumps(config,indent=2))
if __name__=='__main__':main()
