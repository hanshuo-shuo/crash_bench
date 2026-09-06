from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from scripts.expansion.probe_neutral_refresh import trace_difference
HERE=Path(__file__).resolve().parent
one=ROOT/'results/paper_review/448f8d77a5c3_5628711';two=ROOT/'results/paper_review/20c2bd98b0a3_obs_5629071'
a=json.loads((one/'report.json').read_text());b=json.loads((two/'report.json').read_text())
counts={'same_option_pairs':0,'action_diverged':0,'state_diverged':0,'terminal_class_changed':0}
for row in a['anchors']:
 for x,y in [('base_a','base_b'),('refresh_a','refresh_b')]:
  counts['same_option_pairs']+=1
  c=row['comparisons'][x+'_vs_'+y]
  counts['action_diverged']+=c['maximum_absolute_difference']['action']>0
  counts['state_diverged']+=c['maximum_absolute_difference']['state']>0
  counts['terminal_class_changed']+=any(row['outcomes'][x][k]!=row['outcomes'][y][k] for k in ['task_success','catastrophe','safe_noncompletion'])
summary={'kind':'paper_probe_summary','engineering_only':True,'neutral_counts':counts,'input_replay':[],'cross_process':[]}
for row in b['anchors']:
 first=next((r for r in row['observation_differences'] if any(x['changed_elements'] for x in r['fields'].values())),None)
 summary['input_replay'].append({'task_id':row['task_id'],'first_live_input_difference':first,'comparisons':row['comparisons']})
 task=row['task_id'];old=json.loads((one/f'task{task}_fresh_control/base_a.json').read_text())['trace'][:35];new=json.loads((two/f'task{task}_live_a.json').read_text())
 summary['cross_process'].append({'task_id':task,'first_observation_sha_equal':old[0]['observation_sha256']==new[0]['observation_sha256'],
  'first_continuation_sha_equal':old[0]['policy_continuation_sha256']==new[0]['policy_continuation_sha256'],
  'first_action_max_abs_difference':max(abs(x-y) for x,y in zip(old[0]['action'],new[0]['action'])),
  'trace_difference':trace_difference(old,new),'complete_inference_input_and_runtime_identity_proven':False})
(HERE/'evidence/probe_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(json.dumps(counts))
