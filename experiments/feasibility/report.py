"""Compute-side static scientific plots and count-based mechanism report."""
import csv,json,math
from collections import defaultdict
from pathlib import Path
from protocol import STATES,CONDITIONS,CANDIDATES,CHECKPOINTS

def report(root):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 root=Path(root);out=root/'report';out.mkdir(exist_ok=True)
 rows=json.loads((root/'rows.json').read_text());initial=[r for r in rows if r['branch_step'] is None]
 fields=sorted(set(k for r in rows for k in r))
 with (out/'runs.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 stats=[];table=['# 安全未完成状态：可行性见证与受控续接诊断','',
 '这是8个预先固定状态的机制诊断，其中6个诊断、2个正常对照。不是总体性能评估。',
 '身份与几何分开控制；初始VLM文字取自既有repeat0回答并冻结，不是完整部署评价。无新增API调用。',
 '首策略输入和RNG严格配对；共同输入/RNG的原生推理结果仍执行并记录，控制输出复用首次结果。',
 '参考方法使用特权物体位置规划，但所有物理动作仍是同一机器人7D OSC/gripper；仅在官方初始化时加载官方初始状态。',
 '几何条件使用全碰撞网格顶点/保守基本形状包围，数值验证单椭球包含性。它仍保留AEGIS椭球代理局限。','',
 '|状态|原AEGIS|仅身份|仅几何|身份+几何|参考新验证|','|---|---:|---:|---:|---:|---:|']
 def count(group):return '%d/%d'%(sum(r['safe_success'] for r in group),len(group)) if group else '待运行'
 for state in STATES:
  groups={c:[r for r in initial if r['state']==state['id'] and r['condition']==c] for c in ['raw','identity','geometry','identity_geometry']}
  ref=[r for r in initial if r['state']==state['id'] and r['condition']=='reference' and r['validation']]
  table.append('|%s|%s|%s|%s|%s|%s|'%(state['id'],*[count(groups[c]) for c in groups],count(ref)))
  stats.append({'state':state['id'],'role':state['role'],**{c:{'safe_success':sum(r['safe_success'] for r in v),'n':len(v),'safe_noncompletion':sum(not r['success'] and not r['collided'] for r in v)} for c,v in groups.items()},'reference':{'safe_success':sum(r['safe_success'] for r in ref),'n':len(ref)}})
 matrix=np.array([[sum(r['safe_success'] for r in initial if r['state']==s['id'] and r['condition']==c)/max(1,sum(r['state']==s['id'] and r['condition']==c for r in initial)) for c in CONDITIONS] for s in STATES])
 fig,ax=plt.subplots(figsize=(10,5));im=ax.imshow(matrix,vmin=0,vmax=1,cmap='YlGnBu')
 ax.set_xticks(range(len(CONDITIONS)));ax.set_xticklabels(CONDITIONS);ax.set_yticks(range(len(STATES)));ax.set_yticklabels([s['id'] for s in STATES])
 for i,s in enumerate(STATES):
  for j,c in enumerate(CONDITIONS):
   g=[r for r in initial if r['state']==s['id'] and r['condition']==c];ax.text(j,i,count(g),ha='center',va='center',color='white' if matrix[i,j]>.6 else 'black')
 fig.colorbar(im,ax=ax,label='Observed safe completion fraction (diagnostic only)');ax.set_title('Fixed initial-state perception conditions');fig.tight_layout();fig.savefig(out/'perception_conditions.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(2,4,figsize=(13,6),sharex=True)
 for ax,state in zip(axes.flat,STATES):
  for c,color in [('nominal','#cb513a'),('raw','#27658a'),('geometry','#49a578')]:
   record=next((r for r in initial if r['state']==state['id'] and r['condition']==c),None)
   if not record:continue
   trace=[json.loads(l) for l in (root/'runs'/record['run_id']/'steps.jsonl').read_text().splitlines()]
   ax.plot([x['step'] for x in trace],[x['obstacle_l1_m']*1000 for x in trace],label=c,color=color)
  ax.axhline(1,color='black',ls=':',lw=1);ax.set_title(state['id']);ax.set_yscale('symlog',linthresh=1);ax.set_ylabel('Obstacle L1 displacement / mm');ax.set_xlabel('Executed action')
 axes.flat[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'safety_traces.png',dpi=180);plt.close(fig)
 branches=[r for r in rows if r['branch_step'] is not None];bt=[]
 table+=['','## 同前缀续接','', '下表为安全完成次数/执行次数。remaining是原300步总上限；equal300是每检查点给相同300步续接预算。两者均保留失败、碰撞及原AEGIS预算对照。','', '|状态|检查点|预算|原AEGIS|参考|原策略|释放5步|提升后原策略|','|---|---:|---|---:|---:|---:|---:|---:|']
 for state in STATES:
  for step in CHECKPOINTS:
   for extra in sorted(set([0,step])):
    g=[r for r in branches if r['state']==state['id'] and r['branch_step']==step and r['extra_budget']==extra]
    if not g:continue
    counts={c:count([r for r in g if r['condition']==c]) for c in ['raw']+CANDIDATES}
    budget='remaining' if extra==0 else 'equal300'
    bt.append(dict(state=state['id'],step=step,budget=budget,**counts))
    table.append('|%s|%d|%s|%s|%s|%s|%s|%s|'%(state['id'],step,budget,*[counts[c] for c in ['raw']+CANDIDATES]))
 if not branches:table+=['','没有满足分叉前提的状态，或分叉阶段尚未执行。不能把缺少续接结果写成候选失败。']
 if branches:
  bs=sorted({r['state'] for r in branches});fig,axes=plt.subplots(len(bs),2,figsize=(11,3*len(bs)),squeeze=False)
  for i,sid in enumerate(bs):
   for j,budget in enumerate(['remaining','equal300']):
    values=np.full((len(CANDIDATES)+1,len(CHECKPOINTS)),np.nan)
    for a,c in enumerate(['raw']+CANDIDATES):
     for b,step in enumerate(CHECKPOINTS):
      g=[r for r in branches if r['state']==sid and r['condition']==c and r['branch_step']==step and r['extra_budget']==(0 if budget=='remaining' else step)]
      if g:values[a,b]=sum(r['safe_success'] for r in g)/len(g)
    ax=axes[i,j];ax.imshow(values,vmin=0,vmax=1,cmap='YlGnBu');ax.set_xticks(range(len(CHECKPOINTS)));ax.set_xticklabels(CHECKPOINTS);ax.set_yticks(range(len(CANDIDATES)+1));ax.set_yticklabels(['raw']+CANDIDATES);ax.set_title(sid+' / '+budget);ax.set_xlabel('Common-prefix checkpoint / actions')
  fig.tight_layout();fig.savefig(out/'continuation_opportunities.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(len(STATES),3,figsize=(10,3.1*len(STATES)))
 for i,state in enumerate(STATES):
  choices=[next((r for r in initial if r['state']==state['id'] and r['condition']=='raw'),None),next((r for r in initial if r['state']==state['id'] and r['condition']=='identity_geometry'),None),next((r for r in initial if r['state']==state['id'] and r['condition']=='reference' and r['validation'] and r['safe_success']),None)]
  for j,record in enumerate(choices):
   ax=axes[i,j];ax.axis('off');ax.set_title(state['id']+' / '+['raw AEGIS','identity + geometry','validated witness'][j],fontsize=9)
   if record:
    images=sorted((root/'runs'/record['run_id']).glob('frame_*.jpg'))
    if images:ax.imshow(plt.imread(images[-1]));ax.text(.01,.01,'safe success='+str(record['safe_success']),transform=ax.transAxes,color='white',bbox=dict(facecolor='black',alpha=.6),fontsize=8)
   else:ax.text(.5,.5,'No validated witness yet' if j==2 else 'Pending',ha='center')
 fig.tight_layout();fig.savefig(out/'execution_examples.jpg',dpi=120);plt.close(fig)

 discrepancies=[]
 for p in (root/'runs').glob('*/policy.jsonl'):
  for l in p.read_text().splitlines():
   r=json.loads(l)
   if r['native_output_linf_difference']>0:discrepancies.append(r)
 table+=['','## 执行可比性与解释边界','',
 '全部已完成条件均通过首观察与受控首chunk相等检查；原生同输入推理非零差异记录数：%d，最大L∞=%g。'%(len(discrepancies),max([x['native_output_linf_difference'] for x in discrepancies] or [0])),
 '续接前缀比较模拟器state/qpos/qvel/ctrl/warmstart、controller、marker、动作队列、策略RNG、Python/NumPy RNG、AEGIS状态与观察。任何不一致都停止，不删除后重新选结果。',
 '参考验证用10个新策略种子；参考自身确定性，因此不是10个独立物理扰动样本，不能由10/10或4/5认证高概率成功。找到一次安全动作执行是可行性见证；找不到见证仍是未知。','',
 '## 路线判断','', '|证据|研究决策|','|---|---|',
 '|身份/几何修正后多数完成|优先感知或代理表示|',
 '|独立安全见证、修正后仍失败、固定候选重复有效|方向二：选择或介入时机二选一；先报告次数与验证限制|',
 '|早期见证，等额续接预算下后期机会仍下降|方向一可加强；仍须区分参考能力与真实不可行|',
 '|见证未找到且有限候选失败|未知；不训练无解分类器|']
 decision=[]
 for s in stats:
  if s['role']!='diagnostic':continue
  corrected=s['identity_geometry'];raw=s['raw'];ref=s['reference']
  if corrected['n'] and corrected['safe_success']>corrected['n']/2 and corrected['safe_success']>raw['safe_success']:route='感知/椭球代理优先'
  elif ref['n'] and ref['safe_success'] and corrected['safe_noncompletion']:
   successes=[r for r in branches if r['state']==s['state'] and r['safe_success'] and r['condition']!='raw']
   route='已有见证；续接出现有效候选，需按重复与等预算检验方向二/一' if successes else '已有见证但固定续接尚无有效证据；保留未知'
  else:route='未获得独立安全见证；可行性未知'
  decision.append({'state':s['state'],'decision':route});table.append('\n- %s：%s。'%(s['state'],route))
 (out/'REPORT.md').write_text('\n'.join(table)+'\n')
 (out/'statistics.json').write_text(json.dumps({'initial':stats,'branches':bt,'decision':decision,'native_discrepancies':discrepancies},ensure_ascii=False,indent=2))
 import html
 text=html.escape('\n'.join(table))
 (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>安全未完成诊断</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto;line-height:1.6}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>安全未完成状态的可行性见证与受控续接实验</h1><img src="perception_conditions.png"><img src="safety_traces.png"><img src="execution_examples.jpg"><pre>'+text+'</pre>')
 return stats
if __name__=='__main__':
 import sys
 report(Path(sys.argv[1]))
