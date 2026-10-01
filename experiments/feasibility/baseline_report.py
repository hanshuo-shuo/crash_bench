"""Quest-side prior evidence dossier; never present these as new experiments."""
import csv,hashlib,json,os,sys
from pathlib import Path
from protocol import STATES

def main(source,out):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 source=Path(source);out=Path(out);out.mkdir(exist_ok=True)
 data=list(csv.DictReader(source.open()));records=[]
 text=['# 固定诊断集的既有证据','', '以下均来自已完成的600次配对实验，不是本轮可行性见证或受控续接结果。',
 '六个诊断状态及两个正常执行对照已固定。本轮将分开检验身份/几何，再以独立普通机器人动作寻找安全完成见证。','',
 '|状态|角色|π0.5 完成|π0.5 碰撞|π0.5 安全完成|AEGIS 完成|AEGIS 碰撞|AEGIS 安全未完成|配对首chunk差异|',
 '|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for state in STATES:
  scenario='%s/%s/task%d'%(state['suite'],state['level'],state['task'])
  group=[r for r in data if r['scenario']==scenario and int(r['episode'])==state['episode']]
  pairs={m:sorted([r for r in group if r['method']==m],key=lambda r:int(r['repeat'])) for m in ['nominal','aegis']}
  if any(len(g)!=5 for g in pairs.values()):raise RuntimeError('Prior evidence incomplete')
  counts={}
  for m,g in pairs.items():
   counts[m]={'success':sum(r['success']=='True' for r in g),'collision':sum(r['collided']=='True' for r in g),
    'safe_success':sum(r['success']=='True' and r['collided']=='False' for r in g),'safe_noncompletion':sum(r['success']=='False' and r['collided']=='False' for r in g),'n':5}
  mismatches=sum(a['first_action_chunk_sha256']!=b['first_action_chunk_sha256'] for a,b in zip(pairs['nominal'],pairs['aegis']))
  record={'state':state['id'],'role':state['role'],'obstacle':pairs['aegis'][0]['active_obstacle'],'counts':counts,'first_chunk_mismatches':mismatches,'repeat0_vlm':pairs['aegis'][0]['vlm_object']};records.append(record)
  a,b=counts['nominal'],counts['aegis'];text.append('|%s|%s|%d/5|%d/5|%d/5|%d/5|%d/5|%d/5|%d/5|'%(state['id'],state['role'],a['success'],a['collision'],a['safe_success'],b['success'],b['collision'],b['safe_noncompletion'],mismatches))
 fig,ax=plt.subplots(figsize=(12,5));x=np.arange(len(STATES));width=.35
 for offset,m in [(-width/2,'nominal'),(width/2,'aegis')]:
  bottom=np.zeros(len(STATES))
  for category,color in [('safe_success','#56a88c'),('safe_noncompletion','#d6b65c'),('collision','#b96259')]:
   values=np.array([r['counts'][m][category] for r in records]);ax.bar(x+offset,values,width,bottom=bottom,color=color,label=(category if m=='nominal' else None));bottom+=values
 ax.set_xticks(x);ax.set_xticklabels([s['id'] for s in STATES],rotation=25,ha='right');ax.set_ylim(0,5.4);ax.set_ylabel('Observed episodes / 5');ax.set_title('PRIOR evidence only: left pi0.5, right AEGIS');ax.legend(loc='upper left',bbox_to_anchor=(1,1));fig.tight_layout();fig.savefig(out/'prior_outcomes.png',dpi=180);plt.close(fig)
 text+=['','![既有状态结果](prior_outcomes.png)','',
 'Spatial 3、9、15 的旧首chunk差异不能靠删除重复消除。本轮记录首观察输入、原生chunk及RNG，并在共同输入/RNG上冻结受控输出。',
 '六个诊断状态的既有repeat0 VLM均正确识别黑色酒瓶，身份修正可能是空操作；几何条件仍有诊断价值。',
 '正常对照来自两方法均5/5安全完成的状态，但Object II/2既有repeat0身份回答错误，因此也能检查身份错误是否在该状态改变行为。',
 '本轮预检已核验相同机器人动作尺度：平移0.05 m/action、旋转0.5 rad/action，普通7D OSC/gripper。',
 '安全见证、新验证、修正几何后的行为及续接机会尚待新GPU作业。现有数据不回答这些问题。']
 (out/'PRIOR_REPORT.md').write_text('\n'.join(text)+'\n')
 (out/'prior_evidence.json').write_text(json.dumps({'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'prior_runs':600,'selected':records,'plot_job':os.environ['SLURM_JOB_ID']},ensure_ascii=False,indent=2))
 (out/'prior_report.html').write_text('<meta charset="utf-8"><title>既有诊断证据</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto;line-height:1.7}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>固定诊断集：既有证据</h1><img src="prior_outcomes.png"><pre>'+__import__('html').escape('\n'.join(text))+'</pre>')
if __name__=='__main__':main(sys.argv[1],sys.argv[2])
