"""Provisional report and explicitly post-hoc task/class context; never fits a model."""
import argparse
import base64
import csv
import html
import json
import os
from pathlib import Path
from collections import defaultdict
from common import atomic_json,sha
from report import read,verify,find,estimate,display,number
from stat_rules import grouped_weights,classification_metrics,weighted_auc


def prevalence(rows):
    if not rows:return None
    w=grouped_weights([r['scene'] for r in rows],[r['id'] for r in rows])
    return sum(a*int(r['y']) for a,r in zip(w,rows))/sum(w)


def stratify(predictions,models):
    output=[]
    for stage in ['first_infer','first_five_landmark']:
        for proxy in ['disagreement','churn','act_norm','min_dist']:
            rows=[r for r in predictions if r['stage']==stage and r['proxy']==proxy]
            train=[r for r in rows if r['split']=='train'];test=[r for r in rows if r['split']=='test']
            tasks=sorted({r['scene'].rsplit('/',1)[0] for r in rows})
            model=models.get('completion/'+stage+'/'+proxy,{})
            global_prior=model.get('prevalence')
            priors={t:prevalence([r for r in train if r['scene'].rsplit('/',1)[0]==t]) for t in tasks}
            for task in ['pooled']+tasks:
                for arm in ['pooled','nominal','aegis']:
                    chosen=[r for r in test if (task=='pooled' or r['scene'].rsplit('/',1)[0]==task) and (arm=='pooled' or r['arm']==arm)]
                    w=grouped_weights([r['scene'] for r in chosen],[r['id'] for r in chosen]) if chosen else []
                    y=[int(r['y']) for r in chosen];x=[float(r['x']) for r in chosen];p=[float(r['probability']) for r in chosen]
                    metrics=classification_metrics(y,x,p,w)
                    weighted_prev=sum(a*b for a,b in zip(w,y))/sum(w) if w else None
                    def weighted_loss(predict):
                        if not w or any(v is None for v in predict):return None
                        return sum(a*(b-c)**2 for a,b,c in zip(w,predict,y))/sum(w)
                    conditional=[]
                    for label in [0,1]:
                        selected=[i for i,v in enumerate(y) if v==label];mass=sum(w[i] for i in selected)
                        conditional.append(sum(w[i]*(p[i]-y[i])**2 for i in selected)/mass if mass else None)
                    output.append(dict(stage=stage,proxy=proxy,task=task,arm=arm,rows=len(chosen),states=len({r['scene'] for r in chosen}),positive=sum(y),negative=len(y)-sum(y),
                        weighted_positive_prevalence=weighted_prev,**metrics,calibrated_AUROC=weighted_auc(y,p,w),Brier_positive=conditional[1],Brier_negative=conditional[0],
                        global_train_prior_Brier=weighted_loss([global_prior]*len(chosen)),
                        task_train_prior_Brier=weighted_loss([priors[r['scene'].rsplit('/',1)[0]] for r in chosen]),
                        scope='post_hoc_descriptive_no_new_fit_no_CI; original primary metrics unchanged'))
    return output


def main(task2,output,gate_receipt):
    complete=verify(task2,'ANALYSIS_COMPLETE.json')
    if complete['stage']!='full' or complete['test_rollouts']!=360:raise RuntimeError('Complete fixed Task2 required')
    models=json.loads((task2/'PLATT_MODELS.json').read_text());predictions=read(task2/'completion_predictions.csv')
    context=stratify(predictions,models);output.mkdir(exist_ok=False)
    from analyze import write_csv
    write_csv(output/'completion_task_class_context.csv',context)
    collision=read(task2/'collision_metrics.csv');completion=read(task2/'completion_metrics.csv');calibration=read(task2/'calibration_curve.csv')
    outcomes=read(task2/'outcomes.csv');failure=read(task2/'confident_failure.csv');references=read(task2/'confident_failure_reference.csv')
    plan=json.loads(gate_receipt.read_text());science=json.loads((task2/'PROVENANCE.json').read_text())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    figures=[]
    with PdfPages(output/'PROVISIONAL_FIGURES.pdf') as pdf:
        def save(fig,name):
            fig.tight_layout();fig.savefig(output/(name+'.png'),dpi=150,bbox_inches='tight');pdf.savefig(fig,bbox_inches='tight');plt.close(fig);figures.append(name)
        fig,ax=plt.subplots(figsize=(7,4))
        for proxy,color in [('disagreement','#2563eb'),('churn','#7c3aed'),('act_norm','#d97706'),('min_dist','#059669')]:
            rows=[find(collision,proxy=proxy,arm='pooled',metric='AUROC',horizon_actions=h) for h in [5,10,20,40]]
            ax.plot([5,10,20,40],[number(r) for r in rows],'o-',label=proxy,color=color);ax.fill_between([5,10,20,40],[number(r,'low') for r in rows],[number(r,'high') for r in rows],alpha=.1,color=color)
        ax.axhline(.5,color='#94a3b8',ls='--');ax.set(xlabel='Future actions, includes current action',ylabel='Raw-direction collision AUROC',title='Task2 held-out collision forecast');ax.legend();save(fig,'collision_auroc')
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,stage in zip(axes,['first_infer','first_five_landmark']):
            rows=[r for r in calibration if r['stage']==stage and r['proxy']=='disagreement' and r['arm']=='pooled']
            ax.plot([0,1],[0,1],'--',color='#94a3b8');ax.plot([number(r,'predicted') for r in rows],[number(r,'observed') for r in rows],'o-')
            ax.vlines([number(r,'predicted') for r in rows],[number(r,'observed_low') for r in rows],[number(r,'observed_high') for r in rows])
            ax.set(xlim=(0,1),ylim=(0,1),xlabel='Fixed train-fit safe-success probability',ylabel='Observed safe-success probability',title=stage.replace('_',' '))
        save(fig,'calibration')
        tasks=sorted({r['task'] for r in context if r['task']!='pooled' and r['proxy']=='disagreement'})
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for stage,color in [('first_infer','#2563eb'),('first_five_landmark','#d97706')]:
            rows=[find(context,stage=stage,proxy='disagreement',arm='pooled',task=t) for t in tasks]
            axes[0].plot(range(len(tasks)),[r['AUROC'] for r in rows],'o-',color=color,label=stage)
            axes[1].plot(range(len(tasks)),[r['weighted_positive_prevalence'] for r in rows],'o-',color=color,label=stage)
        labels=[t.replace('safelibero_','') for t in tasks]
        for ax in axes:ax.set(xticks=range(len(tasks)),xticklabels=labels,ylim=(0,1));ax.tick_params(axis='x',labelrotation=15);ax.legend(fontsize=8)
        axes[0].axhline(.5,color='#94a3b8',ls='--');axes[0].set(ylabel='Within-task raw AUROC',title='Post-hoc descriptive, no new fit / CI')
        axes[1].set(ylabel='Weighted safe-success prevalence',title='Task mixture and class imbalance')
        save(fig,'task_context')
        fig,axes=plt.subplots(1,3,figsize=(12,3.5))
        for ax,task in zip(axes,tasks):
            rows=[r for r in predictions if r['stage']=='first_infer' and r['proxy']=='disagreement' and r['split']=='test' and r['scene'].rsplit('/',1)[0]==task]
            pooled_values=[-float(r['x']) for r in rows]
            lo,hi=min(pooled_values),max(pooled_values)
            if lo==hi:lo,hi=lo-1e-12,hi+1e-12
            bin_edges=[lo+(hi-lo)*i/12 for i in range(13)]
            for label,color,name in [(0,'#64748b','not safe success'),(1,'#0891b2','safe success')]:
                values=[-float(r['x']) for r in rows if int(r['y'])==label]
                if values:ax.hist(values,bins=bin_edges,color=color,alpha=.55,label=name+' n='+str(len(values)))
            ax.set(xlabel='First-infer disagreement',ylabel='Raw rollout count',title=task.replace('safelibero_',''));ax.legend(fontsize=7)
        save(fig,'first_infer_by_task_and_class')

    blocks=[];md=[]
    def h(title):blocks.append('<h2>'+html.escape(title)+'</h2>');md.append('## '+title+'\n')
    def p(value):blocks.append('<p>'+html.escape(value)+'</p>');md.append(value+'\n')
    def table(headers,rows):
        blocks.append('<table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>')
        md.append('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in rows)+'\n')
    def image(name,caption):
        data=base64.b64encode((output/(name+'.png')).read_bytes()).decode()
        blocks.append('<figure><img src="data:image/png;base64,'+data+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>');md.append('!['+caption+']('+name+'.png)\n\n'+caption+'\n')
    h('状态：Task 3 待完成')
    p('这份临时报告仅包含已完整采集并独立复算的 Task1/2 结果及描述性任务/类别分层。Task3 gate 尚未形成完整结果。任何临时图表均不代表gate效果；完整报告需要300条gate轨迹通过审计后生成。')
    p('Task1 是60固定初始状态×nominal/full AEGIS×10 seed，1200条新rollout。Task2冻结42/18状态切分，每任务14/6状态；主留出N=180/臂。全部19个输出，包括合格集、预测、指标及2000-draw状态bootstrap区间，已独立复算。')
    table(['留出N=180/臂','CAR 无碰撞率','TSR 原任务成功率','安全完成率'],[[arm]+[estimate(find(outcomes,population='test_primary',arm=arm,metric=m)) for m in ['CAR','TSR','safe_success']] for arm in ['nominal','aegis']])
    h('短期碰撞预警')
    p('预定义原方向的U碰撞预测AUROC在h5/10/20/40为.541/.568/.559/.533，全部95%CI覆盖.5，结果未支持可靠短期碰撞预警。h是环境动作数，包含当前动作。在险集包括首次碰撞动作，但排除此前已撞的时点；TTC=-1代表从未碰撞。距离风险的较强排序不等于官方crash标签。')
    image('collision_auroc','Task2主留出碰撞AUROC与固定状态bootstrap区间；原方向AUROC和calibrated_AUROC分别保存在原CSV。')
    h('初始安全完成排序、校准与Brier')
    table(['landmark / proxy','原方向AUROC [95% CI]','BA [95% CI]','Brier [95% CI]','合格数'],
        [[stage+' / '+proxy]+[estimate(find(completion,stage=stage,proxy=proxy,arm='pooled',metric=m)) for m in ['AUROC','BA','Brier']]+[find(completion,stage=stage,proxy=proxy,arm='pooled',metric='AUROC')['rollouts']] for stage in ['first_infer','first_five_landmark'] for proxy in ['disagreement','churn','act_norm','min_dist']])
    p('首次infer的低U安全完成AUROC=.829，描述的是跨三个已知任务、留出状态的排序。前五次landmark AUROC=.779，只纳入动作21之前仍合格的291/360轨迹，是条件关联。首次churn没有先前chunk，AUROC/BA/Brier为NA；不将其当成零分数。安全完成概率由冻结训练集1D Platt拟合；BA使用p≥.5，Brier是概率误差。')
    image('calibration','U的固定训练校准与留出观察率；非空固定箱的类别数、状态数与CI见原calibration_curve.csv。')
    h('任务与类别分层：事后描述性解释')
    p('下面分层和类频率参照是为解释合并AUROC而追加的描述性报告，不改变冻结主分析，也不另拟合校准器、选择proxy、阈值或新增显著性检验。任务/臂分层可能只有极少成功或失败；单类AUROC/BA记NA，Brier仍可定义。')
    def context_rows(arm='pooled'):
        return [r for r in context if r['proxy']=='disagreement' and r['arm']==arm and r['task']!='pooled']
    table(['landmark','任务','N/状态','安全成功/其余','加权成功率','原方向AUROC','BA','Brier','成功类Brier','其余类Brier'],
        [[r['stage'],r['task'],str(r['rows'])+'/'+str(r['states']),str(r['positive'])+'/'+str(r['negative'])]+[display(r,k) for k in ['weighted_positive_prevalence','AUROC','BA','Brier','Brier_positive','Brier_negative']] for r in context_rows()])
    table(['landmark','任务/臂','N','安全成功/其余','原方向AUROC','BA','Brier'],
        [[r['stage'],r['task']+'/'+r['arm'],str(r['rows']),str(r['positive'])+'/'+str(r['negative'])]+[display(r,k) for k in ['AUROC','BA','Brier']] for r in context if r['proxy']=='disagreement' and r['task']!='pooled' and r['arm']!='pooled'])
    p('Object/I测试首次infer仅1/120条安全成功，其中nominal是0/60。该任务AUROC由一个阳性支撑，不适合解释为稳定预测能力。Object/II首次U任务内AUROC=.597，Spatial为.702，低于合并.829。任务类别间的基率和U分布差异会影响合并排序，不能把合并AUROC直接解释为每个任务内都同样有效。')
    image('task_context','事后任务内排序及加权安全完成基率；未新增CI，类别数在表中列出。')
    image('first_infer_by_task_and_class','首次U按任务与最终类别的原始轨迹数分布；同任务两类共用分箱。这是计数图，不是重新加权的模型训练。')
    table(['landmark（合并）','U固定Platt Brier','全局TRAIN类频率常量Brier','任务TRAIN类频率参照Brier'],
        [[stage]+[display(find(context,stage=stage,proxy='disagreement',task='pooled',arm='pooled'),k) for k in ['Brier','global_train_prior_Brier','task_train_prior_Brier']] for stage in ['first_infer','first_five_landmark']])
    p('类频率参照只由对应landmark的TRAIN标签计算，按原等状态/合格轨迹权重、共享两臂；测试标签只用于计算Brier，不决定概率。它是事后背景参照而非新增主模型。首次U Brier=.175优于全局TRAIN常量.239，但未低于任务TRAIN参照.173；前五次分别.191/.240/.176。这使高合并AUROC不能单独作为可迁移、增量安全判断能力的证据。以上均是描述性点估计，未据此调整gate。')
    h('低proxy失败的覆盖与训练参考')
    table(['臂','K动作','低/合格','比例 [95% CI]','测试碰撞总数','不完整窗口排除','无参考排除'],
        [[r['arm'],r['K_actions'],r['n_low']+'/'+r['eligible'],estimate(r),r['total_test_crash'],r['excluded_incomplete_window'],r['excluded_missing_reference']] for r in failure if r['task']=='pooled' and r['weighting']=='raw_rollout_proportion'])
    p('nominal K5是43/60合格、总120碰撞，60条缺同任务同臂TRAIN安全成功参考，未计负例。K10=42/60，K20=39/52；AEGIS=60/68、60/68、56/67。Object/I nominal参考NA；AEGIS参考仅2状态/4条成功轨迹。CI条件于固定训练参考，不含重估这些小样本参考的不确定性。')
    table(['TRAIN任务','臂','K动作','合格成功轨迹/状态','保存中位数'],
        [[r['task'],r['arm'],r['K_actions'],r['eligible_rollouts']+'/'+r['eligible_states'],display(r,'reference_median',6)] for r in references])
    p('窗口是碰撞C之前[C−K,C−1]或安全成功L之前[L−K,L−1]，不含事件动作。一个轻微浮点半权重约定使Spatial AEGIS K20保存参考.11642515466175145与精确下中位数.11479897786603213不同，零测试低proxy判定改变，gate q90不受影响；原输出保留，详细复现见独立审计附注。')
    h('Task3固定规则、smoke与两作业计划')
    p('gate使用原始M8动作采样disagreement，阈值为TRAIN nominal在险infer的全局加权q90='+repr(plan['configuration']['gate_threshold'])+'，来自42状态×10 seed、5590条在险infer；等状态→rollout→infer权重。规则是严格U>阈值才把当前五动作队列的七维环境命令全置零，正常消耗队列并从新观测replan。阈值相等时不gate。它不使用Platt概率，不是训练得到的新控制器。')
    p('总60状态×repeat0–4=300 gate rollout，偶/奇状态分两个150条作业。首作业9321091包含固定smoke：Spatial/I/task1/episode00、Object/II/task1/episode02，各repeat0/1共4条。这些是TRAIN/已暴露工程状态，不覆盖Object/I；smoke不要求自然激活gate。真实激活计数将在smoke/full收据列出。')
    p('首作业继续完成其150条；四条smoke的M8独立归约、原生动作/RNG严格不扰动、与Task1相同初始物理数组等验收通过后，Quest tmux监督器只提交一次另一150条作业。失败/PARTIAL写STOP并仅取消own gate作业。每个作业p33100/gengpu、1A100、8CPU、64GB、6h；先前106366GPU秒+本阶段最大43200秒=149566秒=41.55GPUh，批准48h。72小时仅是队列监督墙钟，不增加GPU时限。')
    p('OSC零位置增量跟踪当前位置，零旋转增量保留旧目标；gripper零命令保留累积执行器目标，动力学/力矩继续，不保证机械静止。确定性合成工程测试会执行冻结队列循环验证归零、消耗、replan；它不代替实际smoke的控制器字节与运行验收。')
    p('关闭Codex桌面或SSH连接不终止已核验detached Quest tmux监督和Slurm。Mac侧原始包下载进程PPID=1，但Mac睡眠/断网可中断传输；保留部分文件并用现有授权连接续传。监督器写阶段收据，聊天通知由监督任务读取后发出。')
    h('方法、成本和证据')
    p('π0.5为flow matching：M8/H10/k5，有效7维，训练资产尺度归一化，跨样本ddof1标准差再对时间/维度平均。共享观测前缀KV，但八个suffix使用原生B1顺序kernel，不宣称八路并行加速。连续state向量未作为固定pi05模型条件；图像+prompt行为保持原样。每infer严格同输入/RNG动作相等证明非扰动，不代表跨新环境渲染一致或零耗时；额外诊断/native耗时比约3.33/3.36，额外严格reference/native约.97/.99。')
    p('官方crash是保护物体L1平移>1mm，不等同接触或负signed distance；success原判定保留，crash优先outcome但TSR保留原成功。min_dist几何核只读且独立验证，act_norm是七维环境命令范数而非速度/力。nominal20步快照仅保存，不从中途分支或宣称完整恢复。95%CI按任务分层状态2000次PCG64 seed20261009，保留抽样multiplicity及原权重，不包含训练拟合/参考重估不确定性。')
    p('Task1采集 '+science['collection_commit']+'；Task2分析 '+science['analysis_code_commit']+'，作业 '+str(science['slurm_job'])+'。原采集计划SHA '+science['collection_source_plan_sha256']+'；最终分析计划SHA '+science['analysis_plan_sha256']+'；输入ParquetSHA '+science['input_parquet_sha256']+'。全部原输出和来源身份保留。本报告代码 '+str(os.environ.get('CB_CODE_COMMIT'))+'，CPU作业 '+str(os.environ.get('SLURM_JOB_ID'))+'。')
    p('系列已结算API $1.17709184（600完整AEGIS+4此前uncertainty smoke），gate与报告无新增API；原始远端证据 '+science['raw_evidence_location']+'。本地包完整性以各自DELIVERY_STATUS/LOCAL_EVIDENCE_VERIFIED.json为准；临时报告不宣称未完成tar已本地验完。')
    title='临时实验报告：Task1/2已验收，Task3待完成'
    style='body{font-family:system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;line-height:1.7;color:#172033}h1,h2{line-height:1.3}h2{margin-top:2em}table{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow:auto}th,td{border:1px solid #dbe2ea;padding:8px}th{background:#eff4f8}img{width:100%;max-width:1000px}figure{margin:24px 0}figcaption{font-size:13px;color:#475569}p{overflow-wrap:anywhere}@media print{figure,table{break-inside:avoid}}'
    (output/'PROVISIONAL_REPORT.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="UTF-8"><title>'+title+'</title><style>'+style+'</style><main><h1>'+title+'</h1>'+''.join(blocks)+'</main></html>')
    (output/'PROVISIONAL_REPORT.md').write_text('# '+title+'\n\n'+'\n'.join(md))
    atomic_json(output/'PROVISIONAL_COMPLETE.json',dict(passed=True,task3_pending=True,task3_actual_results_included=False,task2_complete_sha256=sha(task2/'ANALYSIS_COMPLETE.json'),gate_receipt_sha256=sha(gate_receipt),
        code_commit=os.environ.get('CB_CODE_COMMIT'),slurm_job=os.environ.get('SLURM_JOB_ID'),scope='fixed original Task2 primary outputs + post-hoc descriptive task/class context; no new fit/tuning/rollout/API',files={f.name:sha(f) for f in output.iterdir() if f.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task2',type=Path);p.add_argument('output',type=Path);p.add_argument('gate_receipt',type=Path);a=p.parse_args();main(a.task2.resolve(),a.output.resolve(),a.gate_receipt.resolve())
