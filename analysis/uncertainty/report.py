"""Render frozen Task2/Task3 tables into an illustrated Chinese report; no fitting."""
import argparse
import base64
import csv
import html
import json
from pathlib import Path
from common import atomic_json,sha

LABELS={'disagreement':'Disagreement','churn':'Shifted chunk churn','act_norm':'Applied command norm','min_dist':'Nearest distance risk'}
COLORS={'disagreement':'#2563eb','churn':'#7c3aed','act_norm':'#d97706','min_dist':'#059669'}
ARMCOLORS={'nominal':'#64748b','aegis':'#0891b2','gate':'#c026d3'}


def read(path):return list(csv.DictReader(path.open()))


def number(row,key='value'):return float(row[key]) if row.get(key) not in ['',None] else None


def display(row,key='value',digits=3):
    value=number(row,key);return 'NA' if value is None else format(value,'.'+str(digits)+'f')


def estimate(row):
    return display(row)+' ['+display(row,'low')+', '+display(row,'high')+']'


def find(rows,**keys):
    selected=[r for r in rows if all(r.get(k)==str(v) for k,v in keys.items())]
    if len(selected)!=1:raise RuntimeError('Ambiguous/missing report cell: '+str(keys))
    return selected[0]


def verify(directory,receipt):
    complete=json.loads((directory/receipt).read_text())
    if not complete['passed']:raise RuntimeError('Verified completed analysis required')
    for name,digest in complete['files'].items():
        if sha(directory/name)!=digest:raise RuntimeError('Changed analysis output '+name)
    return complete


def render(task2,gate,output):
    c2=verify(task2,'ANALYSIS_COMPLETE.json');c3=verify(gate,'GATE_ANALYSIS_COMPLETE.json')
    if c2['test_rollouts']!=360 or c3['gate_rollouts']!=300 or c3['test_rollouts_per_arm']!=90:raise RuntimeError('Final complete matrices required')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
    output.mkdir(exist_ok=False)
    collision=read(task2/'collision_metrics.csv');completion=read(task2/'completion_metrics.csv')
    temporal=read(task2/'temporal_curves.csv');calibration=read(task2/'calibration_curve.csv')
    failure=read(task2/'confident_failure.csv');outcomes=read(task2/'outcomes.csv')
    gate_out=read(gate/'outcomes.csv');delta=read(gate/'paired_differences.csv');frequency=read(gate/'gate_frequencies.csv')
    p2=json.loads((task2/'PROVENANCE.json').read_text());p3=json.loads((gate/'PROVENANCE.json').read_text())
    figures=[]
    with PdfPages(output/'figures.pdf') as pdf:
        def save(fig,name):
            fig.tight_layout();fig.savefig(output/(name+'.png'),bbox_inches='tight');pdf.savefig(fig,bbox_inches='tight');plt.close(fig);figures.append(name)
        fig,ax=plt.subplots(figsize=(7.7,4.6))
        for proxy in LABELS:
            rows=[find(collision,arm='pooled',proxy=proxy,metric='AUROC',horizon_actions=h) for h in [5,10,20,40]]
            x=np.asarray([5,10,20,40]);y=np.asarray([number(r) for r in rows]);lo=np.asarray([number(r,'low') for r in rows]);hi=np.asarray([number(r,'high') for r in rows])
            ax.plot(x,y,'o-',label=LABELS[proxy],color=COLORS[proxy]);ax.fill_between(x,lo,hi,color=COLORS[proxy],alpha=.11)
        ax.axhline(.5,ls='--',color='#94a3b8');ax.set(xlabel='Future window (environment actions, current action included)',ylabel='Raw-direction collision AUROC',ylim=(.25,1.02),xticks=[5,10,20,40],title='Held-out pre-action at-risk inference boundaries')
        ax.legend(fontsize=8);save(fig,'collision_prediction')
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for stage,ax in zip(['first_infer','first_five_landmark'],axes):
            for i,proxy in enumerate(LABELS):
                row=find(completion,arm='pooled',proxy=proxy,metric='AUROC',stage=stage);v=number(row)
                if v is None:
                    ax.text(i,.08,'NA',ha='center',color=COLORS[proxy]);continue
                ax.plot(i,v,'o',color=COLORS[proxy]);ax.vlines(i,number(row,'low'),number(row,'high'),color=COLORS[proxy])
            ax.axhline(.5,ls='--',color='#94a3b8');ax.set(xticks=range(4),xticklabels=['U','Churn','Norm','Distance'],ylim=(0,1),ylabel='Safe-completion AUROC',title=stage.replace('_',' '))
        save(fig,'completion_prediction')
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for stage,ax in zip(['first_infer','first_five_landmark'],axes):
            rows=[r for r in calibration if r['stage']==stage and r['proxy']=='disagreement' and r['arm']=='pooled']
            x=[number(r,'predicted') for r in rows];y=[number(r,'observed') for r in rows]
            ax.plot([0,1],[0,1],'--',color='#94a3b8');ax.plot(x,y,'o-');ax.vlines(x,[number(r,'observed_low') for r in rows],[number(r,'observed_high') for r in rows])
            ax.set(xlim=(0,1),ylim=(0,1),xlabel='Train-only Platt predicted safe completion',ylabel='State-balanced observed safe completion',title=stage.replace('_',' '))
        save(fig,'completion_calibration')
        fig,axes=plt.subplots(2,2,figsize=(10,7))
        for proxy,ax in zip(LABELS,axes.flat):
            for arm in ['nominal','aegis']:
                rows=[r for r in temporal if r['population']=='test_primary' and r['arm']==arm and r['proxy']==proxy and number(r) is not None]
                x=[number(r,'step') for r in rows];y=[number(r) for r in rows]
                ax.plot(x,y,label=arm,color=ARMCOLORS[arm]);ax.fill_between(x,[number(r,'low') for r in rows],[number(r,'high') for r in rows],color=ARMCOLORS[arm],alpha=.15)
            ax.set(xlabel='Environment action index',ylabel=LABELS[proxy],title=LABELS[proxy]);ax.legend()
        save(fig,'at_risk_time_curves')
        fig,axes=plt.subplots(1,3,figsize=(11,4))
        for metric,ax in zip(['CAR','TSR','safe_success'],axes):
            for i,arm in enumerate(['nominal','aegis','gate']):
                row=find(gate_out,population='test_primary',arm=arm,metric=metric);v=number(row)
                ax.bar(i,v,color=ARMCOLORS[arm]);ax.vlines(i,number(row,'low'),number(row,'high'),color='#0f172a')
            ax.set(xticks=range(3),xticklabels=['Nominal','AEGIS','Gate'],ylim=(0,1.05),ylabel='Probability',title=metric.replace('_',' '))
        fig.suptitle('Task3: 18 held-out states × 5 matched seeds per arm',y=1.02);save(fig,'gate_outcomes')
        fig,ax=plt.subplots(figsize=(8,4))
        labels=[]
        for i,(reference,metric) in enumerate((r,m) for r in ['nominal','aegis'] for m in ['CAR','TSR','safe_success']):
            row=find(delta,population='test_primary',comparison='gate-minus-'+reference,metric=metric);v=number(row)
            ax.plot(v,i,'o',color=ARMCOLORS[reference]);ax.hlines(i,number(row,'low'),number(row,'high'),color=ARMCOLORS[reference]);labels.append('Gate − '+reference+': '+metric)
        ax.axvline(0,color='#94a3b8',ls='--');ax.set(yticks=range(len(labels)),yticklabels=labels,xlabel='Paired probability difference, 95% state-cluster interval')
        save(fig,'gate_paired_differences')

    blocks=[];markdown=[]
    def heading(title):blocks.append('<h2>'+html.escape(title)+'</h2>');markdown.append('## '+title+'\n')
    def paragraph(value):blocks.append('<p>'+html.escape(value)+'</p>');markdown.append(value+'\n')
    def table(headers,rows):
        blocks.append('<table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>')
        markdown.append('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in rows)+'\n')
    def figure(name,caption):
        encoded=base64.b64encode((output/(name+'.png')).read_bytes()).decode()
        blocks.append('<figure><img src="data:image/png;base64,'+encoded+'" alt="'+html.escape(caption)+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>')
        markdown.append('!['+caption+']('+name+'.png)\n\n'+caption+'\n')
    heading('实验范围与主要结果')
    paragraph('Task 1 已完整采集 60 个固定初始状态 × 2 臂 × 10 seed，共 1,200 条新 rollout；Task 2 使用按状态冻结的 42/18 train/test 切分。Task 3 新增 60×5=300 条 gate rollout，只与 Task 1 同状态、同 repeat 0–4 的 baseline 配对；主结果每臂 18×5=90 条，训练状态及全部 60 状态结果仅作描述。每个阶段的两场景×两 seed smoke 均包含在各自既定流程中。')
    u=[find(collision,arm='pooled',proxy='disagreement',metric='AUROC',horizon_actions=h) for h in [5,10,20,40]]
    paragraph('Disagreement 对未来 5/10/20/40 个环境动作内碰撞的原方向 AUROC 为 '+', '.join(display(r) for r in u)+'；四个 95% 区间均覆盖 0.5。这组结果未支持将该 proxy 当作可靠短期碰撞预警。最近障碍距离的排序表现更强，但几何距离不是官方 crash 标签。')
    first=find(completion,arm='pooled',proxy='disagreement',metric='AUROC',stage='first_infer')
    five=find(completion,arm='pooled',proxy='disagreement',metric='AUROC',stage='first_five_landmark')
    paragraph('低 disagreement 对最终安全完成的排序能力为：首次 infer AUROC '+estimate(first)+'；前五次 infer landmark AUROC '+estimate(five)+'。后者只包含 '+five['rollouts']+' 条合格轨迹，条件是到动作 21 尚未在此前碰撞且轨迹覆盖该时点，不能与全部 360 条首次 infer 分析混用。')
    table(['Task1/2 留出 N=180/臂','CAR 无碰撞率','TSR 原任务成功率','安全完成率'],
        [[arm]+[estimate(find(outcomes,population='test_primary',arm=arm,metric=m)) for m in ['CAR','TSR','safe_success']] for arm in ['nominal','aegis']])
    table(['Task1 全60状态 N=600/臂（描述）','CAR 无碰撞率','TSR 原任务成功率','安全完成率'],
        [[arm]+[estimate(find(outcomes,population='all_states_descriptive',arm=arm,metric=m)) for m in ['CAR','TSR','safe_success']] for arm in ['nominal','aegis']])
    paragraph('上述 Task1/2 baseline 使用全部10 seed；下面 Task3 baseline 仅取配对 repeat0–4，分母分别为180与90，不能混用。AEGIS 与 nominal 的完成率点估计应与无碰撞率一起读；单臂区间重叠与否不能替代配对差值检验。')
    cut=json.loads((task2/'TRAIN_THRESHOLDS.json').read_text())
    paragraph('固定 gate 阈值来自训练 nominal 在险推理的加权 q90：'+repr(cut['gate_q90'])+'，推理样本数 '+str(cut['inferences'])+'。未搜索其他阈值。gate 在 U 严格大于阈值时，连续五个执行队列动作的七维环境控制命令均置零，然后按原计划 replan。')
    table(['臂（留出 N=90）','CAR 无碰撞率','TSR 原任务成功率','安全完成率','300步目标未完成率'],
        [[arm]+[estimate(find(gate_out,population='test_primary',arm=arm,metric=m)) for m in ['CAR','TSR','safe_success','budget_goal_incomplete']] for arm in ['nominal','aegis','gate']])
    table(['配对比较','CAR 差值','TSR 差值','安全完成差值'],
        [[comparison]+[estimate(find(delta,population='test_primary',comparison=comparison,metric=m)) for m in ['CAR','TSR','safe_success']] for comparison in ['gate-minus-nominal','gate-minus-aegis','aegis-minus-nominal']])
    paragraph('差值单位是绝对概率；正 CAR 差值表示更多轨迹无碰撞，正 TSR/安全完成差值表示更多轨迹完成。配对区间按状态重采样，保留同状态的五个 seed 与三臂。碰撞减少若伴随完成率降低，应同时报告，不能据此声称整体改进。')
    figure('gate_outcomes','固定 q90 gate 与匹配 baseline：主留出状态，N=90/臂，95% 状态聚类区间。')
    figure('gate_paired_differences','Gate 配对差值；零线用于判断区间是否覆盖无差异。')
    heading('碰撞预测、校准与安全完成')
    table(['proxy / h动作','原方向 AUROC [95% CI]','Platt BA','Platt Brier'],
        [[LABELS[p]+' / '+str(h),estimate(find(collision,arm='pooled',proxy=p,metric='AUROC',horizon_actions=h)),estimate(find(collision,arm='pooled',proxy=p,metric='BA',horizon_actions=h)),estimate(find(collision,arm='pooled',proxy=p,metric='Brier',horizon_actions=h))] for p in LABELS for h in [5,10,20,40]])
    figure('collision_prediction','预动作、在险 infer 碰撞预测。h 包含当前动作；概率校准不会改变主 AUROC 的预定义原方向。')
    table(['landmark / proxy','AUROC','BA','Brier','合格轨迹'],
        [[stage+' / '+LABELS[p]]+[estimate(find(completion,arm='pooled',proxy=p,metric=m,stage=stage)) for m in ['AUROC','BA','Brier']]+[find(completion,arm='pooled',proxy=p,metric='AUROC',stage=stage)['rollouts']] for stage in ['first_infer','first_five_landmark'] for p in LABELS])
    figure('completion_prediction','安全完成预测与短期碰撞预警是不同目标；首次与前五次 landmark 的合格分母不同。')
    figure('completion_calibration','训练集拟合的 Platt 概率与留出观察率；每个非空固定概率箱都有合格数，详见 calibration_curve.csv。')
    corr=read(task2/'paired_delta_correlations.csv')
    paragraph('同场景 nominal→AEGIS 的前五次安全完成信心差值与安全完成率差值：'+ '; '.join(r['metric']+' '+estimate(r)+'，'+r['states']+' 状态/'+r['paired_seeds']+' 配对seed' for r in corr)+'。这是经过 landmark 合格筛选的干预后条件关联，不能解释为信心变化导致了安全变化。')
    heading('失败前的低 disagreement')
    table(['臂','K动作','低 proxy / 合格碰撞轨迹','比例 [95% CI]','测试碰撞总数','短/不完整窗口排除','无训练成功参考排除'],
        [[r['arm'],r['K_actions'],r['n_low']+'/'+r['eligible'],estimate(r),r['total_test_crash'],r['excluded_incomplete_window'],r['excluded_missing_reference']] for r in failure if r['task']=='pooled' and r['weighting']=='raw_rollout_proportion'])
    paragraph('主统计对每条碰撞轨迹取碰撞动作之前连续 K 个环境动作的 U 均值，不含碰撞动作；与同任务×同臂训练安全成功轨迹、目标达成动作之前 K 动作的加权中位数比较，严格小于才计为低 proxy。缺少完整窗口或训练成功参考即排除，不跨任务、臂或测试集补参考。它描述特定失败前的低分歧，并不证明校准概率高或模型在语义上自信。')
    references=read(task2/'confident_failure_reference.csv')
    table(['训练参考任务','臂','K动作','中位数','合格成功轨迹','合格状态'],
        [[r['task'],r['arm'],r['K_actions'],display(r,'reference_median',6),r['eligible_rollouts'],r['eligible_states']] for r in references])
    paragraph('Object/I/task2 的 nominal 没有任何训练安全成功参考，结果为NA。该任务 AEGIS 的参考仅有2个状态、4条成功轨迹。低proxy失败区间条件于这些已冻结参考，不包含重新估计小样本训练参考所带来的不确定性，因此不能视为完整参考不确定性区间。')
    paragraph('补充 train nominal q10 统计按失败轨迹最后 K 动作内在险 infer 边界的低 U 比例聚合，和主统计定义不同；原始表为 low_boundary_fraction.csv 与 low_boundary_cases.csv。所有任务分层、各类排除、有效 bootstrap 数及每条轨迹窗口都保存在对应 CSV。')
    heading('时间曲线与 gate 频率')
    figure('at_risk_time_curves','每五个动作一次推理的在险曲线；仅纳入该动作仍可观察且此前未碰撞的轨迹。阴影为固定权重状态 bootstrap 的95%区间。')
    paragraph('曲线随动作变化时，其存活且可观察的状态/轨迹组成也在变化；不能把曲线直接解释为同一批机器人随时间的因果趋势。每个时间点 available_states / available_rollouts 在 temporal_curves.csv 中列出。动作横轴是仿真动作时钟，与 GPU 推理墙钟耗时不同。')
    table(['gate频率（主测试）','等状态/轨迹均值 [95% CI]','总分子/分母（描述）'],
        [[r['metric'],estimate(r),r['numerator']+'/'+r['denominator']] for r in frequency if r['population']=='test_primary'])
    heading('定义、验收与可复现性')
    paragraph('π0.5 为 flow matching。M=8，H=10，replan k=5，使用有效七维命令；U 是按训练资产中的固定动作尺度归一化后，跨样本 ddof=1 标准差，再对 horizon 与动作维求均值。churn 比较当前 chunk 前五步与上次 chunk 后五步，首 infer 无值。act_norm 是实际送入环境的归一化七维命令范数，不等同物理速度、力或机械运动幅度。')
    paragraph('每次诊断共享一份观测前缀 KV cache，八个噪声种子通过原生 B=1 suffix 顺序执行。缓存复用已实现，但没有宣称八样本并行矩阵计算加速。采样种子独立，逐 infer 在相同输入与原生 RNG 下动作数组严格相等，原 RNG 链未被诊断推进；新环境 RGB 渲染可能出现灰阶差异，因此不把跨新环境整段轨迹严格相等作为验收。固定公开 pi05 配置不将保存的连续 state 向量作为模型条件，保留图像+prompt 的原行为。')
    paragraph('官方 crash 为受保护物体的平移 L1 位移超过 0.001m；不替换为接触或负 min_dist。crashed 是动作执行后的标记，预测在险集使用此前是否已撞，首次碰撞动作保留；从不碰撞的 TTC=-1，首次碰撞之后 TTC 为缺失。h 的事件窗是 [t,t+h−1]；短且无事件的未完成结尾被删失并明确计数。')
    paragraph('min_dist 是只读几何诊断；box-box 分离时使用精确欧氏距离，重叠时使用负 SAT 最小平移深度，其他凸形使用原生方法。独立分离/重叠校验已通过。它不是 crash 判定。AEGIS 的 raw/proposed/applied 命令与控制器裁剪后的输入分别留存，QP/VLM/原评估评分逻辑保持原行为。nominal 每20动作快照仅保存，未验证完整任意状态恢复，也未从中途分支。')
    paragraph('OSC 零位置增量追踪当前末端位置，零旋转增量保留原方向目标；Panda gripper 零命令保留累积执行器目标，动力学与控制力矩继续。因此 gate 的零命令不能称为机械停止。stall 指标仅代理原300动作预算下目标未完成，不是实际速度停滞检测。')
    paragraph('一维 Platt 仅使用训练行：固定均值/标准差，规范化加权对数损失，加0.5×0.001×slope²惩罚，截距不惩罚；L-BFGS-B 参数冻结。单类或标准差≤1e−12 使用截距常量概率，完整分离由L2保证有限解，优化不收敛即停止，不选择新 fallback。AUROC 原方向、calibrated_AUROC、BA(p≥0.5)和 Brier 分开保留。')
    paragraph('95%区间采用 2,000 次按任务分层的状态 bootstrap，PCG64 seed20261009；使用原先权重乘每次抽中的状态次数，再归一化，保留同状态seed/arm/time，避免重复ID导致重复抽样权重被折叠。区间条件于固定训练拟合、阈值和参考，不包含训练集重采样不确定性；单类等未定义重采样次数明确列出。')
    heading('证据位置与成本')
    paragraph('Task1 采集提交 '+p2['collection_commit']+'；Task2 分析提交 '+p2['analysis_code_commit']+'，Slurm '+str(p2['slurm_job'])+'；Task3 gate 提交 '+p3['gate_commit']+'，作业 '+', '.join(x['job'] for x in p3['gate_gpu_accounting'])+'；本汇总提交 '+str(p3['analysis_code_commit'])+'。')
    paragraph('原采集统计计划 SHA '+p2['collection_source_plan_sha256']+'；最终冻结分析计划 SHA '+p2['analysis_plan_sha256']+'；split SHA '+p2['split_sha256']+'。保留两份统计计划身份，不将事后分析文件伪称为采集时文件。')
    paragraph('Task1 聚合 Parquet SHA '+p2['input_parquet_sha256']+'；完整独立审计覆盖1,200条rollout、54,497次M8诊断和每个Parquet单元；完整来源/上游文件、模型资产、归一化资产及容器 bytes 已核验。Task2紧凑输出可在完整远端审计通过后分析，本地大包传输以各自 DELIVERY_STATUS 与 LOCAL_EVIDENCE_VERIFIED.json 为准。此报告生成于 Quest，不能代替 Mac 本地全量原始证据到达确认。')
    paragraph('Task1 远端原始证据：'+p2['raw_evidence_location']+'；Task3：'+p3['raw_remote']+'。Task2输入 '+str(task2)+'；Task3汇总 '+str(gate)+'。图表来源是这些已通过SHA校验的CSV，报告自身记录输入完成receipt SHA。')
    paragraph('该系列600次完整AEGIS调用与4次此前uncertainty smoke调用累计结算 $'+str(p3['series_committed_usd'])+'；gate与统计分析无新VLM调用。成本证据 '+p3['api_final_path']+'，SHA '+p3['api_final_sha256']+'。此成本来自该系列账本，并不包括历史基线重现的其他独立批次。实际累计GPU时间 '+format(p3['actual_series_gpu_seconds']/3600.,'.3f')+' GPUh，授权上限 '+format(p3['authorized_series_gpu_seconds']/3600.,'.0f')+' GPUh；终态accounting保留在 PROVENANCE.json。')
    style='body{font-family:system-ui,sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;line-height:1.7;color:#172033}h1,h2{line-height:1.3}h2{margin-top:2em}table{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow:auto}th,td{border:1px solid #dbe2ea;padding:8px;text-align:left}th{background:#eff4f8}img{width:100%;max-width:1000px}figure{margin:24px 0}figcaption{font-size:13px;color:#475569}p{overflow-wrap:anywhere}@media print{body{margin:0}figure,table{break-inside:avoid}}'
    title='SafeLIBERO π0.5 / AEGIS 动作采样不确定性实验报告'
    (output/'REPORT.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="UTF-8"><title>'+title+'</title><style>'+style+'</style><main><h1>'+title+'</h1>'+''.join(blocks)+'</main></html>')
    (output/'REPORT.md').write_text('# '+title+'\n\n'+'\n'.join(markdown))
    atomic_json(output/'REPORT_COMPLETE.json',dict(passed=True,task2_receipt_sha256=sha(task2/'ANALYSIS_COMPLETE.json'),task3_receipt_sha256=sha(gate/'GATE_ANALYSIS_COMPLETE.json'),figures=figures,language='Chinese report with English scientific figure labels',inputs_fitted=False,api_calls=0,files={p.name:sha(p) for p in output.iterdir() if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task2',type=Path);p.add_argument('gate',type=Path);p.add_argument('output',type=Path);a=p.parse_args();render(a.task2.resolve(),a.gate.resolve(),a.output.resolve())
