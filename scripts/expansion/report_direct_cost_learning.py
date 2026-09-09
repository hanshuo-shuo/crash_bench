#!/usr/bin/env python3
"""Render the fixed diagnosis without changing predictions, gates or fits."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.expansion.direct_cost_learning import EVIDENCE,CONTEXTS,load_context,features,read,sha,write,TARGETS
OUT=ROOT/'docs/audits/20260908/direct_cost_learning'
RUN=OUT/'run_v1'


def main():
    frozen=read(RUN/'freeze.json');metrics=read(RUN/'metrics.json');timing=read(RUN/'timing.json')
    assert sha(RUN/'freeze.json')==(RUN/'freeze.sha256').read_text().strip()
    anchors=read(EVIDENCE/'anchors.json');b=read(EVIDENCE/'B.json')
    names=['Base','AlwaysRefresh','age1','age3','age5','period10','age_equal1','metadata','observation','metadata_ungated_diagnostic','observation_ungated_diagnostic']
    labels=dict(zip(names,['Base','AlwaysRefresh','age≥1','age≥3','age≥5','period10','age=1（开发规则）','Ridge：任务/age/步数','Ridge：加入观察','元信息去安全门（诊断）','观察去安全门（诊断）']))
    diagnostics={};dataset=[]
    for i,a in enumerate(anchors):
        context=load_context(CONTEXTS/a['panel_id']/'selector_context.pkl')
        dataset.append({'panel_id':a['panel_id'],'source':a['physical_source_id'],'condition_for_reporting_only':a['condition'],
            'metadata':features(a,context,'metadata').tolist(),'observation':features(a,context,'observation').tolist(),
            'A_target_delta':frozen['A_target_deltas'][i],
            'B_target_delta_evaluation_only':metrics['anchor_predictions'][i]['actual_B_delta']})
    write(OUT/'feature_dataset.json',{'warning':'B included only for post-fit evaluation; never fit using B.',
        'target_order':list(TARGETS),'observation_feature_order':['task2,age,anchor_steps','proprio8','full_image_4x4_RGB','wrist_image_4x4_RGB','chunk_length'],'rows':dataset})
    for name in names:
        choice=frozen['choices'][name]
        selected=[r for r in b if r['option']==choice[r['panel_id']]]
        base={(r['panel_id'],r['repeat']):r for r in b if r['option']==0}
        raw={'successes':sum(r['task_success'] for r in selected),'accidents':sum(r['catastrophe'] for r in selected),
             'lost_successes':sum(base[r['panel_id'],r['repeat']]['task_success'] and not r['task_success'] for r in selected),
             'new_accidents':sum(not base[r['panel_id'],r['repeat']]['catastrophe'] and r['catastrophe'] for r in selected),
             'refresh_count':sum(choice[r['panel_id']] for r in selected),'n':len(selected)}
        rows=[r for r in timing['rows'] if r['method']==name]
        # Source-macro overhead from anchor-level medians, consistent with cost aggregation.
        overhead=float(np.mean([np.mean([r['median_ms'] for r in rows if r['source']==s]) for s in sorted({r['source'] for r in rows})]))
        macro=metrics['methods'][name]['source_macro']['all']
        saved=-macro['inference_calls_delta']
        diagnostics[name]={'raw_counts':raw,'source_macro_selector_ms':overhead,
            'median_anchor_median_ms':float(np.median([r['median_ms'] for r in rows])),
            'max_anchor_p95_ms':float(max(r['p95_ms'] for r in rows)),
            'break_even_VLA_call_ms':overhead/saved if saved>0 else None,
            'coefficient_count':sum(np.asarray(m['coefficients']).size for m in frozen['models'].get(name,{}).values())}
    errors=[]
    for kind in ('metadata','observation'):
        for source in sorted({a['physical_source_id'] for a in anchors}):
            rows=[r for r in metrics['anchor_predictions'] if r['source']==source]
            e=np.array([np.array(r['predicted'][kind])-r['actual_B_delta'] for r in rows])
            errors.append({'model':kind,'source':source,**{k+'_mae':float(np.abs(e[:,j]).mean()) for j,k in enumerate(TARGETS)}})
    write(OUT/'diagnostics.json',{'timing_and_counts':diagnostics,'per_source_prediction_error':errors})
    rows=[]
    for name in names:
        for row in metrics['methods'][name]['per_source']:rows.append(dict(method=name,**row))
    with (OUT/'per_source.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# 直接收益学习：八来源留一开发诊断','',
      '已完成固定方案：无新增 rollout、无 GPU、无策略推理；18 个锚点、8 个来源。两个输入版本都使用同一 Ridge（α=1），每折仅用其他七来源的 A 拟合，B 评价。', '',
      '**结论：本轮简单规则取得小幅效率收益；固定 Ridge 未超过它，加入粗粒度观察内容也没有建立额外选择价值。**', '',
      '## 总体结果（200 步，物理来源等权）','',
      '完成代价为成功步数，未成功统一记 200；代价和调用越低越好。全部 stale 与控制纳入。下面的百分比是来源等权均值，不是把 72 次执行当独立样本。','',
      '| 方法 | 成功率 | 事故率 | 完成代价 | 相对 Base 差 | 实际调用 | 调用差 | 新增事故次数 |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in names:
        m=metrics['methods'][name]['source_macro']['all'];d=diagnostics[name]
        lines.append(f"| {labels[name]} | {m['success']:.2%} | {m['accident']:.2%} | {m['completion_cost']:.3f} | {m['completion_cost_delta']:+.3f} | {m['inference_calls']:.3f} | {m['inference_calls_delta']:+.3f} | {d['raw_counts']['new_accidents']} |")
    lines+=['','Base 原始成功 69/72；两个主 Ridge 均 67/72，均损失 3 次 Base 成功、获得 1 次正向转换。事故总数仍为 1，但该事故是一次新增事故，原来另一处事故消失。不能用净事故数相同声称“无新增事故”。','',
      '## 三个问题的答案','',
      '1. **简单信息能拿到多少？** age=1 规则相对 Base 平均少 0.8125 完成代价步、少 0.140625 次调用，成功保持、没有新增事故。只看 stale 时，少 1.625 步、少 0.28125 次调用。period10 的总体收益更小：少 0.265625 步、少 0.03125 次调用。age=1 是看到严重度分布后提出的开发规则，不能称为原协议独立验证；它不是 age≥1，后者仍会刷新有害的 age=3 状态。','',
      '2. **观察能否区分 b00/b02？** 元信息模型两者都刷新。观察模型两者都不刷新：避开 b02 的 +5.5 步/+1 调用损害，却漏掉 b00 的 −6 步/−1 调用收益，没有学出所需的相反选择。b00/b02 的元信息相同，但它们来自不同 LOSO 模型，因此 OOF 预测数值不同不能证明元信息能区分二者。同一模型面对相同元信息只能给相同预测。','',
      '3. **收益抵得上开销和错误吗？** 两个学习器在计入自身开销前已增加完成代价和调用，因此本轮没有净效率优势。轻量观察提取很便宜，但便宜不等于选对。','',
      '## 选择器开销','',
      '本机 CPU、预热后、内存中现有输入，包含特征提取、标准化、四输出预测和决策；不含读取文件和传感器获取。数值不是 Quest GPU 或机器人端到端延迟。','',
      '| 方法 | 各锚点中位耗时的中位数（ms） | 最慢锚点 p95（ms） | 盈亏平衡 VLA 单次延迟（ms） |','|---|---:|---:|---:|']
    for name in ['period10','age_equal1','metadata','observation']:
        d=diagnostics[name];be=d['break_even_VLA_call_ms'];bs='不适用：未节省调用' if be is None else f'{be:.6f}'
        lines.append(f"| {labels[name]} | {d['median_anchor_median_ms']:.6f} | {d['max_anchor_p95_ms']:.6f} | {bs} |")
    lines+=['','盈亏平衡仅把选择器耗时除以平均节省调用数，假设每次 VLA 调用延迟相同且其他开销不变；不包含图像获取成本，也不能证明实际机器人净加速。当前量化的是已有 delivered 输入；fresh 图像没有进入模型。','',
      '## 错误来源与稳定性','',
      '唯一 age=3 来源包含 b06/b08。将它留出时，其余来源 A 的成功差和事故差全为零，两个安全预测头也输出零；它们并不知道这一来源存在有害刷新。两个模型都刷新 b06/b08，导致主要损害。这个事实说明当前点预测安全门缺少支持，不是已经校准的安全保证。','',
      '观察主模型还在 b11 等中性控制上增加小量执行成本。所有控制仍完成且无事故，但“任务未损害”不能替代效率检查。去掉安全门的观察诊断也仍较 Base 更差，故本轮失败不能单独归咎于安全门过严。','',
      '删去一个来源后重新计算总体（不重新拟合）的完成代价差：元信息约 −0.679 到 +1.438，观察约 −0.429 到 +1.509；结果明显受来源组成影响。age=1 规则范围 −1.321 到 −0.500，但这仍是八个历史筛选来源内部的敏感性，不能外推。','',
      '## 数据与验证','',
      '- `PLAN.md`：拟合前固定方案；未运行超参数或阈值搜索。',
      '- `run_v1/freeze.json`：所有折的训练来源、标准化、系数、A 标签、OOF 预测与选择；拟合入口不读取 B。',
      '- `run_v1/metrics.json`、`per_source.csv`：各来源/条件的成功、事故、代价、调用和损害。',
      '- `feature_dataset.json`：可继续利用的允许特征及明确标注的 A/B 标签；B 字段只用于评价。',
      '- `diagnostics.json`：逐来源四目标 MAE、开销和原始转换计数。',
      '- 18 个上下文文件与 Quest 原件 SHA-256 一致；全仓 558 测试通过，repository audit 通过。', '',
      '本轮时间目标在看过 B 后提出，属于开发诊断。来源隔离不消除历史成功筛选偏差；只有八个来源，外层模型训练集彼此重叠。本次粗图像池化 Ridge 未成功，不等于所有观察表示都无效。所有锚点 chunk 为空，未添加 nominal proposal、fresh 输入或额外策略调用。','',
      '这份数据目前支持的下一步是解释并改善收益预测的来源支持与误差，而不是直接宣称学习选择器成功或启动在线触发。本轮没有自动追加训练方案或实验。','']
    with (OUT/'RESULTS_ZH.md').open('x') as f:f.write('\n'.join(lines))

if __name__=='__main__':main()
