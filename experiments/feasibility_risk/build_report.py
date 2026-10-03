"""Compact group-meeting PDF from verified diagnostic records; no new data."""
import argparse,json,math
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor,Color,white
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.graphics.shapes import Drawing,String
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics import renderPDF
W,H=612,792
INK=HexColor('#193448');TEAL=HexColor('#187b78');RED=HexColor('#b64244');LIGHT=HexColor('#eef3f6')
STYLE=ParagraphStyle('body',fontName='Helvetica',fontSize=10,leading=14,textColor=INK)
SMALL=ParagraphStyle('small',parent=STYLE,fontSize=8.3,leading=11)

def read(p):return json.loads(p.read_text())
def paragraph(c,text,x,y,width=524,style=STYLE):
    p=Paragraph(text,style);_,h=p.wrap(width,700);p.drawOn(c,x,y-h);return y-h-9

def header(c,kicker,title,page):
    c.setFillColor(TEAL);c.rect(0,H-12,W,12,fill=1,stroke=0)
    c.setFont('Helvetica-Bold',9);c.drawString(44,H-40,kicker.upper())
    c.setFillColor(INK);c.setFont('Helvetica-Bold',21);c.drawString(44,H-72,title)
    c.setFont('Helvetica',8);c.drawString(44,26,'CrashBench / FR-1 + FR-1B | frozen diagnostics | 3 October 2026');c.drawRightString(W-44,26,str(page))

def table(c,columns,rows,y,widths):
    x=44;c.setFillColor(LIGHT);c.rect(x,y-24,sum(widths),24,fill=1,stroke=0)
    c.setFillColor(INK);c.setFont('Helvetica-Bold',8.5)
    for name,w in zip(columns,widths):c.drawString(x+6,y-16,name);x+=w
    y-=24
    for row in rows:
        x=44;c.setStrokeColor(HexColor('#dbe4ea'));c.line(44,y-25,568,y-25)
        c.setFont('Helvetica',8.8)
        for value,w in zip(row,widths):c.drawString(x+6,y-16,str(value));x+=w
        y-=25
    return y-14

def structural_pages(c,cpu,gpu,out,bcpu,bgpu):
    s=read(out/'FR1B_INDEPENDENT_AUDIT.json')
    header(c,'Prospective structural control / actual policy views','A closed box can be irrelevant',5)
    paragraph(c,'Both translated-enclosure states admit the <b>same 226-action safe witness</b>. Moving the complete enclosure away from the milk changes relevance without changing its dimensions or red colour. This is a counterexample to closure-only infeasibility classification, in one existing construction group.',44,694,style=SMALL)
    columns=[('relevant open',gpu/'e10/open'),('relevant sealed',gpu/'e10/sealed'),('decoy open',bgpu/'e10/decoy_open'),('decoy sealed',bgpu/'e10/decoy_sealed')]
    for j,(label,p) in enumerate(columns):
        x=44+j*134;c.setFillColor(INK);c.setFont('Helvetica-Bold',9);c.drawString(x,619,label)
        for key,yy in [('agentview',486),('robot0_eye_in_hand',341)]:
            c.drawImage(ImageReader(str(p/(key+'_policy_224.png'))),x,yy,width=122,height=122)
            c.setFont('Helvetica',7);c.drawString(x,yy-11,'agentview' if key=='agentview' else 'wrist')
    rows=[]
    for row in s['rows']:
        views=row['views'];rows.append([row['variant'],'feasible','226',str(views['agentview']['target'])+' / '+str(views['robot0_eye_in_hand']['target'])])
    y=table(c,['New state','Existence label','Witness actions','Target pixels A/W'],rows,307,[130,124,115,155])
    y=paragraph(c,f'<b>Independent audit:</b> {s["total_actions"]} legal actions, {s["total_samples"]:,} samples and {s["verified_hashes"]} per-run hashes. Complete initial physics/controller states and all recorded commands match the original witness. The target is outside the decoy; the exclusion theorem is inapplicable. Native and RE-1 success both pass.',44,y,style=SMALL)
    y=paragraph(c,'<b>Scope:</b> two new physical states, zero new independent layouts, zero policy rollouts or policy-risk labels. Target visibility and enclosure position remain shortcuts. The control establishes label validity under changed relevance; it does not establish a learned relevance judgment.',44,y,style=SMALL)
    paragraph(c,'The machine certificate carries an inherited aggregate condition named initial_contact_and_geometry_safe=false when target-inside fails. The separate initial_safe gate passes for both decoys. Neither decoy is labelled infeasible.',44,y,style=SMALL)
    c.showPage()
    header(c,'Frozen features / explicit analysis amendment','Compare within one process',6)
    y=paragraph(c,'The first FR-1B GPU attempt stopped at an exact cross-job equality gate. Identical input and extraction source yielded unequal vectors on different nodes. This is a measured reproducibility discrepancy; its cause is unresolved. The failed realization saved maxima but not vectors, so its relative and cosine differences cannot be recovered.',44,692)
    y=table(c,['Failed check: job 8371672','Maximum absolute difference'],[['Prefix mean','0.0188293457'],['Image embedding mean','0.0634765625'],['Image prefix mean','0.0137729645'],['Seed-7 action chunk','0.0022231306']],y,[290,234])
    y=paragraph(c,'A prospectively approved, one-attempt repair re-extracted all six original saved inputs and two decoy inputs in one process, then repeated the first input. All within-input feature/action invariance checks and the end-anchor repeat are exact. Checkpoint contents were hashed before and after the run; input hashes and precision/device/software receipts are saved. No robot action was executed.',44,y)
    rows=[]
    for r in s['feature_cosine_distances']:
        rows.append([r['a']+' / '+r['b'],f'{r["prefix_final"]:.6f}',f'{r["image_embedding"]:.6f}'])
    y=table(c,['Same-process pair (initial state 10)','Prefix distance','Image distance'],rows,y,[290,117,117])
    ds=s['cross_job_comparisons']['rows'];r=ds[0]['differences']['prefix_final'];a=ds[0]['differences']['action_seed7']
    y=paragraph(c,f'For the repaired e10/open versus its original saved vector, prefix maximum difference is {r["max_absolute"]:.6g}, relative L2 difference {r["relative_l2"]:.6g}, and cosine distance {r["cosine_distance"]:.6g}; action maximum difference is {a["max_absolute"]:.6g}. All six repaired inputs exactly match all saved original feature/action arrays (cosine roundoff can be nonzero). The repair used the same node as the failed job. These are new comparisons, not reconstructed failed-job values.',44,y,style=SMALL)
    y=paragraph(c,'The irrelevant decoy closure produces 5.04 times the relevant-closure prefix distance, but its visible lid occupies 3.09 times as many pixels and lies elsewhere. This is descriptive sensitivity, not evidence against a decodable label direction. No readout, threshold, layer selection or population inference was fitted. The earlier within-process FR-1 measurements remain intact; cross-job values are not silently mixed into the FR-1B contrasts.',44,y,style=SMALL)
    paragraph(c,'Historical jobs did not capture checkpoint content hashes or device/precision receipts. A shared historical checkpoint path is weaker than verified historical bytes. The repair establishes current content identity and common-process comparison, not a retroactive proof of the source of numerical drift.',44,y,style=SMALL)
    c.showPage()
    header(c,'Decision / minimal follow-on choices / receipts','A useful control, no decoding claim',7)
    y=paragraph(c,'<b>Completed result:</b> six safe witnesses and two conditional exclusions across eight physical states in two closely related construction groups. All six tested pi0.5 executions hit the protected wine bottle; no safe-horizon risk class was observed. Feasibility and policy risk are empirically different here, but general feasibility decoding and collision prediction remain unevaluated.',44,692)
    y=paragraph(c,'<b>Why stop before a readout:</b> the two layouts are tiny jitters of one task, the target is hidden in both sealed relevant states, and there is no held-out spatial variation. Fitting eight states would not answer the requested generalization question. These are construct-support gaps, not a proof of unidentifiability or representation incapability.',44,y)
    y=paragraph(c,'<b>Smallest proposed follow-on, not executed:</b> render each of the same eight states with all red panels at fixed alpha 0.25, retaining identical physics and labels. Extract all opaque and transparent inputs together, plus a repeated anchor: 17 inputs, 34 feature forwards and 102 local unexecuted action inferences. Suggested caps: CPU 5 min; one A100 10 min; no robot actions or API spend. Audit actual target influence in preprocessed RGB, not integer segmentation alone.',44,y,style=SMALL)
    y=paragraph(c,'This can break the zero-target-pixels shortcut within the transparent condition and test sensitivity to a label-irrelevant rendering choice. It is an out-of-distribution ablation, may expose easier closure cues, and does not create a full visibility-by-feasibility factorial. It cannot establish ordinary-RGB native reasoning or unseen-layout accuracy. The other valid choice is to retain these controls and stop at the current finding.',44,y,style=SMALL)
    y=paragraph(c,'<b>Prompt versus contract:</b> the shared prompt requests milk-to-basket and no red/wine contact. RE-1 additionally specifies integration-segment restrictions and a fixed-goal conjunction outside that prompt. A future supervised probe would estimate externally supplied labels; even successful decoding would not establish native semantic understanding, introspection, or causal use by control. The witness uses simulator state to construct legal actions; it does not establish that an RGB-only solver can find them. Initial-only data say nothing about runtime recovery.',44,y,style=SMALL)
    receipts=[]
    for p in [cpu,gpu,bcpu,bgpu.parent/'20261003T060127Z_gpu_be5b79d49e97',bgpu]:
        rr=read(p/'receipt.json');receipts.append([rr['job_id'],rr['stage'],rr['code_commit'][:12], 'failed equality gate' if rr['job_id']=='8371672' else 'completed'])
    y=table(c,['Job','Stage','Immutable source','Terminal result'],receipts,y,[84,61,155,224])
    paragraph(c,'Zero paid API calls; no trained estimator or bulk matrix. Full source archives, protocols/amendment, Slurm accounting, label evidence, independent audits, raw features/actions and hashes are included. Claude methods critique was requested in chat, but its response was not retrieved after browser transport detached; no claims rely on it.',44,y,style=SMALL)

def main(cpu,gpu,out,bcpu=None,bgpu=None):
    summary=read(out/'DIAGNOSTIC_SUMMARY.json');audit=read(out/'CPU_INDEPENDENT_AUDIT.json')
    cpu_receipt=read(cpu/'receipt.json');gpu_receipt=read(gpu/'receipt.json')
    pdf=out/'FR1_GROUP_MEETING_REPORT.pdf';c=canvas.Canvas(str(pdf),pagesize=(W,H));c.setTitle('Feasibility and policy risk: FR-1 and FR-1B diagnostics')
    header(c,'Research question / strongest warranted result','Feasibility is not policy risk',1)
    y=690
    y=paragraph(c,'Can frozen VLA features separate <b>an unsafe current policy with a safe solution</b> from <b>no safe solution under an explicit contract</b>? This diagnostic establishes labels and tests observability before fitting a readout. Synthetic fixture dimensions come from each successful reference path.',44,y)
    counts=summary['risk_counts']
    y=paragraph(c,f'<b>New evidence:</b> two preselected official milk-task initial states yielded four actual safe witnesses and two conditional geometric exclusions. Both parked-lid controls were feasible. Of six pi0.5 executions, {counts["violation_within_30"]} violated safety within 30 actions and {counts["horizon_safe_30"]} reached the full horizon safely.',44,y)
    rows=[]
    for r in summary['rows']:
        risk='violation @ '+str(r['policy_actions']) if r['risk_label']=='violation_within_30' else r['risk_label']
        rows.append([str(r['layout'])+' / '+r['variant'],'RE-1 infeasible' if r['feasibility_label']=='infeasible' else 'feasible',r['reference_actions'] or 'certificate',risk])
    y=table(c,['Layout / state','Existence label','Witness actions','30-action policy risk'],rows,y,[119,117,111,177])
    y=paragraph(c,'<b>The valid control:</b> the parked lid has the same solid dimensions and colour as the sealed lid, but it does not seal the target. Lid presence alone gives '+str(summary['shortcut_rules']['lid_presence_rule_correct'])+'/6 correct on these construction states. The privileged rules "lid over target" and "target visibility" each give '+str(summary['shortcut_rules']['privileged_lid_over_target_rule_correct'])+'/6; this is a geometry sanity check, not learned or held-out accuracy.',44,y)
    y=paragraph(c,'<b>Feature caution:</b> moving the irrelevant parked lid changes the pooled frozen prefix more than closing the lid in both layouts (cosine-distance ratios '+', '.join(format(x['parked_over_closed_prefix_distance_ratio'],'.2f') for x in summary['irrelevant_cue_sensitivity'])+'). This shows sensitivity to irrelevant appearance; it does not rule out a decodable feasibility direction.',44,y,style=SMALL)
    y=paragraph(c,'<b>Decision: '+summary['bulk_recommendation']+'.</b> No readout was trained; there are zero train/validation/test layouts and no claimed AUROC or held-out accuracy. Camera evidence, risk-class support and real layout diversity must pass before scaling.',44,y)
    y=paragraph(c,'<b>Label contract.</b> Ordinary 7D OSC/gripper actions; 300-action witness budget; no robot or milk contact with red panels or wine bottle; per-integration material-root/actor-centre segment checks; native, synchronized and frozen-world goal conjunction. Sealed exclusion is a conditional digital theorem, not pure MuJoCo physical impossibility. Native success is instantaneous root containment and need not involve release.',44,y,style=SMALL)
    paragraph(c,f'CPU independent audit: {audit["total_actions"]:,} legal actions; {audit["total_samples"]:,} persisted samples; {audit["verified_hashes"]} file hashes; both paired initial physics states identical. No construction failures were removed.',44,y,style=SMALL)
    c.showPage()
    for page,e in [(2,10),(3,11)]:
        header(c,'Actual policy input / no added information','Layout '+str(e)+': what the model saw',page)
        y=694
        paragraph(c,'Native 224 x 224 policy inputs after the original preprocessing. Top: agentview. Bottom: wrist camera. The privileged certificate and segmentation labels were never supplied to pi0.5 or its feature extractor.',44,y,style=SMALL)
        for j,v in enumerate(('open','sealed','parked')):
            x=44+j*178;d=gpu/('e%d'%e)/v;c.setFont('Helvetica-Bold',11);c.setFillColor(INK);c.drawString(x,639,v.upper())
            for key,yy in [('agentview',462),('robot0_eye_in_hand',268)]:
                c.drawImage(ImageReader(str(d/(key+'_policy_224.png'))),x,yy,width=168,height=168)
                c.setFont('Helvetica',8);c.drawString(x,yy-12,key)
        relevant=[r for r in summary['rows'] if r['layout']==e]
        rows=[]
        for r in relevant:
            av=r['views']['agentview'];wr=r['views']['robot0_eye_in_hand'];p=av['pixels'];w=wr['pixels']
            rows.append([r['variant'],f'{p["target"]} / {w["target"]}',f'{p["goal"]} / {w["goal"]}',f'{p["lid"]} / {w["lid"]}',str(all(v['exact_policy_rgb_match'] for v in r['views'].values()))])
        y=table(c,['State','Target pixels A/W','Goal pixels A/W','Lid pixels A/W','RGB aligned'],rows,227,[70,122,122,120,90])
        paragraph(c,'Counts use nearest-neighbour geometry-ID masks in the native view. Zero target pixels means the target is occluded or outside that view; a valid hidden-state certificate does not make its premises directly observable. Clipping and projections are recorded in VISIBILITY.json. This audit does not by itself prove two differently labelled states have identical observations.',44,y,style=SMALL)
        c.showPage()
    header(c,'Measurement / limits / next decision','What this supports - and what it cannot',4)
    y=693
    y=paragraph(c,'<b>Frozen extraction:</b> final normalized PaliGemma prefix masked mean and pre-fusion SigLIP image embeddings, initial input only. All model weights stayed fixed. Two repeat feature passes and same-seed native actions bracket extraction; the action RNG is restored. Four independent fixed-seed proposals measure action disagreement without extra execution.',44,y)
    rr=[]
    for d in summary['feature_cosine_distances']:
        rr.append([str(d['layout']),d['a']+' / '+d['b'],f'{d["prefix_final"]:.6f}',f'{d["image_embedding"]:.6f}'])
    y=table(c,['Layout','Pair','Prefix cosine distance','Image cosine distance'],rr,y,[54,176,147,147])
    y=paragraph(c,'These distances show representation sensitivity, not feasibility decoding. They do not establish native introspection, a causal role in control, or predictive performance. Readouts, thresholds and normalization were not selected on these six states. Simple RGB pooling, red-pixel cues, proprioception, relational geometry and uncalibrated action-magnitude/disagreement proxies are preserved for audit.',44,y,style=SMALL)
    y=paragraph(c,'<b>Generalization and uncertainty:</b> two construction groups from one official task; milk positions differ by about 4 mm and basket positions by 28 mm. The prior full benchmark may have exposed these initial states. All paired variants belong to the same split. No population accuracy interval is warranted here; unknown coverage is an evidence property, not a model performance score.',44,y,style=SMALL)
    gates=summary['gates'];y=paragraph(c,'<b>Frozen gates:</b> '+', '.join(k.replace('_',' ')+': '+str(v) for k,v in gates.items())+'. No bulk matrix was submitted. A joint follow-up needs observable relations, both risk classes and frozen whole-layout splits. A feasibility-only study can separately freeze its own validity gates.',44,y,style=SMALL)
    y=paragraph(c,'<b>Relation to prior work:</b> Yuan, Wu &amp; Bajcsy, <i>When to Act, Ask, or Learn</i>, RSS 2026 (<link href="https://arxiv.org/abs/2602.22474v2" color="#187b78">arXiv:2602.22474v2</link>), calibrate task/proposal uncertainty and policy incapability. Our existence label requires external witnesses or a conditional exclusion over an explicit contract. Generic risk detection and asking for help are not novelty claims. Kintsugi-VLA, RoboPilot and Li-Dantam references are retained with verified URLs in the evidence package.',44,y,style=SMALL)
    paragraph(c,'<b>Provenance:</b> CPU '+cpu_receipt['job_id']+' / '+cpu_receipt['code_commit'][:12]+'; GPU '+gpu_receipt['job_id']+' / '+gpu_receipt['code_commit'][:12]+'. Upstream 2457feed5968. Seed 7. API spend $0. Complete Slurm accounting, source archives, protocol hashes, raw actions, full physics and independent auditors accompany this report.',44,y,style=SMALL)
    if bcpu and bgpu:
        c.showPage();structural_pages(c,cpu,gpu,out,bcpu,bgpu)
    c.save();print(pdf)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cpu',type=Path);p.add_argument('gpu',type=Path);p.add_argument('out',type=Path);p.add_argument('--structural-cpu',type=Path);p.add_argument('--structural-gpu',type=Path);a=p.parse_args();main(a.cpu,a.gpu,a.out,a.structural_cpu,a.structural_gpu)
