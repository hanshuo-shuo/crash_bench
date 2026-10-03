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
    c.setFont('Helvetica',8);c.drawString(44,26,'CrashBench / FR-1 | frozen diagnostic | 3 October 2026');c.drawRightString(W-44,26,str(page))

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

def main(cpu,gpu,out):
    summary=read(out/'DIAGNOSTIC_SUMMARY.json');audit=read(out/'CPU_INDEPENDENT_AUDIT.json')
    cpu_receipt=read(cpu/'receipt.json');gpu_receipt=read(gpu/'receipt.json')
    pdf=out/'FR1_GROUP_MEETING_REPORT.pdf';c=canvas.Canvas(str(pdf),pagesize=(W,H));c.setTitle('Feasibility is not policy risk: FR-1 diagnostic')
    header(c,'Research question / strongest warranted result','Feasibility is not policy risk',1)
    y=690
    y=paragraph(c,'Can frozen VLA features separate <b>an unsafe current policy with a safe solution</b> from <b>no safe solution under an explicit contract</b>? This diagnostic establishes labels and tests observability before fitting a readout.',44,y)
    counts=summary['risk_counts']
    y=paragraph(c,f'<b>New evidence:</b> two preselected official milk-task initial states yielded four actual safe witnesses and two conditional geometric exclusions. Both parked-lid controls were feasible. Of six pi0.5 executions, {counts["violation_within_30"]} violated safety within 30 actions and {counts["horizon_safe_30"]} reached the full horizon safely.',44,y)
    rows=[]
    for r in summary['rows']:
        risk='violation @ '+str(r['policy_actions']) if r['risk_label']=='violation_within_30' else r['risk_label']
        rows.append([str(r['layout'])+' / '+r['variant'],r['feasibility_label'],r['reference_actions'] or 'certificate',risk])
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
    y=paragraph(c,'These distances show representation sensitivity, not feasibility decoding. They do not establish native introspection, a causal role in control, or predictive performance. Readouts, thresholds and normalization were not selected on these six states. Simple RGB pooling, red-pixel cues, proprioception, relational geometry and initial action-risk scores are preserved for audit.',44,y,style=SMALL)
    y=paragraph(c,'<b>Generalization and uncertainty:</b> two construction groups from one official task; milk positions differ by about 4 mm and basket positions by 28 mm. The prior full benchmark may have exposed these initial states. All paired variants belong to the same split. No population accuracy interval is warranted here; unknown coverage is an evidence property, not a model performance score.',44,y,style=SMALL)
    gates=summary['gates'];y=paragraph(c,'<b>Frozen gates:</b> '+', '.join(k.replace('_',' ')+': '+str(v) for k,v in gates.items())+'. No bulk matrix was submitted. A follow-up must establish visual evidence sufficient for the relevant relation, contain both short-horizon risk classes without changing the task to create balance, and prefreeze whole-layout splits.',44,y,style=SMALL)
    y=paragraph(c,'<b>Relation to prior work:</b> Yuan, Wu &amp; Bajcsy, <i>When to Act, Ask, or Learn</i>, RSS 2026 (arxiv.org/abs/2602.22474v2), calibrate task/proposal uncertainty and policy incapability. Our existence label requires external witnesses or a conditional exclusion over an explicit contract. Generic risk detection and asking for help are not novelty claims. Kintsugi-VLA, RoboPilot and Li-Dantam references are retained with verified URLs in the evidence package.',44,y,style=SMALL)
    paragraph(c,'<b>Provenance:</b> CPU '+cpu_receipt['job_id']+' / '+cpu_receipt['code_commit'][:12]+'; GPU '+gpu_receipt['job_id']+' / '+gpu_receipt['code_commit'][:12]+'. Upstream 2457feed5968. Seed 7. API spend $0. Complete Slurm accounting, source archives, protocol hashes, raw actions, full physics and independent auditors accompany this report.',44,y,style=SMALL)
    c.save();print(pdf)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cpu',type=Path);p.add_argument('gpu',type=Path);p.add_argument('out',type=Path);a=p.parse_args();main(a.cpu,a.gpu,a.out)
