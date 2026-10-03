"""Two-page addendum for the preserved, incomplete FR-1C diagnostic."""
import argparse,json
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from build_report import paragraph,table,INK,TEAL,SMALL,W,H

def read(p):return json.loads(p.read_text())
def header(c,kicker,title,page):
    c.setFillColor(TEAL);c.rect(0,H-12,W,12,fill=1,stroke=0)
    c.setFont('Helvetica-Bold',9);c.drawString(44,H-40,kicker.upper())
    c.setFillColor(INK);c.setFont('Helvetica-Bold',21);c.drawString(44,H-72,title)
    c.setFont('Helvetica',8);c.drawString(44,26,'CrashBench / FR-1C addendum | 3 October 2026');c.drawRightString(W-44,26,str(page))

def main(cpu,gpu,out):
    s=read(out/'FR1C_PERSISTED_EVIDENCE_AUDIT.json')
    assert s['persisted_evidence_audit_passed'] and not s['experiment_completed']
    assert s['transparent_result'] is None
    pdf=out/'FR1C_TRANSPARENCY_ADDENDUM.pdf';c=canvas.Canvas(str(pdf),pagesize=(W,H))
    c.setTitle('FR-1C addendum: transparency diagnostic stopped at image integrity check')
    header(c,'One-attempt stop / original report unchanged','Transparency remains untested',1)
    y=paragraph(c,'<b>Outcome:</b> the fixed alpha 0.25 diagnostic stopped at an image-integrity assertion. CPU checks passed for all eight physical states, and one opaque input completed feature extraction. No transparent input completed validation or extraction. This is an incomplete observation diagnostic, not a failed transparency visibility result.',44,692)
    y=paragraph(c,'<b>Main research finding remains FR-1/FR-1B:</b> six safe witnesses and two conditional RE-1 exclusions, in two closely related groups. All six tested pi0.5 rollouts hit the protected wine bottle; four had independently witnessed safe alternatives. The irrelevant closed-box control defeats closure-only classification. Hidden targets, tiny layout variation and a missing safe-risk class still prevent a valid held-out readout study.',44,y,style=SMALL)
    y=paragraph(c,'<b>Prospective FR-1C design:</b> change only all red-panel alpha values from 1.0 to 0.25 in the same eight states, preserving physics, cameras, prompt and inherited labels. Planned: 16 inputs plus one end anchor, 34 feature forwards and 102 unexecuted local action inferences. No alpha tuning, new labels, policy rollout or trained estimator.',44,y,style=SMALL)
    rows=[[str(x['episode'])+'/'+x['variant'],'F' if x['inherited_label']=='feasible' else 'RE-1 I',str(x['model_array_count']),'passed'] for x in s['cpu_rows']]
    y=table(c,['Existing state','Inherited label','Model arrays checked','Alpha-only / state exact'],rows,y,[133,101,139,151])
    y=paragraph(c,'F = independently witnessed feasible; RE-1 I = conditional exclusion under the external digital contract, not pure MuJoCo impossibility. In every headless CPU check, all 386 model-array fingerprints and numeric model/option fields were compared. Only the intended red alpha entries changed; complete restored dynamic/controller/random state stayed exact. These checks do not validate rendered pixels.',44,y,style=SMALL)
    y=table(c,['FR-1C job','Terminal status','Actual wall time','Authorized cap'],[['8375743 CPU','COMPLETED / 0:0','1m49s','1 CPU, 8 GiB, 5 min'],['8376030 GPU','FAILED / 1:0','3m29s','1 A100, 4 CPU, 32 GiB, 10 min']],y,[103,139,115,167])
    paragraph(c,'<b>Resource totals:</b> FR-1C used 2 feature forwards, 6 unexecuted local action inferences, zero environment actions, zero attempted integration steps and $0 API spend. Across FR-1/FR-1B/FR-1C, CPU-only jobs used 13m10s and GPU allocations 13m44s, including both preserved failures. All seven jobs are terminal; no retry or bulk run is queued.',44,y,style=SMALL)
    c.showPage();header(c,'Persisted opaque evidence / unresolved render mismatch','What completed, and what did not',2)
    y=paragraph(c,'The only completed input is e10/open at alpha 1.0. Its RGB and proprioception exactly match the saved original input. Repeated frozen features and same-seed action bracketing have zero maximum difference. Diagnostic target colours were never sent to the model; normal-repeat images and model/state restoration passed.',44,692,style=SMALL)
    sub=gpu/'e10/open/opaque'
    pictures=[('Actual agentview','agentview_policy_224.png'),('Diagnostic cyan','agentview_cyan_224.png'),('Diagnostic magenta','agentview_magenta_224.png'),('Actual wrist view','robot0_eye_in_hand_policy_224.png')]
    top=y-20
    for j,(label,name) in enumerate(pictures):
        x=44+j*134;c.setFillColor(INK);c.setFont('Helvetica-Bold',8.2);c.drawString(x,top,label)
        c.drawImage(ImageReader(str(sub/name)),x,top-132,width=122,height=122)
    y=top-149
    y=paragraph(c,'Opaque target-appearance influence: <b>69 agentview pixels; 0 wrist pixels</b>, using a fixed one-level RGB difference threshold inside the target projection. Counts were recomputed from saved cyan/magenta images. This is colour influence, not object recognition; it is a different measurement from the earlier integer segmentation count.',44,y,style=SMALL)
    y=paragraph(c,'<b>Exact stop:</b> during e10/open/transparent, opacity.py line 116 asserted that a directly rendered, preprocessed RGB image equalled the refreshed observation image. They were unequal. The compared pair was not saved before the assertion, so difference size and root cause cannot be recovered. The empty transparent directory and fixed camera order localize the failure to the first camera (agentview); the exception itself did not log a camera identifier.',44,y,style=SMALL)
    y=paragraph(c,'<b>Diagnosis boundary:</b> source inspection shows force-refresh requests a fresh camera render; the apparent benchmark observation override is a delegating method in unused DemoRenderEnv. A stale cache, render-order effect or transparency-specific rendering issue is not demonstrated. No transparent visibility gate, paired feature contrast, end-anchor repeat or post-job checkpoint verification completed. The pre-job checkpoint manifest is retained.',44,y,style=SMALL)
    y=paragraph(c,'<b>Next engineering correction, before any future scientific use:</b> save both compared raw and preprocessed images, hashes, camera/condition identifiers and difference metrics before asserting equality. Then diagnose the observation and direct-render paths under a separately authorized bounded check. Keep one consistent validated render path and the exact equality gate; do not tune alpha, relax the gate or infer missing results from the opaque case. No repair run was submitted this round.',44,y,style=SMALL)
    y=paragraph(c,'<b>Next research study:</b> after observability is established, prospectively freeze a feasibility-only design with spatially distinct whole-layout splits, independently witnessed/certified labels, explicit unknowns and closure-irrelevance controls in every split. Compare one fixed readout with equal-capacity image/RGB/cue baselines and report layout-level uncertainty. Collision prediction separately requires both safe and violating full-horizon examples. Neither successful decoding nor this OOD rendering change would establish native introspection or causal control use.',44,y,style=SMALL)
    paragraph(c,'<b>Provenance:</b> run source dd3e3db7325735177226539d7fd66a0fd66a723b; upstream 2457feed5968; seed 7. Separate evidence contains immutable source snapshots, protocols, Slurm receipts, all eight CPU checks, the opaque arrays/images, failure logs and a file-based audit. The original seven-page FR-1/FR-1B report is unchanged.',44,y,style=SMALL)
    c.save();print(pdf)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('cpu','gpu','out'):p.add_argument(name,type=Path)
    a=p.parse_args();main(a.cpu,a.gpu,a.out)
