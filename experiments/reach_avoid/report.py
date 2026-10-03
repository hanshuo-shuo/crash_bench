"""Group-meeting report built only from persisted scientific records."""
import argparse,collections,json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Image,Table,TableStyle
from PIL import Image as PILImage


def read(path):return json.loads(path.read_text())
def num(value):return 'not estimable' if value is None else f'{value:.3f}'
def interval(value):return 'not estimable' if not value else '['+', '.join(f'{x:.3f}' for x in value)+']'


def build(session,analysis,destination):
    result=read(analysis/'READOUT.json');inc=read(analysis/'INCREMENTAL.json');manifest=read(analysis/'MANIFEST.json')
    rows=manifest['rows'];primary=result['primary_vla_representation'];res=result['results'];counts=collections.Counter(r['label'] for r in rows)
    outcomes=collections.Counter(r['outcome'] for r in rows);fig=session/'deliverables/figures'
    styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='TitleRA',fontName='Helvetica-Bold',fontSize=26,leading=30,textColor=colors.HexColor('#143E4C'),spaceAfter=16))
    styles.add(ParagraphStyle(name='SubRA',fontSize=13,leading=17,textColor=colors.HexColor('#167D8D'),spaceAfter=12))
    styles.add(ParagraphStyle(name='BodyRA',fontName='Helvetica',fontSize=10.2,leading=14,spaceAfter=9))
    styles.add(ParagraphStyle(name='SmallRA',fontName='Helvetica',fontSize=8.3,leading=11,spaceAfter=6))
    styles.add(ParagraphStyle(name='HeadRA',fontName='Helvetica-Bold',fontSize=17,leading=21,textColor=colors.HexColor('#143E4C'),spaceAfter=12))
    story=[]
    def p(text,style='BodyRA'):story.append(Paragraph(text,styles[style]))
    def heading(text):p(text,'HeadRA')
    def figure(name,width=495):
        path=fig/name
        if path.exists():
            w,h=PILImage.open(path).size;story.append(Image(str(path),width=width,height=width*h/w));story.append(Spacer(1,7))
    def table(data,widths):
        cells=[[Paragraph(escape(str(v)),styles['SmallRA']) for v in row] for row in data]
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E9F0F2')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#69828B')),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6)]));story.append(t);story.append(Spacer(1,8))
    def page():story.append(PageBreak())
    p('Safe goal-reaching<br/>versus policy failure','TitleRA')
    p('Reach-avoid feasibility study | pi0.5 | 3 October 2026','SubRA')
    p('<b>Question.</b> Can a frozen policy representation distinguish independently impossible states from solvable states on which that policy fails? The execution horizon is T=300 environment commands, never a solver-compute budget.')
    p(f'<b>Completed evidence.</b> The grouped extension contains {len(rows)} states: {counts["feasible"]} with complete safe witnesses, {counts["infeasible"]} with conditional geometric certificates, and {counts["unknown"]} UNKNOWN. All UNKNOWN cases remain in coverage and are excluded from binary accuracy.')
    metric=res[primary]['failed_subset'];p(f'<b>Primary held-out readout.</b> Validation selected <b>{escape(primary)}</b>. Among fixed-policy failures in the held-out cage construction, infeasibility AUROC is <b>{num(metric["auroc"])}</b> on {metric["n"]} independently labeled states. Four test layout groups support exploratory evidence only.')
    if 'not_estimable' not in inc:
        p(f'<b>Incremental test.</b> SAFE-score-only AUROC {num(inc["safe_only"]["auroc"])}; SAFE plus feasibility {num(inc["safe_plus_feasibility"]["auroc"])}. Paired group-bootstrap 95% interval for the AUROC difference: {interval(inc["paired_group_bootstrap"]["delta_auroc_interval"])}.')
    else:p('<b>Incremental test.</b> '+escape(inc['not_estimable']))
    p('<b>Scope.</b> Constructor transfer within one separator proof family. This is not independent-mechanism generalization, unrestricted physical impossibility, native semantic understanding, or an AEGIS rescue result.')
    figure('01_baseline.png')
    p('Baseline: the one predeclared clear wine placement yields three safe pi0.5 completions and one safe timeout. All four have independent safe Reference witnesses from the exact same initial states. Thus policy failure is demonstrably different from infeasibility.','SmallRA')
    page();heading('1. What the task and certificate actually mean')
    p('The controlled extension uses SafeLIBERO Object I/task2 (milk into basket), the pinned public simulator and pi0.5 checkpoint. Twelve target/goal layout combinations, two visible obstacle constructors and seven fixed variants are recorded before collection. The wine bottle uses one fixed clear xy placement; neither bottle nor fixture parameters were tuned against policy outcomes.')
    p('<b>Capabilities.</b> The native 7D OSC/gripper action interface remains available with each command in [-1,1]. All target orientations, routes, ordinary grasp/support contacts and pushing of unprotected movable objects are allowed. No speed limit is inferred from clipping. All robot and milk collision geometries are protected against contact with the wine and fixed red obstacles.')
    p('<b>Goal.</b> A witness must satisfy the native basket predicate, its independently synchronized evaluation, and the basket containment volume frozen in world coordinates at initialization. Moving the basket cannot move the fixed goal. Every native and synchronized integration sample is audited; no new command is issued after the first violating command.')
    p('<b>RA-SEP-1 digital contract.</b> Actual target collision geometry contains a root-centered material ball with radius about 25.88mm, invariant under target rotation. A world-aligned cube of half-width r/sqrt(3)-1 micrometre lies inside it. A declared swept-root-cube guard forbids crossing fixed red boxes between integration samples. If their expanded AABBs cover all six faces of a containing region and the fixed goal lies entirely outside, every allowed continuous root path to the goal crosses a forbidden region. This excludes all rotations/routes/contact modes under this contract. It is not a theorem that unmodified soft-contact MuJoCo or unrestricted physical robots cannot escape.')
    figure('03_visible_constructions.png',495)
    p('Development examples, native camera. A hidden target can be feasible (slit 180mm), and a visibly localized target can be certified infeasible (cage 20mm). Fixtures can be partially off-frame; pixel visibility and clipping diagnostics are retained. Slit roof and bar cage share the same separator proof family.','SmallRA')
    page();heading('2. Independent labels, unresolved gaps, and outcomes')
    table([['Label','Rule','Count'],['Feasible','Full safe successful native-action witness within T=300',counts['feasible']],['Conditionally infeasible','Independent RA-SEP-1 geometry certificate',counts['infeasible']],['UNKNOWN','Neither complete safe witness nor certificate',counts['unknown']]],[110,320,65])
    p('Each new group is independently settled, snapshotted, restored into every paired fixture, certified against actual executing geometry and attempted by the fixed Reference controller. Its failure never creates an infeasible label. Solver/rollout limits do not change the execution-time definition of T.')
    p('The development numeric-gap controls have certified negatives at 20/28mm, UNKNOWN at 34/60mm, and a witnessed positive at 180mm. The observed unresolved bracket is28–180mm, width 152mm. The sufficient certificate cutoff near 29.88mm is not a true feasibility boundary. The grouped state table retains per-layout evidence and numeric-gap brackets; no near-boundary accuracy is inferred from separated cases.')
    figure('08_unknown_bracket.png',465)
    table([['Fixed pi0.5 endpoint','Count'],*[[k,outcomes[k]] for k in ['safe_completion','safe_timeout','collision','invalid','not_run']]],[340,155])
    p('Unknown initial labels receive read-only features but no matrix policy rollout. A collision is an observed safety-first failure. A safe timeout requires exactly300 commands. Invalid/illegal, incomplete or missing policy outcomes are not silently recoded as failures; valid initial-state feasibility observations are retained even when their policy endpoint is excluded.')
    table([['Split','Layout groups','Primary constructor','Known states','UNKNOWN states'],['Train','6','slit',len(result['indices']['train']),sum(r['split']=='train' and r['family']=='slit' and r['label']=='unknown' for r in rows)],['Validation','2','slit',len(result['indices']['validation']),sum(r['split']=='validation' and r['family']=='slit' and r['label']=='unknown' for r in rows)],['Test','4','cage',len(result['indices']['test']),sum(r['split']=='test' and r['family']=='cage' and r['label']=='unknown' for r in rows)]],[75,85,110,100,125])
    p('All paired fixtures, variants, seeds, frames and renders of a layout stay together. Split seed 741 is fixed. The grid reuses some position components across splits, so these are held-out layout combinations and constructors, not unseen task or component-position generalization. Other constructor/split cells remain descriptive and never select the reported primary model.','SmallRA')
    page();heading('3. Frozen representations and matched readouts')
    figure('04_layer_readouts.png');figure('05_comparators.png',470)
    p('Every representation uses training-only standardization and PCA (at most 32 dimensions), four logistic regularization values and four 32-unit MLP regularization values, selected by validation log loss. Linear and nonlinear results and fixed training-group learning curves are saved separately. The primary layer/head is selected without test outcomes. Error bars resample the four test layout groups 2,000 times; they are exploratory, not population precision.','SmallRA')
    p('Own SigLIP tower output, projected visual tokens, all 18 residual layers (whole-prefix and image-token pools), and normalized final prefix use frozen native weights. DINOv2-small receives the same two actual 224px RGB views with no crop/resize and frozen weights. Primary DINO patch pooling matches the two-view averaging of the own visual tower; this external comparison cannot establish information destruction. Differences between native layers are likewise finite-probe and finite-data results, not proof that information is irretrievably absent.','SmallRA')
    page();heading('4. Does feasibility add beyond a failure score?')
    if 'not_estimable' not in inc:figure('07_incremental.png')
    else:p(escape(inc['not_estimable']))
    p('The SAFE-style probe is a fixed MLP32 predicting the full-T policy failure endpoint from final native features. It is not an official SAFE reproduction and failure does not mean only short-horizon collision. Both feasibility stackers use identical initial decision time and training failures: a scalar SAFE score alone versus that score plus a feasibility score.')
    p('Training scores use leave-one-layout-group-out cross-fitting. Each fold refits all transforms and repeats feasibility representation AND head selection using only the other train groups and fixed validation groups. Final test remains untouched. An independent audit found and corrected internal selection leakage before any grouped GPU rollout; the issue was not outer-test contamination.')
    if len(inc['training_failure_classes'])<2:p('<b>Failure-probe limitation.</b> The training failure endpoint has only one class. The recorded constant-prior fallback is not a learned discriminative SAFE model; any incremental result must be interpreted against that limited baseline.')
    p('Undefined policy-outcome exclusions: '+escape(str(inc['undefined_outcome_exclusion_counts']))+'. Exclusion case IDs, fold membership, chosen fold models, raw predictions and paired group-bootstrap distributions are reproducible from the supplied analysis records.')
    figure('06_learning_curves.png')
    page();heading('5. Attention, prompt sensitivity, and interpretation')
    p('The KNOWS-inspired comparator uses actual native action-query/vision-key attention at initial diffusion t=1 with fixed RNG, averaged over action queries and K=1 in time. Target mass, area-normalized density and entropy are recorded. A projected target ellipsoid from simulator geometry replaces the tracker; this is privileged localization and is disclosed. Fixed one-based layer 12 / head 3 is interpreted as array indices11/2. It is an initial-time adaptation, not the paper\'s trajectory-level experiment.')
    p('Native suffix velocity and the traced attention execution agree exactly on both adapter-development states. The native layer trace agrees exactly on 28 earlier initial states; comparison arrays are saved before acceptance checks. Feature extraction never advances environment physics or policy RNG.')
    figure('02_prompt_sensitivity.png')
    p('At each of four fixed clear baseline states, the original instruction, explicit safety sentence and matched original duplicate share scene/state/RNG. Actual token IDs differ at 19 positions; final features and action chunks change, while duplicate arrays match exactly. Action maximum absolute changes are 0.195,0.540,0.851,0.203. These show sensitivity, not comprehension. Finite equality would likewise not prove information absent.')
    p('<b>Claims this study does not support.</b> There are not three independent physical obstruction mechanisms, no train-two/test-third test, no validated runtime AEGIS blockage trigger or state-specific rescue alternatives, and no rescue-improvement or reporting-delay result. Initial feasibility can be lost after motion. Existence of a safe solution never licenses an unsafe action. Hidden labels train external probes; they do not show spontaneous semantic understanding of uncommunicated simulator rules.')
    page();heading('6. Reproduction, failures, and evidence integrity')
    p('CPU matrix source:87c5b06ef2d5e7b0f21f38350044e9f495097a59; GPU matrix source:465f112ea18bf12204c2c87f1df99c75dd67bd4a. Pinned public upstream:2457feed5968ae803926e178c8ce8243b9ecdcf9. Every job uses an immutable git archive with source hashes, exact protocol, seed 7, T=300 and Slurm provenance. DINO revision:ed25f3a31f01632728cabb09d1542f84ab7b0056.')
    p('Main jobs: baseline CPU8414531 and GPU8417233; development mechanisms8416208; native layer/visibility audit8418441; grouped CPU array8419904; adapter8421049; grouped GPU array8422120. Complete terminal accounting, any repairs and per-case manifests accompany the report. Account p33100; CPU short; GPU gengpu; at most one matrix A100 active. Zero paid API calls; no base-VLA training.')
    p('Recoverable faults were retained: explicit bottle identity/support repair; a pending native-support job cancellation; prompt timing metadata excluded from numeric arrays; bfloat16 WebSocket conversion with server-side array preservation; stale Slurm dependency repaired only after completed accounting and28 verified label records. Historical adapter receipt protocol-hash fields point to the older observer protocol; the full immutable archives and scientific hashes preserve their actual configuration. No failure supplied a scientific label.')
    p('Independent audit regression covers nested selection, undefined outcome exclusion through the saved-record loader and OOF path, full-T timeout validation, persisted native/fixed goal predicates, and exact separator coverage (including one-ULP gaps). Library copies are uploaded only after local content/render verification, with a separate delivery receipt. Raw physics, actions, inputs, source archives and labels remain recoverable at their original Quest roots and local retrieval paths.')
    heading('Primary sources and distinct estimands')
    for title,url,desc in [
        ('KNOWS','https://arxiv.org/html/2606.09749v1','Target attention, tracking and a CBF; success association does not establish general goal feasibility.'),
        ('SAFE','https://arxiv.org/abs/2506.09937','VLA failure prediction; this study uses a declared SAFE-style head, not an official reproduction.'),
        ('ShieldVLA','https://arxiv.org/html/2609.13231v1','Safe operating-region/indefinite safety is distinct from safe goal-reaching.'),
        ('DINOv2-small model','https://huggingface.co/facebook/dinov2-small','Frozen external vision baseline with pinned artifact hashes.')]:
        p(f'<link href="{url}" color="#167D8D">{title}</link>: {escape(desc)}','SmallRA')
    p('Prepared from saved records. Interpret the numerical readout as a small, controlled constructor-transfer result with explicit UNKNOWN coverage and digital-contract assumptions.','SmallRA')
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#687B83'));canvas.drawString(48,28,'CrashBench / SafeLIBERO controlled extension | RA-1 / RA-2');canvas.drawRightString(564,28,str(doc.page))
    destination.parent.mkdir(parents=True,exist_ok=True)
    SimpleDocTemplate(str(destination),pagesize=(612,792),rightMargin=48,leftMargin=48,topMargin=43,bottomMargin=45,title='Safe goal-reaching versus policy failure',author='CrashBench research study').build(story,onFirstPage=footer,onLaterPages=footer)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('session',type=Path);p.add_argument('analysis',type=Path);p.add_argument('destination',type=Path);a=p.parse_args();build(a.session,a.analysis,a.destination)
