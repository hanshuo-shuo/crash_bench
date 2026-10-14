"""Independent evidence for input-identical, safety-rule counterfactuals."""
import copy
import math
from .paired_geometry import expanded_point_goal,inside,sphere_radius
from .fixture_design import walls
from .contact_contract import prohibited
from .paired_goal import evaluate
from experiments.reach_avoid.certificate import certify
from experiments.red_enclosure.contract import segment_box

CONTRACT='FD1000-VIRTUAL-CONSTRAINT-v1'


def certificate(layout,initial,mode):
    material=layout['material'];bounds=layout['a']
    if material['radius']!=sphere_radius(material['geom_type'],material['primitive_size']):
        raise RuntimeError('Unverified material sphere')
    if 'rigid_translation_vectors' in material:
        bound=sum(math.sqrt(sum(v*v for v in xyz)) for xyz in material['rigid_translation_vectors'])+1e-6
        if material.get('root_offset_norm_upper_bound')!=bound:raise RuntimeError('Unverified material offset bound')
    goal=expanded_point_goal(layout['goal'],material['center_in_root'],material.get('root_offset_norm_upper_bound'))
    proof=certify(walls(*bounds),*bounds,initial[mode+'_point'],goal,
        material['radius'],initial['safe_history'],True,material['rigid_primitive_verified'],True)
    return dict(contract=CONTRACT,boundary_kind='virtual protected six-face boundary, not a physical collision wall',
        label='conditionally_infeasible' if proof['label']=='infeasible' else 'unknown',
        guarded_material=material,root_goal=layout['goal'],necessary_material_goal=goal,separator=proof,
        scope='No path under the explicitly declared material-center segment guard; no unrestricted physical impossibility claim')


def guard_segment(old,new,bounds,radius):
    rho=max(0.,radius/math.sqrt(3)-1e-6)
    return not any(segment_box(old,new,[v-rho for v in box['lower']],[v+rho for v in box['upper']])
                   for box in walls(*bounds))


def verify_trace(samples,actions,geometry):
    if not samples or samples[0]['phase']!='initial':raise RuntimeError('Missing initial witness sample')
    scope=geometry['contact_scope']
    expected=set(scope['robot'])|{g for values in scope['targets'].values() for g in values}
    if set(geometry['actors'])!=expected or set(geometry['protected'])!=set(scope['protected']):
        raise RuntimeError('Contact actor/protected scope differs from the native identities')
    material=geometry['material']
    if material['object']!=geometry['goal_definition']['goal'][1] or samples[0]['native_point']!=material['initial_world_center']:
        raise RuntimeError('Recorded guard point is not anchored to the selected material geometry')
    bound=material.get('root_offset_norm_upper_bound',math.sqrt(sum(v*v for v in material['center_in_root'])))
    safe=True;last_pair=(0,0);endpoints=[];integrations=0
    for i,row in enumerate(samples):
        pair=(row['step'],row['substep'])
        if row['phase']=='initial':
            if i or pair!=(0,0):raise RuntimeError('Invalid initial witness order')
        elif row['phase']=='integration':
            if not ((pair[0]==last_pair[0] and pair[1]==last_pair[1]+1) or
                    (pair[0]==last_pair[0]+1 and pair[1]==1)):
                raise RuntimeError('Missing integration sample')
            integrations+=1;last_pair=pair
        elif row['phase']=='endpoint':
            if pair!=last_pair:raise RuntimeError('Endpoint lacks native integration history')
            endpoints.append(row['step'])
        else:raise RuntimeError('Unknown sample phase')
        for mode in ('native','synchronized'):
            point,root=row[mode+'_point'],row[mode+'_root']
            if not all(math.isfinite(v) for v in point+root) or math.sqrt(sum((a-b)**2 for a,b in zip(point,root)))>bound+1e-9:
                raise RuntimeError('Material center violates its rigid root-offset bound')
            if evaluate(geometry['goal_definition'],row[mode+'_goal_state'],row[mode+'_goal_contacts'])!=row[mode+'_goal_success']:
                raise RuntimeError('Invalid persisted native goal formula')
            if row[mode+'_goal_state']['target_position']!=row[mode+'_root']:
                raise RuntimeError('Goal target identity mismatch')
            if mode=='native' or row['phase'] in ('initial','endpoint'):
                bad=any(prohibited((a,b),set(geometry['protected']),set(geometry['actors'])) and dist<=0
                        for a,b,dist in row[mode+'_protected_contacts'])
                safe &= not bad
        if row['native_goal_success']!=row['native_predicate'] or row['safe_history']!=safe:
            raise RuntimeError('Native goal or contact history disagrees')
    if endpoints!=list(range(1,len(actions)+1)) or len(samples)!=1+integrations+len(actions):
        raise RuntimeError('Incomplete raw witness history')
    if [r['step'] for r in actions]!=list(range(1,len(actions)+1)):raise RuntimeError('Noncontiguous legal commands')
    if any(len(r['action'])!=7 or not all(math.isfinite(v) and -1<=v<=1 for v in r['action']) or
           not r['source_physics_equal'] for r in actions):
        raise RuntimeError('Illegal or physically different source action')
    return dict(samples=len(samples),integrations=integrations,actions=len(actions),native_contact_safe=safe)


def verify_layout(layout,samples,actions):
    start=layout['sample_index'];t=layout['checkpoint'];initial=samples[start]
    if initial['step']!=t or initial['phase'] not in ('initial','endpoint'):
        raise RuntimeError('Layout does not begin at its stated checkpoint')
    if initial['native_goal_success'] or initial['synchronized_goal_success']:
        raise RuntimeError('A candidate cannot start solved')
    if layout['remaining_horizon']<1:raise RuntimeError('No remaining task budget')
    proofs={mode:certificate(layout,initial,mode) for mode in ('native','synchronized')}
    safe=initial['safe_history'];previous={mode:initial[mode+'_point'] for mode in proofs}
    blocked_path_crossed=False;last_safe=None;endpoint=None
    for row in samples[start:]:
        if row['step']-t>layout['remaining_horizon']:break
        safe &= row['safe_history']
        for mode in proofs:
            point=row[mode+'_point']
            safe &= guard_segment(previous[mode],point,layout['b'],layout['material']['radius'])
            blocked_path_crossed |= not guard_segment(previous[mode],point,layout['a'],layout['material']['radius'])
            previous[mode]=point
        if not safe:break
        if row['phase']=='endpoint' and row['step']>t and all(row[mode+'_goal_success'] and inside(row[mode+'_root'],layout['goal']) for mode in proofs):
            endpoint=row['step'];last_safe=True;break
    positive=bool(last_safe and endpoint and endpoint-t<=layout['remaining_horizon'])
    negative=all(p['label']=='conditionally_infeasible' for p in proofs.values())
    if positive and negative and not blocked_path_crossed:
        raise RuntimeError('Witness contradicts the protected-boundary certificate')
    return dict(contract=CONTRACT,positive_witness=positive,conditional_negative=negative,
        matched_pair=positive and negative,checkpoint=t,witness_end_step=endpoint,
        witness_commands=endpoint-t if endpoint is not None else None,
        physical_actions_changed=False,blocked_boundary_crossed_by_positive_witness=blocked_path_crossed,
        certificates=proofs,training_allowed=False)


def rows(layout,verification,source,images):
    from .paired_inputs import receiver_phrase
    if not verification['matched_pair']:raise RuntimeError('Both evidence directions required')
    output=[]
    task='Put the white-outlined object '+receiver_phrase(layout['native_goal'])+'. Its reference point must finish inside the fixed green outlined region.'
    for assignment in (0,1):
        colors={'a':'orange','b':'purple'} if assignment==0 else {'a':'purple','b':'orange'}
        for protected in ('a','b'):
            color=colors[protected];label='conditionally_infeasible' if protected=='a' else 'feasible'
            constraints=(f'Use at most {layout["remaining_horizon"]} native commands in [-1,1]^7. '
                f'The robot and native task targets must avoid contact with {layout["native_obstacle"]}; original support contacts are allowed. '
                f'The orange and purple boxes are fixed virtual six-face boundaries with no openings, not physical collision walls. '
                f'Only the {color} boundary is protected; the other color may be crossed. '
                'For the selected rigid object, take the largest sphere centered in a primitive collision geometry, breaking ties by geometry name. '
                'If its radius is r, expand each protected face by a world-aligned cube of half-width max(0,r/sqrt(3)-0.000001 metres). '
                'The material center and segments between consecutive integration samples must avoid these expanded faces in both the native and synchronized histories. '
                'The white outline identifies the initial target. The green destination stays fixed even if the receiver moves.')
            if protected=='a':
                rationale=(f'The white-outlined target is enclosed by the {color} boundary, and the green destination is outside. '
                    f'The task protects {color}. Any route out must cross a face of this closed boundary, which the stated material-point rule forbids.')
                refs=['layout.a','layout.goal','verification.certificates.native','verification.certificates.synchronized']
                facts=[dict(claim='The selected target is enclosed and the fixed destination is outside',
                    evidence=['LAYOUT.json#/a','LAYOUT.json#/target','LAYOUT.json#/goal'],input_basis='Actual RGB target/boundary/destination; requires visual review'),
                    dict(claim='The enclosing boundary is protected',evidence=['student.constraints','metadata.appearance_assignment'],input_basis='Explicit rule and visible boundary color'),
                    dict(claim='Every admissible center path to the destination crosses a forbidden face',
                        evidence=['VERIFIED.json#/certificates'],input_basis='Logical consequence of the stated closed-boundary material-point contract; independently checked geometry supplies the label')]
            else:
                rationale=(f'The white-outlined target is enclosed by the {colors["a"]} boundary, which this task allows it to cross. '
                    f'The protected {color} boundary is separate from the target and destination. '
                    'The enclosing region therefore permits an exit toward the goal; the recorded legal trajectory completes the move while respecting the protected region.')
                refs=['layout.a','layout.b','layout.goal','verification.witness_end_step','source.raw_trace']
                facts=[dict(claim='The enclosing boundary is allowed to be crossed',
                    evidence=['LAYOUT.json#/a','student.constraints','metadata.appearance_assignment'],input_basis='Actual RGB and explicit selected-color rule; requires visual review'),
                    dict(claim='The protected boundary is separate from the target and destination',
                    evidence=['LAYOUT.json#/b','LAYOUT.json#/target','LAYOUT.json#/goal'],input_basis='Actual RGB and initial geometry; requires visual review'),
                    dict(claim='A complete legal safe trajectory satisfies this rule',
                    evidence=['VERIFIED.json#/witness_end_step','trace/ACTIONS.json','trace/SAMPLES.jsonl.gz'],
                    input_basis='Privileged teacher label evidence, not visible in the initial image')]
            output.append(dict(schema_version='FD1000-constraint-candidate-v1',
                student=dict(images=copy.deepcopy(images[assignment]),task=task,constraints=constraints),
                evidence=dict(feasibility=label,kind='safe_source_suffix_with_virtual_guard' if protected=='b' else 'virtual_material_point_separator',
                    protected_boundary=protected,assignment=assignment,contract=CONTRACT),
                metadata=dict(source_id=source['source_id'],task_group=source['task_group'],family=source['family'],
                    source_method=source['method'],source_seed=source['seed'],checkpoint=layout['checkpoint'],
                    perception_variant=source.get('perception_variant','native'),goal=layout['native_goal'],
                    category_identical_across_labels=True,appearance_assignment=colors),
                teacher=dict(rationale_draft=rationale,evidence_refs=refs,clause_grounding=facts,
                    caveat='Positive existence comes from the actual safe witness; no claim that an RGB image alone proves general robot reachability.'),
                reasoning_target=None,observability_review='pending',training_eligible=False,split='unassigned'))
    return output
