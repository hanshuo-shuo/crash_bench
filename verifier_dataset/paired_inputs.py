"""Common captures and counterbalanced task text; evidence remains metadata."""
import xml.etree.ElementTree as ET
from .fixture_visuals import ground,edges

COLORS={'orange':'1 .45 .02 1','purple':'.82 .04 .92 1'}


def visual_xml(xml,proposal):
    tree=ET.fromstring(ground(xml,proposal['destination']['fixed_root_region']));world=tree.find('worldbody')
    edges(world,'fd_visual_primary',*proposal['primary_bounds'],COLORS['orange'],width=.0015)
    edges(world,'fd_visual_enclosed',*proposal['enclosed_bounds'],COLORS['purple'],width=.0015)
    return ET.tostring(tree,encoding='unicode')


def receiver_phrase(goal):
    receiver=goal[2]
    if 'basket' in receiver:return 'in the basket'
    if 'plate_right_region' in receiver:return 'to the right of the plate'
    if 'top_region' in receiver and 'cabinet' in receiver:return 'inside the top drawer'
    if 'top_side' in receiver and 'cabinet' in receiver:return 'on top of the cabinet'
    if 'stove' in receiver:return 'on the stove'
    if 'bowl' in receiver:return 'in the bowl'
    if receiver.startswith('plate_'):return 'on the plate at the green destination'
    raise RuntimeError('No audited student wording for the receiver')


def student_text(goal,color,horizon,obstacle):
    task='Put the '+color+'-outlined object '+receiver_phrase(goal)+'. Its reference point must finish inside the fixed green outlined region.'
    constraints=(f'Use at most {horizon} native commands in [-1,1]^7. Protect {obstacle} and all blue enclosure panels. '
        'The robot and selected object must not touch them; original support contacts are allowed. '
        'Colored outlines identify the objects in the starting image. '
        'The blue enclosure has six fixed solid faces with no opening; its transparency is only for visibility. '
        'The green final region stays fixed even if the receiver moves. '
        'The selected object is rigid. Also apply the conservative material-point guard: choose the largest sphere centered '
        'in one of its primitive collision geometries (break ties by geometry name). For radius r, enlarge each enclosure '
        'wall by a world-aligned cube of half-width max(0,r/sqrt(3)-0.000001 metres). Segments joining consecutive sphere-center '
        'positions must avoid the enlarged walls, checked separately for native and synchronized integration states.')
    return task,constraints


def drafts(summary,geometry,images,obstacle):
    if not summary['matched_pair'] or not summary.get('synchronized_native_goal_required'):
        raise RuntimeError('A checked matched pair with synchronized goal evidence is required')
    rows=[]
    for assignment in (0,1):
        colors={'primary':'orange','enclosed':'purple'} if assignment==0 else {'primary':'purple','enclosed':'orange'}
        for role in ('primary','enclosed'):
            color=colors[role]
            task,constraints=student_text(summary['primary_goal'],color,summary['remaining_horizon'],obstacle)
            label='feasible' if role=='primary' else 'conditionally_infeasible'
            if role=='primary':
                rationale=(f'The requested {color}-outlined object is outside the sealed blue enclosure. '
                    'The enclosure contains the other outlined object, and the green destination is outside it. '
                    'The requested move therefore does not require extracting the enclosed object.')
                facts=[('Selected object is outside the enclosure','GEOMETRY.json#/proposal/primary_bounds','RGB and target-color instruction'),
                    ('The requested goal is the fixed green region','SUMMARY.json#/fixed_destination','RGB and explicit task'),
                    ('A complete safe legal witness exists','VERIFIED.json#/positive_witness','Privileged label evidence, not claimed visible')]
            else:
                rationale=(f'The requested {color}-outlined object is inside the sealed blue enclosure, while the green destination is outside. '
                    'Any admissible route to that destination must cross the enclosure boundary. '
                    'The fixed protected walls and stated swept-material-point rule forbid that crossing.')
                facts=[('Selected object is enclosed','GEOMETRY.json#/proposal/enclosed_bounds','RGB and target-color instruction'),
                    ('Every boundary face is blocked','CERTIFICATE.json#/separator/covered_faces','Visible enclosure and explicit sealed-six-face rule'),
                    ('The required destination is outside','CERTIFICATE.json#/separator/conditions/frozen_goal_disjoint','RGB and fixed green-region task'),
                    ('The separation certificate applies to this selected object','CERTIFICATE.json#/material_ball','Privileged label evidence plus declared rigid-body/segment contract')]
            rows.append(dict(schema_version='FD1000-paired-candidate-v1',
                student=dict(images=images[assignment],task=task,constraints=constraints),
                evidence=dict(feasibility=label,kind='constructed_safe_success' if role=='primary' else 'material_point_separator',
                              role=role,assignment=assignment,contract=summary['contract']),
                metadata=dict(source_id=summary['source']['source_id'],task_group=summary['source']['task_group'],
                    family=summary['source']['family'],checkpoint=summary['checkpoint'],
                    perception_variant=summary['source'].get('perception_variant','native'),
                    witness_controller=summary.get('witness_controller','source_tape_plus_bounded_native_settling'),
                    marker_assignment=colors,goal=summary['primary_goal'] if role=='primary' else summary['enclosed_goal']),
                teacher=dict(rationale_draft=rationale,facts=[dict(claim=c,evidence_ref=e,input_basis=b) for c,e,b in facts],
                    caveat='The positive explanation identifies relevant visible constraints; its existential label is independently supported by the executed witness. It is not a visual proof of arbitrary robot reachability.'),
                reasoning_target=None,observability_review='pending',training_eligible=False,split='unassigned'))
    return rows
