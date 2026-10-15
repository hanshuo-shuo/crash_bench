"""Remove whole contrast units only for recorded, image-bound visibility faults."""

PREMISES={'target_identifiable','target_boundary_relation_visible',
          'destination_relation_visible','both_rule_colors_identifiable'}


def selection_settings(review,protocol,protocol_digest):
    """Allow a prospectively frozen smaller, fully balanced reviewed subset."""
    plan=review.get('final_selection')
    if plan is None:return 250,protocol.get('selection_quotas')
    if (review.get('model_predictions_used') is not False or
        plan.get('basis')!='completed_visibility_reviews_before_v4_model_fitting' or
        plan.get('reviewed_visible_only') is not True or
        plan.get('original_protocol_sha256')!=protocol_digest):
        raise RuntimeError('Reduction must precede fitting and retain the frozen protocol')
    quotas=plan['selection_quotas'];original=protocol['selection_quotas']
    if set(quotas)!=set(original):raise RuntimeError('Reduction changed source splits')
    for split,cells in original.items():
        q=quotas[split]
        if (len(cells)!=4 or set(q)!=set(cells) or len(set(q.values()))!=1 or
            any(type(n) is not int or n<=0 or n>cells[c] for c,n in q.items())):
            raise RuntimeError('Reduction must preserve all four balanced cells without increasing quotas')
    target=sum(sum(cells.values()) for cells in quotas.values())
    if type(plan.get('target_units')) is not int or plan['target_units']!=target or not 0<target<250:
        raise RuntimeError('Invalid reduced target')
    return target,quotas


def reviewed_only(units,review):
    indexed={(u[0]['metadata']['source_id'],u[0]['metadata']['checkpoint']):u for u in units}
    if len(indexed)!=len(units):raise RuntimeError('Repeated source checkpoint')
    seen=set();visible=set()
    for r in review['all_reviewed_units']:
        key=(r['source_id'],r['checkpoint'])
        if key in seen:raise RuntimeError('Repeated visual review')
        seen.add(key)
        if r['status']=='excluded':continue
        if r['status']!='premises_visible' or r.get('failed_premises') or key not in indexed:
            raise RuntimeError('Invalid visible review')
        images=sorted({im['sha256'] for row in indexed[key] for im in row['student']['images']})
        if sorted(r['image_sha256'])!=images:raise RuntimeError('Visible review refers to different pixels')
        visible.add(key)
    return [u for u in units if (u[0]['metadata']['source_id'],u[0]['metadata']['checkpoint']) in visible]


def exclude(units,review):
    if review.get('purpose')!='observable_premises_only' or review.get('model_predictions_used') is not False:
        raise RuntimeError('Curation must concern observable premises, not model outcomes')
    indexed={}
    for unit in units:
        m=unit[0]['metadata'];key=(m['source_id'],m['checkpoint'])
        if key in indexed:raise RuntimeError('Repeated source checkpoint')
        indexed[key]=unit
    rejected=set()
    for record in review['excluded_units']:
        key=(record['source_id'],record['checkpoint'])
        if key in rejected or key not in indexed:raise RuntimeError('Unknown or repeated exclusion')
        unit=indexed[key]
        images=sorted({im['sha256'] for row in unit for im in row['student']['images']})
        if sorted(record['image_sha256'])!=images:raise RuntimeError('Exclusion refers to different pixels')
        if not record['failed_premises'] or not set(record['failed_premises'])<=PREMISES or not record['reason'].strip():
            raise RuntimeError('Visibility exclusion needs a concrete failed premise and reason')
        rejected.add(key)
    return [u for u in units if (u[0]['metadata']['source_id'],u[0]['metadata']['checkpoint']) not in rejected]
