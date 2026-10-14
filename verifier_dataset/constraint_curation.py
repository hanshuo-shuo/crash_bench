"""Remove whole contrast units only for recorded, image-bound visibility faults."""

PREMISES={'target_identifiable','target_boundary_relation_visible',
          'destination_relation_visible','both_rule_colors_identifiable'}


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
