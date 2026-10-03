"""Export saved, preregistered analysis values as inspectable CSV tables."""
import argparse
import collections
import csv
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def write(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export(analysis, output):
    output.mkdir(parents=True, exist_ok=True)
    result = read(analysis / 'READOUT.json')
    rows = []
    for representation, value in result['results'].items():
        if 'unavailable' in value:
            continue
        for population, interval_key in [('test', 'auroc_interval'),
                                         ('failed_subset', 'failed_subset_interval')]:
            interval = value[interval_key]['interval']
            rows.append(dict(representation=representation,
                primary=representation == result['primary_vla_representation'],
                selected_head=value['selection']['selected'], population=population,
                **value[population], auroc_ci_low=interval[0] if interval else None,
                auroc_ci_high=interval[1] if interval else None,
                bootstrap_unit='scene_group', test_groups=value[interval_key]['groups']))
    write(output / 'readout_metrics.csv', rows)
    rows = []
    for representation, value in result['results'].items():
        for family, record in value.get('head_families', {}).items():
            rows.append(dict(representation=representation, head_family=family,
                selected_head=record['selected']['name'], **record['test']))
    write(output / 'head_family_metrics.csv', rows)
    rows = []
    for representation, points in read(analysis / 'LEARNING_CURVES.json').items():
        for point in points:
            if 'test' in point:
                rows.append(dict(representation=representation, training_groups=point['groups'],
                    training_states=point['n'], selected_head=point['selection'], **point['test']))
    write(output / 'learning_curves.csv', rows)
    manifest = read(analysis / 'MANIFEST.json')['rows']
    counts = collections.Counter((r['scene_group'], r['split'], r['family'], r['label'], r['outcome'])
                                 for r in manifest)
    write(output / 'group_outcomes.csv', [dict(scene_group=k[0], split=k[1], constructor=k[2],
        independent_label=k[3], policy_outcome=k[4], states=n) for k, n in sorted(counts.items())])
    incremental = read(analysis / 'INCREMENTAL.json')
    rows = []
    if 'not_estimable' not in incremental:
        for model in ('safe_only', 'safe_plus_feasibility'):
            rows.append(dict(scene_group='pooled', model=model, **incremental[model]))
        for group, values in incremental['raw_groups'].items():
            for model, metrics in values.items():
                rows.append(dict(scene_group=group, model=model, **metrics))
        write(output / 'incremental_metrics.csv', rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('analysis', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    export(args.analysis, args.output)
