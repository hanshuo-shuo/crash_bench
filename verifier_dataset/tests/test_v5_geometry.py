import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from verifier_dataset import v5_geometry as g
from verifier_dataset import v5_scan


def obj(x, y, half=.02, active=True):
    return dict(lo=[x - half, y - half, .9], hi=[x + half, y + half, 1.], center=[x, y, .95], active=active)


class Names(unittest.TestCase):
    def test_head_drops_index_obstacle_and_small(self):
        self.assertEqual(g.category('moka_pot_obstacle_1'), 'moka_pot_obstacle')
        self.assertEqual(g.head('red_coffee_mug_obstacle_1'), 'mug')
        self.assertEqual(g.head('milk_small_obstacle_1'), 'milk')
        self.assertEqual(g.head('glazed_rim_porcelain_ramekin_1'), 'ramekin')

    def test_anchor_exclusions(self):
        objects = dict(akita_black_bowl_1=obj(0, 0), plate_1=obj(.2, 0), cookies_1=obj(0, .2),
                       flat_stove_1=obj(-.3, 0), box_base_1=obj(.1, .1), moka_pot_obstacle_1=obj(.1, -.2),
                       porcelain_mug_1=obj(.3, .3), red_coffee_mug_obstacle_1=obj(-.2, -.2),
                       wine_bottle_obstacle_1=obj(5, 5, active=False))
        names = g.anchors(objects, 'pick up the black bowl on the stove and place it on the plate',
                          {'akita_black_bowl_1', 'plate_1'})
        # stove is named, the base is a pedestal, the two mugs are ambiguous, the bottle is parked.
        self.assertEqual(names, ['cookies_1', 'moka_pot_obstacle_1'])


    def test_goal_parent_and_lookalikes_are_not_anchors(self):
        objects = dict(akita_black_bowl_1=obj(0, 0), wooden_cabinet_1=obj(.3, 0), wine_rack_1=obj(-.3, 0),
                       butter_1=obj(0, .3), chocolate_pudding_1=obj(.2, .3), ketchup_1=obj(-.2, .3))
        self.assertEqual(g.goal_objects(objects, 'wooden_cabinet_1_top_region'), {'wooden_cabinet_1_top_region', 'wooden_cabinet_1'})
        names = g.anchors(objects, 'open the top drawer and put the bowl inside',
                          {'akita_black_bowl_1'} | g.goal_objects(objects, 'wooden_cabinet_1_top_region'))
        self.assertEqual(names, ['ketchup_1', 'wine_rack_1'])

    def test_lifted_rows_need_a_wider_margin(self):
        self.assertAlmostEqual(g.required_clearance(.05, .3), .08)
        self.assertAlmostEqual(g.required_clearance(.05, -.1), .05)
        rows = [row('s1', 0, 't', 'r', 1, clearance=.06, lift=0.), row('s2', 0, 't', 'r', 0, clearance=.06, lift=.3)]
        self.assertEqual(g.summarize(rows)['0.05']['all']['rows'], 1)


class Geometry(unittest.TestCase):
    def test_line_sides_and_straddle(self):
        a, b = [0, 0], [1, 0]
        self.assertAlmostEqual(g.line_signed(g.point([.5, .2]), a, b), .2)
        self.assertAlmostEqual(g.line_signed(g.point([.5, -.3]), a, b), -.3)
        self.assertEqual(g.line_signed(([.4, -.1], [.6, .1]), a, b), 0.)
        self.assertAlmostEqual(g.line_signed(([.4, .1], [.6, .3]), a, b), .1)

    def test_zone_inside_outside_and_partial(self):
        zone = ([0, 0], [1, 1])
        self.assertAlmostEqual(g.zone_signed(g.point([.2, .5]), zone), .2)
        self.assertAlmostEqual(g.zone_signed(g.point([1.3, 1.4]), zone), -.5)
        self.assertEqual(g.zone_signed(([.9, .4], [1.2, .6]), zone), 0.)
        self.assertAlmostEqual(g.zone_signed(([.1, .1], [.3, .3]), zone), .1)

    def test_rules_respect_spacing_and_radii(self):
        objects = dict(a=obj(0, 0), b=obj(.05, 0), c=obj(.3, 0))
        rules = g.rules(objects, ['a', 'b', 'c'])
        lines = [r['anchors'] for r in rules if r['template'] == 'line']
        self.assertEqual(lines, [['a', 'c'], ['b', 'c']])
        self.assertEqual(sum(r['template'].startswith('zone') for r in rules), 3 * len(g.ZONE_RADII))

    def test_label_any_pair_separates_and_min_clearance(self):
        rule = dict(template='line', params=[[0, 0], [1, 0]])
        same = (g.point([.2, .3]), g.point([.8, .1]))
        split = (g.point([.2, .3]), g.point([.8, -.2]))
        self.assertEqual(g.label(rule, [same]), (0, .1))
        self.assertEqual(g.label(rule, [same, split])[0], 1)
        self.assertAlmostEqual(g.label(rule, [same, split])[1], .1)

    def test_crossing(self):
        rule = dict(template='line', params=[[0, 0], [1, 0]])
        self.assertFalse(g.crosses(rule, g.point([0, .2]), [g.point([.5, .1]), g.point([1, .3])]))
        self.assertTrue(g.crosses(rule, g.point([0, .2]), [g.point([.5, -.01])]))


def row(source, unit, task, rule, label, **extra):
    return dict(dict(source=source, unit=unit, task=task, rule=rule, label=label, family='f', template='line',
                     clearance=.1, time_bin=0), **extra)


class Statistics(unittest.TestCase):
    def test_text_bound_and_balanced_yield(self):
        rows = [row('s1', 0, 't', 'r1', 1), row('s2', 0, 't', 'r1', 1), row('s3', 0, 't', 'r1', 0),
                row('s1', 0, 't', 'r2', 0), row('s2', 0, 't', 'r2', 0)]
        self.assertAlmostEqual(g.text_bound(rows, g.text_key), 4 / 5)
        kept = g.balanced(rows, g.text_key)
        self.assertEqual(sorted((r['source'], r['label']) for r in kept), [('s1', 1), ('s3', 0)])

    def test_twins_and_depth_use_variable_groups_only(self):
        rows = [row('s1', 0, 't', 'r1', 1), row('s2', 0, 't', 'r1', 0),
                row('s1', 0, 't', 'r2', 0), row('s2', 0, 't', 'r2', 1),
                row('s1', 0, 't', 'r3', 0), row('s2', 0, 't', 'r3', 0)]
        out = g.readouts(rows)
        self.assertEqual(out['twin_units'], 2)
        self.assertEqual(out['depth_pairs'], {'0': 0, '1': 2, '2': 0})
        self.assertEqual(out['text_only_bound'], 4 / 6)

    def test_witnessed_fraction_only_on_free_variable_rows(self):
        rows = [row('s1', 0, 't', 'r1', 1), row('s2', 0, 't', 'r1', 0, witnessed=True),
                row('s3', 0, 't', 'r1', 0, witnessed=False), row('s4', 0, 't', 'r2', 0, witnessed=False)]
        self.assertAlmostEqual(g.readouts(rows)['witnessed_fraction'], .5)

    def test_readings_follow_protocol(self):
        def block(bound, sources, per, witnessed=1.):
            r = dict(text_only_bound=bound, balanced_sources=sources, balanced_rows_per_source=per,
                     witnessed_fraction=witnessed)
            return {'0.05': dict(all=r, all_text_time=r)}
        self.assertEqual(g.readings(block(.95, 500, 10), block(.6, 20, 10)),
                         dict(initial='initial_layout_fixed', checkpoint='checkpoint_design_insufficient',
                              overall='natural_anchor_insufficient'))
        self.assertEqual(g.readings(block(.95, 500, 10), block(.6, 40, 8))['overall'], 'proceed_to_render_pilot')
        self.assertEqual(g.readings(block(.8, 150, 3), block(.6, 40, 8, .5))['initial'], 'initial_layout_viable')


class Selection(unittest.TestCase):
    def test_round_robin_over_tasks_with_one_rule_of_each_label(self):
        rows = []
        for task, sources in (('a', 3), ('b', 1)):
            for i in range(sources):
                for unit in (0, 5):
                    rows += [row('%s%d' % (task, i), unit, task + '|text', 'line:x|y', 1),
                             row('%s%d' % (task, i), unit, task + '|text', 'line:x|z', 0),
                             row('%s%d' % (task, i), unit, task + '|text', 'line:y|z', 0)]
        rows.append(row('a0', 9, 'a|text', 'line:x|y', 1))
        chosen = g.select_units(rows, n=3, strict_n=0)
        self.assertEqual([c['task'].split('|')[0] for c in chosen], ['a', 'b', 'a'])
        self.assertEqual(len({c['source'] for c in chosen}), 3)
        for c in chosen:
            self.assertEqual([r['label'] for r in c['rules']], [1, 0])
            self.assertEqual(c['candidate_rules'], ['line:x|y', 'line:x|z', 'line:y|z'])
            # Each rule keeps one label within its task and progress bin, so none is in the strict stratum.
            self.assertFalse(c['strict'])

    def test_strict_needs_both_labels_within_text_and_progress(self):
        rows = [row('s1', 0, 't|x', 'line:x|y', 1), row('s1', 0, 't|x', 'line:x|z', 0),
                row('s2', 0, 't|x', 'line:x|y', 0), row('s2', 0, 't|x', 'line:x|z', 1)]
        self.assertTrue(all(c['strict'] for c in g.select_units(rows, n=2)))

    def test_strict_units_come_first_then_broad_fill(self):
        rows = [row('s1', 0, 't|x', 'line:x|y', 1), row('s1', 0, 't|x', 'line:x|z', 0),
                row('s2', 0, 't|x', 'line:x|y', 0), row('s2', 0, 't|x', 'line:x|z', 1),
                row('s3', 5, 'u|x', 'line:x|y', 1), row('s3', 5, 'u|x', 'line:x|z', 0)]
        chosen = g.select_units(rows, n=3, strict_n=1)
        # The broad pass still visits task t first; units keep their strict flag.
        self.assertEqual([(c['source'][:2], c['strict']) for c in chosen][0][1], True)
        self.assertEqual(sorted((c['task'][0], c['strict']) for c in chosen), [('t', True), ('t', True), ('u', False)])
        self.assertEqual(len({(c['source'], c['unit']) for c in chosen}), 3)

    def test_rule_text_uses_natural_phrases(self):
        self.assertEqual(g.rule_text('line:ketchup|wine_bottle_obstacle'),
                         'Do not let anything cross the straight line through the ketchup bottle and the wine bottle.')


class Scan(unittest.TestCase):
    def state(self):
        objects = dict(bowl_1=obj(0, 0), plate_1=dict(obj(.4, 0, half=.08)), cookies_1=obj(.2, .3),
                       ramekin_1=obj(.2, -.3), moka_pot_obstacle_1=obj(-.3, .3))
        return dict(source='safelibero_goal/I/task0/episode0', family='safelibero_goal', task='safelibero_goal/I/task0',
                    language='put the bowl on the plate', goal_atoms=[['on', 'bowl_1', 'plate_1']],
                    goals={'plate_1': dict(kind='object', name='plate_1', lo=objects['plate_1']['lo'],
                                           hi=objects['plate_1']['hi'])}, objects=objects)

    def test_initial_rows_label_the_wall_between_target_and_goal(self):
        rows, names = v5_scan.initial_rows(self.state())
        self.assertEqual(names, ['cookies_1', 'moka_pot_obstacle_1', 'ramekin_1'])
        wall = next(r for r in rows if r['rule'] == 'line:cookies|ramekin')
        self.assertEqual(wall['label'], 1)
        self.assertAlmostEqual(wall['clearance'], .12)

    def test_checkpoint_rows_witness_and_time(self):
        state = self.state()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)/'v4'; case = base/'case_000'; (case/'execution').mkdir(parents=True)
            witness = Path(tmp)/'witness'; witness.mkdir()
            (witness/'SUMMARY.json').write_text(json.dumps(dict(source=dict(scene=dict(
                suite='safelibero_goal', level='I', task=0, episode=0)))))
            (case/'INPUT.json').write_text(json.dumps(dict(evidence=dict(witness_root=str(witness)),
                                                          student=dict(task='put the bowl on the plate'))))
            (case/'execution/GEOMETRY.json').write_text(json.dumps(dict(goal_definition=dict(goal=['on', 'bowl_1', 'plate_1']))))
            xs = [0, .05, .1, .15, .25, .3, .35, .4] + [.4] * 12
            samples = [dict(step=0, phase='initial', native_point=[0, 0, .95],
                            native_goal_state=dict(receiver_position=[.4, 0, .9]))]
            for step, x in enumerate(xs, start=1):
                samples.append(dict(step=step, phase='integration', native_point=[x, 0, 1.]))
                samples.append(dict(step=step, phase='endpoint', native_point=[x, 0, 1.],
                                    native_goal_state=dict(receiver_position=[.4, 0, .9])))
            with gzip.open(case/'execution/SAMPLES.jsonl.gz', 'wt') as stream:
                stream.write('\n'.join(json.dumps(s) for s in samples))
            with patch.object(v5_scan, 'V4_ROOTS', [base]):
                rows, inputs = v5_scan.checkpoint_rows({state['source']: state})
        self.assertEqual(len(inputs), 1)
        wall = {r['unit']: r for r in rows if r['rule'] == 'line:cookies|ramekin'}
        self.assertEqual(sorted(wall), [0, 5])
        self.assertEqual(wall[0]['label'], 1)
        self.assertTrue(wall[0]['suffix_crosses'])
        self.assertEqual(wall[5]['label'], 0)
        self.assertTrue(wall[5]['witnessed'])
        self.assertEqual(wall[5]['time_bin'], 1)


if __name__ == '__main__':
    unittest.main()
