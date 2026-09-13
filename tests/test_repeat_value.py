"""Synthetic measurement-contract tests; no benchmark execution or fitted model."""
import unittest
from crashbench.repeat_value import (arms, evaluate, freeze_choice, measure_cell,
                                    source_average, terminal_sample, validate_panel, forecast_comparison)


def cell(name='x', source='s', a_base=(0, 0), a_other=(1, 1), b_base=(0, 0), b_other=(1, 1)):
    c = dict(id=name, source=source, condition='glass', role='fitting', samples={})
    for phase, base, other, offset in [('A', a_base, a_other, 0), ('B', b_base, b_other, len(a_base))]:
        for o, values in enumerate([base, other]):
            c['samples'][phase, o] = [terminal_sample(v, 0, 10, i+offset) for i, v in enumerate(values)]
    return c


class RepeatValueTests(unittest.TestCase):
    def test_future_block_can_favor_either_forecast(self):
        favorable = forecast_comparison([dict(A_gain=1, B_gain=0, C_gain=0)], bootstrap=20)
        unfavorable = forecast_comparison([dict(A_gain=1, B_gain=0, C_gain=1)], bootstrap=20)
        assert favorable['error_improvement'] == 1
        assert unfavorable['error_improvement'] == -1

    def test_c_completeness_preserves_shared_prefix(self):
        from scripts.expansion.run_repeat_value_c import check_records
        anchors = [dict(episode_id='x', triggered=True), dict(episode_id='early', triggered=False)]
        records = [dict(episode_id='x', repeat=r, option=o, phase='C') for r in [4, 5] for o in [0, 1]]
        check_records(anchors, records, [4, 5], 'detour')
        with self.assertRaises(ValueError):
            check_records(anchors, records[:-1], [4, 5], 'detour')

    def test_stable_real_rescue_survives_separate_execution(self):
        c = cell()
        rows = evaluate(dict(cells=[c]), 'real_full', 20, 20)
        assert rows[0]['A_gain'] == rows[0]['B_gain'] == 1
        assert freeze_choice([c], 'pseudo_one', 20) == {'x': 0}


    def test_same_policy_pseudo_winner_can_disappear_without_real_effect(self):
        c = cell(a_base=(0, 1), a_other=(0, 1), b_base=(1, 1), b_other=(1, 1))
        rows = evaluate(dict(cells=[c]), 'pseudo_one', 20, 20)
        assert rows[0]['A_gain'] == 1
        assert rows[0]['B_gain'] == 0
        assert freeze_choice([c], 'real_full', 20) == {'x': 0}


    def test_deadline_curve_keeps_choice_and_distinguishes_late_catchup(self):
        c = cell(a_base=(1, 1), b_base=(1, 1))
        for phase in ['A', 'B']:
            for s in c['samples'][phase, 0]:
                s['success_step'] = 21
        choice = freeze_choice([c], 'real_full', 20)
        assert choice == {'x': 1}
        assert evaluate(dict(cells=[c]), 'real_full', 20, 20, choice)[0]['B_gain'] == 1
        assert evaluate(dict(cells=[c]), 'real_full', 20, 21, choice)[0]['B_gain'] == 0


    def test_sources_not_anchors_define_macro_weight(self):
        result = source_average([dict(source='a', gain=1), dict(source='a', gain=1),
                                 dict(source='b', gain=0)], ['gain'])
        assert sum(r['gain'] for r in result)/len(result) == .5


    def test_shared_early_accident_kept_without_pseudo_repetitions(self):
        c = dict(id='early', source='s', condition='offpath', role='calibration', samples={},
                 shared=terminal_sample(0, 1, 0, -1))
        assert freeze_choice([c], 'pseudo_one', 20) == {'early': 0}
        assert measure_cell(c, 'B', 'pseudo_one', 20, 0)['selected_accident'] == 1
        with self.assertRaises(ValueError):
            arms(c, 'B', 'pseudo_one', 20)
        with self.assertRaises(ValueError):
            measure_cell(c, 'B', 'real_full', 20, 1)


    def test_missing_repeat_fails_without_silent_replacement(self):
        c = cell()
        c['samples']['B', 1].pop()
        with self.assertRaises(ValueError):
            validate_panel(dict(cells=[c], repeats=2, historical={}))


    def test_competing_terminal_and_new_accident_accounting(self):
        with self.assertRaises(ValueError):
            terminal_sample(1, 1, 5, 0)
        c = cell(a_base=(1, 1), b_base=(1, 1), b_other=(0, 0))
        c['samples']['B', 1] = [terminal_sample(0, 1, 4, i+2) for i in range(2)]
        m = measure_cell(c, 'B', 'real_full', 20, 1)
        assert m['gain'] == -1 and m['loss'] == 1 and m['new_accident'] == 1


    def test_pseudo_two_uses_disjoint_equal_sized_base_subsets(self):
        c = cell(a_base=(0, 0, 1, 1), a_other=(1, 1, 1, 1),
                 b_base=(1, 1, 1, 1), b_other=(1, 1, 1, 1))
        x, y = arms(c, 'A', 'pseudo_two', 20)
        assert x.shape == y.shape == (2, 2)
        assert x[:, 0].mean() == 0 and y[:, 0].mean() == 1


if __name__ == '__main__':
    unittest.main()
