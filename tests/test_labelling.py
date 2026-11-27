"""Hand-made cases for the AHE definition.   Run:  python -m unittest discover -s tests"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ahe import config as C                     # noqa: E402
from ahe.labelling import assign_labels         # noqa: E402


def labels(map_values, drug_at=None, drug='norepinephrine'):
    n = len(map_values)
    df = pd.DataFrame({'patient_id': 1, 'period': np.arange(n), 'map': np.asarray(map_values, float)})
    rates = {d: np.zeros(n) for d in C.DRUGS}
    if drug_at is not None:
        rates[drug][drug_at] = 0.1
    return assign_labels(df, rates)['ahe'].tolist()


class TestSustainedHypotension(unittest.TestCase):
    def test_single_low_period_is_not_an_event(self):
        self.assertEqual(sum(labels([80] * 5 + [60] + [80] * 5)), 0)

    def test_two_consecutive_low_periods_are_an_event(self):
        self.assertEqual(labels([80, 80, 60, 60, 80, 80]), [0, 0, 1, 1, 0, 0])

    def test_episode_ends_after_two_periods_at_or_above_65(self):
        # one period ≥ 65 does not end the episode: the following period < 65 still belongs to it
        self.assertEqual(labels([80, 60, 60, 70, 60, 80, 80, 80]), [0, 1, 1, 0, 1, 0, 0, 0])

    def test_after_two_normal_periods_a_single_low_period_is_not_an_event(self):
        self.assertEqual(labels([80, 60, 60, 80, 80, 60, 80, 80]), [0, 1, 1, 0, 0, 0, 0, 0])

    def test_periods_without_monitoring_are_never_events(self):
        self.assertEqual(sum(labels([80, 0, 0, 0, 80])), 0)


class TestClinicallySignificantHypotension(unittest.TestCase):
    def test_single_low_period_with_infusion_in_the_same_period(self):
        self.assertEqual(labels([80] * 10 + [60] + [80] * 10, drug_at=10)[10], 1)

    def test_infusion_within_32_minutes_after(self):
        self.assertEqual(labels([80] * 10 + [60] + [80] * 10, drug_at=18)[10], 1)      # 8 periods later

    def test_infusion_within_32_minutes_before(self):
        self.assertEqual(labels([80] * 10 + [60] + [80] * 10, drug_at=2)[10], 1)       # 8 periods earlier

    def test_infusion_more_than_32_minutes_away(self):
        self.assertEqual(labels([80] * 10 + [60] + [80] * 10, drug_at=19)[10], 0)      # 9 periods later
        self.assertEqual(labels([80] * 10 + [60] + [80] * 10, drug_at=1)[10], 0)       # 9 periods earlier

    def test_any_vasoactive_drug_counts(self):
        for drug in C.DRUGS:
            self.assertEqual(labels([80] * 10 + [60] + [80] * 10, drug_at=12, drug=drug)[10], 1, drug)

    def test_normal_map_with_infusion_is_not_an_event(self):
        self.assertEqual(sum(labels([80] * 20, drug_at=10)), 0)


if __name__ == '__main__':
    unittest.main()
