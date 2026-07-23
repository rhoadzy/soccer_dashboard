import unittest

import pandas as pd

from data.metrics import calculate_shot_on_target_percentages


class ShotOnTargetPercentageTests(unittest.TestCase):
    def test_percentages_use_aggregate_on_target_over_total_shots(self):
        matches = pd.DataFrame(
            [
                {
                    "shots_for": 10,
                    "shots_target": 4,
                    "shots_against": 8,
                    "shots_against_target": 2,
                },
                {
                    "shots_for": 20,
                    "shots_target": 11,
                    "shots_against": 12,
                    "shots_against_target": 8,
                },
            ]
        )

        shots_target_pct, shots_against_target_pct = calculate_shot_on_target_percentages(matches)

        self.assertAlmostEqual(shots_target_pct, 50.0)
        self.assertAlmostEqual(shots_against_target_pct, 50.0)

    def test_missing_or_zero_totals_return_zero(self):
        matches = pd.DataFrame(
            [{"shots_for": 0, "shots_target": 3, "shots_against": 0, "shots_against_target": 2}]
        )

        self.assertEqual(calculate_shot_on_target_percentages(matches), (0.0, 0.0))
        self.assertEqual(calculate_shot_on_target_percentages(pd.DataFrame()), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
