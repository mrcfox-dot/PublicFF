"""Unit tests for fpl_rival.calibration.stage3_impact."""

import unittest

from fpl_rival.calibration.stage3_impact import Stage3Scenario, compare_stage3_impact
from fpl_rival.fixtures import prediction_fixtures as pfx
from fpl_rival.prediction.models import PredictionConfig
from fpl_rival.simulation.models import SimulationConfig


def _scenario(history_builder, label):
    chris, league, projections, candidates, dave, dave_history, ep = pfx.full_pipeline_scenario(history_builder())
    return Stage3Scenario(
        label=label, chris=chris, league=league, projections=projections, candidate_captain_ids=candidates,
        rival=dave, rival_history=dave_history, target_gameweek=11, expected_points=ep,
    )


class TestStage3ImpactComparison(unittest.TestCase):
    def test_produces_one_row_per_scenario(self):
        scenarios = [_scenario(pfx.dave_salah_loyal_history, "loyal"), _scenario(pfx.dave_balanced_history, "balanced")]
        rows = compare_stage3_impact(
            scenarios, PredictionConfig(), PredictionConfig(weight_personal_loyalty=3.0),
            SimulationConfig(num_simulations=5000, random_seed=1, rival_captain_mode="probabilistic"),
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual({r.scenario_label for r in rows}, {"loyal", "balanced"})

    def test_probability_distributions_are_valid(self):
        scenarios = [_scenario(pfx.dave_salah_loyal_history, "loyal")]
        rows = compare_stage3_impact(
            scenarios, PredictionConfig(), PredictionConfig(),
            SimulationConfig(num_simulations=5000, random_seed=1, rival_captain_mode="probabilistic"),
        )
        row = rows[0]
        self.assertAlmostEqual(sum(row.manual_probabilities.values()), 1.0, places=6)
        self.assertAlmostEqual(sum(row.fitted_probabilities.values()), 1.0, places=6)

    def test_identical_configs_never_change_the_recommendation(self):
        scenarios = [_scenario(pfx.dave_salah_loyal_history, "loyal")]
        config = PredictionConfig()
        rows = compare_stage3_impact(
            scenarios, config, config, SimulationConfig(num_simulations=5000, random_seed=1, rival_captain_mode="probabilistic")
        )
        self.assertFalse(rows[0].recommendation_changed)


if __name__ == "__main__":
    unittest.main()
