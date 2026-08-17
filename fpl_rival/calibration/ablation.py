"""Feature-family ablation: zero one component's weight at a time and
measure the change in validation performance.

Run on the VALIDATION split (not test) - ablation is diagnostic, meant to
explain the fitted model's behaviour, not a second model-selection step
that would eat into the held-out test set's integrity.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

from fpl_rival.prediction.evaluation import evaluate
from fpl_rival.prediction.models import PredictionConfig

from .harness import ExpectedPointsLookup, filter_phase, run_stage4_over_dataset, to_eval_pairs
from .models import DatasetSplit, HistoricalRecord, VALIDATION

# feature family -> the config weight(s) zeroing it out corresponds to
FEATURE_FAMILIES = {
    "no_personal_history": ("weight_personal_loyalty",),
    "no_recency": ("weight_recency",),
    "no_concentration": ("weight_concentration",),
    "no_consensus": ("weight_consensus",),
    "no_expected_points": ("weight_expected_points",),
}


@dataclass(frozen=True)
class AblationRow:
    feature_family: str
    removed_parameters: Tuple[str, ...]
    top1_accuracy: float
    log_loss: float
    brier_score: float
    log_loss_delta: float  # positive = worse (removing this feature hurt)
    num_predictions: int


def run_ablation(
    records: List[HistoricalRecord],
    split: DatasetSplit,
    fitted_config: PredictionConfig,
    expected_points: Optional[ExpectedPointsLookup] = None,
    max_managers: Optional[int] = None,
    only_families: Optional[List[str]] = None,
) -> List[AblationRow]:
    baseline_results = run_stage4_over_dataset(records, fitted_config, split=split, expected_points=expected_points, max_managers=max_managers)
    baseline_pairs = to_eval_pairs(filter_phase(baseline_results, VALIDATION))
    baseline_metrics = evaluate(baseline_pairs)

    families = only_families or list(FEATURE_FAMILIES)
    rows = []
    for family in families:
        params_to_zero = FEATURE_FAMILIES[family]
        # skip ablating a feature the fitted config never used anyway (e.g.
        # weight_expected_points when no EP data exists) - nothing to remove.
        if all(getattr(fitted_config, p) == 0 for p in params_to_zero):
            continue
        ablated_config = replace(fitted_config, **{p: 0.0 for p in params_to_zero})
        results = run_stage4_over_dataset(records, ablated_config, split=split, expected_points=expected_points, max_managers=max_managers)
        pairs = to_eval_pairs(filter_phase(results, VALIDATION))
        metrics = evaluate(pairs)
        rows.append(
            AblationRow(
                feature_family=family,
                removed_parameters=params_to_zero,
                top1_accuracy=metrics["top1_accuracy"],
                log_loss=metrics["log_loss"],
                brier_score=metrics["brier_score"],
                log_loss_delta=metrics["log_loss"] - baseline_metrics["log_loss"],
                num_predictions=metrics["num_predictions"],
            )
        )
    return rows
