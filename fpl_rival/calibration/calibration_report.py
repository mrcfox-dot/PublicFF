"""Calibration measurement (reusing Stage 4's bucket computation directly)
plus Expected Calibration Error and an optional temperature-scaling
correction.

Measurement first, correction only if justified (Stage 4.5 brief item 12):
``fit_temperature`` searches for a temperature on the VALIDATION split;
the caller decides whether to actually apply it based on whether it
improves validation log loss over T=1 (no change) by a meaningful margin -
this module never silently recalibrates on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from fpl_rival.prediction.captain_model import softmax_probabilities
from fpl_rival.prediction.evaluation import evaluate
from fpl_rival.prediction.models import CaptainPrediction

DEFAULT_TEMPERATURE_GRID: Tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0)


def expected_calibration_error(buckets: List[dict]) -> float:
    """Weighted mean absolute gap between predicted probability and actual
    occurrence rate, across non-empty buckets - the standard ECE."""
    total = sum(b["count"] for b in buckets)
    if total == 0:
        return float("nan")
    return sum(
        (b["count"] / total) * abs(b["mean_predicted_probability"] - b["actual_occurrence_rate"])
        for b in buckets
        if b["count"] > 0
    )


def _rescale_with_temperature(prediction: CaptainPrediction, temperature: float) -> Dict[int, float]:
    """Recomputes softmax from the prediction's stored PRE-softmax scores
    (``CandidateFeatures.raw_score`` - the true logits) at a new
    temperature, independent of whatever temperature the original config
    used. At ``temperature`` equal to that original value, this reproduces
    the original probabilities exactly."""
    scores = {cid: f.raw_score for cid, f in prediction.candidate_features.items()}
    return softmax_probabilities(scores, temperature)


@dataclass(frozen=True)
class TemperatureFitResult:
    temperature: float
    baseline_validation_log_loss: float  # at T=1 (i.e. the model's own raw-score softmax, no correction)
    fitted_validation_log_loss: float
    improved: bool


def fit_temperature(
    results_with_predictions: List[dict], grid: Tuple[float, ...] = DEFAULT_TEMPERATURE_GRID
) -> TemperatureFitResult:
    """``results_with_predictions`` items need ``prediction`` (a full
    ``CaptainPrediction``, as produced by
    ``harness.run_stage4_over_dataset``) and ``actual_captain_id``.
    Searched on whatever split the caller filtered to first - pass
    VALIDATION-phase results only."""

    def loss_at(temperature: float) -> float:
        pairs = [
            {"candidate_probabilities": _rescale_with_temperature(r["prediction"], temperature), "actual_captain_id": r["actual_captain_id"]}
            for r in results_with_predictions
        ]
        return evaluate(pairs)["log_loss"]

    baseline_loss = loss_at(1.0)
    best_temperature, best_loss = 1.0, baseline_loss
    for temperature in grid:
        loss = loss_at(temperature)
        if loss < best_loss - 1e-9:
            best_loss, best_temperature = loss, temperature

    return TemperatureFitResult(
        temperature=best_temperature,
        baseline_validation_log_loss=baseline_loss,
        fitted_validation_log_loss=best_loss,
        improved=best_temperature != 1.0 and best_loss < baseline_loss - 1e-9,
    )


def apply_temperature(results_with_predictions: List[dict], temperature: float) -> List[dict]:
    """Returns evaluation-ready {candidate_probabilities, actual_captain_id}
    pairs with the temperature-rescaled probabilities."""
    return [
        {"candidate_probabilities": _rescale_with_temperature(r["prediction"], temperature), "actual_captain_id": r["actual_captain_id"]}
        for r in results_with_predictions
    ]
