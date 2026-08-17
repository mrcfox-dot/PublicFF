"""Manager-level captain predictability - evidence-based, not a reuse of
Stage 2's risk classification (that's derived from squad/transfer
behaviour with no ground truth; this is derived from how well the model
ACTUALLY predicted that specific manager's real historical captain
choices, in backtest).

predictability_score blends three signals, each already in [0, 1]:

* concentration_index - Stage 4's Herfindahl-style index over the
  manager's full captain history (repeatedly captaining a small pool -> high).
* 1 - mean_normalized_entropy - how confident/peaked the model's own
  predicted distributions were for this manager (low entropy -> the model
  found a clear signal -> high).
* top1_accuracy - how often the model's top pick was actually right, for
  this manager specifically, in backtest (ground truth, not a proxy).

All three weights are configurable and this stays a measurement, not a
black box - every manager's row exposes all three components plus the
blended score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from fpl_rival.prediction import features
from fpl_rival.prediction.evaluation import top1_accuracy

LOW, MODERATE, HIGH, INSUFFICIENT_DATA = "Low", "Moderate", "High", "INSUFFICIENT_DATA"


@dataclass(frozen=True)
class PredictabilityConfig:
    weight_concentration: float = 1 / 3
    weight_confidence: float = 1 / 3
    weight_accuracy: float = 1 / 3
    min_predictions: int = 5
    low_max_score: float = 0.4
    high_min_score: float = 0.7


@dataclass(frozen=True)
class ManagerPredictability:
    manager_id: int
    manager_name: Optional[str]
    num_predictions: int
    concentration_index: float
    mean_normalized_entropy: float
    top1_accuracy: float
    predictability_score: float
    label: str


def _normalized_entropy(probabilities: Dict[int, float]) -> float:
    n = len(probabilities)
    if n <= 1:
        return 0.0
    entropy = -sum(p * math.log(p) for p in probabilities.values() if p > 0)
    return entropy / math.log(n)


def compute_manager_predictability(
    manager_id: int,
    manager_name: Optional[str],
    full_history: tuple,  # tuple[CaptainObservation, ...] - the manager's WHOLE observed history
    stage4_results_for_manager: List[dict],  # this manager's rows from run_stage4_over_dataset (any phase)
    config: PredictabilityConfig = PredictabilityConfig(),
) -> ManagerPredictability:
    n = len(stage4_results_for_manager)
    if n < config.min_predictions:
        return ManagerPredictability(
            manager_id=manager_id, manager_name=manager_name, num_predictions=n,
            concentration_index=0.0, mean_normalized_entropy=0.0, top1_accuracy=float("nan"),
            predictability_score=float("nan"), label=INSUFFICIENT_DATA,
        )

    concentration = features.concentration_index(full_history)
    mean_entropy = sum(_normalized_entropy(r["candidate_probabilities"]) for r in stage4_results_for_manager) / n
    pairs = [{"candidate_probabilities": r["candidate_probabilities"], "actual_captain_id": r["actual_captain_id"]} for r in stage4_results_for_manager]
    accuracy = top1_accuracy(pairs)

    score = (
        config.weight_concentration * concentration
        + config.weight_confidence * (1 - mean_entropy)
        + config.weight_accuracy * accuracy
    )
    score = max(0.0, min(1.0, score))

    if score <= config.low_max_score:
        label = LOW
    elif score >= config.high_min_score:
        label = HIGH
    else:
        label = MODERATE

    return ManagerPredictability(
        manager_id=manager_id, manager_name=manager_name, num_predictions=n,
        concentration_index=round(concentration, 3), mean_normalized_entropy=round(mean_entropy, 3),
        top1_accuracy=round(accuracy, 3), predictability_score=round(score, 3), label=label,
    )


def compute_all_manager_predictability(
    stage4_results: List[dict],
    histories_by_manager: Dict[int, tuple],
    names_by_manager: Optional[Dict[int, str]] = None,
    config: PredictabilityConfig = PredictabilityConfig(),
) -> List[ManagerPredictability]:
    by_manager: Dict[int, List[dict]] = {}
    for r in stage4_results:
        by_manager.setdefault(r["manager_id"], []).append(r)

    rows = []
    for manager_id, rows_for_manager in by_manager.items():
        rows.append(
            compute_manager_predictability(
                manager_id,
                (names_by_manager or {}).get(manager_id),
                histories_by_manager.get(manager_id, ()),
                rows_for_manager,
                config,
            )
        )
    def sort_key(row: ManagerPredictability) -> float:
        # NaN (insufficient data) sorts last, not first.
        return -row.predictability_score if row.predictability_score == row.predictability_score else float("inf")

    return sorted(rows, key=sort_key)
