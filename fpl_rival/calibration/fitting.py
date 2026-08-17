"""Parameter fitting via coordinate search, minimizing VALIDATION log loss.

Deliberately simple (Stage 4.5 brief item 6 explicitly asks for grid /
random / coordinate / basic numerical search, not a machine-learning
framework): for each fittable parameter in turn, try every value in its
grid (holding all other parameters fixed at their current best), keep
whichever value minimised validation log loss, then move to the next
parameter. Repeat for a few passes until a full pass makes no
improvement. This is fully deterministic - no randomness anywhere - so
running it twice on the same data always produces the same fitted config
(the "parameter search reproducibility" the brief's test list asks for).

Only fits parameters the dataset can actually support (see
``fittable_parameters``): e.g. ``weight_consensus`` is excluded when there
is only one manager in the dataset (no consensus signal exists to fit
against), and ``weight_expected_points``/``weight_global_consensus`` are
excluded unless the caller actually supplies that input. Data-sufficiency
thresholds (``prior_driven_max_shrinkage`` etc.) are never included here -
they classify a prediction after the fact and have no effect on the
probabilities log loss is computed from, so "fitting" them against log
loss would be meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

from fpl_rival.prediction.evaluation import evaluate
from fpl_rival.prediction.models import PredictionConfig

from .harness import ExpectedPointsLookup, filter_phase, run_stage4_over_dataset, to_eval_pairs
from .models import DatasetError, DatasetSplit, HistoricalRecord, VALIDATION

DEFAULT_GRIDS: Dict[str, Tuple[float, ...]] = {
    "weight_personal_loyalty": (0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
    "weight_recency": (0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
    "weight_concentration": (0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
    "weight_consensus": (0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
    "weight_global_consensus": (0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
    "weight_expected_points": (0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
    "personal_shrinkage_k": (2.0, 4.0, 6.0, 10.0, 15.0, 25.0),
    "recency_decay_rate": (0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99),
    "softmax_temperature": (0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
}

ALWAYS_FITTABLE = (
    "weight_personal_loyalty",
    "weight_recency",
    "weight_concentration",
    "personal_shrinkage_k",
    "recency_decay_rate",
    "softmax_temperature",
)


def fittable_parameters(records: List[HistoricalRecord], expected_points: Optional[ExpectedPointsLookup]) -> List[str]:
    params = list(ALWAYS_FITTABLE)
    num_managers = len({r.manager_id for r in records})
    if num_managers >= 2:
        params.append("weight_consensus")
    if expected_points:
        params.append("weight_expected_points")
    # weight_global_consensus needs an external global-popularity input this
    # module has no legitimate local source for - never included automatically.
    return params


@dataclass(frozen=True)
class FitResult:
    fitted_config: PredictionConfig
    base_validation_log_loss: float
    final_validation_log_loss: float
    fitted_parameters: Tuple[str, ...]
    search_log: Tuple[dict, ...] = field(default_factory=tuple)


def _validation_log_loss(
    records: List[HistoricalRecord], config: PredictionConfig, split: DatasetSplit,
    expected_points: Optional[ExpectedPointsLookup], max_managers: Optional[int],
) -> float:
    results = run_stage4_over_dataset(records, config, split=split, expected_points=expected_points, max_managers=max_managers)
    pairs = to_eval_pairs(filter_phase(results, VALIDATION))
    if not pairs:
        raise DatasetError(
            "No validation-phase predictions available - check the dataset split boundaries "
            "(train_through/validation_through) against the data's actual gameweek range."
        )
    return evaluate(pairs)["log_loss"]


def coordinate_search(
    records: List[HistoricalRecord],
    split: DatasetSplit,
    base_config: PredictionConfig = PredictionConfig(),
    expected_points: Optional[ExpectedPointsLookup] = None,
    grids: Optional[Dict[str, Tuple[float, ...]]] = None,
    max_passes: int = 3,
    max_managers: Optional[int] = None,
) -> FitResult:
    grids = grids or DEFAULT_GRIDS
    params = fittable_parameters(records, expected_points)

    current = base_config
    best_loss = _validation_log_loss(records, current, split, expected_points, max_managers)
    base_loss = best_loss
    search_log = [{"pass": 0, "parameter": None, "value": None, "validation_log_loss": best_loss}]

    for pass_num in range(1, max_passes + 1):
        improved_this_pass = False
        for param in params:
            best_value = getattr(current, param)
            for candidate_value in grids[param]:
                candidate_config = replace(current, **{param: candidate_value})
                loss = _validation_log_loss(records, candidate_config, split, expected_points, max_managers)
                search_log.append({"pass": pass_num, "parameter": param, "value": candidate_value, "validation_log_loss": loss})
                if loss < best_loss - 1e-9:
                    best_loss = loss
                    best_value = candidate_value
                    improved_this_pass = True
            current = replace(current, **{param: best_value})
        if not improved_this_pass:
            break

    return FitResult(
        fitted_config=current,
        base_validation_log_loss=base_loss,
        final_validation_log_loss=best_loss,
        fitted_parameters=tuple(params),
        search_log=tuple(search_log),
    )
