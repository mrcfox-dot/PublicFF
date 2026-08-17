"""Buckets predictions by how much personal history was available at
prediction time, so we can see exactly when personal rival behaviour
starts pulling its weight versus the prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from fpl_rival.prediction.evaluation import evaluate

DEFAULT_BUCKETS: Tuple[Tuple[str, int, int], ...] = (
    ("0-2", 0, 2),
    ("3-5", 3, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21+", 21, 10_000),
)


@dataclass(frozen=True)
class HistoryDepthRow:
    bucket_label: str
    sample_size: int
    top1_accuracy: float
    log_loss: float
    brier_score: float


def bucket_by_history_depth(results: List[dict], buckets: Tuple[Tuple[str, int, int], ...] = DEFAULT_BUCKETS) -> List[HistoryDepthRow]:
    """``results`` items need ``gameweeks_observed``, ``candidate_probabilities``
    and ``actual_captain_id`` (the shape ``harness.run_stage4_over_dataset``
    produces)."""
    rows = []
    for label, lo, hi in buckets:
        bucketed = [r for r in results if lo <= r["gameweeks_observed"] <= hi]
        pairs = [{"candidate_probabilities": r["candidate_probabilities"], "actual_captain_id": r["actual_captain_id"]} for r in bucketed]
        metrics = evaluate(pairs)
        rows.append(
            HistoryDepthRow(
                bucket_label=label,
                sample_size=len(bucketed),
                top1_accuracy=metrics["top1_accuracy"],
                log_loss=metrics["log_loss"],
                brier_score=metrics["brier_score"],
            )
        )
    return rows
