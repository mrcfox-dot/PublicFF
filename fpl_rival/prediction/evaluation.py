"""Evaluation metrics for captain predictions.

Every function takes a list of "pairs": dicts with
``candidate_probabilities: Dict[player_id, float]`` and
``actual_captain_id: int`` - the shape both ``backtest.run_backtest`` and
``history_store.prediction_outcome_pairs`` produce.

Accuracy alone is not enough (Stage 4 brief item 11): a model saying
"Salah 51% / Palmer 49%" and one saying "Salah 99% / Palmer 1%" both score
identically on top-1 accuracy when Salah is captained, but are very
different predictions. ``log_loss`` and ``brier_score`` are what
distinguish them.
"""

from __future__ import annotations

import math
from typing import Iterable

EPSILON = 1e-9  # clamp for log(0) - documented, not hidden


def top1_accuracy(pairs: Iterable[dict]) -> float:
    pairs = list(pairs)
    if not pairs:
        return float("nan")
    correct = sum(
        1 for p in pairs if max(p["candidate_probabilities"], key=p["candidate_probabilities"].get) == p["actual_captain_id"]
    )
    return correct / len(pairs)


def top2_accuracy(pairs: Iterable[dict]) -> float:
    pairs = list(pairs)
    if not pairs:
        return float("nan")
    correct = 0
    for p in pairs:
        ranked = sorted(p["candidate_probabilities"], key=p["candidate_probabilities"].get, reverse=True)
        if p["actual_captain_id"] in ranked[:2]:
            correct += 1
    return correct / len(pairs)


def log_loss(pairs: Iterable[dict]) -> float:
    """Mean of -log(P(actual captain)). Lower is better; 0 is a perfect,
    fully-confident-and-correct prediction. Probabilities are clamped away
    from exactly 0 to avoid -inf on a single wrong-with-certainty call."""
    pairs = list(pairs)
    if not pairs:
        return float("nan")
    total = 0.0
    for p in pairs:
        prob = p["candidate_probabilities"].get(p["actual_captain_id"], 0.0)
        prob = max(prob, EPSILON)
        total += -math.log(prob)
    return total / len(pairs)


def brier_score(pairs: Iterable[dict]) -> float:
    """Mean multiclass Brier score: mean over predictions of
    sum((p_i - y_i)^2) across every candidate i, where y_i is 1 for the
    actual captain and 0 otherwise. 0 is perfect; higher is worse. Unlike
    top-1 accuracy this rewards well-calibrated confidence, not just
    picking the right winner."""
    pairs = list(pairs)
    if not pairs:
        return float("nan")
    total = 0.0
    for p in pairs:
        actual = p["actual_captain_id"]
        total += sum((prob - (1.0 if cid == actual else 0.0)) ** 2 for cid, prob in p["candidate_probabilities"].items())
    return total / len(pairs)


def evaluate(pairs: Iterable[dict]) -> dict:
    pairs = list(pairs)
    return {
        "num_predictions": len(pairs),
        "top1_accuracy": top1_accuracy(pairs),
        "top2_accuracy": top2_accuracy(pairs),
        "log_loss": log_loss(pairs),
        "brier_score": brier_score(pairs),
    }
