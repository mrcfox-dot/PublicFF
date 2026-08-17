"""Calibration measurement: for predictions assigned ~70%, did the
predicted outcome actually happen ~70% of the time?

Every (candidate, probability) pair across every prediction is bucketed
(not just the top pick) - this is the standard reliability-diagram
approach for a multiclass probabilistic model. Stage 4 scope is measurement
infrastructure only (brief item 12) - no reweighting/rescaling of the
model based on this is implemented yet.
"""

from __future__ import annotations

from typing import Iterable, List


def compute_calibration_buckets(pairs: Iterable[dict], num_buckets: int = 10) -> List[dict]:
    """Buckets are ``num_buckets`` equal-width bins over [0, 1]
    (default 10 -> 0-10%, 10-20%, ..., 90-100%). For each bucket: how many
    (candidate, probability) observations fell in it, the mean predicted
    probability, and the actual rate at which that candidate really was
    the captain.
    """
    if num_buckets < 1:
        raise ValueError("num_buckets must be >= 1.")

    bucket_width = 1.0 / num_buckets
    buckets = [
        {
            "bucket_index": i,
            "range_low": round(i * bucket_width, 4),
            "range_high": round((i + 1) * bucket_width, 4),
            "count": 0,
            "sum_predicted": 0.0,
            "sum_actual": 0.0,
        }
        for i in range(num_buckets)
    ]

    for pair in pairs:
        actual = pair["actual_captain_id"]
        for candidate_id, prob in pair["candidate_probabilities"].items():
            index = min(int(prob * num_buckets), num_buckets - 1)  # prob==1.0 goes in the last bucket
            bucket = buckets[index]
            bucket["count"] += 1
            bucket["sum_predicted"] += prob
            bucket["sum_actual"] += 1.0 if candidate_id == actual else 0.0

    result = []
    for bucket in buckets:
        count = bucket["count"]
        result.append(
            {
                "bucket_index": bucket["bucket_index"],
                "range_low": bucket["range_low"],
                "range_high": bucket["range_high"],
                "count": count,
                "mean_predicted_probability": (bucket["sum_predicted"] / count) if count else None,
                "actual_occurrence_rate": (bucket["sum_actual"] / count) if count else None,
            }
        )
    return result


def format_calibration_report(buckets: List[dict]) -> str:
    lines = ["Calibration report:"]
    for b in buckets:
        low_pct, high_pct = int(b["range_low"] * 100), int(b["range_high"] * 100)
        if b["count"] == 0:
            lines.append(f"  Predicted {low_pct}-{high_pct}%: no observations")
            continue
        actual_pct = b["actual_occurrence_rate"] * 100
        lines.append(f"  Predicted {low_pct}-{high_pct}% (n={b['count']}): actual occurrence {actual_pct:.1f}%")
    return "\n".join(lines)
