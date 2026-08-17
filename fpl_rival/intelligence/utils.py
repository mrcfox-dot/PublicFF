"""Small deterministic helpers shared across the analytics modules."""

from __future__ import annotations

from typing import Optional


def safe_pct(numerator: float, denominator: float) -> Optional[float]:
    """Percentage, or None if the denominator is zero/unknown."""
    if not denominator:
        return None
    return round(100.0 * numerator / denominator, 1)


def safe_div(numerator: float, denominator: float) -> Optional[float]:
    if not denominator:
        return None
    return numerator / denominator


def overlap_stats(a_ids: set, b_ids: set) -> dict:
    """Symmetric overlap between two element-id sets."""
    shared = a_ids & b_ids
    only_a = a_ids - b_ids
    only_b = b_ids - a_ids
    union = a_ids | b_ids
    return {
        "shared_count": len(shared),
        "shared_ids": shared,
        "only_a_ids": only_a,
        "only_b_ids": only_b,
        "a_count": len(a_ids),
        "b_count": len(b_ids),
        "overlap_pct_of_a": safe_pct(len(shared), len(a_ids)),
        "overlap_pct_of_b": safe_pct(len(shared), len(b_ids)),
        "jaccard_pct": safe_pct(len(shared), len(union)) if union else None,
    }


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))
