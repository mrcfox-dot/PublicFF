"""Presentation layer for the calibration CLI - deterministic text only,
no LLM prose, matching the section layout the Stage 4.5 brief specifies."""

from __future__ import annotations

from typing import List, Optional

SYNTHETIC_BANNER = "*** SYNTHETIC DATASET - NOT REAL MANAGER HISTORY - INFRASTRUCTURE VALIDATION ONLY ***"


def _metrics_row(label: str, metrics: dict, coverage: Optional[int] = None, total: Optional[int] = None) -> str:
    if metrics["num_predictions"] == 0:
        coverage_str = f" (0/{total} - skipped, no legitimate signal)" if total else " (no predictions)"
        return f"{label:<26}{coverage_str}"
    top1 = f"{100 * metrics['top1_accuracy']:.1f}%"
    top2 = f"{100 * metrics['top2_accuracy']:.1f}%"
    ll = f"{metrics['log_loss']:.3f}"
    brier = f"{metrics['brier_score']:.3f}"
    cov = f" n={metrics['num_predictions']}" + (f"/{total}" if total and metrics["num_predictions"] != total else "")
    return f"{label:<26}Top1 {top1:<8}Top2 {top2:<8}LogLoss {ll:<8}Brier {brier:<8}{cov}"


def render_dataset_summary(summary) -> str:
    lines = ["DATASET SUMMARY", "-" * 40]
    if summary.is_synthetic:
        lines.append(SYNTHETIC_BANNER)
    lines.append(f"Managers: {summary.num_managers}")
    lines.append(f"Gameweeks: {summary.num_gameweeks}")
    lines.append(f"Captain decisions: {summary.num_records}")
    lines.append(f"Seasons: {', '.join(summary.seasons) if summary.seasons else 'none'}")
    lines.append(f"Gameweek range: {summary.min_gameweek}-{summary.max_gameweek}")
    if summary.warnings:
        lines.append("Warnings:")
        for w in summary.warnings:
            lines.append(f"  - {w}")
    return "\n".join(lines)


def render_split(split) -> str:
    return "SPLIT\n" + "-" * 40 + "\n" + split.describe()


def render_performance_table(title: str, rows: List[tuple], total: Optional[int] = None) -> str:
    """``rows``: list of (label, metrics_dict)."""
    lines = [title, "-" * 40]
    for label, metrics in rows:
        lines.append(_metrics_row(label, metrics, total=total))
    return "\n".join(lines)


def render_calibration(buckets: List[dict], ece: float) -> str:
    lines = ["CALIBRATION", "-" * 40]
    for b in buckets:
        low, high = int(b["range_low"] * 100), int(b["range_high"] * 100)
        if b["count"] == 0:
            lines.append(f"  {low:>3}-{high:>3}%: no observations")
            continue
        lines.append(
            f"  {low:>3}-{high:>3}% (n={b['count']:>4}): mean predicted {100*b['mean_predicted_probability']:5.1f}%  "
            f"actual {100*b['actual_occurrence_rate']:5.1f}%"
        )
    lines.append(f"Expected Calibration Error: {ece:.4f}")
    return "\n".join(lines)


def render_history_depth(rows) -> str:
    lines = ["HISTORY DEPTH ANALYSIS", "-" * 40]
    for r in rows:
        if r.sample_size == 0:
            lines.append(f"  {r.bucket_label:<8} n=0 (no observations in this bucket)")
            continue
        lines.append(
            f"  {r.bucket_label:<8} n={r.sample_size:<5} Top1 {100*r.top1_accuracy:5.1f}%  "
            f"LogLoss {r.log_loss:.3f}  Brier {r.brier_score:.3f}"
        )
    return "\n".join(lines)


def render_ablation(rows) -> str:
    lines = ["ABLATION RESULTS (validation split; positive delta = removing the feature made log loss worse)", "-" * 40]
    if not rows:
        lines.append("  No ablations run (fitted model does not use any weighted feature to ablate).")
        return "\n".join(lines)
    for r in rows:
        sign = "+" if r.log_loss_delta >= 0 else ""
        lines.append(
            f"  {r.feature_family:<20} LogLoss {r.log_loss:.3f} ({sign}{r.log_loss_delta:.3f} vs fitted)  "
            f"Top1 {100*r.top1_accuracy:.1f}%  n={r.num_predictions}"
        )
    return "\n".join(lines)


def render_predictability(rows) -> str:
    lines = ["MANAGER PREDICTABILITY", "-" * 40]
    for r in rows:
        name = r.manager_name or f"Manager {r.manager_id}"
        if r.label == "INSUFFICIENT_DATA":
            lines.append(f"  {name:<28} INSUFFICIENT_DATA (n={r.num_predictions})")
            continue
        lines.append(
            f"  {name:<28} {r.label:<10} score={r.predictability_score:.2f}  "
            f"(concentration={r.concentration_index:.2f}, top1_acc={r.top1_accuracy:.2f}, n={r.num_predictions})"
        )
    return "\n".join(lines)


def render_stage3_impact(rows) -> str:
    lines = ["STAGE 3 IMPACT", "-" * 40]
    for r in rows:
        changed = "CHANGED" if r.recommendation_changed else "unchanged"
        lines.append(f"  {r.scenario_label}: manual -> {r.manual_objective_winner} | fitted -> {r.fitted_objective_winner}  [{changed}]")
    num_changed = sum(1 for r in rows if r.recommendation_changed)
    lines.append(f"Recommendation changed in {num_changed}/{len(rows)} scenario(s).")
    return "\n".join(lines)


def render_recommendation(text_lines: List[str]) -> str:
    return "RECOMMENDATION\n" + "-" * 40 + "\n" + "\n".join(text_lines)
