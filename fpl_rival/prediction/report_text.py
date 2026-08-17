"""Presentation layer: deterministic template text for a CaptainPrediction.

No LLM prose anywhere in this module - every sentence is a fixed template
filled in from numbers the engine already computed, matching the Stage 4
brief's example output exactly. If a template's trigger condition isn't
met by any candidate, that line is simply omitted - nothing is invented to
fill a gap.
"""

from __future__ import annotations

from typing import Dict, List

from .models import CaptainPrediction, PredictionConfig

SYNTHETIC_BANNER = "*** SYNTHETIC HISTORY - NOT REAL FPL DATA ***"


def _name(player_names: Dict[int, str], player_id: int) -> str:
    return player_names.get(player_id, f"Player {player_id}")


def primary_reasons(prediction: CaptainPrediction, player_names: Dict[int, str], config: PredictionConfig) -> List[str]:
    """One deterministic template sentence per feature category, included
    only when that category's raw value clearly favours the top candidate
    (it is both the maximum among all candidates AND past a configured
    threshold where one applies)."""
    top_id = prediction.most_likely_captain
    top = prediction.candidate_features[top_id]
    top_name = _name(player_names, top_id)
    others = [f for cid, f in prediction.candidate_features.items() if cid != top_id]

    reasons: List[str] = []

    if top.personal_loyalty_rate >= config.strong_preference_rate and (
        not others or top.personal_loyalty_rate == max(f.personal_loyalty_rate for f in others + [top])
    ):
        reasons.append(f"Personal captain frequency: strong {top_name} preference")

    if top.recent_window_size > 0 and top.recent_window_captained == max(
        (f.recent_window_captained for f in others + [top]), default=0
    ) and top.recent_window_captained > 0:
        reasons.append(
            f"Recent behaviour: {top_name} captained {top.recent_window_captained} of last {top.recent_window_size}"
        )

    if top.league_consensus_rate >= config.dominant_consensus_rate and top.league_consensus_rate == max(
        f.league_consensus_rate for f in others + [top]
    ):
        reasons.append(f"League consensus: {top_name} dominant")

    ep_values = {cid: f.expected_points_value for cid, f in prediction.candidate_features.items() if f.expected_points_value is not None}
    if ep_values and max(ep_values, key=ep_values.get) == top_id:
        reasons.append(f"Expected points: {top_name} highest")

    return reasons


def render_captain_prediction(
    prediction: CaptainPrediction,
    player_names: Dict[int, str],
    config: PredictionConfig = PredictionConfig(),
    is_synthetic: bool = True,
) -> str:
    lines = ["FPL RIVAL, RIVAL PREDICTION"]
    if is_synthetic:
        lines.append(SYNTHETIC_BANNER)
    lines.append("")
    lines.append("Rival:")
    lines.append(prediction.rival_name)
    lines.append("")
    lines.append("Historical observations:")
    lines.append(f"{prediction.data_sufficiency.gameweeks_observed} gameweeks")
    lines.append("")
    lines.append("Prediction state:")
    lines.append(prediction.data_sufficiency.data_state)
    if prediction.data_sufficiency.low_personal_data:
        lines.append("LOW PERSONAL DATA")
    lines.append("")
    lines.append("Expected captain:")
    lines.append("")

    ranked = sorted(prediction.probabilities.items(), key=lambda kv: -kv[1])
    name_width = max((len(_name(player_names, cid)) for cid, _ in ranked), default=8) + 4
    for cid, prob in ranked:
        name = _name(player_names, cid)
        lines.append(f"{name:<{name_width}}{100 * prob:5.1f}%")

    lines.append("")
    lines.append("Most likely:")
    lines.append(_name(player_names, prediction.most_likely_captain))
    lines.append("")
    lines.append("Confidence:")
    lines.append(prediction.data_sufficiency.confidence_label)
    lines.append("")

    reasons = primary_reasons(prediction, player_names, config)
    if reasons:
        lines.append("Primary reasons:")
        lines.append("")
        lines.extend(reasons)

    return "\n".join(lines)


def render_contribution_breakdown(prediction: CaptainPrediction, player_names: Dict[int, str]) -> str:
    """Per-candidate contribution breakdown - the debugging/transparency
    view the brief's item 4 example shows (e.g. "Personal history: +1.4")."""
    lines = []
    for cid, prob in sorted(prediction.probabilities.items(), key=lambda kv: -kv[1]):
        features = prediction.candidate_features[cid]
        lines.append(_name(player_names, cid).upper())
        lines.append("")
        lines.append(f"Probability: {100 * prob:.0f}%")
        lines.append("")
        lines.append("Contributions:")
        lines.append("")
        for component, value in features.contributions.items():
            label = component.replace("_", " ").capitalize()
            sign = "+" if value >= 0 else ""
            lines.append(f"{label}: {sign}{value:.2f}")
        lines.append("")
    return "\n".join(lines).rstrip()
