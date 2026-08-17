"""Presentation layer: renders Captain Battle (and related) results as CLI text.

This module only formats numbers the engine already computed - it never
computes a metric, and it never tells Chris to "buy" or "sell" anyone. Per
Stage 3 scope, this is a captaincy comparison tool, not a recommendation
engine with a chat voice.
"""

from __future__ import annotations

from .captain_battle import CaptainBattleResult

SYNTHETIC_BANNER = "*** SYNTHETIC PROJECTIONS - NOT REAL FPL PREDICTIONS ***"


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def render_captain_battle_report(result: CaptainBattleResult, num_managers: int, is_synthetic: bool = True) -> str:
    lines = ["FPL RIVAL, CAPTAIN BATTLE"]
    if is_synthetic:
        lines.append(SYNTHETIC_BANNER)
    lines.append("")
    lines.append("Current league position:")
    lines.append(f"{result.starting_position} / {num_managers}")
    lines.append("")
    lines.append("Gap to leader:")
    lines.append(f"{result.starting_gap_to_leader:.0f} points")
    lines.append("")
    lines.append("Simulations:")
    lines.append(f"{result.num_simulations:,}")
    lines.append("")
    lines.append("Captain options:")
    lines.append("")

    for candidate in result.candidates:
        r = candidate.result
        lines.append(candidate.player_name.upper())
        lines.append("")
        lines.append(f"Expected GW points: {r.expected_gameweek_points:.1f}")
        lines.append(f"P(finish 1st): {_pct(r.prob_finish_first)}")
        lines.append(f"P(move up): {_pct(r.prob_move_up)}")
        lines.append(f"P(move down): {_pct(r.prob_move_down)}")
        lines.append(f"Expected position: {r.expected_position:.1f}")
        lines.append(f"Expected leader gap: {r.expected_leader_gap:.1f}")
        lines.append("")

    points_winner = result.expected_points_winner
    objective_winner = result.objective_winner

    lines.append("EXPECTED POINTS WINNER:")
    lines.append(points_winner.player_name)
    lines.append("")
    lines.append(f"MINI LEAGUE WINNER ({result.objective}):")
    lines.append(objective_winner.player_name)

    if points_winner.player_id != objective_winner.player_id:
        lines.append("")
        lines.append(
            "(Highest expected points and highest mini-league win probability "
            "point to different captains here - see the Stage 3 report.)"
        )

    return "\n".join(lines)


def render_shared_squad_report(low_diff_label: str, low_diff_gap_std: float, high_diff_label: str, high_diff_gap_std: float, is_synthetic: bool = True) -> str:
    lines = ["FPL RIVAL, SHARED PLAYER EXPOSURE CHECK"]
    if is_synthetic:
        lines.append(SYNTHETIC_BANNER)
    lines.append("")
    lines.append(f"{low_diff_label}: gap std dev = {low_diff_gap_std:.2f} points")
    lines.append(f"{high_diff_label}: gap std dev = {high_diff_gap_std:.2f} points")
    lines.append("")
    if high_diff_gap_std > low_diff_gap_std:
        lines.append(
            "As expected: more differing players between the two squads produces a "
            "larger spread in the points gap between them, because shared players' "
            "outcomes cancel out of the comparison and only the differing players "
            "drive relative movement."
        )
    else:
        lines.append(
            "Unexpected: the more-differentiated squad did not show a larger gap "
            "spread - see the Stage 3 report for investigation."
        )
    return "\n".join(lines)
