"""Presentation layer: renders an engine report dict as CLI text.

This module only formats data the engine already computed - it never
computes a metric itself, and it never adds a recommendation ("buy",
"sell", "should"). That is explicitly out of scope for Stage 2.
"""

from __future__ import annotations

TOP_N_DISPLAY = 5


def render_report(report: dict) -> str:
    if report.get("mode") == "preseason":
        return _render_preseason(report)
    return _render_active(report)


def _fmt_pct(value) -> str:
    return f"{value}%" if value is not None else "n/a"


def _render_active(report: dict) -> str:
    lines = ["FPL RIVAL INTELLIGENCE"]
    if report.get("is_synthetic"):
        lines.append("*** SYNTHETIC TEST DATA - NOT LIVE FPL DATA ***")
    lines.append("")

    profiles = report["profiles"]
    chris = profiles[report["chris_entry_id"]]
    lines.append("Chris:")
    lines.append(f"Position: {chris['league_position']} / {len(profiles)}")
    lines.append(f"Points: {chris['total_points']}")

    strategy = report["strategy"]
    leader_gap = strategy["metrics"].get("points_behind_leader")
    lines.append(f"Leader gap: {leader_gap if leader_gap is not None else 'n/a'}")
    lines.append("")
    lines.append("Strategy state:")
    lines.append(strategy["state"])
    lines.append(f"({strategy['reasoning']})")
    lines.append("")

    primary_id = report.get("primary_rival_entry_id")
    if primary_id is not None:
        comp = report["comparisons"][primary_id]
        cls = report["classifications"][primary_id]
        relevant = next(
            (r for r in report["relevance"]["relevant_rivals"] if r["entry_id"] == primary_id), None
        )
        lines.append("Primary rival:")
        lines.append(comp.get("team_name") or comp.get("manager_name") or str(primary_id))
        if relevant is not None:
            gap = relevant["points_gap"]
            sign = "+" if gap >= 0 else ""
            lines.append(f"Gap: {sign}{gap} points")
        overlap = comp.get("squad_overlap_15")
        lines.append(
            f"Squad overlap: {_fmt_pct(overlap['shared_pct_of_chris_squad'])}" if overlap else "Squad overlap: n/a"
        )
        cap_exposure = comp.get("captain_exposure")
        cap_pct = cap_exposure.get("captain_overlap_pct") if cap_exposure else None
        lines.append(f"Captain overlap: {_fmt_pct(cap_pct)}")
        lines.append(f"Risk profile: {cls['risk_behaviour']['classification']}")
        lines.append(f"Transfer profile: {cls['transfer_behaviour']['classification']}")
    else:
        lines.append("Primary rival: none identified yet (insufficient scored data).")
    lines.append("")

    ownership_by_element = {
        row["element"]: row["mini_league_ownership_percentage"] for row in report["consensus"]["player_ownership"]
    }

    lines.append("Chris's weapons:")
    weapons = report["threats"]["weapons"]
    if weapons:
        for w in weapons[:TOP_N_DISPLAY]:
            pct = ownership_by_element.get(w["element"], w["ownership_pct_among_relevant_rivals"])
            lines.append(f"{w['name']}, {_fmt_pct(pct)} league ownership")
    else:
        lines.append("None identified.")
    lines.append("")

    lines.append("Major threats:")
    threats = report["threats"]["threats"]
    if threats:
        for t in threats[:TOP_N_DISPLAY]:
            lines.append(f"{t['name']}, {_fmt_pct(t['ownership_pct_among_relevant_rivals'])} ownership among relevant rivals")
    else:
        lines.append("None identified.")
    lines.append("")

    lines.append("League template:")
    template = report["consensus"]["most_common_players"]
    if template:
        for row in template[:TOP_N_DISPLAY]:
            lines.append(f"{row['name']}, {_fmt_pct(row['mini_league_ownership_percentage'])}")
    else:
        lines.append("None identified.")
    lines.append("")
    lines.append("(Relative league exposure only - this is not a buy/sell recommendation.)")

    return "\n".join(lines)


def _render_preseason(report: dict) -> str:
    pre = report["preseason"]
    lines = ["FPL RIVAL INTELLIGENCE - PRE-SEASON"]
    if report.get("is_synthetic"):
        lines.append("*** SYNTHETIC TEST DATA - NOT LIVE FPL DATA ***")
    lines.append("")
    lines.append(f"League: {pre['league_name']} (ID {pre['league_id']})")
    lines.append(f"Competitors: {pre['number_of_competitors']} (plus Chris = {pre['total_managers_in_league']} total)")
    lines.append("")
    lines.append("Membership:")
    for m in pre["membership"]:
        label = m["manager_name"] or "Unknown manager"
        team = m["team_name"] or "Unknown team"
        lines.append(f"  - {label} ({team}) [entry {m['entry_id']}]")
    lines.append("")

    chris_info = pre.get("chris_public_info")
    if chris_info and chris_info.get("past_seasons"):
        lines.append("Chris's public season history (career, not this league):")
        for s in chris_info["past_seasons"][-5:]:
            lines.append(f"  - {s['season_name']}: {s['total_points']} pts, rank {s['rank']}")
        lines.append("")

    lines.append("Analysis that will activate after Gameweek 1 is scored:")
    for item in pre["activates_after_gw1"]:
        lines.append(f"  - {item}")
    lines.append("")

    lines.append("Calculations that currently lack sufficient data:")
    for blocked in pre["currently_blocked"]:
        lines.append(f"  - {blocked['area']}: {blocked['reason']}")

    return "\n".join(lines)
