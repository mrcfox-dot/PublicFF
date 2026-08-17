"""All tunable thresholds for the Rival Intelligence Engine in one place.

Nothing in the analytics modules should hardcode a magic number that
belongs here. Construct a custom ``EngineConfig`` and pass it through to
override any default (e.g. for experimentation or a different league's
scoring quirks).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EngineConfig:
    # -- Data sufficiency gates ------------------------------------------------
    # Minimum number of scored gameweeks before a manager's transfer/risk
    # behaviour is classified at all. Below this: "Insufficient data".
    min_gameweeks_for_classification: int = 3
    # Minimum number of comparable captain data points (gameweeks where both
    # managers have known picks) before a captain-overlap % is reported.
    min_captain_datapoints: int = 2

    # -- Transfer behaviour bands ------------------------------------------------
    # (label, minimum average transfers per completed gameweek), checked in
    # order - first band whose minimum the manager's average meets or beats.
    transfer_bands: tuple = (
        ("Very active", 3.0),
        ("Active", 2.0),
        ("Typical", 1.0),
        ("Patient", 0.3),
        ("Very patient", 0.0),
    )

    # -- Risk behaviour ------------------------------------------------------------
    # risk_score in [0, 1] is a weighted blend of five normalized components;
    # weights should sum to 1.0 but are not forcibly normalized so the
    # relative contribution of each is easy to read off directly.
    risk_weight_hits: float = 0.25
    risk_weight_differential: float = 0.25
    risk_weight_captain_variety: float = 0.25
    risk_weight_transfer_frequency: float = 0.15
    risk_weight_squad_deviation: float = 0.10
    risk_aggressive_threshold: float = 0.60  # score >= this -> Aggressive
    risk_conservative_threshold: float = 0.35  # score <= this -> Conservative
    # A manager taking this many hits (in -4 units) or more over the season
    # scores the maximum "hits" component.
    risk_hits_saturation: int = 6
    # A manager averaging this many transfers/gw or more scores the maximum
    # "transfer frequency" component.
    risk_transfer_freq_saturation: float = 3.0
    # A manager using this many unique captains or more (relative to gws
    # played) scores the maximum "captain variety" component.
    risk_captain_variety_saturation: float = 0.6  # unique captains / gws played

    # -- Ownership thresholds (used by consensus / threats / weapons) -----------
    high_ownership_threshold_pct: float = 50.0
    low_ownership_threshold_pct: float = 20.0
    consensus_top_n: int = 10

    # -- Relevant rival selection (Section 6) --------------------------------------
    # relevance_score = 1 - (gap_weight * normalized_points_gap
    #                        + position_weight * normalized_position_distance)
    relevance_gap_weight: float = 0.6
    relevance_position_weight: float = 0.4
    relevance_score_threshold: float = 0.55
    # Immediate neighbours in the table are always considered relevant
    # regardless of points gap, up to this rank distance.
    always_relevant_position_distance: int = 2
    # Safety bounds so "relevant rivals" stays a manageable group even in a
    # very large or very tightly-packed league.
    min_relevant_rivals: int = 1
    max_relevant_rivals: int = 8

    # -- Effective exposure (Section 2) --------------------------------------------
    exposure_weight_squad_overlap: float = 0.7
    exposure_weight_captain_overlap: float = 0.3

    # -- Strategy state (Section 7) -------------------------------------------------
    season_total_gameweeks: int = 38
    # Leader is DEFEND if their margin over the next relevant challenger is
    # at least this many points.
    defend_min_margin_points: float = 15.0
    # A trailing manager is ATTACK once their deficit to the leader is at
    # least this many points (and not yet DESPERATE - see below).
    attack_min_deficit_points: float = 10.0
    # A trailing manager is DESPERATE once the deficit-per-remaining-gameweek
    # exceeds this - i.e. the required swing is no longer plausible.
    desperate_points_per_remaining_gw: float = 15.0

    # -- Transfer timing (Section 1) -------------------------------------------------
    late_transfer_threshold_hours: float = 6.0

    def transfer_band_labels(self) -> tuple:
        return tuple(label for label, _ in self.transfer_bands)
