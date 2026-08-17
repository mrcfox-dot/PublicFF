"""Data models for the Stage 4 Rival Prediction Engine.

Nothing here touches the network. These are the only shapes the prediction
engine understands - they can be populated from a legitimate historical
dataset (see ``CaptainObservation``'s docstring for the exact fields Stage
4 expects, matching the "historical dataset interface" required by the
brief) or from synthetic fixtures, without the engine itself changing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class CaptainObservation:
    """One historical gameweek's decision for one manager.

    This is the generic "historical dataset interface" (Stage 4 brief item
    13): manager ID, gameweek, squad, captain, vice-captain, transfers,
    points, rank. Only ``gameweek``, ``owned_player_ids`` and
    ``captain_id`` are used by the current feature set; the rest are
    carried for a future backtester/feature that needs them, and because a
    real historical dataset will naturally have them.
    """

    gameweek: int
    owned_player_ids: Tuple[int, ...]
    captain_id: int
    vice_captain_id: Optional[int] = None
    transfers: int = 0
    points: Optional[int] = None
    rank: Optional[int] = None

    def __post_init__(self):
        if self.captain_id not in self.owned_player_ids:
            raise ValueError(
                f"GW{self.gameweek}: captain_id {self.captain_id} is not in owned_player_ids - "
                "a historical observation must be internally consistent."
            )


@dataclass(frozen=True)
class LeagueHistory:
    """Every observed manager's chronological decision history.

    ``.before(gameweek)`` is the ONE mechanism the prediction engine uses
    to prevent future leakage - see prediction_engine.py. It is exposed
    here (not buried in the engine) so backtesting and any caller can
    reuse the identical, tested leakage boundary.
    """

    observations_by_entry: Dict[int, Tuple[CaptainObservation, ...]] = field(default_factory=dict)

    def before(self, gameweek: int) -> "LeagueHistory":
        """Every observation with ``gameweek < gameweek`` (strictly), for every manager."""
        return LeagueHistory(
            {
                entry_id: tuple(o for o in obs if o.gameweek < gameweek)
                for entry_id, obs in self.observations_by_entry.items()
            }
        )

    def for_entry(self, entry_id: int) -> Tuple[CaptainObservation, ...]:
        return self.observations_by_entry.get(entry_id, ())

    def other_entries(self, exclude_entry_id: int) -> Dict[int, Tuple[CaptainObservation, ...]]:
        return {eid: obs for eid, obs in self.observations_by_entry.items() if eid != exclude_entry_id}


# -- Data sufficiency / confidence states ------------------------------------------------

PRIOR_DRIVEN = "PRIOR_DRIVEN"
MIXED = "MIXED"
BEHAVIOUR_DRIVEN = "BEHAVIOUR_DRIVEN"

CONFIDENCE_LOW = "Low"
CONFIDENCE_MODERATE = "Moderate"
CONFIDENCE_HIGH = "High"


@dataclass(frozen=True)
class CandidateFeatures:
    """Every quantitative feature computed for one captain candidate,
    exposed raw (before weighting) so a caller can see exactly what the
    model saw - "debugging and future calibration" per the brief."""

    player_id: int
    times_owned: int
    times_captained: int
    personal_loyalty_rate: float  # times_captained / times_owned, in [0, 1]
    recency_weighted_rate: float  # same idea, exponentially recency-weighted
    concentration_share: float  # times_captained / total_captain_decisions_ever, in [0, 1]
    league_consensus_rate: float  # fraction of OTHER relevant rivals' past decisions that captained this player
    global_consensus_value: Optional[float]  # optional external input, None if not supplied
    expected_points_value: Optional[float]  # optional external input, None if not supplied
    recent_window_captained: int = 0  # times captained within the last `recent_window_size` observed gameweeks
    recent_window_size: int = 0  # actual number of recent observations available (<= config.recent_window_size)
    contributions: Dict[str, float] = field(default_factory=dict)  # component_name -> weighted contribution to score
    raw_score: float = 0.0


@dataclass(frozen=True)
class DataSufficiency:
    gameweeks_observed: int
    captain_decisions_observed: int
    shrinkage_weight: float  # 0 (pure prior) .. 1 (pure personal), see captain_model.py
    data_state: str  # PRIOR_DRIVEN / MIXED / BEHAVIOUR_DRIVEN
    low_personal_data: bool
    confidence_label: str  # Low / Moderate / High
    concentration_index: float  # Herfindahl-style index over the manager's full captain history, in [0, 1]


@dataclass(frozen=True)
class CaptainPrediction:
    rival_entry_id: int
    rival_name: str
    target_gameweek: int
    probabilities: Dict[int, float]  # player_id -> probability, sums to ~1
    candidate_features: Dict[int, CandidateFeatures]
    data_sufficiency: DataSufficiency
    config_version: str

    @property
    def most_likely_captain(self) -> int:
        return max(self.probabilities, key=self.probabilities.get)


@dataclass(frozen=True)
class PredictionConfig:
    """Every tunable weight/threshold in the prediction model. Nothing in
    features.py or captain_model.py should hardcode a magic number that
    belongs here."""

    version: str = "stage4-v1"

    # -- score component weights (all in the same additive score) ---------------
    weight_personal_loyalty: float = 2.0
    weight_recency: float = 1.5
    weight_concentration: float = 1.0
    weight_consensus: float = 1.0
    weight_global_consensus: float = 0.8
    weight_expected_points: float = 1.2

    # -- recency weighting -------------------------------------------------------
    # per-gameweek exponential decay applied to older observations:
    # weight(age) = recency_decay_rate ** age, age = target_gw - observation_gw.
    recency_decay_rate: float = 0.85

    # -- shrinkage toward the prior (Stage 4 brief item 5) -----------------------
    # shrinkage_weight = n / (n + k), n = total personal captain decisions
    # observed. At n=0 -> 0 (pure prior); at n=k -> 0.5; as n grows -> 1.
    # This is what makes the personal-history components fade in gradually
    # rather than an arbitrary hard switch like "after GW5".
    personal_shrinkage_k: float = 6.0

    # -- softmax --------------------------------------------------------------------
    softmax_temperature: float = 1.0

    # -- data sufficiency thresholds (all quantitative, all configurable) -------------
    low_personal_data_gw_threshold: int = 3  # gameweeks_observed below this -> LOW PERSONAL DATA
    prior_driven_max_shrinkage: float = 0.35  # shrinkage_weight <= this -> PRIOR_DRIVEN
    behaviour_driven_min_shrinkage: float = 0.65  # shrinkage_weight >= this -> BEHAVIOUR_DRIVEN
    high_confidence_min_gw: int = 10  # gameweeks_observed >= this (and BEHAVIOUR_DRIVEN) -> High confidence

    # -- feature-specific thresholds used by the deterministic text templates ---------
    strong_preference_rate: float = 0.6  # personal_loyalty_rate >= this -> "strong preference" wording
    dominant_consensus_rate: float = 0.5  # league_consensus_rate >= this -> "dominant" wording
    recent_window_size: int = 4  # "captained X of last N" window size for the text report
