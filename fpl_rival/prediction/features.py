"""Explicit, quantitative per-candidate features - the inputs to captain_model.py.

Every function here takes a slice of history that is ALREADY safe (the
caller - prediction_engine.py - is responsible for calling
``LeagueHistory.before(target_gameweek)`` first). These functions do not
themselves know what "the future" is; they just summarise whatever
observations they are given. Keeping the leakage boundary in one place
(models.py's ``LeagueHistory.before``) rather than scattered through this
module is deliberate - see prediction_engine.py's docstring.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Optional, Tuple

from .models import CaptainObservation


def personal_loyalty_rate(history: Tuple[CaptainObservation, ...], candidate_id: int) -> Tuple[int, int, float]:
    """(times_owned, times_captained, times_captained / times_owned).

    "Owned" here means the candidate appeared in that gameweek's squad -
    this is the conditional "when I had the option, how often did I take
    it" rate. Returns rate=0.0 (not NaN) when never owned, since a
    candidate the manager has literally never had the chance to captain
    contributes nothing to this feature - the shrinkage mechanism handles
    down-weighting it further via low overall data volume.
    """
    owned = [o for o in history if candidate_id in o.owned_player_ids]
    captained = [o for o in owned if o.captain_id == candidate_id]
    times_owned = len(owned)
    times_captained = len(captained)
    rate = times_captained / times_owned if times_owned else 0.0
    return times_owned, times_captained, rate


def recency_weighted_rate(
    history: Tuple[CaptainObservation, ...],
    candidate_id: int,
    target_gameweek: int,
    decay_rate: float,
) -> float:
    """Same conditional rate as ``personal_loyalty_rate``, but each past
    gameweek's contribution is discounted by ``decay_rate ** age`` where
    ``age = target_gameweek - observation.gameweek`` (age >= 1, since
    history is already strictly before target_gameweek). A gameweek from
    last week counts far more than one from 20 gameweeks ago.
    """
    owned = [o for o in history if candidate_id in o.owned_player_ids]
    if not owned:
        return 0.0
    weighted_owned = 0.0
    weighted_captained = 0.0
    for o in owned:
        age = max(1, target_gameweek - o.gameweek)
        w = decay_rate**age
        weighted_owned += w
        if o.captain_id == candidate_id:
            weighted_captained += w
    return weighted_captained / weighted_owned if weighted_owned else 0.0


def recent_window_stats(history: Tuple[CaptainObservation, ...], candidate_id: int, window_size: int) -> Tuple[int, int]:
    """(times captained in the last ``window_size`` observed gameweeks,
    actual number of gameweeks in that window - may be less than
    ``window_size`` if history is shorter). Powers the "captained X of
    last N" wording in the text report."""
    recent = sorted(history, key=lambda o: o.gameweek, reverse=True)[:window_size]
    captained = sum(1 for o in recent if o.captain_id == candidate_id)
    return captained, len(recent)


def concentration_share(history: Tuple[CaptainObservation, ...], candidate_id: int) -> Tuple[int, float]:
    """(total_captain_decisions_ever, this candidate's unconditional share
    of them). Distinct from ``personal_loyalty_rate``: this asks "of every
    captaincy decision this manager has ever made, what fraction went to
    this specific player" - it does not condition on ownership, so it
    captures "keeps going back to the same small pool" even across squad
    changes.
    """
    total = len(history)
    if total == 0:
        return 0, 0.0
    times_captained = sum(1 for o in history if o.captain_id == candidate_id)
    return total, times_captained / total


def concentration_index(history: Tuple[CaptainObservation, ...]) -> float:
    """Herfindahl-style concentration index over the manager's WHOLE
    captain-choice distribution (not per-candidate): sum(share_i^2) over
    every player ever captained. 1.0 = always captains the same single
    player; close to 0 = spreads captaincy across many different players.
    Used only for the confidence/data-sufficiency report, not the score.
    """
    if not history:
        return 0.0
    counts = Counter(o.captain_id for o in history)
    total = len(history)
    return sum((c / total) ** 2 for c in counts.values())


def league_consensus_rate(
    other_entries_history: Dict[int, Tuple[CaptainObservation, ...]], candidate_id: int
) -> float:
    """Fraction of every OTHER relevant rival's past captain decisions
    (pooled across all of them) that went to this candidate. This is a
    legitimate pre-deadline signal: it only uses other managers' OWN past
    gameweeks (never this gameweek's actual picks, which are not public
    before the deadline - see Stage 1's documented data boundaries), so it
    is best read as "how often has this player been the league's captaincy
    choice historically", a track-record proxy for consensus rather than
    a live snapshot of this week's picks.
    """
    all_observations = [o for obs in other_entries_history.values() for o in obs]
    if not all_observations:
        return 0.0
    matches = sum(1 for o in all_observations if o.captain_id == candidate_id)
    return matches / len(all_observations)


def historical_response_to_expected_points(
    history: Tuple[CaptainObservation, ...],
    historical_ep_by_gw: Optional[Dict[int, Dict[int, float]]],
    historical_template_captain_by_gw: Optional[Dict[int, int]] = None,
) -> Dict[str, Optional[float]]:
    """Measures (does not classify) whether this manager's past captain
    choices tracked the highest-projected player and/or the league
    template captain, using only gameweeks where the relevant historical
    data was supplied. Returns None for any rate whose required data is
    missing - never a fabricated number. These rates are exposed for
    transparency/future calibration; per Stage 4 scope they are not yet
    fed back into the scoring model (see the completion report).
    """
    result: Dict[str, Optional[float]] = {
        "follows_highest_projected_rate": None,
        "follows_template_rate": None,
        "differential_rate": None,
        "gameweeks_with_ep_data": 0,
    }
    if historical_ep_by_gw:
        comparable = [o for o in history if o.gameweek in historical_ep_by_gw]
        if comparable:
            follows_highest = 0
            for o in comparable:
                ep_this_gw = historical_ep_by_gw[o.gameweek]
                owned_with_ep = {pid: ep_this_gw[pid] for pid in o.owned_player_ids if pid in ep_this_gw}
                if owned_with_ep and max(owned_with_ep, key=owned_with_ep.get) == o.captain_id:
                    follows_highest += 1
            result["follows_highest_projected_rate"] = follows_highest / len(comparable)
            result["differential_rate"] = 1.0 - result["follows_highest_projected_rate"]
            result["gameweeks_with_ep_data"] = len(comparable)

    if historical_template_captain_by_gw:
        comparable_t = [o for o in history if o.gameweek in historical_template_captain_by_gw]
        if comparable_t:
            follows_template = sum(
                1 for o in comparable_t if o.captain_id == historical_template_captain_by_gw[o.gameweek]
            )
            result["follows_template_rate"] = follows_template / len(comparable_t)

    return result
