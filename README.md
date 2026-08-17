# FPL Rival - Stages 1, 2 & 3

A small local CLI that proves out retrieval and structuring of **public**
Fantasy Premier League (FPL) data for a manager and one of their classic
mini leagues. This is Stage 1 only: no website, no auth, no payments, no
AI/LLM calls, no simulations. Just: can we get the data, and is it enough?

## What it does

1. Looks up manager entry `5234355` (default; override with `--entry`).
2. Lists every classic mini league that manager belongs to (ID, name,
   current position, manager count where available).
3. Lets you pick one league (interactively, or via `--league-id`).
4. Pulls the full manager list for that league (handles multi-page
   standings and multi-page membership lists).
5. For every manager (up to `--max-managers`, default 50), pulls:
   gameweek-by-gameweek history, chips used, transfer history, overall
   points/rank, and squad picks (captain/vice/bench) for the latest
   **completed** gameweek.
6. Saves everything as plain JSON files under `data/run_<entry>_<league>_<timestamp>/`.
7. Prints a summary: your position, points behind the leader, points
   ahead of the manager below you, league size, squad overlap with each
   rival, players only you own vs. only they own, and captain choices -
   all for the latest completed gameweek.
8. Prints a final report distinguishing public data retrieved, data that
   needs FPL authentication, and data that cannot be obtained at all.

## Run it

```bash
pip install -r requirements.txt
python3 run.py
```

You'll be shown a numbered list of leagues and prompted to pick one.

Useful flags:

```bash
python3 run.py --entry 5234355 --league-id 1132827
python3 run.py --league-id 1132827 --max-managers 20
python3 run.py --include-system-leagues   # also show FPL's automatic Overall/country/club leagues
```

## Output layout

```
data/run_<entry>_<league>_<timestamp>/
  entry.json              raw manager entry payload
  league_standings.json   raw + merged standings/membership for the league
  managers/<entry_id>.json  per-manager history, transfers, chips, latest picks
  summary.json             the position/squad summary printed at the end
```

## A note on the data available right now

The 2026/27 season had not started as of this prototype's testing
(Gameweek 1's deadline is in the future), so **no manager anywhere has a
scored gameweek yet**. That means position/points and squad-overlap
sections will legitimately report "not available yet" rather than
inventing numbers - this is the intended behaviour (see requirement 10
in the brief: don't invent unavailable data), and it also happens to be
a good live test of the missing-data handling. Once gameweeks are
played, the same code paths populate normally.

## Known data boundaries

- **Public, no login needed:** entry profiles, league standings/membership,
  per-manager history/chips/transfers, and picks for any gameweek *after*
  that gameweek's deadline has passed.
- **Requires FPL login (not implemented here):** making transfers, changing
  your captain, joining/leaving leagues - none of this is needed for
  read-only analysis.
- **Not obtainable at all, even with login:** another manager's squad for
  a gameweek *before* that gameweek's deadline - FPL does not expose this
  to anyone, by design (keeps the deadline "blind").

## Design choices

- Plain files, no database - this is a structure-proving exercise, not a
  product.
- `requests` with retries/backoff is the only third-party dependency.
- `--max-managers` caps per-manager deep fetches so a very large league
  doesn't turn into thousands of API calls by accident; the full manager
  *listing* is still shown regardless of the cap.
- Every failed or missing API call is recorded in `data_gaps` in
  `summary.json` and echoed in the final report, instead of being
  swallowed silently.

---

# Stage 2 - Rival Intelligence Engine

Stage 2 adds a **deterministic analytics engine** on top of Stage 1's data
retrieval. It does not touch any Stage 1 file. The goal: build the
analytical foundations for eventually optimising *probability of winning
the mini league*, not just raw expected points - but Stage 2 itself makes
no recommendations. No LLM, no website, no auth, no database.

## Architecture

Three layers, each independently testable:

```
fpl_rival/api.py, league.py, manager.py     Stage 1 retrieval (unchanged)
fpl_rival/intelligence/collect.py            Stage 2 retrieval: adds full
                                              picks-history fetch, reuses
                                              Stage 1's league/standings code
fpl_rival/intelligence/models.py             Shared data model (ManagerData,
                                              LeagueData, GameContext) - the
                                              only thing the engine sees
fpl_rival/intelligence/{profiles,rival_analysis,
  classification,consensus,relevance,
  threats,strategy,preseason}.py             Pure analytics - no network,
                                              no I/O, fully unit-testable
fpl_rival/intelligence/engine.py             Orchestrates the above into one report
fpl_rival/intelligence/report_text.py        Presentation - formats the report as text
fpl_rival/fixtures/synthetic.py              Synthetic test data ONLY - never
                                              mixed with live data (is_synthetic
                                              flag, entry IDs >= 900000)
intelligence_cli.py                          Single-command launcher
tests/                                        51 unittest cases, all synthetic
```

The engine (`fpl_rival.intelligence.engine.run`) takes a `LeagueData` +
`GameContext` and returns a report dict - it never calls the network, so it
runs identically whether that data came from a live FPL run or a test
fixture.

## Run it

```bash
python3 intelligence_cli.py                        # live, interactive league pick
python3 intelligence_cli.py --league-id 1132827     # live, specific league
python3 intelligence_cli.py --fixture scenario_a    # synthetic demo, zero network calls
python3 intelligence_cli.py --fixture scenario_a --json  # + full JSON report
```

Available `--fixture` names: `scenario_a` (DEFEND), `scenario_b` (ATTACK),
`scenario_c` (DESPERATE), `scenario_d` (near-identical squads),
`scenario_e` (threat detection), `preseason`.

## Run the tests

```bash
python3 -m unittest discover -s tests -v
```

No `pytest`, no other test framework - stdlib `unittest` only, per the "no
unnecessary frameworks" brief. No test touches the network.

## What each section computes

1. **Manager profiles** (`profiles.py`) - position, rank, points, squad/XI/
   bench, captain/vice, chips used **and remaining** (derived from FPL's own
   per-half chip windows in bootstrap-static), transfers/gw, hits, average
   transfers/gw, captain history/concentration/unique count, formation
   history, bench points (sourced from gameweek history, not picks - so it
   works even with only the latest gameweek's picks), transfer timing
   (median hours before deadline), team value.
2. **Chris vs. rival** (`rival_analysis.py`) - squad overlap (15-man and
   XI separately), differentials both directions, captain overlap % across
   however many comparable gameweeks exist, and an explicit "effective
   exposure" blend (`0.7 * squad_overlap_fraction + 0.3 *
   captain_overlap_fraction`, weights configurable) - documented as a
   simple linear model, not a fitted one.
3. **Behavioural classification** (`classification.py`) - transfer
   behaviour (5 configurable bands) and risk behaviour (a transparent,
   weighted blend of hits/differential-ownership/captain-variety/transfer-
   frequency/squad-deviation, every component exposed). Both return
   `"Insufficient data"` below `min_gameweeks_for_classification`.
4. **League consensus** (`consensus.py`) - player/captain ownership %,
   most common players/captain, most template/differentiated managers
   (by average pairwise squad overlap), unique-to-one-manager players.
5. **Threat analysis** (`threats.py`) - shields (Chris owns, heavily owned
   around/above him), threats (heavily owned/captained by relevant rivals,
   Chris lacks), weapons (Chris owns, lightly owned by relevant rivals).
   Explicitly not a claim about player quality - relative exposure only.
6. **Relevant rivals** (`relevance.py`) - a points-gap-and-position score
   (`1 - (gap_weight * normalized_gap + position_weight *
   normalized_position)`), not a hardcoded "top 5"; immediate table
   neighbours are always included regardless of score. All weights/
   thresholds/caps live in `EngineConfig`.
7. **Strategy state** (`strategy.py`) - DEFEND / BALANCED / ATTACK /
   DESPERATE, or `INSUFFICIENT_SEASON_DATA` before GW1 is scored. Does not
   drive any recommendation. See `strategy.py`'s docstring for the exact
   decision tree and thresholds.
8. **Pre-season handling** (`preseason.py`) - when no gameweek is
   complete, the engine returns membership, competitor count, whatever
   public info exists (e.g. career season history), what activates after
   GW1, and which calculations are currently blocked and why. No dummy
   numbers.

## Synthetic fixtures vs. live data

`fpl_rival/fixtures/synthetic.py` is used **only** by tests and by
`--fixture` demos. Every synthetic `LeagueData`/`ManagerData` is stamped
`is_synthetic=True` and uses entry/league IDs >= 900000, and
`report_text.py` prints a loud `*** SYNTHETIC TEST DATA ***` banner
whenever `is_synthetic` is set - so it can never be mistaken for a real
run, on screen or in the saved JSON.

---

# Stage 3 - League Win Simulator (Captain Battle)

Stage 3 adds a **Monte Carlo simulation layer**, architecturally separate
from retrieval (Stage 1) and analytics (Stage 2). It tests the core
product hypothesis: a mini-league's win probability is not always
maximised by the same decision that maximises expected points. No LLM, no
website, no live gameweek data (the 2026/27 season hasn't started) - every
scenario here runs on synthetic, clearly-labelled fixtures.

## Run it

```bash
python3 simulation_cli.py --fixture captain-attack
python3 simulation_cli.py --fixture captain-defend
python3 simulation_cli.py --fixture shared-squad
python3 simulation_cli.py --fixture rival-uncertainty
python3 simulation_cli.py --fixture captain-attack --simulations 5000 --seed 7
```

## Run the tests

```bash
python3 -m unittest discover -s tests -v
```

66 new Stage 3 test cases (117 total across all three stages), stdlib
`unittest` only, no network access anywhere.

## Architecture

```
fpl_rival/simulation/
  models.py             PlayerProjection, ManagerState, LeagueState, SimulationConfig
  exceptions.py         InvalidCaptainError, MissingProjectionError, etc.
  player_outcomes.py    samples one gameweek score per unique player (shared across managers)
  rival_behaviour.py    fixed vs. probabilistic rival captain choice
  manager_outcomes.py   XI sum + captain doubling + vice-captain fallback
  monte_carlo.py        sample -> score everyone -> rerank -> aggregate
  captain_battle.py     runs every candidate captain under identical randomness, ranks results
  relevance_adapter.py  thin adapter reusing Stage 2's relevant-rival scoring
  report_text.py        presentation only - formats results, adds no analysis
fpl_rival/fixtures/simulation_fixtures.py   synthetic scenarios (>= entry ID 900000)
simulation_cli.py                            single-command launcher
```

`numpy` is used here (the one new dependency in this project) because the
engine draws `num_simulations` x `num_unique_players` random numbers and
reranks a whole league every simulation - vectorising that is what keeps
50,000 simulations at well under a second; a pure-Python nested loop over
managers x simulations would be materially slower for no benefit.

## Distribution methodology

```
simulated_points = round(clip(Normal(mean, std), floor=min_points, ceiling=None))
```

A plain Gaussian draw per player per simulation, floored at -4 (real FPL
scores are very rarely worse than that) and rounded to the nearest integer
(real FPL scores are always whole numbers). Zero is a normal, reachable
value - no special-casing. This is a deliberately simple distribution to
test the *league strategy mechanism*, not a claim about being a good
football scoring model - see "known modelling weaknesses" below.

**Independence limitation (documented, not hidden):** every player is
sampled independently. Real teammates are positively correlated (same
match) and some players are negatively correlated (playing-time
competition) - none of that is modelled. The architecture keeps this
addable later without a rewrite: `player_outcomes.py`'s only contract is
"return one array per player_id, all the same length"; everything
downstream only ever consumes that dict, never the sampling internals.

**The critical mechanism - shared player outcomes:** every manager's score
in a given simulation is built from the *same* per-player arrays. If Chris
and a rival both own Salah, `player_points[salah_id][i]` is Salah's one
and only simulated score for simulation `i`, used by both of them. This is
what makes the acceptance tests below work: when two managers share a
player, that player's outcome cancels out of their pairwise comparison
*by construction*, leaving only the players that actually differ to drive
relative movement.

**Captain Battle's "common random numbers" design:** when comparing
candidate captains, player outcomes and rival captain choices are sampled
**once** and reused for every candidate - only Chris's captain changes
between runs. This removes simulation noise from the comparison itself,
which is what makes the acceptance-test effects below resolve cleanly
rather than being drowned in Monte Carlo noise.

**Tie rule (documented, not invented):** a manager's simulated position is
`1 + (number of other managers with a strictly greater simulated total)`
- "competition ranking" (1, 1, 3, 4, ...), same as a sports table before
any secondary tiebreaker. FPL's real tiebreakers (e.g. total team value)
are not implemented; a genuine points tie is left as a tie.

## Synthetic fixtures

`fpl_rival/fixtures/simulation_fixtures.py` - entry IDs >= 900000, real
player names (Salah, Palmer, ...) used as readable labels only, with
projections that are entirely made up for testing (never claimed as real
predictions):

* `captain_attack()` - Chris trails the leader; both own Salah (leader
  captains him with certainty); Chris also owns Palmer, which the leader
  doesn't.
* `captain_defend()` - mirror image: Chris leads; the closest rival owns
  and captains Salah with certainty; Chris also owns Palmer.
* `shared_squad_scenario(num_shared)` / `_low_diff()` / `_high_diff()` -
  two managers whose XIs share a configurable number of players.
* `rival_uncertainty_scenario_a()` / `_b()` - identical attacking setup,
  with the leader's captain probability shifted from 90/10 to 50/50.

## Known modelling weaknesses (explicit, not hidden)

* Player projections are entirely synthetic/illustrative, not derived from
  any real statistical model.
* The scoring distribution is a plain clipped/rounded Normal - no
  appearance/goal/assist/bonus decomposition, no position-specific shape.
* Player outcomes are sampled independently - no team/fixture correlation.
* No transfer simulation, no chip simulation, no future gameweeks.
* No bench auto-substitution - bench players never score, regardless of
  what the starting XI does. The only "substitution" implemented is: if a
  captain's simulated score rounds to exactly 0, the vice-captain's score
  is used for the captain bonus instead (a simplification of FPL's real
  "captain didn't play" rule - it does not model injuries or rotation).
* Rival captain probabilities are supplied manually, not inferred from
  Stage 2's behavioural history (a natural Stage 4+ candidate).
* No injury/news modelling, no validated commercial projection feed.

See the Stage 3 completion report (delivered in-conversation) for the
acceptance test results, performance benchmark, and Stage 4
recommendation.
