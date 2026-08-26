# FPL Rival - Stages 1, 2, 3, 4 & 4.5

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
acceptance test results and performance benchmark.

---

# Stage 4 - Rival Prediction Engine

Stage 4 estimates "P(rival captains player X)" from a rival's historical
captaincy decisions, as an explicit, fully-explained probability
distribution - never an unexplained number. It feeds directly into Stage
3's probabilistic rival-captain mode via a thin adapter. No LLM, no
website, no live gameweek data yet - every scenario runs on synthetic,
clearly-labelled fixtures.

## Run it

```bash
python3 prediction_cli.py --fixture template
python3 prediction_cli.py --fixture loyal
python3 prediction_cli.py --fixture differential
python3 prediction_cli.py --fixture ep-follower
python3 prediction_cli.py --fixture erratic
python3 prediction_cli.py --fixture pipeline-loyal      # full Stage4 -> Stage3 pipeline
python3 prediction_cli.py --fixture pipeline-balanced   # same, Dave's history changed - strategy can flip
```

## Run the tests

```bash
python3 -m unittest discover -s tests -v
```

94 new Stage 4 test cases (211 total across all four stages), no network
access anywhere.

## Architecture

```
fpl_rival/prediction/
  models.py            CaptainObservation, LeagueHistory (the ONE leakage boundary), PredictionConfig
  exceptions.py         EmptyCandidateSetError, InvalidExternalInputError
  features.py            personal loyalty, recency-weighted rate, concentration, league consensus, historical EP-response
  captain_model.py       additive weighted score -> softmax; shrinkage toward the prior; data-sufficiency classification
  prediction_engine.py    orchestrates features + model; candidates restricted to owned players
  backtest.py             chronological replay, reuses the same leakage boundary as live prediction
  history_store.py        JSONL prediction log + actual-outcome recording (no database)
  evaluation.py            top-1/top-2 accuracy, log loss, Brier score
  calibration.py           probability-bucket reliability reporting
  stage3_adapter.py        the one seam to Stage 3 - turns a prediction into ManagerState.captain_probabilities
  report_text.py           presentation only - deterministic templates, no LLM prose
fpl_rival/fixtures/prediction_fixtures.py   5 personas + end-to-end pipeline scenario (entry IDs/config synthetic, reuses Stage 3's player-id pool)
prediction_cli.py                            single-command launcher
```

## Methodology

```
captain_score(candidate) =
      shrinkage_weight * [ w_personal * personal_loyalty_rate
                          + w_recency  * recency_weighted_rate
                          + w_concentration * concentration_share ]
    +                     [ w_consensus * league_consensus_rate
                          + w_global_consensus * global_consensus_value   (optional input)
                          + w_expected_points   * expected_points_value   (optional input) ]

probabilities = softmax(score / temperature)
```

**Shrinkage toward the prior (brief item 5):** `shrinkage_weight = n / (n + k)`,
where `n` is the number of personal captaincy decisions observed and `k`
(`personal_shrinkage_k`, default 6) is configurable. This is 0 with zero
personal history (pure prior - consensus/EP only), 0.5 at `n = k`, and
approaches 1 as history accumulates - a smooth curve, never a hard
gameweek-based switch. The prior components (consensus, global consensus,
expected points) always contribute at full weight; only the
personal-behaviour components are scaled by this weight, which is what
makes "manager-specific signal progressively matters more" literally true
without ever disabling the prior.

**Data sufficiency states** (`PRIOR_DRIVEN` / `MIXED` / `BEHAVIOUR_DRIVEN`)
are quantitative thresholds on `shrinkage_weight`, all configurable in
`PredictionConfig`. A separate `LOW PERSONAL DATA` flag fires below a
configurable gameweek count, independent of the state classification.

**Every prediction exposes:** the full candidate probability distribution,
each candidate's raw feature values, each component's weighted
contribution to the score, and the data-sufficiency block above - nothing
is an opaque number.

## The critical mechanism: one leakage boundary

`LeagueHistory.before(target_gameweek)` is called exactly once, at the top
of `prediction_engine.predict_captain`, before any feature is computed.
Backtesting reuses this identical function rather than re-implementing its
own slicing - there is only one place in the whole system that decides
"what counts as the past", and it is tested directly (see
`tests/test_prediction_engine.py`'s defensive leakage test, which feeds a
history containing gameweeks at/after the target and confirms they're
ignored).

## Stage 3 integration

```python
prediction = predict_captain(rival.entry_id, rival.name, rival.starting_xi, target_gw, history, config, expected_points=ep)
rival_updated = stage3_adapter.apply_prediction_to_manager_state(rival, prediction)
# rival_updated.captain_probabilities is now ready for Stage 3's
# SimulationConfig(rival_captain_mode="probabilistic") - no Stage 3 file changes.
```

The adapter reuses Stage 3's own `rival_behaviour.validate_captain_distribution`
to confirm the output is immediately usable, the same reuse-without-coupling
pattern `fpl_rival/simulation/relevance_adapter.py` already uses for Stage 2.

## Known limitations (explicit, not hidden)

* "League consensus" is a track-record proxy (other managers' own past
  captaincy), not a live snapshot of this week's picks - genuinely
  unavailable before the deadline (see Stage 1's documented data
  boundaries), and clearly labelled as such in the code.
* "Historical response to expected points" (follows-highest-projected /
  differential rate) is computed and exposed for transparency, but not yet
  fed back into the scoring model - a deliberate Stage 4 scope boundary,
  not an oversight.
* All component weights, the shrinkage constant, the recency decay rate,
  and the data-sufficiency thresholds are manually configured defaults,
  not fitted to any real data - there is no real captaincy dataset yet to
  fit them against.
* Global consensus and expected-points inputs are optional, synthetic/
  manually-supplied interfaces - no live source is wired up.
* Candidates are restricted to the rival's currently-owned squad, so no
  transfer-in captaincy risk is modelled (out of scope per the brief).

See the Stage 4 completion report (delivered in-conversation) for the
persona/acceptance test results and backtesting details.

---

# Stage 4.5 - Historical Calibration

Stage 4.5 answers one question: does Stage 4's manually-weighted rival
captain model actually predict real manager behaviour better than simple
baselines? It adds dataset ingestion, no-leakage backtesting against
baselines, chronological train/validation/test splitting, coordinate-search
parameter fitting, ablation, history-depth analysis, a manager
predictability score, and calibration measurement (with an optional
temperature-scaling correction) - all reusing Stage 4's prediction engine
and leakage boundary unmodified.

**No real historical FPL manager dataset was available in this
environment.** Every number this stage produces without `--dataset` is
from a synthetic dataset (Stage 4's own behavioural personas, run through
the real CSV importer) and is clearly labelled as such wherever it's
printed - see the Stage 4.5 completion report for exactly what real data
would be needed and why this is an honest "Outcome C" for the real-world
question, even though the infrastructure itself is fully validated.

## Run it

```bash
python3 calibration_cli.py                              # synthetic dataset, no --fit
python3 calibration_cli.py --fit                         # + coordinate-search parameter fitting
python3 calibration_cli.py --dataset /path/to/data.csv --fit
python3 calibration_cli.py --dataset /path/to/dir --long-format --fit
```

`--dataset` must be a local file or directory already on disk - nothing is
ever downloaded or scraped. See `fpl_rival/calibration/importer.py` for
the native CSV/JSONL schema and an example adaptor for a differently-shaped
raw export.

## Run the tests

```bash
python3 -m unittest discover -s tests -v
```

83 new Stage 4.5 test cases (294 total across all stages), no network
access anywhere.

## Architecture

```
fpl_rival/calibration/
  models.py              HistoricalRecord (the dataset interface), DatasetSplit (season+gameweek chronology)
  importer.py             local file/dir CSV+JSONL import, one example adaptor, dataset summary
  baselines.py             Last Captain / Personal Frequency / Consensus / EP-Favourite / Uniform
  harness.py                runs the Stage 4 model OR a baseline over a whole dataset, phase-tagged
  splits.py                  chronological train/validation/test boundary construction
  fitting.py                  coordinate-search parameter fitting, minimizing validation log loss
  ablation.py                  single-feature-family zeroing, measured on validation
  history_depth.py              buckets predictions by gameweeks_observed
  predictability.py              evidence-based manager predictability score (from backtest accuracy, not Stage 2 reuse)
  calibration_report.py           ECE + optional temperature-scaling correction (fit on validation, reused on test)
  stage3_impact.py                 manual vs fitted config fed into Stage 3 Captain Battle, same scenarios
fpl_rival/fixtures/calibration_fixtures.py   synthetic dataset generator (deterministic - see note below)
calibration_cli.py                            single-command launcher
```

## Key design point: one leakage boundary, reused everywhere

`harness.py` never re-implements "what counts as the past" - it calls
Stage 4's `LeagueHistory.before(target_gameweek)` for baselines exactly the
way `predict_captain` already uses it internally for the model. Backtest
replay, parameter fitting, and ablation all go through this same harness,
so there is exactly one leakage boundary in the whole system, tested
directly with deliberate leak-attempt cases (see
`tests/test_calibration_harness.py`).

## A bug this stage's own tests caught

An earlier version of the synthetic dataset generator seeded each
persona's random history with Python's built-in `hash()` on a
`(name, index)` tuple. `hash()` on strings is randomized per process
(`PYTHONHASHSEED`) unless explicitly disabled, so two runs of the CLI
silently produced two *different* synthetic datasets and therefore
different, non-reproducible numbers - caught by re-running the CLI twice
and diffing the output, then locked down with
`tests/test_calibration_dataset_export.py`, which runs the generator in
separate subprocesses and asserts identical results. Fixed by deriving
seeds from a plain integer formula instead of `hash()`. Left in this
README as a reminder that "looks deterministic" and "is deterministic"
are different claims worth actually testing.

## Known limitations (explicit, not hidden)

* No real historical dataset was available to validate against - every
  synthetic-data result is infrastructure validation, not evidence about
  real manager behaviour.
* Personal history does not carry across season boundaries in the harness
  (gameweek numbers reset each season) - a documented simplification, not
  a claim that real rival loyalty resets every season.
* Manager predictability's three weights (concentration / prediction
  confidence / backtest accuracy) are equal by default, not fitted -
  reasonable, not derived from data.
* Baseline D (expected-points favourite) and the EP feature/weight are
  only exercised when EP data is supplied - real datasets very likely
  won't have historical pre-deadline EP data, and this stage does not
  fabricate any.
* Ablation and parameter fitting both run on the validation split only,
  by design - test-split numbers are reported exactly once, after every
  choice is already locked in.

See the Stage 4.5 completion report (delivered in-conversation) for the
full results table, ablation findings, and recommendation.

---

# Real rival history (live, post-Stage-4.5)

Once a real gameweek exists, `real_history_cli.py` builds/grows a REAL
historical captain-decision dataset for one mini league, in the exact CSV
schema `fpl_rival/calibration/importer.py` already reads - so every
Stage 4.5 tool (`calibration_cli.py`, baselines, fitting, ablation) works
on it completely unchanged the moment enough real gameweeks exist.

```bash
python3 real_history_cli.py --league-id <id>
```

Safe to re-run every week: it re-fetches whichever gameweeks are finished
and merges them into the existing file, keyed by (season, manager,
gameweek), so running it after each deadline grows the dataset one
gameweek at a time without duplicating rows.

This is NOT the same thing as `importer.py`'s "local file only, never
auto-downloaded" rule for third-party datasets - `fpl_rival/calibration/live_import.py`
calls the exact same public FPL endpoints Stage 1 already uses (via Stage
2's retrieval, reused unmodified), for leagues the user has legitimate
real-time access to. See `data/real_history/` (gitignored - this is real
data about real people in your leagues, not something to publish).
