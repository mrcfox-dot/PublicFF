# FPL Rival - Stages 1 & 2

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
