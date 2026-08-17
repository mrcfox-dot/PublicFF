# FPL Rival - Stage 1 Prototype

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
