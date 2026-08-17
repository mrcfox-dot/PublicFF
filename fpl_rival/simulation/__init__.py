"""Stage 3: League Win Simulator (Captain Battle).

A Monte Carlo simulation layer, kept architecturally separate from:

* FPL data retrieval (fpl_rival.api / .league / .manager / .intelligence.collect)
* Stage 2 intelligence analytics (fpl_rival.intelligence.*)
* presentation (report_text.py here, and fpl_rival.intelligence.report_text)

This package never calls the network and never depends on live data - every
input is a plain dataclass in ``models.py``, populated either from a future
live adapter or (for now, since GW1 hasn't happened) from the synthetic
fixtures in ``fpl_rival/fixtures/simulation_fixtures.py``.

Core hypothesis under test: a mini-league's win probability is not always
maximised by the same decision that maximises expected points. See
``captain_battle.py`` and the Stage 3 completion report for the evidence.
"""
