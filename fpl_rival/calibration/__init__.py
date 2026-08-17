"""Stage 4.5: Historical Calibration.

Answers one question: does the Stage 4 rival captain prediction model
actually predict real manager behaviour better than simple baselines?

This package adds infrastructure to ingest a legitimate LOCAL historical
manager dataset (never downloaded or scraped automatically - see
``importer.py``), run chronological no-future-leakage backtests, compare
against simple baselines, and fit/tune Stage 4's parameters where the
available data justifies it. It does not modify Stage 1, 2, 3 or the
Stage 4 prediction engine's behaviour - it only calls Stage 4's existing,
already-leakage-safe machinery (``prediction_engine.predict_captain`` via
``backtest.run_backtest``) with different configs and different (real or
synthetic) input data.

If no real dataset is supplied, every tool here still works against the
project's synthetic personas (Stage 4's own fixtures, exported through the
same importer path a real dataset would use) - see
``fpl_rival/fixtures/calibration_fixtures.py``. Results from synthetic
data are clearly labelled as such everywhere they're printed; they
validate the INFRASTRUCTURE, not real-world predictive power.
"""
