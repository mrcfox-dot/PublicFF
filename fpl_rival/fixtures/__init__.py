"""Synthetic test fixtures ONLY.

Nothing in this package ever calls the live FPL API or touches data saved
by a live run. Every ``LeagueData``/``ManagerData`` built here is stamped
``is_synthetic=True`` and uses entry IDs from a clearly fake range
(900000+) and a league ID from the same range, so it can never be confused
with a real manager or league. Use this data only for automated tests and
for demonstrating the analytics engine before real gameweeks exist.
"""
