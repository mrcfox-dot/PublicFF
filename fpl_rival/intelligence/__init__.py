"""Stage 2: deterministic Rival Intelligence Engine.

This package is pure analytics - it never calls the network. It consumes
the data models defined in ``models.py``, which can be populated either
from live FPL data (see ``collect.py``) or from synthetic test fixtures
(see ``fpl_rival/fixtures``). No LLM calls, no web framework.
"""
