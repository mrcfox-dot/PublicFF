"""Stage 4: Rival Prediction Engine.

Estimates P(rival captains player X) for the upcoming gameweek, as an
explicit, interpretable probability distribution - never an unexplained
black box number. This package never calls the network and never depends
on live data; every input is a plain dataclass in ``models.py``, populated
from a legitimate historical-decision dataset (see ``models.py``'s
``CaptainObservation``) - live or synthetic, the engine doesn't care.

Kept architecturally separate from:

* retrieval (fpl_rival.api / .league / .manager / .intelligence.collect)
* intelligence (fpl_rival.intelligence.*) - Stage 2
* simulation (fpl_rival.simulation.*) - Stage 3
* presentation (report_text.py here, and the other stages' report_text.py)

``stage3_adapter.py`` is the one deliberate seam to Stage 3: it turns a
``CaptainPrediction`` into the ``{player_id: probability}`` shape Stage 3's
``ManagerState.captain_probabilities`` already accepts, without either
package importing the other's internals.
"""
