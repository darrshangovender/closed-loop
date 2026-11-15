"""When is an asset stalled? One definition for every proposer (0003 A18, ruling R5).

An asset is **stalled** when it has been in ``Op.STOP`` with no work order open for at least
``STALL_Q = DAY_Q + 2`` quarters. A shield (or an interlock) may stop an asset on a bound and a
controller that raises work orders only for assets it believes are failing never sees it again; the
operator's restart (``proposers.restart``) is offered once a day, and ``DAY_Q + 2`` is long enough
for one restart to have been offered and refused. After that the asset needs a work order, as a
failed one does: the part is the only way the bound can move again.

The watch reads views only (``Observation``): the quarter of the first view that shows the asset
stopped with no job, forgotten as soon as it runs, has a job or is anything else. A baseline or a
planner without a shield or an interlock never stops an asset except for a job, so it never
reaches this path (``tests/proposers/test_stall.py`` pins that for B2, B3 and P0).
"""

from __future__ import annotations

from closed_loop.core.timegrid import DAY_Q
from closed_loop.core.types import Op
from closed_loop.plant.plant import Observation

__all__ = ["STALL_Q", "StallWatch"]

STALL_Q: int = DAY_Q + 2


class StallWatch:
    """Tracks, from the views alone, how long each asset has stood stopped with no job."""

    def __init__(self) -> None:
        self._since: dict[str, int] = {}

    def reset(self) -> None:
        self._since = {}

    def observe(self, obs: Observation) -> None:
        """Note this quarter's views. Call once per quarter, before asking."""
        for aid, view in obs.assets.items():
            if view.op is Op.STOP and view.job is None:
                self._since.setdefault(aid, obs.quarter)
            else:
                self._since.pop(aid, None)

    def stalled(self, aid: str, obs: Observation) -> bool:
        since = self._since.get(aid)
        return since is not None and obs.quarter - since >= STALL_Q

    def stalled_assets(self, obs: Observation) -> frozenset[str]:
        return frozenset(a for a in self._since if self.stalled(a, obs))
