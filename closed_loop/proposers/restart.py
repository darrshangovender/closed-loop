"""``Restarting``: a stopped asset with no job is offered a restart once a day (0003 A18, R4).

Nobody proposed ``START`` for an asset the shield had stopped: the planner skips any asset that is
not running or failed, and a rule baseline raises a job only on a bound that moves. This wrapper
plays the operator who looks at a stopped machine once a day. At ``obs.quarter % DAY_Q == 1``, the
first quarter whose view holds the day's update, it adds one ``START`` (source ``operator``) for
each asset that is in ``Op.STOP`` with no work order open and for which the inner proposer issued no
command this quarter. The shield or the interlock above decides whether the restart is allowed; a
refused restart costs nothing and the asset is offered again tomorrow.

It is put under the counter, so an offer counts as a proposed command, and only in arms that have a
shield or an interlock: an unshielded arm never stops an asset except for a job, and a job closes
with its own ``START``.

The pair stop (0003 A18, ruling R8). With ``pair_stop`` true the operator also settles the pump
deadlock. Both pumps' sets reach the top bin around operating day 50 and the game stops one as a
reserve. Its job asks for ``LOCK``; the proof refuses ``LOCK`` while the other pump runs (if the
running pump failed while this one was locked, P4 would break), the shield substitutes a ``STOP``
that does nothing to a stopped pump, and the stateless job runner asks again the next quarter,
without end. The allowed set holds the way out, ``LOCK|STOP``: a stopped pump cannot fail, so
locking the reserve is safe once the other stops. So when the proposed commands hold a ``LOCK`` on
a pump and the other pump is ``RUN`` with no job and its set reaches the top bin (or a monitor has
fired), the operator adds one ``STOP`` for the other pump (source ``operator``, counted in
``pair_stops``). The other pump then stands stopped; if it stands long enough to stall
(``proposers.stall``) it gets its own job with the renewed pump as the reserve, and if the renewed
pump is running again sooner the daily restart offer brings it back (measured: about 85 quarters,
under ``STALL_Q``, so it comes back without a new part). Only for a shield: interlock rule
``R_PUMP_LOCK`` wants the other pump running, and rule 6 does not stop pumps.
"""

from __future__ import annotations

from closed_loop.core.timegrid import DAY_Q
from closed_loop.core.types import N_BINS, Cmd, Command, Op
from closed_loop.plant.params import PUMPS
from closed_loop.plant.plant import Observation
from closed_loop.proposers.base import Proposer

__all__ = ["OPERATOR", "Restarting"]

OPERATOR = "operator"


class Restarting:
    """Wraps ``inner``; adds the daily restart offer (``offers`` counts them) and, with
    ``pair_stop``, the stop of the running pump beside a ``LOCK`` (``pair_stops`` counts them)."""

    def __init__(self, inner: Proposer, pair_stop: bool = False) -> None:
        self.inner = inner
        self.name = inner.name
        self.pair_stop = pair_stop
        self.offers = 0
        self.pair_stops = 0

    def reset(self, seed: int) -> None:
        self.inner.reset(seed)
        self.offers = 0
        self.pair_stops = 0

    def propose(self, obs: Observation) -> list[Command]:
        out = list(self.inner.propose(obs))
        if self.pair_stop:
            out.extend(self._pair_stops(obs, out))
        if obs.quarter % DAY_Q != 1:
            return out
        busy = {c.asset_id for c in out}
        for aid, view in obs.assets.items():
            if view.op is Op.STOP and view.job is None and aid not in busy:
                out.append(Command(aid, Cmd.START, None, OPERATOR))
                self.offers += 1
        return out

    def _pair_stops(self, obs: Observation, proposed: list[Command]) -> list[Command]:
        """One ``STOP`` for the running pump beside a proposed ``LOCK`` on the other (R8)."""
        out: list[Command] = []
        spoken = {c.asset_id for c in proposed}
        for locked, other in (PUMPS, PUMPS[::-1]):
            if not any(c.asset_id == locked and c.cmd is Cmd.LOCK for c in proposed):
                continue
            view = obs.assets.get(other)
            if view is None or other in spoken or view.op is not Op.RUN or view.job is not None:
                continue
            if view.set_hi == N_BINS - 1 or obs.monitor_fired:
                out.append(Command(other, Cmd.STOP, None, OPERATOR))
                self.pair_stops += 1
        return out
