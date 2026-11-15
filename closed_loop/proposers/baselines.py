"""The four baseline controllers of the SYNTHETIC plant (packet P6a): B0 to B3.

All four are rule controllers over the same plant, the same ``Observation`` and the same job
runner. They differ only in *when* they raise a work order:

``B0`` run to failure
    after the asset has failed (``op`` is ``FAIL``);
``B1`` calendar
    when the component is older than a fixed fraction of the class's characteristic life (the scale
    of the Weibull the estimator uses as its class prior), counted from the start of the episode or
    from the last completed work order;
``B2`` fixed limit
    when the last health index reading reaches a fixed level;
``B3`` bound rule
    when the lower bound of the perception layer's RUL interval falls below ``k`` days (or, with
    ``bin_limit``, when its health bin reaches that bin). With a diagnoser the work order carries
    the diagnosed failure mode, and a diagnosis of a sensor fault or a process upset raises none
    (ADR 0003 A3), unless the asset has already failed (see below).

What every baseline shares (decisions the interface text leaves open):

* Once a work order is open the job runner (``plant.jobs.JobRunner``) carries it out, one command
  per quarter, and the baseline never issues a command outside ``jobs.ALPHABET`` for it.
* A stalled asset (stopped with no work order for a day and two quarters, ``proposers.stall``;
  ADR 0003 A18) is treated like a failed one: a shield that stopped it on a bound has left it
  there and the bound will not move while it stands. Without a shield or an interlock nothing
  stops an asset except a work order, so an unshielded baseline never reaches this.
* A failed asset (``op`` is ``FAIL``) gets a work order under every rule, whatever the rule says
  about its readings and its date (ADR 0003 A17): a failed asset is not waiting for a forecast.
  Before this a calendar baseline left it broken until its date, and B3 could talk itself out of
  the repair by diagnosing a sensor fault on an asset that had stopped. The pump serialisation and
  the cooldown still apply.
* The failure mode on a work order is the most likely mode of the asset class, unless B3 has a
  diagnoser. A failed asset takes the diagnosed mode if the cause is degradation, else the class's
  most likely mode. A wrong mode completes a job without renewing the component (ADR 0003 A3);
  that is the consequence being measured, not something a baseline avoids.
* Work on the two pumps is serialised: no work order is raised for a pump while the other has one
  open. It is the rule a planner applies by hand, and without it the baselines would violate the
  pump-availability property (P4) by construction.
* A completed work order is taken to have renewed the component (an operator cannot tell a wrong
  repair from a right one). The day of completion is a blackout: the daily reading and the
  perception output are one day old until the next day boundary, and acting on them would order a
  second job for a part that has just been replaced.
* At most one work order per asset per ``cooldown_q`` quarters (a day by default), so that a work
  order the approver withdrew is not re-raised every quarter.
* A spare that has run out is ordered again (ADR 0003 A17): when a part of the asset shows a stock
  of 0 and this proposer has not already ordered it, one ``ORDER`` goes to the plant; the memory of
  it is dropped once the stock shows above 0, so a later stock-out orders again, and also after
  ``ORDER_RETRY_DAYS`` (the planner's own retry) in case the plant refused the order. The reorder
  point is zero, the plainest stores rule there is, so that a planner is not credited merely for
  ordering at all. Nothing is ordered ahead of need, and a work order for a part that is out of
  stock waits with the asset running (the job runner does not take the asset out of service until
  the part is here); it is not withdrawn, and the baseline does not reason about a wrong repair.
  Tuning of ``k``, the limit and the calendar fraction belongs to the development seeds
  (ADR 0001 section 11); the defaults here are starting points, not tuned values.

A proposer reads only the ``Observation``. Everything here is SYNTHETIC.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from closed_loop.core.timegrid import DAY_Q
from closed_loop.core.types import BIN_EDGES_DAYS, N_BINS, Command, Op, PlanCmd, bin_of_rul
from closed_loop.perception.estimator import class_prior
from closed_loop.plant.jobs import JobRunner, work_order_command
from closed_loop.plant.params import ASSET_CLASS, MODES_BY_CLASS, PUMPS, PlantParams
from closed_loop.plant.plant import AssetView, Observation
from closed_loop.proposers.stall import StallWatch

if TYPE_CHECKING:
    from closed_loop.causal.diagnoser import Diagnoser, Diagnosis
    from closed_loop.causal.records import HistoryRecorder, LifeRecord
    from closed_loop.proposers.base import Proposer

__all__ = [
    "BASELINES",
    "ORDER_RETRY_DAYS",
    "BoundRule",
    "Calendar",
    "FixedLimit",
    "RunToFailure",
    "default_mode",
    "make_baseline",
]


# A part ordered and still showing no stock this many days later is ordered again: the plant refuses
# an order with probability ``p_refuse``, and refuses a second one for a part already on order, so
# the retry is harmless. The planner's own default (``order_retry_days``).
ORDER_RETRY_DAYS = 30


def default_mode(asset_id: str) -> str:
    """The most likely failure mode of the asset's class (the register prior)."""
    modes = MODES_BY_CLASS[ASSET_CLASS[asset_id]]
    return max(modes, key=lambda m: m.weight).mode


class _RuleProposer:
    """Shared mechanics; a subclass says when to raise a work order and for which mode."""

    name = "rule"

    def __init__(
        self,
        *,
        params: PlantParams | None = None,
        runner: JobRunner | None = None,
        cooldown_q: int = DAY_Q,
        serialise_pumps: bool = True,
    ) -> None:
        if cooldown_q < 0:
            raise ValueError(f"cooldown_q must be non-negative, got {cooldown_q}")
        self.params = params if params is not None else PlantParams()
        self.runner = runner if runner is not None else JobRunner(source=self.name)
        self.cooldown_q = cooldown_q
        self.serialise_pumps = serialise_pumps
        self._last_raise: dict[str, int] = {}
        self._jobs_done: dict[str, int] = {}
        self._renewed_day: dict[str, int] = {}
        self._ordered: dict[str, int] = {}  # part -> the day it was ordered
        self._stall = StallWatch()

    # ------------------------------------------------------------------ the Proposer protocol
    def reset(self, seed: int) -> None:
        self._last_raise = {}
        self._jobs_done = {}
        self._renewed_day = {}
        self._ordered = {}
        self._stall.reset()
        self._on_reset(seed)

    def propose(self, obs: Observation) -> list[Command]:
        out: list[Command] = []
        raised: set[str] = set()
        for aid, view in obs.assets.items():
            self._look(aid, view, obs)
        self._stall.observe(obs)
        self._before_deciding(obs)
        for aid, view in obs.assets.items():
            if view.job is not None:
                cmd = self.runner.next_command(view, view.job, obs.quarter)
                if cmd is not None:
                    out.append(cmd)
                continue
            if not self._may_raise(aid, obs, raised):
                continue
            if not self.is_down(view, obs) and not self._wants_job(view, obs):
                continue
            mode = self._mode_for(view, obs)
            if mode is None:
                continue
            out.append(work_order_command(aid, mode, source=self.name))
            self._last_raise[aid] = obs.quarter
            raised.add(aid)
        out.extend(self._reorder(obs))
        return out

    def _reorder(self, obs: Observation) -> list[Command]:
        """One ``ORDER`` for a part that shows no stock and that this proposer has not ordered."""
        out: list[Command] = []
        for aid, view in obs.assets.items():
            for part, stock in view.spares.items():
                if stock > 0:
                    self._ordered.pop(part, None)
                    continue
                since = self._ordered.get(part)
                if since is not None and obs.day - since < ORDER_RETRY_DAYS:
                    continue
                self._ordered[part] = obs.day
                out.append(Command(aid, PlanCmd.ORDER, part, self.name))
        return out

    def is_down(self, view: AssetView, obs: Observation) -> bool:
        """Failed, or stalled (0003 A18): it needs a work order whatever the rule says."""
        return view.op is Op.FAIL or self._stall.stalled(view.asset_id, obs)

    # ------------------------------------------------------------------ hooks
    def _on_reset(self, seed: int) -> None:
        pass

    def _on_renewal(self, aid: str, obs: Observation) -> None:
        pass

    def _before_deciding(self, obs: Observation) -> None:
        pass

    def _wants_job(self, view: AssetView, obs: Observation) -> bool:
        raise NotImplementedError

    def _mode_for(self, view: AssetView, obs: Observation) -> str | None:
        return default_mode(view.asset_id)

    # ------------------------------------------------------------------ bookkeeping
    def _look(self, aid: str, view: AssetView, obs: Observation) -> None:
        """Note a completed work order (``jobs_done`` moved) as a renewal."""
        seen = self._jobs_done.setdefault(aid, view.jobs_done)
        if view.jobs_done != seen:
            self._jobs_done[aid] = view.jobs_done
            self._renewed_day[aid] = obs.day
            self._on_renewal(aid, obs)

    def _may_raise(self, aid: str, obs: Observation, raised: set[str]) -> bool:
        if obs.day <= self._renewed_day.get(aid, -1):
            return False
        last = self._last_raise.get(aid)
        if last is not None and obs.quarter - last < self.cooldown_q:
            return False
        if self.serialise_pumps and aid in PUMPS:
            other = PUMPS[1] if aid == PUMPS[0] else PUMPS[0]
            other_view = obs.assets.get(other)
            if other in raised or (other_view is not None and other_view.job_open):
                return False
        return True


class RunToFailure(_RuleProposer):
    """B0: raise a work order only after the asset has failed."""

    name = "B0"

    def _wants_job(self, view: AssetView, obs: Observation) -> bool:
        return view.op is Op.FAIL


class Calendar(_RuleProposer):
    """B1: replace on a calendar, ``fraction`` of the class's characteristic life apart."""

    name = "B1"

    def __init__(self, fraction: float = 0.5, **kw: Any) -> None:
        if not 0.0 < fraction:
            raise ValueError(f"fraction must be positive, got {fraction}")
        super().__init__(**kw)
        self.fraction = fraction
        onset = self.params.renewal_onset_days
        self.interval_days: dict[str, float] = {
            aid: fraction * class_prior(cls, onset).scale for aid, cls in ASSET_CLASS.items()
        }
        self._since: dict[str, int] = {}

    def _on_reset(self, seed: int) -> None:
        self._since = {}

    def _on_renewal(self, aid: str, obs: Observation) -> None:
        self._since[aid] = obs.day

    def _wants_job(self, view: AssetView, obs: Observation) -> bool:
        since = self._since.setdefault(view.asset_id, obs.day)
        return obs.day - since >= self.interval_days[view.asset_id]


class FixedLimit(_RuleProposer):
    """B2: raise a work order when the last health index reading reaches ``limit``."""

    name = "B2"

    def __init__(self, limit: float | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        # The plant raises its second alert at the midpoint of the alert and failure levels.
        self.limit = (
            limit if limit is not None else 0.5 * (self.params.hi_alert + self.params.hi_fail)
        )

    def _wants_job(self, view: AssetView, obs: Observation) -> bool:
        return view.hi_last is not None and view.hi_last >= self.limit


class BoundRule(_RuleProposer):
    """B3: raise a work order when the lower bound of the RUL interval is under ``k`` days.

    ``bin_limit`` states the same rule in health bins (the shield's unit): raise when
    ``bin_of_rul(lo)`` is at least ``bin_limit``. Give one of ``k`` and ``bin_limit``; with neither,
    ``k`` is the second bin edge (10 days).

    ``diagnoser`` is a ``Diagnoser`` or the name of one (``"scm"`` is the default, as 0003 A3 has it
    for B3). A name is built with ``make_diagnoser`` and is **not fitted**: call ``fit_diagnoser``
    (or pass a fitted one), or the first diagnosis raises. ``None`` means no diagnosis layer: the
    work order carries the class's most likely mode and nothing is ever ruled out.
    """

    name = "B3"

    def __init__(
        self,
        k: float | None = None,
        *,
        bin_limit: int | None = None,
        diagnoser: Diagnoser | str | None = "scm",
        **kw: Any,
    ) -> None:
        if k is not None and bin_limit is not None:
            raise ValueError("give k (days) or bin_limit (a health bin), not both")
        if bin_limit is not None and not 1 <= bin_limit < N_BINS:
            raise ValueError(f"bin_limit must be in 1..{N_BINS - 1}, got {bin_limit}")
        super().__init__(**kw)
        self.k = float(k) if k is not None else float(BIN_EDGES_DAYS[1])
        self.bin_limit = bin_limit
        self._diagnoser: Diagnoser | None
        if isinstance(diagnoser, str):
            from closed_loop.causal.diagnoser import make_diagnoser

            self._diagnoser = make_diagnoser(diagnoser)
        else:
            self._diagnoser = diagnoser
        self._recorder: HistoryRecorder | None = None
        self._diagnosed: dict[str, tuple[int, Diagnosis]] = {}
        self.diagnoses: list[Diagnosis] = []

    @property
    def diagnoser(self) -> Diagnoser | None:
        return self._diagnoser

    def fit_diagnoser(self, lives: Sequence[LifeRecord]) -> None:
        if self._diagnoser is None:
            raise ValueError("this B3 has no diagnoser")
        self._diagnoser.fit(lives)

    def _on_reset(self, seed: int) -> None:
        self._recorder = None
        self._diagnosed = {}
        self.diagnoses = []

    def _on_renewal(self, aid: str, obs: Observation) -> None:
        if self._recorder is not None:
            self._recorder.reset_life(aid, obs.day)
        self._diagnosed.pop(aid, None)

    def _before_deciding(self, obs: Observation) -> None:
        if self._diagnoser is None:
            return
        if self._recorder is None:
            from closed_loop.causal.records import HistoryRecorder

            self._recorder = HistoryRecorder()
        # A reading taken before a completed work order describes the old component: keep it out
        # of the new history (the daily reading catches up at the next day boundary).
        stale = {a for a, v in obs.assets.items() if self._reading_predates_renewal(a, v)}
        if stale:
            kept = {a: v for a, v in obs.assets.items() if a not in stale}
            obs = dataclasses.replace(obs, assets=MappingProxyType(kept))
        self._recorder.record(obs)

    def _reading_predates_renewal(self, aid: str, view: AssetView) -> bool:
        return view.last_obs is not None and view.last_obs.day <= self._renewed_day.get(aid, -1)

    def _wants_job(self, view: AssetView, obs: Observation) -> bool:
        lo = view.rul_lo
        if lo is None:
            return False
        if self.bin_limit is not None:
            return bin_of_rul(lo) >= self.bin_limit
        return lo < self.k

    def _mode_for(self, view: AssetView, obs: Observation) -> str | None:
        aid = view.asset_id
        if self._diagnoser is None or self._recorder is None:
            return default_mode(aid)
        cached = self._diagnosed.get(aid)
        if cached is None or cached[0] != obs.day:
            # The history is rebuilt only on a stale cache: when the diagnosis says "not
            # degradation" no work order follows, this runs every quarter, and a rebuild costs
            # four array copies and a validating constructor.
            history = self._recorder.history(aid)
            if history.n == 0:
                return default_mode(aid)
            cached = (obs.day, self._diagnoser.diagnose(aid, obs.day, history))
            self._diagnosed[aid] = cached
            self.diagnoses.append(cached[1])
        diagnosis = cached[1]
        if diagnosis.cause == "degradation":
            return diagnosis.mode
        # a failed or stalled asset needs a job whatever the diagnoser thinks of its readings
        return default_mode(aid) if self.is_down(view, obs) else None


BASELINES: dict[str, Callable[..., Proposer]] = {
    "B0": RunToFailure,
    "B1": Calendar,
    "B2": FixedLimit,
    "B3": BoundRule,
}


def make_baseline(name: str, **kw: Any) -> Proposer:
    """``"B0"`` run to failure, ``"B1"`` calendar, ``"B2"`` fixed limit, ``"B3"`` bound rule."""
    try:
        factory = BASELINES[name]
    except KeyError:
        raise ValueError(
            f"unknown baseline {name!r}; expected one of {sorted(BASELINES)}"
        ) from None
    return factory(**kw)
