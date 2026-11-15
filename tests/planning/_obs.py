"""Hand-built observations and a stand-in belief tracker for the planner tests (SYNTHETIC).

The planner reads three things: the ``Observation``, a tracker that answers ``particles``,
``day`` and ``load_forecast``, and the constraints. These builders make the first two without a
simulator, so that each test states exactly the situation it is about.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from closed_loop.core.timegrid import DAY_Q
from closed_loop.core.types import Op, Permit
from closed_loop.planning.belief import BeliefTracker
from closed_loop.planning.worldmodel import Particles
from closed_loop.plant.des import JobView
from closed_loop.plant.observe import DailyObs
from closed_loop.plant.plant import AssetView, Observation

ASSETS = ("CR-501", "CV-101", "PU-201", "PU-202")


def job_view(
    asset_id: str, *, part_no: str | None = "SYN-IMP-2001", status: str = "AWAITING_APPROVAL"
) -> JobView:
    return JobView(
        asset_id=asset_id,
        wo_id=f"WO-{asset_id}-1",
        mode="IMPELLER_WEAR",
        status=status,
        ready=False,
        confined=False,
        work_total_q=16,
        work_done_q=0,
        part_no=part_no,
        part_available=True,
    )


def view(
    asset_id: str,
    *,
    op: Op = Op.RUN,
    job_open: bool = False,
    spares: Mapping[str, int] | None = None,
    job: JobView | None = None,
    hi: float | None = 1.0,
    day: int = 30,
) -> AssetView:
    return AssetView(
        asset_id=asset_id,
        op=op,
        permit=Permit.NONE,
        occupied=False,
        gas_age_q=99,
        alarm=False,
        soon=False,
        set_lo=0,
        set_hi=3,
        rul_lo=None,
        rul_hi=None,
        prior_only=False,
        hi_last=hi,
        job_open=job_open or job is not None,
        spares=MappingProxyType(dict(spares or {})),
        job=job,
        last_obs=DailyObs(day, hi, 0.0, 0.0, 0.0, 1.0),
    )


def observation(
    views: Mapping[str, AssetView],
    *,
    day: int = 30,
    quarter: int | None = None,
    crew_free: int = 2,
) -> Observation:
    q = day * DAY_Q if quarter is None else quarter
    return Observation(
        quarter=q,
        day=q // DAY_Q,
        shift=(q % DAY_Q) // 32,
        assets=MappingProxyType(dict(views)),
        note=None,
        alerts=(),
        crew_free=crew_free,
        safety_officer_on_shift=(q % DAY_Q) < 32,
        monitor_fired=False,
    )


class FakeTracker:
    """Answers what ``BeliefPlanner`` asks of a ``BeliefTracker``: belief, day and load."""

    def __init__(self, beliefs: Mapping[str, Particles], day: int = 30, load: float | None = None):
        self._beliefs = dict(beliefs)
        self._day = day
        self._load = load

    def particles(self, asset_id: str) -> Particles:
        return self._beliefs[asset_id]

    def day(self, asset_id: str) -> int:
        return self._day

    def load_forecast(self, asset_id: str) -> float | None:
        return self._load

    def as_tracker(self) -> BeliefTracker:
        return cast(BeliefTracker, cast(Any, self))


def plant_particles(x: float, n: int = 64, *, mode: int = 1, dur: float = 40.0, k: float = 2.0):
    """A tight belief about a pump: every particle at progress ``x`` of ``dur`` days."""
    return Particles(
        {
            "mode": np.full(n, mode, dtype=np.int64),
            "x": np.full(n, x),
            "dur": np.full(n, dur),
            "k": np.full(n, k),
            "rate": np.ones(n),
        },
        np.full(n, 1.0 / n),
    )
