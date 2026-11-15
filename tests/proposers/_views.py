"""Hand-built observations for the proposer tests (SYNTHETIC)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType

from closed_loop.core.timegrid import DAY_Q
from closed_loop.core.types import Op, Permit
from closed_loop.plant.des import JobView
from closed_loop.plant.observe import DailyObs
from closed_loop.plant.params import ASSET_IDS
from closed_loop.plant.plant import AssetView, Observation


def job(
    asset_id: str,
    status: str = "APPROVED",
    mode: str = "IMPELLER_WEAR",
    *,
    ready: bool = True,
    confined: bool = False,
    total: int = 12,
    done: int = 0,
) -> JobView:
    return JobView(
        asset_id=asset_id,
        wo_id=f"WO-{asset_id}-1",
        mode=mode,
        status=status,
        ready=ready,
        confined=confined,
        work_total_q=total,
        work_done_q=done,
        part_no="SYN-IMP-2001",
        part_available=True,
    )


def view(
    asset_id: str,
    *,
    op: Op = Op.RUN,
    permit: Permit = Permit.NONE,
    hi_last: float | None = None,
    rul_lo: float | None = None,
    rul_hi: float | None = None,
    day: int = 0,
    jobs_done: int = 0,
    job_view: JobView | None = None,
    signature: tuple[float, float, float] = (0.0, 0.0, 0.0),
    load: float = 1.0,
    occupied: bool = False,
    gas_age_q: int = 99,
    alarm: bool = False,
    soon: bool = False,
    set_lo: int = 0,
    set_hi: int = 3,
    spares: Mapping[str, int] | None = None,
) -> AssetView:
    return AssetView(
        asset_id=asset_id,
        op=op,
        permit=permit,
        occupied=occupied,
        gas_age_q=gas_age_q,
        alarm=alarm,
        soon=soon,
        set_lo=set_lo,
        set_hi=set_hi,
        rul_lo=rul_lo,
        rul_hi=rul_hi,
        prior_only=False,
        hi_last=hi_last,
        job_open=job_view is not None,
        spares=MappingProxyType(dict(spares or {})),
        job=job_view,
        last_obs=DailyObs(day, hi_last, *signature, load),
        jobs_done=jobs_done,
    )


def observation(
    views: Iterable[AssetView] | Mapping[str, AssetView] = (),
    *,
    quarter: int = 0,
    note: str | None = None,
    fired: bool = False,
) -> Observation:
    """An observation of the four assets; any asset not in ``views`` is a quiet running one."""
    given = dict(views) if isinstance(views, Mapping) else {v.asset_id: v for v in views}
    day = quarter // DAY_Q
    assets = {a: given.get(a, view(a, day=day)) for a in ASSET_IDS}
    return Observation(
        quarter=quarter,
        day=day,
        shift=(quarter % DAY_Q) // 32,
        assets=MappingProxyType(assets),
        note=note,
        alerts=(),
        crew_free=2,
        safety_officer_on_shift=True,
        monitor_fired=fired,
    )
