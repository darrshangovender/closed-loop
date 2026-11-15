"""The telemetry side of an F2 episode: pieces of each asset's life, the pipeline cache and the
daily estimate (packet P11c). Everything is SYNTHETIC.

The plant owns the latent failures (onset, failure, shape, mode of each component life). This
module is the simulation acting as the telemetry source: it reads the plant's wear clock at the
instant an asset starts running, turns it into the wall-clock days at which the component would
reach its onset and its failure if nothing stopped it again, and has ``render_life`` write the
telemetry of that life once. That is the part of F2 where sim-only truth is read, and it is read
here and only here (``tests/f2/test_episode.py`` pins that the controller side never touches it).

The caching rule of 0001 section 3, as built
--------------------------------------------
* A life is rendered once, from the plant's latent, for the whole episode; later pieces of the life
  are cuts of the same record (so what the controller has seen never depends on what it chose to do
  afterwards).
* ``run_asset`` runs when the history the controller is entitled to changes, and not otherwise: a
  history is the list of pieces (one per unbroken run of the asset), so it changes when an asset
  *starts running again* after a stop (a stop that opens a gap, a completed job that adds a repair
  record, a replacement that starts a new life), and once at the start. While an asset is stopped
  or failed nothing new reaches the pipeline, so the estimate is held and nothing is run. A restart
  is picked up at the next daily update, so several stops and restarts inside one day cost one run.
* Every later day is answered by slicing the cached run (``estimate_rul`` on the cached health
  index, about 2 ms) and ``analyse_run`` is called only when the controller asks for an alert.

What the cut cannot do (a limitation, recorded as a finding): the generator ages on the wall clock,
the plant's wear clock stops when the asset stops. A stopped asset therefore keeps degrading in the
telemetry it will show after the restart (``docs/FINDINGS.md``, the item on stops).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from numpy.typing import NDArray

from closed_loop.core.timegrid import DAY_Q
from closed_loop.core.types import Op
from closed_loop.f2._siblings import load
from closed_loop.f2.alert_source import RunCache, aware, naive
from closed_loop.f2.render import (
    LifeLatent,
    RenderedTelemetry,
    asset_health_id,
    render_life,
    sim_time,
)
from closed_loop.perception.estimator import (
    STATUS_ESTIMATED,
    STATUS_INSUFFICIENT,
    STATUS_NOT_DEGRADING,
    STATUS_PRIOR,
    EstimatorOutput,
    class_prior,
)
from closed_loop.plant.degradation import Physical
from closed_loop.plant.observe import build_obs_streams
from closed_loop.plant.params import ASSET_CLASS, ASSET_IDS, PlantParams
from closed_loop.plant.plant import AssetView, F1Plant
from closed_loop.plant.scenario import FAULT_NONE, FAULT_SPIKE, FAULT_STUCK, asset_index

__all__ = [
    "AssetTelemetry",
    "FaultPlan",
    "FrameTransform",
    "F2Telemetry",
    "fault_plan",
    "inject_faults",
    "wall_day",
]

# A change to the rows of a rendered piece (a pandas frame in, a frame out), deterministic in the
# timestamps so that every cut of a life is altered alike. For tests and for replaying a scenario.
FrameTransform = Callable[[Any], Any]

_Z90 = 1.2815515655446004
# channels a gain fault, a stuck sensor or a spike acts on: the condition-monitoring ones
_MEASURED_PREFIXES = ("vib_", "temp_c")


def wall_day(cum: NDArray[np.float64], q0: int, clock: float) -> float:
    """The plant day at which a component running continuously from quarter ``q0`` has earned
    ``clock`` more nominal days. The same rule as ``plant.observe.simulate_lives``."""
    idx = int(np.searchsorted(cum, cum[q0] + max(clock, 0.0), side="left"))
    return idx / DAY_Q


# ----------------------------------------------------------------------------- scenario faults


@dataclass(frozen=True, eq=False)
class FaultPlan:
    """The sensor faults of one asset in a resolved scenario, in the plant's terms (0001 section 9).

    ``kind`` is the scenario's placed fault (``FAULT_STUCK`` or ``FAULT_SPIKE``) at ``day``;
    ``drift`` is the plant's own unmasked gain error per calendar day as a fraction of full scale
    (``ObsStreams.drift_frac``: the asset's random gain fault, the shifted condition, a scenario
    drift fault), or ``None`` when there is none. The plant's random stuck and spike days are not
    rendered: the generator draws data faults of its own (mismatch S4).
    """

    kind: int = FAULT_NONE
    day: float = 0.0
    stuck_days: int = 3
    drift: NDArray[np.float64] | None = None

    @property
    def active(self) -> bool:
        return self.kind in (FAULT_STUCK, FAULT_SPIKE) or (
            self.drift is not None and bool(np.any(self.drift > 0.0))
        )


def fault_plan(phys: Physical, asset_id: str, params: PlantParams, seed: int) -> FaultPlan:
    """The faults the plant's exogenous streams put on ``asset_id`` (same streams as F1)."""
    a = asset_index(asset_id)
    resolved = phys.resolved
    streams = build_obs_streams(params, seed, asset_id, phys)
    kind = int(resolved.fault_type[a])
    return FaultPlan(
        kind if kind in (FAULT_STUCK, FAULT_SPIKE) else FAULT_NONE,
        float(resolved.fault_day[a]),
        params.stuck_days,
        np.asarray(streams.drift_frac, dtype=np.float64),
    )


def inject_faults(frame: Any, plan: FaultPlan) -> Any:
    """A copy of a rendered historian frame with the scenario's sensor faults written into it.

    This is the documented bypass of 0002 P11a: asset-health's generator cannot place a data fault
    at a chosen time or add a slow gain error (mismatch S4), so the fault is written into the
    exported rows. A stuck sensor holds the last good value of the measured channels for
    ``stuck_days`` days, a spike multiplies three hourly rows by six, a gain error multiplies the
    measured channels by ``1 + drift`` of the calendar day. The pipeline's own data-quality gate
    decides what it makes of them.
    """
    if not plan.active:
        return frame
    out = frame.copy()
    cols = [c for c in out.columns if c.startswith(_MEASURED_PREFIXES)]
    origin = np.datetime64(sim_time(0).replace(tzinfo=None))
    days = (
        (out["timestamp"].to_numpy(dtype="datetime64[ns]") - origin) / np.timedelta64(1, "D")
    ).astype(np.float64)
    if plan.kind == FAULT_STUCK:
        window = (days >= plan.day) & (days < plan.day + plan.stuck_days)
        if window.any():
            first = int(np.flatnonzero(window)[0])
            for c in cols:
                out.loc[window, c] = out[c].iloc[max(first - 1, 0)]
    elif plan.kind == FAULT_SPIKE:
        rows = np.flatnonzero(days >= plan.day)[:3]
        for c in cols:
            at = out.columns.get_loc(c)
            out.iloc[rows, at] = out.iloc[rows, at].to_numpy(dtype=float) * 6.0
    if plan.drift is not None and np.any(plan.drift > 0.0):
        idx = np.clip(np.floor(days).astype(np.int64), 0, len(plan.drift) - 1)
        gain = 1.0 + plan.drift[idx]
        for c in cols:
            out[c] = out[c].to_numpy(dtype=float) * gain
    return out


# ----------------------------------------------------------------------------- one asset


@dataclass
class _Piece:
    life_id: int
    start_q: int
    end_q: int | None
    repair: bool  # a job was completed on this asset since the previous piece ended


class AssetTelemetry:
    """The pieces of one asset's history in one episode, built from its op sequence.

    ``observe`` is called with every decision quarter's view. A piece opens when the asset starts
    running and closes when it stops or fails. The latent of a life is made when its first piece
    opens (``_latent_of``) and never again.
    """

    def __init__(
        self,
        plant: F1Plant,
        phys: Physical,
        asset_id: str,
        seed: int,
        params: PlantParams,
        faults: FaultPlan | None = None,
        transform: FrameTransform | None = None,
    ) -> None:
        self.asset_id = asset_id
        self._phys = phys
        self._transform = transform
        self.ah_id = asset_health_id(asset_id)
        self._plant = plant
        self._seed = seed
        self._params = params
        self._faults = faults if faults is not None and faults.active else None
        self._records: dict[int, LifeLatent] = {}
        self._pieces: list[_Piece] = []
        self._open: _Piece | None = None
        self._prev_op: Op | None = None
        self._jobs_at_end = 0
        self._rendered: dict[tuple[Any, ...], RenderedTelemetry] = {}
        self._dropped: set[tuple[int, int]] = set()  # (life id, start quarter) of dropped pieces

    @property
    def dropped_pieces(self) -> int:
        """Pieces ``history`` dropped: the asset restarted past its life's failure day."""
        return len(self._dropped)

    # ------------------------------------------------------------------ the op sequence
    @property
    def running(self) -> bool:
        return self._open is not None

    def observe(self, view: AssetView, quarter: int) -> bool:
        """Note the asset's state at the start of ``quarter``. True if its history changed."""
        op = view.op
        prev, self._prev_op = self._prev_op, op
        if prev is None:
            if op is Op.RUN:
                self._start(view, quarter)
                return True
            return False
        if prev is Op.RUN and op is not Op.RUN:
            piece = self._open
            if piece is not None:
                piece.end_q = quarter
                self._pieces.append(piece)
                self._open = None
                self._jobs_at_end = view.jobs_done
            return False  # a stop adds no row: the pipeline hears of it when the asset restarts
        if prev is not Op.RUN and op is Op.RUN:
            self._start(view, quarter)
            return True
        return False

    def _start(self, view: AssetView, quarter: int) -> None:
        s = self._plant.des.by_id[self.asset_id]  # sim-only: which component is fitted
        life_id = int(s.life_id)
        if life_id not in self._records:
            self._records[life_id] = self._latent_of(s, quarter)
        self._open = _Piece(life_id, quarter, None, view.jobs_done > self._jobs_at_end)
        self._jobs_at_end = view.jobs_done

    def _latent_of(self, s: Any, quarter: int) -> LifeLatent:
        """Wall-clock onset and failure of the component fitted in ``s``, if it ran on from
        ``quarter`` without another stop (the plant's rule: the clock earns ``cum``
        while it runs)."""
        cum = self._phys.cum[self.asset_id]
        life = s.life
        tau_now = float(s.clock.tau(quarter))
        onset = wall_day(cum, quarter, life.tau_onset - tau_now)
        failure = wall_day(cum, quarter, life.tau_fail - tau_now)
        return LifeLatent(
            self.asset_id,
            int(s.life_id),
            life.mode,
            onset,
            max(failure, onset + 1e-6),
            float(life.shape),
            float(life.downtime_hours),
        )

    # ------------------------------------------------------------------ history
    @property
    def latents(self) -> dict[int, LifeLatent]:
        """The latent of every life that has been rendered, by plant life id (for the tests)."""
        return dict(self._records)

    def history(self) -> list[RenderedTelemetry]:
        """The pieces the pipeline is shown now, as rendered telemetry, in order.

        A piece too short to hold a row is dropped, and the repair record it carried moves to the
        next piece (a completed job must not vanish because the asset blinked).

        So is a piece that would start at or after the life's failure day. That day is fixed when
        the life's first piece opens, on the assumption that the asset does not stop again, so a
        long stop can put a restart past it, and the pipeline then sees no more rows of that life.
        The condition is tested **before** the piece is built: ``LifeLatent.__post_init__`` refuses
        such a piece (``RenderError``), so a test made afterwards on the built latent never runs
        (the first full run lost an F2 episode to that, 0003 A18 / P12g). The drops are counted in
        ``dropped_pieces``, once per piece however often the history is asked for.
        """
        pieces = [*self._pieces, *([self._open] if self._open is not None else [])]
        out: list[RenderedTelemetry] = []
        carry = False
        for piece in pieces:
            base = self._records[piece.life_id]
            repair = piece.repair or carry
            start_day = piece.start_q / DAY_Q
            end_day = None if piece.end_q is None else piece.end_q / DAY_Q
            stop_day = base.failure_day if end_day is None else min(end_day, base.failure_day)
            if stop_day <= start_day:
                self._dropped.add((piece.life_id, piece.start_q))
                carry = repair
                continue
            latent = dataclasses.replace(
                base,
                start_day=start_day,
                end_day=end_day,
                repair_at_start=repair if out else False,
            )
            rendered = self._render(latent, len(out))
            if rendered.n_rows == 0:
                carry = repair
                continue
            carry = False
            out.append(rendered)
        return out

    def _render(self, latent: LifeLatent, segment: int) -> RenderedTelemetry:
        key = (latent, segment)
        hit = self._rendered.get(key)
        if hit is None:
            hit = render_life(latent, self._params, self._seed, segment)
            if self._faults is not None:
                hit = dataclasses.replace(hit, frame=inject_faults(hit.frame, self._faults))
            if self._transform is not None:
                hit = dataclasses.replace(hit, frame=self._transform(hit.frame))
            self._rendered[key] = hit
        return hit


# ----------------------------------------------------------------------------- the four assets


@dataclass
class _Held:
    output: EstimatorOutput


@dataclass
class TelemetryStats:
    """Counters of the pipeline side of one episode."""

    daily_estimates: int = 0
    held_estimates: int = 0
    pipeline_runs: int = 0
    restarts: int = 0
    # pieces dropped because the asset restarted past the failure day fixed at its life's first
    # start: the pipeline sees no more rows of that life (see ``AssetTelemetry.history``)
    dropped_pieces: int = 0
    estimated_status: dict[str, int] = field(default_factory=dict)


class F2Telemetry:
    """Pieces, pipeline runs and the daily estimate of every asset of one episode."""

    def __init__(
        self,
        plant: F1Plant,
        phys: Physical,
        seed: int,
        params: PlantParams,
        cache: RunCache,
        *,
        faults: bool = True,
        transforms: Mapping[str, FrameTransform] | None = None,
    ) -> None:
        self._plant = plant
        self._params = params
        self.cache = cache
        self.assets: dict[str, AssetTelemetry] = {
            aid: AssetTelemetry(
                plant,
                phys,
                aid,
                seed,
                params,
                fault_plan(phys, aid, params, seed) if faults else None,
                None if transforms is None else transforms.get(aid),
            )
            for aid in ASSET_IDS
        }
        self._pd = load("pandas")
        self._policy = load("asset_health.alerting.policy")
        self._health = load("asset_health.rul.health_index")
        self._estimate = load("asset_health.rul.estimate")
        self._hi: dict[str, tuple[int, Any]] = {}
        self._last: dict[str, EstimatorOutput] = {}
        self._changed: set[str] = set()
        self.stats = TelemetryStats()

    def observe(self, assets: Any, quarter: int) -> None:
        for aid, tel in self.assets.items():
            if tel.observe(assets[aid], quarter):
                self._changed.add(aid)

    def _health_index(self, aid: str) -> Any:
        run = self.cache.run(aid)
        hit = self._hi.get(aid)
        if hit is None or hit[0] != id(run):
            cfg = self.cache.analysis_cfg
            hit = (id(run), self._health.health_index(run.z24, cfg.rul.top_k))
            self._hi[aid] = hit
        return hit[1]

    def may_alert(self, aid: str, instant: datetime) -> bool:
        """False only if no alert can be open for ``aid`` at ``instant``.

        An alert episode opens when the fused score reaches the asset type's threshold, so while the
        fused score of the current history has not reached it by ``instant`` no work order exists
        and ``analyse_run`` (0.05 to 0.3 s) need not be called. A superset test: after a repair
        record it stays true, and then the real analysis decides. This is the "or the fused score
        first crosses its threshold" half of the calling rule of 0001 section 3.
        """
        if not self.cache.has(aid):
            return False
        run = self.cache.run(aid)
        cfg = self.cache.analysis_cfg
        threshold = self._policy.threshold_for(run.spec.asset_type, cfg.policy)
        fused = run.frame["fused"].loc[: self._pd.Timestamp(naive(instant))]
        peak = fused.max()
        return bool(peak == peak and float(peak) >= float(threshold))

    def estimate(
        self, aid: str, day: int, view: AssetView, life_age_days: float
    ) -> EstimatorOutput:
        """The daily estimator output of asset ``aid`` at the start of plant day ``day``.

        asset-health's own ``estimate_rul`` on the cached health index when the asset is running;
        the previous output (held) when it is not; the class prior when the pipeline reports no
        degradation, too few points, or is still learning its baseline.
        """
        tel = self.assets[aid]
        self.stats.daily_estimates += 1
        if not tel.running:
            self.stats.held_estimates += 1
            held = self._last.get(aid)
            if held is not None:
                return dataclasses.replace(held, day=day)
            return self._prior(aid, day, life_age_days, STATUS_PRIOR)
        history = tel.history()
        self.stats.dropped_pieces = sum(a.dropped_pieces for a in self.assets.values())
        if not history:  # the asset has run for less than an hour
            return self._prior(aid, day, life_age_days, STATUS_PRIOR)
        if self.cache.set_history(aid, history):
            self.stats.pipeline_runs += 1
        hi = self._health_index(aid)
        cfg = self.cache.analysis_cfg
        stamp = self._pd.Timestamp(naive(sim_time(day)))
        est = self._estimate.estimate_rul(
            history[0].ah_asset_id,
            hi,
            stamp,
            self.cache.rul_model,
            cfg.rul,
            history[0].spec.asset_type,
        )
        status = est.status.value
        self.stats.estimated_status[status] = self.stats.estimated_status.get(status, 0) + 1
        if status == "estimated":
            p10, p50, p90 = float(est.p10_days), float(est.p50_days), float(est.p90_days)
            out = EstimatorOutput(
                aid, day, STATUS_ESTIMATED, p50, max((p90 - p10) / (2.0 * _Z90), 1e-6), None
            )
        else:
            out = self._prior(
                aid,
                day,
                life_age_days,
                STATUS_NOT_DEGRADING if status == "not_degrading" else STATUS_INSUFFICIENT,
            )
        self._last[aid] = out
        return out

    def _prior(self, aid: str, day: int, age_days: float, status: str) -> EstimatorOutput:
        prior = class_prior(ASSET_CLASS[aid], self._params.renewal_onset_days)
        m, sigma = prior.point_and_scale(max(age_days, 0.0))
        return EstimatorOutput(aid, day, status, m, sigma, None)


def sim_instant(quarter: int) -> datetime:
    """The aware simulation instant at the start of plant quarter ``quarter``."""
    return aware(sim_time(0) + timedelta(minutes=15 * quarter))
