"""``Plant`` for the SYNTHETIC F1 simulator: observations, ``step``, the ``Trace`` and twin worlds.

``F1Plant`` wires the pieces of P3a (degradation SCM, observation chain) to the DES of P3b. One call
of ``step`` advances exactly one quarter. The ``Observation`` it returns is what a proposer sees
when it decides at the *next* quarter; it holds no truth field (a reflection test enforces that).
Truth is on ``Plant.truth`` and in ``Plant.trace()``.

Interfaces beyond the frozen text of docs/adr/0002 section 3.5, all optional or additive:

* ``set_perception`` records the perception layer's set and interval for an asset (the plant does
  not estimate anything); it flows into the views and into ``Trace.set_lo/set_hi``;
* ``set_monitor_fired`` sets ``Observation.monitor_fired``;
* ``step(commands, overridden=())`` marks commands the shield replaced, for ``Trace.commands``;
* ``stats()``, ``alert_truth()`` and ``life_start_day()`` give the cost counters and the sim-only
  labels of alerts;
* extra defaulted fields on ``AssetView`` (``job``, ``last_obs``, ``jobs_done``, ``inspection``,
  ``derated``) carry the daily signature channels, the load and the open work order.
"""

import dataclasses
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from closed_loop.core.timegrid import DAY_Q, SHIFT_Q
from closed_loop.core.trace import Trace
from closed_loop.core.types import Command, Fidelity, Op, Permit, bin_of_rul
from closed_loop.plant.degradation import Physical, build_physical
from closed_loop.plant.des import (
    AssetState,
    Des,
    DesStreams,
    JobView,
    SpecLike,
    StepOutcome,
    build_des_streams,
)
from closed_loop.plant.notes import NOTE_BANK, bank_index, draw_note
from closed_loop.plant.observe import (
    DailyObs,
    ObsStreams,
    alert_cause,
    blind_until_day,
    build_obs_streams,
    observe_day,
)
from closed_loop.plant.params import ASSET_CLASS, ASSET_IDS, MODES_BY_CLASS, PlantParams
from closed_loop.plant.scenario import Scenario

_N_BINS_MAX = 3
_GAS_AGE_CAP_Q = 99


@dataclass(frozen=True)
class InspectResult:
    """What an inspection found: a low-noise health-index reading and a drift flag."""

    day: int
    hi: float  # no sensor gain drift in it: it measures the asset, not the transmitter
    gain_fault: bool


@dataclass(frozen=True, slots=True)
class AssetView:
    asset_id: str
    op: Op
    permit: Permit
    occupied: bool
    gas_age_q: int
    alarm: bool
    soon: bool
    set_lo: int  # perception set (health bins)
    set_hi: int
    rul_lo: float | None
    rul_hi: float | None
    prior_only: bool
    hi_last: float | None
    job_open: bool
    spares: Mapping[str, int]
    # ---- additive fields (docs/adr/0003 A3: signature channels and load are observed)
    job: JobView | None = None
    last_obs: DailyObs | None = None  # y_d = (hi, sig_bearing, sig_proc, sig_hyd) plus the load
    jobs_done: int = 0  # completed work orders, right or wrong; a proposer cannot see which
    inspection: InspectResult | None = None
    derated: bool = False
    # quarters until the permit is spent for good (12 h from its issue, 0003 A12.1); None without
    # one. It is printed on a real permit, so it is observed, not truth.
    permit_limit_left_q: int | None = None


@dataclass(frozen=True, slots=True)
class Observation:
    quarter: int
    day: int
    shift: int
    assets: Mapping[str, AssetView]
    note: str | None  # shift-handover note, if one was posted this quarter
    alerts: tuple[str, ...]  # alert ids opened since the last call
    crew_free: int
    safety_officer_on_shift: bool
    monitor_fired: bool


@dataclass(frozen=True, slots=True)
class StepResult:
    obs: Observation
    executed: tuple[Command, ...]
    refused: tuple[Command, ...]
    violations: tuple[
        tuple[str, str, int], ...
    ]  # (property id, asset id, quarter), the plant's checks
    done: bool


@dataclass(frozen=True)
class TruthView:
    """Sim-only truth. Proposers never receive it."""

    true_rul_days: Mapping[
        str, float
    ]  # realised: failure time with no further intervention, minus now
    health_bin: Mapping[
        str, int
    ]  # bin of the rate RUL (remaining clock over the current wear rate)
    life_id: Mapping[str, int]
    failure_time_q: Mapping[str, float]  # T in the no-intervention continuation
    rate_rul_days: Mapping[str, float] = dataclasses.field(default_factory=dict)
    life_mode: Mapping[str, str] = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class AlertTruth:
    alert_id: str
    asset_id: str
    day: int
    life_id: int
    mode: str
    cause: str  # "degradation" | "sensor_fault" | "process_upset"


@dataclass(frozen=True)
class PlantStats:
    planned_jobs: int
    unplanned_failures: int
    wrong_mode_jobs: int
    wo_rejected: int
    inspections: int
    orders: int
    cost: float


class Plant(Protocol):
    truth: TruthView

    def reset(self, seed: int, scenario: Scenario) -> Observation: ...
    def step(self, commands: Sequence[Command]) -> StepResult: ...
    def trace(self) -> Trace: ...
    def twin(self, intervention: Sequence[tuple[int, Command]]) -> "Plant": ...


class F1Plant:
    """The owned F1 plant. One instance can be ``reset`` many times."""

    def __init__(
        self,
        params: PlantParams | None = None,
        *,
        fidelity: Fidelity = Fidelity.F1,
        spec: SpecLike | None = None,
    ):
        params = params if params is not None else PlantParams()
        if fidelity is Fidelity.F1D:
            params = dataclasses.replace(params, derate_enabled=True)
        elif fidelity is not Fidelity.F1:
            raise ValueError(f"F1Plant implements F1 and F1+D, not {fidelity}")
        params.validate()
        self.params = params
        self.fidelity = fidelity
        self._spec = spec
        self._seed: int | None = None
        self._scenario: Scenario | None = None
        self._des: Des | None = None
        self._interventions: dict[int, list[Command]] = {}

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int, scenario: Scenario | None = None) -> Observation:
        scenario = scenario if scenario is not None else Scenario()
        p = self.params
        self._seed, self._scenario = seed, scenario
        resolved = scenario.resolve(seed, p)
        self._resolved = resolved
        phys: Physical = build_physical(p, seed, resolved)
        self._phys = phys
        streams: DesStreams = build_des_streams(p, seed)
        des = Des(p, seed, resolved, phys, streams, self._spec)
        des.inspection_hook = self._on_inspection
        self._des = des
        self._obs_streams: dict[str, ObsStreams] = {
            aid: build_obs_streams(p, seed, aid, phys) for aid in ASSET_IDS
        }
        self._last_obs: dict[str, DailyObs | None] = dict.fromkeys(ASSET_IDS)
        self._inspect: dict[str, InspectResult | None] = dict.fromkeys(ASSET_IDS)
        self._perc: dict[str, tuple[int, int, float | None, float | None, bool]] = {
            aid: (0, _N_BINS_MAX, None, None, False) for aid in ASSET_IDS
        }
        self._alert_level: dict[str, int] = dict.fromkeys(ASSET_IDS, 0)
        self._alert_life: dict[str, int] = dict.fromkeys(ASSET_IDS, 0)
        self._alert_truth: dict[str, AlertTruth] = {}
        self._new_alerts: list[str] = []
        self._monitor_fired = False
        self._interventions = {}  # a reset starts a fresh world; a twin sets its own after this
        if resolved.note_q is not None:
            self._note_handover = max(round(resolved.note_q / SHIFT_Q), 1)
        else:
            self._note_handover = -1
        self._views: dict[str, AssetView] = {}
        self._dirty: set[int] = set(range(len(ASSET_IDS)))
        self._crew_free = self.params.crew_per_shift
        self._timed: set[int] = set()  # views that change every quarter (open job, recent gas test)
        self._assets_map: Mapping[str, AssetView] | None = None
        self._parts = {
            aid: tuple(sorted({m.part_no for m in MODES_BY_CLASS[ASSET_CLASS[aid]]}))
            for aid in ASSET_IDS
        }
        self._spares: dict[str, tuple[int, Mapping[str, int]]] = {}
        self._stock_seen = self.des.stock_version
        self._daily(0)
        self.initial_observation = self._observation(None)
        return self.initial_observation

    @property
    def des(self) -> Des:
        if self._des is None:
            raise RuntimeError("call reset() before using the plant")
        return self._des

    # ------------------------------------------------------------------ daily observations, alerts
    def _daily(self, q: int) -> None:
        p = self.params
        des = self.des
        day = q // DAY_Q
        if day >= p.horizon_days:
            return
        levels = (p.hi_alert, 0.5 * (p.hi_alert + p.hi_fail))
        for s in des.states:
            aid = s.aid
            load_arr = self._phys.derated(aid)[0] if s.clock.derated else self._phys.load_day[aid]
            load = float(load_arr[day])
            sev = s.clock.severity(q)
            obs = observe_day(p, self._obs_streams[aid], day, sev, s.life.signature, load)
            self._last_obs[aid] = obs
            self._dirty.add(s.idx)
            if self._alert_life[aid] != s.life_id:
                self._alert_life[aid], self._alert_level[aid] = s.life_id, 0
            lvl = self._alert_level[aid]
            life_start_day = s.life_start_q / DAY_Q
            if (
                obs.hi is not None
                and lvl < len(levels)
                and obs.hi >= levels[lvl]
                and day >= blind_until_day(life_start_day, p)
            ):
                self._alert_level[aid] = lvl + 1
                alert_id = f"{aid}-L{s.life_id}-A{lvl + 1}"
                self._new_alerts.append(alert_id)
                self._alert_truth[alert_id] = AlertTruth(
                    alert_id,
                    aid,
                    day,
                    s.life_id,
                    s.life.mode,
                    alert_cause(p, self._obs_streams[aid], day, sev, load),
                )

    def _on_inspection(self, s: AssetState, nxt: int, z: float) -> None:
        p = self.params
        day = min(nxt // DAY_Q, p.horizon_days - 1)
        streams = self._obs_streams[s.aid]
        sev = s.clock.severity(nxt)
        load_arr = self._phys.derated(s.aid)[0] if s.clock.derated else self._phys.load_day[s.aid]
        hi = (
            p.hi_base
            + (p.hi_fail - p.hi_base) * sev
            + p.hi_load_coef * (float(load_arr[day]) - 1.0)
            + z * p.hi_noise_sd * p.inspect_noise_factor
        )
        self._inspect[s.aid] = InspectResult(day, float(hi), bool(streams.drift_frac[day] > 0.01))
        self._dirty.add(s.idx)

    # ------------------------------------------------------------------ views and observations
    def _view(self, s: AssetState) -> AssetView:
        des = self.des
        q = des.q
        lo, hi, rul_lo, rul_hi, prior = self._perc[s.aid]
        obs = self._last_obs[s.aid]
        cached = self._spares.get(s.aid)
        if cached is None or cached[0] != des.stock_version:
            cached = (
                des.stock_version,
                MappingProxyType({p: des.stock[p] for p in self._parts[s.aid]}),
            )
            self._spares[s.aid] = cached
        return AssetView(
            asset_id=s.aid,
            op=s.op,
            permit=s.permit,
            occupied=s.occ,
            gas_age_q=min(des.gas_age_q(s, q), _GAS_AGE_CAP_Q),
            alarm=s.alarm,
            soon=des.soon(s, q),
            set_lo=lo,
            set_hi=hi,
            rul_lo=rul_lo,
            rul_hi=rul_hi,
            prior_only=prior,
            hi_last=None if obs is None else obs.hi,
            job_open=s.job is not None,
            spares=cached[1],
            job=des.job_view(s, q),
            last_obs=obs,
            jobs_done=s.jobs_done,
            inspection=self._inspect[s.aid],
            derated=s.clock.derated,
            permit_limit_left_q=des.permit_limit_left_q(s, q),
        )

    def _note_for(self, q: int) -> str | None:
        """The note posted at the start of quarter ``q`` (a shift start), if any."""
        if q <= 0 or q % SHIFT_Q != 0 or q >= self.params.horizon_q:
            return None
        k = q // SHIFT_Q
        if k == self._note_handover and self._resolved.note_u is not None:
            return NOTE_BANK[bank_index(self._resolved.note_u)].text
        u = self.des.st.notes[k]
        return draw_note(float(u[0]), float(u[1]), self.params.p_malicious_note).text

    @staticmethod
    def _is_timed(s: AssetState, q: int) -> bool:
        return (
            s.job is not None
            or s.permit is not Permit.NONE  # its time left, and `soon`, move with the clock
            or (s.last_pass_q > -(10**8) and q - s.last_pass_q <= _GAS_AGE_CAP_Q)
        )

    def _observation(self, note: str | None, touched: frozenset[int] = frozenset()) -> Observation:
        des = self.des
        q = des.q
        rem = q % SHIFT_Q
        if rem == 0 or rem == SHIFT_Q - 1:
            self._dirty.update(range(len(ASSET_IDS)))  # `soon` flips on both sides of a handover
        timed = self._timed
        for i in touched:
            if self._is_timed(des.states[i], q):
                timed.add(i)
        if timed:
            for i in tuple(timed):
                self._dirty.add(i)  # a view that changes with time alone
                if not self._is_timed(des.states[i], q):
                    timed.discard(i)  # rebuilt one last time, then left alone
        if self._dirty or self._assets_map is None:
            for i in self._dirty:
                s = des.states[i]
                self._views[s.aid] = self._view(s)
            self._dirty.clear()
            self._assets_map = MappingProxyType(dict(self._views))
        alerts = tuple(self._new_alerts)
        self._new_alerts.clear()
        self._crew_free = des.crew_free()
        return Observation(
            quarter=q,
            day=q // DAY_Q,
            shift=(q % DAY_Q) // SHIFT_Q,
            assets=self._assets_map,
            note=note,
            alerts=alerts,
            crew_free=self._crew_free,
            safety_officer_on_shift=des.safety_officer_on_shift(q),
            monitor_fired=self._monitor_fired,
        )

    # ------------------------------------------------------------------ the loop
    def step(self, commands: Sequence[Command], overridden: Sequence[Command] = ()) -> StepResult:
        des = self._des
        if des is None:
            raise RuntimeError("call reset() before using the plant")
        q = des.q
        horizon_q = self.params.horizon_q
        if q >= horizon_q:
            raise RuntimeError("the episode is over")
        cmds = commands
        if self._interventions:
            pending = self._interventions.get(q)
            if pending:
                cmds = list(pending) + list(commands)
        out: StepOutcome = des.step(cmds, overridden)
        if des.stock_version != self._stock_seen:
            # Stock is shared: a part taken for one pump shows in the view of the other at once.
            self._stock_seen = des.stock_version
            self._dirty.update(range(len(ASSET_IDS)))
        nq = q + 1
        rem = nq % SHIFT_Q
        if (
            not out.touched
            and rem != 0
            and rem != SHIFT_Q - 1
            and not self._dirty
            and not self._timed
        ):
            # A quiet quarter: no asset view changed, no handover, no day boundary. The assets
            # mapping is the one already handed out, and nothing else in the observation moves.
            obs = Observation(
                nq,
                nq // DAY_Q,
                (nq % DAY_Q) // SHIFT_Q,
                self._assets_map,  # type: ignore[arg-type]
                None,
                (),
                self._crew_free,
                nq % DAY_Q < SHIFT_Q,
                self._monitor_fired,
            )
            return StepResult(obs, out.executed, out.refused, out.violations, nq >= horizon_q)
        self._dirty.update(out.touched)
        done = nq >= horizon_q
        if nq % DAY_Q == 0 and not done:
            self._daily(nq)
        note = self._note_for(nq) if nq % SHIFT_Q == 0 else None
        obs = self._observation(note, out.touched)
        return StepResult(obs, out.executed, out.refused, out.violations, done)

    # ------------------------------------------------------------------ perception, monitor
    def set_perception(
        self,
        asset_id: str,
        *,
        set_lo: int,
        set_hi: int,
        rul_lo: float | None = None,
        rul_hi: float | None = None,
        prior_only: bool = False,
    ) -> None:
        """Record the perception output for ``asset_id`` from the next decision quarter on."""
        if not 0 <= set_lo <= set_hi <= _N_BINS_MAX:
            raise ValueError(
                f"perception set must satisfy 0 <= lo <= hi <= 3, got {(set_lo, set_hi)}"
            )
        s = self.des.by_id[asset_id]
        self._perc[asset_id] = (set_lo, set_hi, rul_lo, rul_hi, prior_only)
        self.des.set_sets(s, self.des.q, set_lo, set_hi)
        self._dirty.add(s.idx)

    def set_monitor_fired(self, fired: bool) -> None:
        self._monitor_fired = fired

    # ------------------------------------------------------------------ truth and records
    @property
    def truth(self) -> TruthView:
        des = self.des
        q = des.q
        rul, bins, life, ft, rate, mode = {}, {}, {}, {}, {}, {}
        for s in des.states:
            r = s.clock.realised_rul_days(q)
            rr = s.clock.rate_rul_days(q)
            rul[s.aid] = r
            rate[s.aid] = rr
            bins[s.aid] = bin_of_rul(rr)
            life[s.aid] = s.life_id
            ft[s.aid] = q + r * DAY_Q
            mode[s.aid] = s.life.mode
        return TruthView(rul, bins, life, ft, rate, mode)

    def trace(self) -> Trace:
        return self.des.build_trace()

    def stats(self) -> PlantStats:
        st = self.des.stats
        return PlantStats(
            st.planned_jobs,
            st.unplanned_failures,
            st.wrong_mode_jobs,
            st.wo_rejected,
            st.inspections,
            st.orders,
            st.cost,
        )

    def alert_truth(self, alert_id: str) -> AlertTruth:
        return self._alert_truth[alert_id]

    def life_start_day(self, asset_id: str) -> float:
        return self.des.by_id[asset_id].life_start_q / DAY_Q

    @property
    def quarter(self) -> int:
        return self.des.q

    @property
    def done(self) -> bool:
        return self.des.q >= self.params.horizon_q

    # ------------------------------------------------------------------ twin worlds
    def twin(self, intervention: Sequence[tuple[int, Command]]) -> "F1Plant":
        """The same world (seed, scenario, every keyed stream) with ``do()`` interventions added.

        Returns a fresh plant that has been reset and not stepped. Each ``(q, command)`` is injected
        at quarter ``q`` before the commands passed to ``step``. A command with
        ``source == DO_REPLACE`` and ``cmd == RAISE_WO`` is an instant component replacement.
        With no intervention, stepping the twin with the commands the factual run received
        reproduces the factual trace bit for bit.
        """
        if self._seed is None or self._scenario is None:
            raise RuntimeError("call reset() before twin()")
        other = F1Plant(self.params, fidelity=self.fidelity, spec=self._spec)
        other.reset(self._seed, self._scenario)
        table: dict[int, list[Command]] = defaultdict(list)
        for q, cmd in intervention:
            if q < 0:
                raise ValueError("intervention quarter must be non-negative")
            table[q].append(cmd)
        other._interventions = dict(table)
        return other


def make_plant(
    params: PlantParams | None = None,
    *,
    fidelity: Fidelity = Fidelity.F1,
    spec: SpecLike | None = None,
) -> F1Plant:
    """Build an F1 (or F1+D) plant. Call ``reset(seed, scenario)`` before stepping."""
    return F1Plant(params, fidelity=fidelity, spec=spec)
