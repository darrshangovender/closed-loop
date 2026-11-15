"""The degradation SCM of the SYNTHETIC plant (0001 section 2, "hidden degradation").

For component life ``l`` of asset ``a`` the exogenous terms are the mode, the duration, the shape
and the onset (one keyed stream per asset and life), plus a process load shared by the assets of a
circuit (one keyed stream per circuit) and a fixed per-asset load offset. Structure:

* the wear clock runs at ``rho(load) = rate * exp(beta (load - 1))`` and only while the asset runs;
* severity is ``s = ((tau - tau_onset) / (tau_fail - tau_onset))^k`` once the clock ``tau`` passes
  the onset, 0 before it, and the component fails when ``tau`` reaches ``tau_fail``;
* a first life starts mid-episode-life: with probability ``p_fail_in_episode`` it fails inside the
  horizon with no action, and its onset is never before ``first_onset_min_day`` (0001 section 2).

Two remaining-life quantities are kept apart, because they answer different questions:

``realised_rul_days``
    the failure time in the no-intervention continuation, minus now: the asset keeps running and the
    realised future load applies. This is the label of conformal coverage (0001 section 6.1) and
    what ``Trace.true_rul_days`` holds.
``rate_rul_days``
    remaining clock divided by the current wear rate: how long the asset has if conditions stay as
    they are now. A load step moves it at once, which is how a load shift can cut the health bin by
    more than one in a tick (the A_phys violation of 0001 section 4.7). ``Trace.health_bin`` is its
    bin.

Randomness: every draw is keyed through ``rng_for(seed, stream, *ids)`` so that a policy change
cannot move the noise of another asset, another life or a later event.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from closed_loop.core.rng import rng_for
from closed_loop.core.timegrid import DAY_Q
from closed_loop.plant.params import (
    ASSET_CLASS,
    ASSET_IDS,
    CIRCUIT,
    MODES_BY_CLASS,
    N_CIRCUITS,
    ModeSpec,
    PlantParams,
)
from closed_loop.plant.scenario import ResolvedScenario, asset_index

# Days of load beyond the horizon, so that failure times past the episode end are still defined.
EXTRA_DAYS = 210
_BIG_Q = 1 << 60


def _lerp(lo: float, hi: float, u: float) -> float:
    return lo + (hi - lo) * u


@dataclass(frozen=True)
class LifeDraw:
    """One component life. Clock quantities are nominal days (wear rate 1.0)."""

    asset_id: str
    life_id: int
    mode: str
    signature: str
    shape: float
    duration: float  # tau_fail - tau_onset
    tau_start: float
    tau_onset: float
    tau_fail: float
    part_no: str
    downtime_hours: float  # extra repair work if this life ends in an unplanned failure
    will_fail: bool | None  # first life only: designed to fail inside the horizon


def mode_spec(asset_class: str, mode: str) -> ModeSpec | None:
    for spec in MODES_BY_CLASS.get(asset_class, ()):
        if spec.mode == mode:
            return spec
    return None


def cum_at_day(cum: NDArray[np.float64], day: float) -> float:
    """Clock earned by running from the start to calendar ``day`` (linear inside a quarter)."""
    x = day * DAY_Q
    i = min(int(x), len(cum) - 2)
    return float(cum[i] + (x - i) * (cum[i + 1] - cum[i]))


def draw_life(
    params: PlantParams,
    seed: int,
    asset_id: str,
    life_id: int,
    *,
    tau_start: float,
    first: bool,
    load_level: float,
    resolved: ResolvedScenario | None = None,
    cum_design: NDArray[np.float64] | None = None,
    cum_actual: NDArray[np.float64] | None = None,
) -> LifeDraw:
    """Draw life ``life_id`` of ``asset_id`` (always the same number of keyed draws).

    A first life is designed in calendar days against ``cum_design``, the clock without the
    scenario's load shift and wear-rate change, so that ``p_fail_in_episode`` of first lives fail
    inside the horizon in the nominal condition. The scenario's shifts then move the realised
    failure, which is the point of a shift. The onset floor is enforced on ``cum_actual``, the clock
    the asset really follows.
    """
    rng = rng_for(seed, "life", asset_id, life_id)
    u = rng.random(8)
    modes = MODES_BY_CLASS[ASSET_CLASS[asset_id]]
    weights = np.array(
        [
            m.weight
            * np.exp(params.mode_load_tilt * load_level * (1.0 if m.signature == "wear" else -1.0))
            if len(modes) > 1
            else m.weight
            for m in modes
        ],
        dtype=np.float64,
    )
    cdf = np.cumsum(weights / weights.sum())
    spec = modes[min(int(np.searchsorted(cdf, u[0], side="right")), len(modes) - 1)]
    shape = _lerp(spec.shape[0], spec.shape[1], float(u[1]))
    downtime = _lerp(params.downtime_hours[0], params.downtime_hours[1], float(u[5]))
    h_days = float(params.horizon_days)
    will_fail: bool | None = None
    if first:
        if cum_design is None or cum_actual is None:
            raise ValueError("a first life needs cum_design and cum_actual")
        onset_min = params.first_onset_min_day
        will_fail = bool(u[2] < params.p_fail_in_episode)
        d_hi = spec.duration_days[1]
        if will_fail:
            d_hi = min(
                d_hi, h_days - onset_min - 1.0
            )  # leave room for a failure inside the horizon
        d_lo = min(spec.duration_days[0], d_hi)
        duration = _lerp(d_lo, d_hi, float(u[3]))
        if will_fail:
            fail_day = _lerp(onset_min + duration, h_days - 0.5, float(u[4]))
        else:
            fail_day = _lerp(
                max(h_days + 1.0, onset_min + duration), h_days + duration, float(u[4])
            )
        tau_fail = cum_at_day(cum_design, fail_day)
        tau_onset = max(tau_fail - duration, cum_at_day(cum_design, onset_min))
        if resolved is not None:
            a = asset_index(asset_id)
            tau_onset += resolved.onset_shift_days[a]
            tau_fail += resolved.residual_shift_days[a]
        tau_onset = max(tau_onset, cum_at_day(cum_actual, onset_min))
        tau_fail = max(tau_fail, tau_onset + 10.0)
        onset_off = tau_onset - tau_start
        duration = tau_fail - tau_onset
    else:
        lo, hi = params.renewal_onset_days
        onset_off = _lerp(lo, hi, float(u[4]))
        duration = _lerp(spec.duration_days[0], spec.duration_days[1], float(u[3]))
    tau_onset = tau_start + onset_off
    return LifeDraw(
        asset_id=asset_id,
        life_id=life_id,
        mode=spec.mode,
        signature=spec.signature,
        shape=shape,
        duration=duration,
        tau_start=tau_start,
        tau_onset=tau_onset,
        tau_fail=tau_onset + duration,
        part_no=spec.part_no,
        downtime_hours=downtime,
        will_fail=will_fail,
    )


def severity(life: LifeDraw, tau: float) -> float:
    """s in [0, 1]: 0 before the onset, 1 at failure, ``x^k`` between."""
    x = (tau - life.tau_onset) / (life.tau_fail - life.tau_onset)
    if x <= 0.0:
        return 0.0
    return min(x, 1.0) ** life.shape


@dataclass(frozen=True)
class Physical:
    """The pre-drawn physical streams of one episode: load, wear rates and the cumulative clocks.

    The derated clocks (F1+D only) are built on first use.
    """

    seed: int
    params: PlantParams
    resolved: ResolvedScenario
    n_ext_days: int
    load_level: dict[str, float]
    load_day: dict[str, NDArray[np.float64]]  # effective load per day, shift applied, clipped
    rho_day: dict[str, NDArray[np.float64]]  # wear rate per day
    cum: dict[str, NDArray[np.float64]]  # cum[q]: nominal clock days earned in quarters 0..q-1
    cum_design: dict[str, NDArray[np.float64]]  # no shift, wear rate 1: a first life's design clock
    shift_day: int  # first day the scenario's load shift applies; n_ext_days when none
    base_load: dict[str, NDArray[np.float64]] = field(default_factory=dict, repr=False)
    _derated: dict[str, tuple[NDArray, NDArray, NDArray]] = field(default_factory=dict, repr=False)

    def derated(self, asset_id: str) -> tuple[NDArray, NDArray, NDArray]:
        """(load per day, wear rate per day, cum) while the asset is derated."""
        hit = self._derated.get(asset_id)
        if hit is None:
            lo, hi = self.resolved.load_clip
            load = np.clip(self.base_load[asset_id] * (1.0 - self.params.derate_fraction), lo, hi)
            rho = self.resolved.rate_mult[ASSET_IDS.index(asset_id)] * np.exp(
                self.params.beta_load * (load - 1.0)
            )
            hit = (load, rho, _cum_of(rho))
            self._derated[asset_id] = hit
        return hit

    def cum_of(self, asset_id: str, derated: bool) -> NDArray[np.float64]:
        return self.derated(asset_id)[2] if derated else self.cum[asset_id]

    def rho_of(self, asset_id: str, derated: bool) -> NDArray[np.float64]:
        return self.derated(asset_id)[1] if derated else self.rho_day[asset_id]


def _cum_of(rho_day: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cumulative clock at each quarter boundary from a per-day wear rate."""
    out = np.empty(len(rho_day) * DAY_Q + 1)
    out[0] = 0.0
    np.cumsum(np.repeat(rho_day, DAY_Q) / DAY_Q, out=out[1:])
    return out


def build_physical(params: PlantParams, seed: int, resolved: ResolvedScenario) -> Physical:
    n_days = params.horizon_days + EXTRA_DAYS
    h_days = params.horizon_days
    lo_clip, hi_clip = resolved.load_clip
    # circuit load deviation: AR(1) plus at most one upset, shared by the assets of the circuit
    dev = np.zeros((N_CIRCUITS, n_days))
    phi = params.load_phi
    sd = params.load_noise_sd
    innov_sd = sd * np.sqrt(1.0 - phi * phi)
    for c in range(N_CIRCUITS):
        z = rng_for(seed, "load", c).standard_normal(n_days).tolist()
        x = sd * z[0]
        series = [x]
        for d in range(1, n_days):
            x = phi * x + innov_sd * z[d]
            series.append(x)
        row = np.array(series)
        ev = rng_for(seed, "upset", c).random(4)
        if ev[0] < params.upset_p_per_circuit:
            start = int(_lerp(10.0, h_days - 10.0, float(ev[1])))
            d_lo, d_hi = params.upset_days
            dur = min(int(_lerp(d_lo, d_hi + 1, float(ev[2]))), d_hi)
            row[start : start + dur] += _lerp(*params.upset_magnitude, float(ev[3]))
        dev[c] = row
    shift_day = n_days
    if resolved.load_shift_frac > 0.0:
        shift_day = int(np.ceil(resolved.load_shift_day))
    level: dict[str, float] = {}
    load_day: dict[str, NDArray[np.float64]] = {}
    base_load: dict[str, NDArray[np.float64]] = {}
    rho_day: dict[str, NDArray[np.float64]] = {}
    cum: dict[str, NDArray[np.float64]] = {}
    cum_des: dict[str, NDArray[np.float64]] = {}
    n_design = params.horizon_days + 110
    for a, aid in enumerate(ASSET_IDS):
        level[aid] = float(rng_for(seed, "level", aid).standard_normal()) * params.load_level_sd
        base = 1.0 + dev[CIRCUIT[aid]] + level[aid]
        design_rho = np.exp(params.beta_load * (np.clip(base[:n_design], lo_clip, hi_clip) - 1.0))
        cum_des[aid] = _cum_of(design_rho)
        if shift_day < n_days:
            base[shift_day:] *= 1.0 + resolved.load_shift_frac
        base_load[aid] = base
        eff = np.clip(base, lo_clip, hi_clip)
        load_day[aid] = eff
        rho_day[aid] = resolved.rate_mult[a] * np.exp(params.beta_load * (eff - 1.0))
        cum[aid] = _cum_of(rho_day[aid])
    return Physical(
        seed=seed,
        params=params,
        resolved=resolved,
        n_ext_days=n_days,
        load_level=level,
        load_day=load_day,
        rho_day=rho_day,
        cum=cum,
        cum_design=cum_des,
        shift_day=shift_day,
        base_load=base_load,
    )


def first_life(phys: Physical, asset_id: str) -> LifeDraw:
    return draw_life(
        phys.params,
        phys.seed,
        asset_id,
        0,
        tau_start=0.0,
        first=True,
        load_level=phys.load_level[asset_id],
        resolved=phys.resolved,
        cum_design=phys.cum_design[asset_id],
        cum_actual=phys.cum[asset_id],
    )


def next_life(phys: Physical, asset_id: str, life_id: int, tau_start: float) -> LifeDraw:
    return draw_life(
        phys.params,
        phys.seed,
        asset_id,
        life_id,
        tau_start=tau_start,
        first=False,
        load_level=phys.load_level[asset_id],
        resolved=phys.resolved,
    )


def bin_of_rul_array(rul_days: NDArray[np.float64]) -> NDArray[np.int8]:
    """Vectorised ``core.types.bin_of_rul`` (same edges and tie rules)."""
    return (3 - (rul_days >= 3.0) - (rul_days > 10.0) - (rul_days > 30.0)).astype(np.int8)


class AssetClock:
    """The wear clock of one asset: life, run state, derate state and a change log.

    Time is the quarter index. ``cum[q]`` is the clock the asset earns by running from quarter 0 up
    to (not including) quarter ``q`` at the current derate state; the clock advances only while
    running. Every change of life, run state or derate state appends a log entry, from which
    ``truth_arrays`` rebuilds the dense sim-only truth for ``Trace``.
    """

    def __init__(self, phys: Physical, asset_id: str, life: LifeDraw):
        self._phys = phys
        self.asset_id = asset_id
        self.life = life
        self.running = True
        self.derated = False
        self._q_mark = 0
        self._tau_acc = life.tau_start
        self.log: list[tuple[int, int, float, bool, bool, float]] = []
        self._append(0)

    # -------------------------------------------------------------- state changes
    def _cum(self, derated: bool | None = None) -> NDArray[np.float64]:
        d = self.derated if derated is None else derated
        return self._phys.cum_of(self.asset_id, d)

    def _append(self, q: int) -> None:
        self.log.append(
            (q, self.life.life_id, self._tau_acc, self.running, self.derated, self.life.tau_fail)
        )

    def tau(self, q: int) -> float:
        """Clock reading at the start of quarter ``q``."""
        if not self.running:
            return self._tau_acc
        cum = self._cum()
        return self._tau_acc + float(cum[q] - cum[self._q_mark])

    def _sync(self, q: int) -> None:
        self._tau_acc = self.tau(q)
        self._q_mark = q

    def set_running(self, q: int, running: bool) -> None:
        if running == self.running:
            return
        self._sync(q)
        self.running = running
        self._append(q)

    def set_derated(self, q: int, derated: bool) -> None:
        if derated == self.derated:
            return
        self._sync(q)
        self.derated = derated
        self._append(q)

    def start_life(self, q: int, life: LifeDraw) -> None:
        """Renewal at quarter ``q``: the new component starts at the current derate-free state."""
        self.life = life
        self._tau_acc = life.tau_start
        self._q_mark = q
        self.derated = False
        self._append(q)

    # -------------------------------------------------------------- queries
    def fail_q(self) -> int:
        """First quarter whose start sees the clock at ``tau_fail``; a huge value if not running."""
        if not self.running:
            return _BIG_Q
        cum = self._cum()
        target = cum[self._q_mark] + (self.life.tau_fail - self._tau_acc)
        return int(np.searchsorted(cum, target, side="left"))

    def severity(self, q: int) -> float:
        return severity(self.life, self.tau(q))

    def remaining_clock(self, q: int) -> float:
        return self.life.tau_fail - self.tau(q)

    def realised_rul_days(self, q: int) -> float:
        rem = self.remaining_clock(q)
        if rem <= 0.0:
            return 0.0
        cum = self._cum()
        idx = int(np.searchsorted(cum, cum[q] + rem, side="left"))
        return max(min(idx, len(cum) - 1) - q, 0) / DAY_Q

    def rate_rul_days(self, q: int) -> float:
        rem = self.remaining_clock(q)
        if rem <= 0.0:
            return 0.0
        rho = self._phys.rho_of(self.asset_id, self.derated)
        return float(rem / rho[q // DAY_Q])

    # -------------------------------------------------------------- dense truth for the Trace
    def truth_arrays(
        self, n_q: int
    ) -> tuple[NDArray[np.float32], NDArray[np.int8], NDArray[np.int64]]:
        """Dense (realised RUL in days, health bin of the rate RUL, life id) for rows 0..n_q-1."""
        rul = np.zeros(n_q, dtype=np.float64)
        rate = np.zeros(n_q, dtype=np.float64)
        life_ids = np.zeros(n_q, dtype=np.int64)
        log = self.log
        for i, (q0, life_id, tau_acc, running, derated, tau_fail) in enumerate(log):
            q1 = log[i + 1][0] if i + 1 < len(log) else n_q
            if q0 >= n_q or q1 <= q0:
                continue
            q1 = min(q1, n_q)
            rows = np.arange(q0, q1)
            qs = rows + 1  # row r is the end of step r, so the truth is read at the start of r + 1
            cum = self._phys.cum_of(self.asset_id, derated)
            rho_day = self._phys.rho_of(self.asset_id, derated)
            if running:
                rem = tau_fail - tau_acc - (cum[qs] - cum[q0])
                fail_idx = int(np.searchsorted(cum, cum[q0] + (tau_fail - tau_acc), side="left"))
                seg_rul = np.maximum(np.minimum(fail_idx, len(cum) - 1) - qs, 0) / DAY_Q
            else:
                rem = np.full(qs.shape, tau_fail - tau_acc)
                idx = np.searchsorted(cum, cum[qs] + max(tau_fail - tau_acc, 0.0), side="left")
                seg_rul = np.maximum(np.minimum(idx, len(cum) - 1) - qs, 0) / DAY_Q
            seg_rul = np.where(rem <= 0.0, 0.0, seg_rul)
            seg_rate = np.where(rem <= 0.0, 0.0, rem / rho_day[qs // DAY_Q])
            rul[q0:q1] = seg_rul
            rate[q0:q1] = seg_rate
            life_ids[q0:q1] = life_id
        return rul.astype(np.float32), bin_of_rul_array(rate), life_ids
