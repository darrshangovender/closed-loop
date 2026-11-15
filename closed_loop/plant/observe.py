"""The SYNTHETIC observation chain: the daily reading ``y_d`` and the no-intervention life table.

Docs/adr/0003 A3 extends the daily observation to ``y_d = (hi, sig_bearing, sig_proc, sig_hyd)`` and
the plant also reports the load, which is observed. With severity ``s``, load ``L`` and the life's
signature family:

* ``hi = hi_base + (hi_fail - hi_base) s + hi_load_coef (L - 1) + drift * hi_fail + noise``
* ``sig_i = gain_i s + sig_load_coef_i (L - 1) + noise`` with gains ``sig_loading_bearing`` for a
  bearing defect and ``sig_loading_wear`` for impeller or liner wear

So a bearing fault moves ``sig_bearing``; wear moves ``sig_proc`` and ``sig_hyd``; a sensor gain
drift moves ``hi`` alone (the signatures come from other measurement chains); and a process upset, a
rise of the load, moves all four channels. The coefficients are fields of ``PlantParams``.

Data-quality events act on ``hi`` only. A stuck day and a spike day are masked by the gate (``hi``
is ``None``). A slow gain drift is not masked (S: asset-health README limitation). After a life
starts the estimator is blind for ``blind_days`` (21 d baseline plus 48 h settle); this module keeps
that bookkeeping but still reports ``hi``: the estimator decides what to do with it.

Everything is SYNTHETIC.
"""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from closed_loop.core.rng import rng_for
from closed_loop.core.timegrid import DAY_Q
from closed_loop.plant.degradation import (
    LifeDraw,
    Physical,
    build_physical,
    first_life,
    severity,
)
from closed_loop.plant.params import ASSET_IDS, SIG_BEARING, PlantParams
from closed_loop.plant.scenario import FAULT_DRIFT, FAULT_SPIKE, FAULT_STUCK, Scenario, asset_index

CAUSES = ("degradation", "sensor_fault", "process_upset")


@dataclass(frozen=True, slots=True)
class DailyObs:
    """The reading of one asset on one day. ``hi`` is ``None`` on a day the gate masked."""

    day: int
    hi: float | None
    sig_bearing: float
    sig_proc: float
    sig_hyd: float
    load: float

    @property
    def y(self) -> tuple[float, float, float, float]:
        """``y_d = (hi, sig_bearing, sig_proc, sig_hyd)``, NaN where ``hi`` is masked."""
        return (
            float("nan") if self.hi is None else self.hi,
            self.sig_bearing,
            self.sig_proc,
            self.sig_hyd,
        )


@dataclass(frozen=True)
class ObsStreams:
    """Pre-drawn noise and data-quality events of one asset, indexed by calendar day."""

    hi_noise: NDArray[np.float64]
    sig_noise: NDArray[np.float64]  # (3, n_days)
    load_noise: NDArray[np.float64]
    masked: NDArray[np.bool_]  # stuck or spike days: hi is None
    drift_frac: NDArray[np.float64]  # unmasked gain drift as a fraction of full scale, per day
    baseline_fault_day: float | None  # onset of the asset's own gain fault, if any


def build_obs_streams(params: PlantParams, seed: int, asset_id: str, phys: Physical) -> ObsStreams:
    n = params.horizon_days
    r = phys.resolved
    a = asset_index(asset_id)
    hi_noise = rng_for(seed, "hi_noise", asset_id).standard_normal(n)
    sig_noise = rng_for(seed, "sig_noise", asset_id).standard_normal((3, n))
    load_noise = rng_for(seed, "load_obs", asset_id).standard_normal(n)
    dq = rng_for(seed, "dq", asset_id)
    u_stuck = dq.random(n)
    u_spike = dq.random(n)
    g = dq.random(4)
    masked = (u_stuck < params.p_stuck_day) | (u_spike < params.p_spike_day)
    days = np.arange(n, dtype=np.float64)
    drift = np.zeros(n)
    base_day: float | None = None
    if g[0] < params.p_gain_fault:
        base_day = float(10.0 + g[1] * (n - 15.0))
        rate = params.gain_rate[0] + (params.gain_rate[1] - params.gain_rate[0]) * float(g[2])
        drift += rate * np.maximum(days - base_day, 0.0)
    if r.drift_all_per_day > 0.0:
        drift += r.drift_all_per_day * np.maximum(days - r.drift_all_day, 0.0)
    kind = r.fault_type[a]
    fday = r.fault_day[a]
    if kind == FAULT_STUCK:
        masked[(days >= fday) & (days < fday + params.stuck_days)] = True
    elif kind == FAULT_SPIKE:
        masked[int(min(round(fday), n - 1))] = True
    elif kind == FAULT_DRIFT:
        drift += params.shifted_drift_per_day * np.maximum(days - fday, 0.0)
    for arr in (hi_noise, sig_noise, load_noise, masked, drift):
        arr.setflags(write=False)
    return ObsStreams(hi_noise, sig_noise, load_noise, masked, drift, base_day)


def observe_day(
    params: PlantParams,
    streams: ObsStreams,
    day: int,
    sev: float,
    signature: str,
    load: float,
) -> DailyObs:
    """The daily reading for severity ``sev`` and effective load ``load``.

    Pure: the plant and the life table share this one code path.
    """
    gains = params.sig_loading_bearing if signature == SIG_BEARING else params.sig_loading_wear
    dl = load - 1.0
    hi = (
        params.hi_base
        + (params.hi_fail - params.hi_base) * sev
        + params.hi_load_coef * dl
        + streams.drift_frac[day] * params.hi_fail
        + params.hi_noise_sd * streams.hi_noise[day]
    )
    s = streams.sig_noise[:, day]
    c = params.sig_load_coef
    sd = params.sig_noise_sd
    return DailyObs(
        day=day,
        hi=None if streams.masked[day] else float(hi),
        sig_bearing=float(gains[0] * sev + c[0] * dl + sd * s[0]),
        sig_proc=float(gains[1] * sev + c[1] * dl + sd * s[1]),
        sig_hyd=float(gains[2] * sev + c[2] * dl + sd * s[2]),
        load=float(load + params.load_obs_sd * streams.load_noise[day]),
    )


def alert_cause(params: PlantParams, streams: ObsStreams, day: int, sev: float, load: float) -> str:
    """Truth label of an alert: which term contributes most to the health index on ``day``.

    Sim-only. Ties go to ``degradation``, then ``sensor_fault``.
    """
    deg = (params.hi_fail - params.hi_base) * sev
    sens = float(streams.drift_frac[day]) * params.hi_fail
    proc = max(params.hi_load_coef * (load - 1.0), 0.0)
    best, name = deg, "degradation"
    if sens > best:
        best, name = sens, "sensor_fault"
    if proc > best:
        name = "process_upset"
    return name


def blind_until_day(life_start_day: float, params: PlantParams) -> float:
    """First day the estimator is not blind after a life that began on ``life_start_day``."""
    return life_start_day + params.blind_days


def is_blind(day: float, life_start_day: float, params: PlantParams) -> bool:
    return day < blind_until_day(life_start_day, params)


@dataclass(frozen=True)
class LifeTable:
    """First lives of the four assets in the world where nobody acts (the no-intervention twin).

    Arrays are indexed by day ``0 .. n_days - 1``: the reading at the start of the day.
    ``failure_day`` may lie beyond the horizon. SYNTHETIC.
    """

    seed: int
    n_days: int
    lives: "MappingProxyType[str, LifeDraw]"
    onset_day: "MappingProxyType[str, float]"  # calendar day the clock reaches the onset
    failure_day: "MappingProxyType[str, float]"  # calendar day of failure with no intervention
    hi: "MappingProxyType[str, NDArray[np.float64]]"  # NaN where masked
    sig: "MappingProxyType[str, NDArray[np.float64]]"  # (3, n_days): bearing, proc, hyd
    load: "MappingProxyType[str, NDArray[np.float64]]"  # observed load
    load_true: "MappingProxyType[str, NDArray[np.float64]]"  # effective load, sim-only
    severity: "MappingProxyType[str, NDArray[np.float64]]"  # sim-only
    true_rul_days: "MappingProxyType[str, NDArray[np.float64]]"  # realised, sim-only

    def fails_in_horizon(self, asset_id: str) -> bool:
        return self.failure_day[asset_id] < self.n_days


def simulate_lives(params: PlantParams, seed: int, scenario: Scenario) -> LifeTable:
    """Per-asset first lives: onset, failure, shape, mode, daily readings and the true RUL."""
    params.validate()
    resolved = scenario.resolve(seed, params)
    phys = build_physical(params, seed, resolved)
    n = params.horizon_days
    lives, onset, fail = {}, {}, {}
    hi, sig, load, load_true, sev, rul = {}, {}, {}, {}, {}, {}
    for aid in ASSET_IDS:
        life = first_life(phys, aid)
        streams = build_obs_streams(params, seed, aid, phys)
        cum = phys.cum[aid]
        lives[aid] = life
        onset[aid] = float(np.searchsorted(cum, life.tau_onset, side="left")) / DAY_Q
        fail_q = int(np.searchsorted(cum, life.tau_fail, side="left"))
        fail[aid] = fail_q / DAY_Q
        h = np.empty(n)
        sg = np.empty((3, n))
        lo = np.empty(n)
        s_arr = np.empty(n)
        r_arr = np.empty(n)
        for d in range(n):
            q = d * DAY_Q
            s = severity(life, float(cum[q]))
            obs = observe_day(params, streams, d, s, life.signature, float(phys.load_day[aid][d]))
            h[d] = np.nan if obs.hi is None else obs.hi
            sg[:, d] = (obs.sig_bearing, obs.sig_proc, obs.sig_hyd)
            lo[d] = obs.load
            s_arr[d] = s
            r_arr[d] = max(min(fail_q, len(cum) - 1) - q, 0) / DAY_Q
        hi[aid], sig[aid], load[aid], sev[aid], rul[aid] = h, sg, lo, s_arr, r_arr
        load_true[aid] = phys.load_day[aid][:n].copy()
    return LifeTable(
        seed=seed,
        n_days=n,
        lives=MappingProxyType(lives),
        onset_day=MappingProxyType(onset),
        failure_day=MappingProxyType(fail),
        hi=MappingProxyType(hi),
        sig=MappingProxyType(sig),
        load=MappingProxyType(load),
        load_true=MappingProxyType(load_true),
        severity=MappingProxyType(sev),
        true_rul_days=MappingProxyType(rul),
    )
