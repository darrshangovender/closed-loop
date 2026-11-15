"""Records of observed lives and daily histories (packet P7). Everything here is SYNTHETIC.

``LifeRecord`` is the frozen text of docs/adr/0002 section 3.9 plus defaulted fields, added because
the three-channel signature, the load and the post-mortem labels of 0003 A3 have to live somewhere:

``days``, ``sig``, ``load``
    every day the asset was read (``days``), the three signature channels (shape ``(3, n)``, in the
    order ``sig_bearing, sig_proc, sig_hyd``) and the observed load. ``hi_days`` and ``hi`` hold
    only the days the health index was not masked by the data-quality gate.
``derate_start``
    the exact time, in days, at which the DERATE command took effect (``None`` if never). The days
    in ``derate_days`` are those whose start-of-day reading shows the asset derated, so they begin
    on the day after the one it happened on; the clock of that day ran derated for the part of
    it after ``derate_start``.
``mode`` and ``alerts``
    the failure mode of the component and, for every alert the plant raised on it, what the
    post-mortem found (``AlertLabel``). A real plant has these from teardown reports and closed
    work orders; the diagnosers train on them.

``ObservationHistory`` is what ``Diagnoser.diagnose`` is given (0003 A3): the daily readings of one
asset since its current life began, up to and including the day asked about. A controller builds it
with ``HistoryRecorder`` from the ``Observation`` stream, and nothing in it is truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from closed_loop.plant.plant import Observation

CAUSES = ("degradation", "sensor_fault", "process_upset")
N_CHANNELS = 4  # y_d = (hi, sig_bearing, sig_proc, sig_hyd)


def _frozen(a: NDArray) -> NDArray:
    out: NDArray = np.array(a, dtype=np.float64)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class AlertLabel:
    """What the post-mortem found behind one alert (sim truth, as a teardown report would give)."""

    alert_id: str
    day: int
    level: int  # 1 for the first alert of a life, 2 for the second
    cause: str  # one of CAUSES
    mode: str  # the failure mode of the component


@dataclass(frozen=True, eq=False)
class LifeRecord:
    """One component life as the plant's records show it. Arrays are read-only copies."""

    asset_id: str
    hi_days: NDArray[np.float64]
    hi: NDArray[np.float64]
    derate_days: NDArray[np.float64]
    fail_day: float | None
    censored_day: float | None
    # ---- additive, defaulted (see the module docstring)
    days: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    sig: NDArray[np.float64] = field(default_factory=lambda: np.zeros((3, 0)))
    load: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    derate_start: float | None = None
    mode: str | None = None
    seed: int | None = None
    life_id: int = 0
    alerts: tuple[AlertLabel, ...] = ()
    life_start_day: float = 0.0

    def __post_init__(self) -> None:
        if (self.fail_day is None) == (self.censored_day is None):
            raise ValueError("exactly one of fail_day and censored_day must be set")
        for name in ("hi_days", "hi", "derate_days", "days", "load"):
            object.__setattr__(self, name, _frozen(getattr(self, name)))
        object.__setattr__(self, "sig", _frozen(self.sig))
        if self.hi_days.shape != self.hi.shape:
            raise ValueError("hi_days and hi must have the same length")
        if self.hi_days.size and np.any(np.diff(self.hi_days) <= 0):
            raise ValueError("hi_days must be strictly increasing")
        if self.days.size:
            if self.sig.shape != (3, self.days.size) or self.load.shape != self.days.shape:
                raise ValueError("days, sig and load must be aligned")
            if np.any(np.diff(self.days) <= 0):
                raise ValueError("days must be strictly increasing")
        elif self.sig.size or self.load.size:
            raise ValueError("sig and load need days")
        if np.any(~np.isfinite(self.hi)):
            raise ValueError("hi holds masked days as absent, not as NaN")

    @property
    def end_day(self) -> float:
        """The day the record ends: the failure, or the day observation stopped."""
        end = self.fail_day if self.fail_day is not None else self.censored_day
        assert end is not None  # one of the two is set (checked on construction)
        return float(end)

    @property
    def failed(self) -> bool:
        return self.fail_day is not None

    def first_alert_day(self, hi_alert: float = 2.0, blind_days: float = 23.0) -> int | None:
        """First day after the blind window on which the health index reached ``hi_alert``."""
        ok = (self.hi_days >= self.life_start_day + blind_days) & (self.hi >= hi_alert)
        return int(self.hi_days[np.argmax(ok)]) if ok.any() else None

    def history_at(self, day: int) -> ObservationHistory:
        """The history a controller would hold on ``day``: readings of days up to ``day``."""
        if not self.days.size:
            raise ValueError("this record has no daily signature and load arrays")
        n = int(np.searchsorted(self.days, day, side="right"))
        hi = np.full(n, np.nan)
        pos = np.searchsorted(self.days[:n], self.hi_days[self.hi_days <= day])
        hi[pos] = self.hi[self.hi_days <= day]
        y = np.column_stack([hi, self.sig[:, :n].T])
        return ObservationHistory(
            self.asset_id,
            self.days[:n].astype(np.int64),
            y,
            self.load[:n],
            int(self.life_start_day),
        )


@dataclass(frozen=True, eq=False)
class ObservationHistory:
    """The readings of one asset since its life began, oldest first. No truth field."""

    asset_id: str
    days: NDArray[np.int64]
    y: NDArray[np.float64]  # (n, 4): hi (NaN on a masked day), sig_bearing, sig_proc, sig_hyd
    load: NDArray[np.float64]  # (n,) observed load
    life_start_day: int = 0

    def __post_init__(self) -> None:
        n = self.days.size
        if self.y.shape != (n, N_CHANNELS) or self.load.shape != (n,):
            raise ValueError(f"y must be (n, {N_CHANNELS}) and load (n,) for n = {n}")
        if n and np.any(np.diff(self.days) <= 0):
            raise ValueError("days must be strictly increasing")
        for arr in (self.days, self.y, self.load):
            arr.setflags(write=False)

    @property
    def n(self) -> int:
        return int(self.days.size)

    def window(self, k: int) -> ObservationHistory:
        """The last ``k`` readings."""
        return ObservationHistory(
            self.asset_id, self.days[-k:], self.y[-k:], self.load[-k:], self.life_start_day
        )


class HistoryRecorder:
    """Builds ``ObservationHistory`` objects from the stream of plant ``Observation`` objects.

    ``record(obs)`` appends, for every asset, the reading of a new day (``AssetView.last_obs``);
    the same day twice is ignored. ``reset_life`` starts a fresh history for an asset, as a
    controller does when it has renewed a component.
    """

    def __init__(self) -> None:
        self._days: dict[str, list[int]] = {}
        self._y: dict[str, list[tuple[float, float, float, float]]] = {}
        self._load: dict[str, list[float]] = {}
        self._start: dict[str, int] = {}

    def record(self, obs: Observation) -> None:
        for aid, view in obs.assets.items():
            last = view.last_obs
            if last is None:
                continue
            days = self._days.setdefault(aid, [])
            if days and last.day <= days[-1]:
                continue
            days.append(last.day)
            self._y.setdefault(aid, []).append(last.y)
            self._load.setdefault(aid, []).append(last.load)

    def reset_life(self, asset_id: str, day: int) -> None:
        self._days[asset_id] = []
        self._y[asset_id] = []
        self._load[asset_id] = []
        self._start[asset_id] = day

    def history(self, asset_id: str) -> ObservationHistory:
        days = self._days.get(asset_id, [])
        return ObservationHistory(
            asset_id,
            np.array(days, dtype=np.int64),
            np.array(self._y.get(asset_id, []), dtype=np.float64).reshape(len(days), N_CHANNELS),
            np.array(self._load.get(asset_id, []), dtype=np.float64),
            self._start.get(asset_id, 0),
        )
