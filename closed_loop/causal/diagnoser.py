"""Diagnosis in the loop (docs/adr/0003 A3): what is behind an alert, and which mode is failing.

``Diagnoser.diagnose(asset_id, day, history)`` returns a ``Diagnosis``: the failure ``mode`` and the
``cause`` of the alert, ``degradation``, ``sensor_fault`` or ``process_upset``, with a posterior
over the joint labels ``"<cause>:<mode>"`` (it sums to one over the modes the asset's class can
have). The daily observation is ``y_d = (hi, sig_bearing, sig_proc, sig_hyd)`` and the load is
observed. The plant's consequence (0003 A3): a ``RAISE_WO`` with the wrong mode completes a job and
does not renew the failing component; a diagnosis of ``sensor_fault`` or ``process_upset`` raises
no job.

Three kinds, ``make_diagnoser(kind)``:

``"scm"``
    **Diagnosis by inverting a structural causal model.** The graph is handed to it and stated here:
    the mode and the cause are exogenous; severity ``z`` (in health-index units) is caused by
    degradation; a sensor gain error ``g`` adds to the health index only; the load enters the health
    index and every signature channel; a bearing mode loads ``sig_bearing`` and a wear mode loads
    ``sig_proc`` and ``sig_hyd``. In equations, with ``dl`` the observed load less one::

        hi    = h0 + c_h dl + z + g + e_h
        sig_i = b_i + c_i dl + a_{family,i} z + e_i

    Which numbers fill the equations is **estimated from the training lives**, not read from the
    plant: the baseline and load coefficients and the noise levels by regression on the days before
    any wear; the signature gains of each mode family by regression of the signatures on the
    load-corrected health index; the cause frequencies and the distributions of ``z`` and ``g``
    given the cause from the labelled alerts; the mode frequencies of each asset class from the
    labelled lives; the load excursion ``dl`` is one more effect of the cause (an upset is what
    makes it large), so each cause has its own distribution of it. The failure mode is **not** made
    to depend on the load: the graph has no such edge. In training data where heavy load and wear
    modes go together (``PlantParams.mode_load_tilt``) the SCM therefore does not learn that
    association; it gives the mode its marginal frequency and lets the signatures decide. The
    posterior over cause and mode is Bayes' rule on a grid of ``(z, g)``,
    using the last ``window`` days for the mode (the mode is one fact about the life) and the day
    asked about for the cause.
``"corr"``
    A multinomial logistic regression of the joint label on the same observations, with no
    structure: today's reading, window means of the health index and signatures, the load today, in
    the window and over the whole history, and the asset class. It learns whatever predicts the
    labels in the training lives, including the load-to-mode association.
``"oracle"``
    Returns the truth. It needs a lookup bound with ``bind``; it exists to bound what a perfect
    diagnosis is worth (arm P1-oracle) and is meaningless outside the simulator.

Honest reading of the comparison (C-DIAG): the SCM is handed the graph, not the coefficients, and
the graph encodes the one thing that matters under a shift in how the mode depends on the load. The
correlational model has no way to know which of its associations will survive. Where the training
association still holds, the correlational model may win; ``mdiag.py`` measures both settings.

Interfaces: the frozen text of 0003 A3 (``Diagnosis``, ``Diagnoser``, ``make_diagnoser``);
``ObservationHistory`` is the type the text leaves open (``records.py``); ``bind``, ``window`` and
the fitted attributes are additive. Everything is SYNTHETIC.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from closed_loop.causal.correlational import SoftmaxRegression
from closed_loop.causal.records import CAUSES, LifeRecord, ObservationHistory
from closed_loop.plant.params import ASSET_CLASS, MODES_BY_CLASS

FAMILIES = ("bearing", "wear")  # the signature families (plant catalogue: ModeSpec.signature)
_Z = np.arange(0.0, 9.51, 0.1)  # severity in health-index units
_G = np.arange(0.0, 6.01, 0.1)  # sensor gain error in health-index units
_BASELINE_DAYS = 20  # a new component's first days carry no wear (S: 21 d baseline)
_MIN_SD = 1e-6


@dataclass(frozen=True)
class Diagnosis:
    asset_id: str
    day: int
    mode: str
    cause: str  # one of CAUSES
    posterior: Mapping[str, float]  # "<cause>:<mode>" -> probability


class Diagnoser(Protocol):
    name: str  # "scm" | "corr" | "oracle"

    def fit(self, lives: Sequence[LifeRecord]) -> None: ...

    def diagnose(self, asset_id: str, day: int, history: ObservationHistory) -> Diagnosis: ...


def make_diagnoser(kind: str) -> Diagnoser:
    """``"scm"``, ``"corr"`` or ``"oracle"`` (not yet fitted or bound)."""
    if kind == "scm":
        return ScmDiagnoser()
    if kind == "corr":
        return CorrDiagnoser()
    if kind == "oracle":
        return OracleDiagnoser()
    raise ValueError(f"unknown diagnoser {kind!r}; expected scm, corr or oracle")


# ------------------------------------------------------------------------------ catalogue facts
def class_modes(asset_id: str) -> tuple[tuple[str, str], ...]:
    """The ``(mode, family)`` pairs the asset's class can have (asset register, not truth)."""
    return tuple((m.mode, m.signature) for m in MODES_BY_CLASS[ASSET_CLASS[asset_id]])


def labels_for(asset_id: str) -> tuple[str, ...]:
    return tuple(f"{c}:{m}" for c in CAUSES for m, _ in class_modes(asset_id))


def _npdf(x: NDArray | float, mu: NDArray | float, sd: float) -> NDArray:
    z = (np.asarray(x) - mu) / sd
    return np.exp(-0.5 * z * z) / (sd * math.sqrt(2.0 * math.pi))


def _trim(history: ObservationHistory, day: int) -> ObservationHistory:
    n = int(np.searchsorted(history.days, day, side="right"))
    if n == 0:
        raise ValueError("the history holds no reading on or before the day asked about")
    if n == history.n:
        return history
    return ObservationHistory(
        history.asset_id, history.days[:n], history.y[:n], history.load[:n], history.life_start_day
    )


def _make(asset_id: str, day: int, post: NDArray[np.float64], lab: Sequence[str]) -> Diagnosis:
    i = int(np.argmax(post))
    cause, mode = lab[i].split(":")
    return Diagnosis(
        asset_id, day, mode, cause, {k: float(p) for k, p in zip(lab, post, strict=True)}
    )


# ------------------------------------------------------------------------------------ the SCM
class ScmDiagnoser:
    """Diagnosis by Bayes' rule on a structural model fitted to the training lives (see above)."""

    name = "scm"

    def __init__(self, *, window: int = 5, bandwidth: float = 0.25) -> None:
        if window < 1:
            raise ValueError("window must be at least 1")
        self.window = window
        self.bandwidth = bandwidth
        self.fitted = False

    # -------------------------------------------------------------------------- fitting
    def fit(self, lives: Sequence[LifeRecord]) -> None:
        usable = [lf for lf in lives if lf.days.size and lf.mode is not None]
        if not usable:
            raise ValueError("fit needs lives with daily readings and a labelled mode")
        self._fit_baseline(usable)
        self._fit_gains(usable)
        self._fit_priors(usable)
        self.fitted = True

    def _fit_baseline(self, lives: Sequence[LifeRecord]) -> None:
        hi_x, hi_y, sig_x, sig_y = [], [], [], []
        for lf in lives:
            early = lf.days < lf.life_start_day + _BASELINE_DAYS
            dl = lf.load[early] - 1.0
            sig_x.append(dl)
            sig_y.append(lf.sig[:, early])
            m = (lf.hi_days >= lf.life_start_day) & (
                lf.hi_days < lf.life_start_day + _BASELINE_DAYS
            )
            idx = (lf.hi_days[m] - lf.days[0]).astype(int)
            hi_x.append(lf.load[idx] - 1.0)
            hi_y.append(lf.hi[m])
        x, y = np.concatenate(hi_x), np.concatenate(hi_y)
        self.h0, self.c_h, self.sd_h = _line(x, y)
        xs, ys = np.concatenate(sig_x), np.concatenate(sig_y, axis=1)
        fits = [_line(xs, ys[i]) for i in range(3)]
        self.sig_b = np.array([f[0] for f in fits])
        self.sig_c = np.array([f[1] for f in fits])
        self.sig_sd = np.array([f[2] for f in fits])

    def _fit_gains(self, lives: Sequence[LifeRecord]) -> None:
        """Signature gain per unit of degradation, by family (signatures on the corrected index)."""
        self.gains: dict[str, NDArray[np.float64]] = {}
        clean = [lf for lf in lives if not any(a.cause == "sensor_fault" for a in lf.alerts)]
        for fam in FAMILIES:
            zs, rs = [], []
            for lf in clean:
                if _family_of(lf) != fam:
                    continue
                idx = (lf.hi_days - lf.days[0]).astype(int)
                z = lf.hi - self.h0 - self.c_h * (lf.load[idx] - 1.0)
                r = (
                    lf.sig[:, idx]
                    - self.sig_b[:, None]
                    - self.sig_c[:, None] * (lf.load[idx] - 1.0)
                )
                keep = z >= 1.0
                zs.append(z[keep])
                rs.append(r[:, keep])
            z_all = np.concatenate(zs) if zs else np.zeros(0)
            if z_all.size < 5:
                raise ValueError(f"too few degraded days of the {fam} family to estimate its gains")
            r_all = np.concatenate(rs, axis=1)
            self.gains[fam] = (r_all @ z_all) / float(z_all @ z_all)

    def _latents_at_alert(self, lf: LifeRecord, day: int, fam: str) -> tuple[float, float, float]:
        """Load-corrected health-index excess, severity from the signatures, and the sensor gap."""
        i = int(np.searchsorted(lf.days, day))
        j = int(np.searchsorted(lf.hi_days, day))
        dl = lf.load[i] - 1.0
        excess = float(lf.hi[j] - self.h0 - self.c_h * dl)
        a = self.gains[fam]
        r = lf.sig[:, i] - self.sig_b - self.sig_c * dl
        z_sig = float(max(a @ r / float(a @ a), 0.0))
        return excess, z_sig, max(excess - z_sig, 0.0)

    def _fit_priors(self, lives: Sequence[LifeRecord]) -> None:
        zs: dict[str, list[float]] = {c: [] for c in CAUSES}
        dls: dict[str, list[float]] = {c: [] for c in CAUSES}
        gs: list[float] = []
        counts = dict.fromkeys(CAUSES, 0)
        for lf in lives:
            fam = _family_of(lf)
            for a in lf.alerts:
                counts[a.cause] += 1
                dls[a.cause].append(float(lf.load[int(np.searchsorted(lf.days, a.day))] - 1.0))
                excess, z_sig, gap = self._latents_at_alert(lf, a.day, fam)
                if a.cause == "degradation":
                    zs[a.cause].append(max(excess, 0.0))
                else:
                    zs[a.cause].append(z_sig)
                if a.cause == "sensor_fault":
                    gs.append(gap)
        total = sum(counts.values()) + len(CAUSES)
        self.pi = np.array([(counts[c] + 1.0) / total for c in CAUSES])
        self.pz = {c: _smooth(zs[c], _Z, self.bandwidth, np.exp(-_Z)) for c in CAUSES}
        self.pg = _smooth(gs, _G, self.bandwidth, np.exp(-_G))
        # the load excursion given the cause; a cause with few alerts borrows the degradation one
        base = dls["degradation"] if len(dls["degradation"]) >= 5 else [0.0, 0.1]
        self.dl_mu: dict[str, float] = {}
        self.dl_sd: dict[str, float] = {}
        for c in CAUSES:
            v = np.asarray(dls[c] if len(dls[c]) >= 5 else base, dtype=float)
            self.dl_mu[c], self.dl_sd[c] = float(v.mean()), max(float(v.std()), 0.02)
        self.mode_prior: dict[str, dict[str, float]] = {}
        for cls in sorted(set(ASSET_CLASS.values())):
            modes = [m.mode for m in MODES_BY_CLASS[cls]]
            n = {m: 1.0 for m in modes}  # Laplace
            for lf in lives:
                if ASSET_CLASS[lf.asset_id] == cls and lf.mode in n:
                    n[lf.mode] += 1.0
            tot = sum(n.values())
            self.mode_prior[cls] = {m: v / tot for m, v in n.items()}
        self.n_alerts = dict(counts)

    # ------------------------------------------------------------------------ inference
    def _day_likelihood(self, y: NDArray[np.float64], load: float, family: str) -> dict[str, float]:
        """``p(y_day | cause, family)`` with severity and sensor gap integrated out."""
        dl = load - 1.0
        a = self.gains[family]
        sig = np.ones_like(_Z)
        for i in range(3):
            sig = sig * _npdf(
                y[1 + i], self.sig_b[i] + self.sig_c[i] * dl + a[i] * _Z, self.sig_sd[i]
            )
        out: dict[str, float] = {}
        load_like = {c: float(_npdf(dl, self.dl_mu[c], self.dl_sd[c])) for c in CAUSES}
        if np.isnan(y[0]):  # the gate masked the health index: only the signatures speak
            for c in CAUSES:
                out[c] = load_like[c] * float(self.pz[c] @ sig)
            return out
        excess = y[0] - self.h0 - self.c_h * dl
        h_direct = _npdf(excess - _Z, 0.0, self.sd_h)
        for c in ("degradation", "process_upset"):
            out[c] = load_like[c] * float((self.pz[c] * sig) @ h_direct)
        h_joint = _npdf(excess - _Z[:, None] - _G[None, :], 0.0, self.sd_h)
        out["sensor_fault"] = load_like["sensor_fault"] * float(
            (self.pz["sensor_fault"] * sig) @ (h_joint @ self.pg)
        )
        return out

    def diagnose(self, asset_id: str, day: int, history: ObservationHistory) -> Diagnosis:
        if not self.fitted:
            raise RuntimeError("call fit() first")
        h = _trim(history, day)
        cls = ASSET_CLASS[asset_id]
        modes = class_modes(asset_id)
        k = h.n
        first = max(k - self.window, 0)
        mode_ev = {m: self.mode_prior[cls][m] for m, _ in modes}
        for j in range(first, k - 1):  # earlier days speak for the mode only
            for m, fam in modes:
                like = self._day_likelihood(h.y[j], float(h.load[j]), fam)
                mode_ev[m] *= float(sum(self.pi[i] * like[c] for i, c in enumerate(CAUSES)))
        post = np.zeros((len(CAUSES), len(modes)))
        for jm, (m, fam) in enumerate(modes):
            like = self._day_likelihood(h.y[k - 1], float(h.load[k - 1]), fam)
            for ic, c in enumerate(CAUSES):
                post[ic, jm] = self.pi[ic] * mode_ev[m] * like[c]
        z = post.sum()
        post = post / z if z > 0.0 and np.isfinite(z) else np.full(post.shape, 1.0 / post.size)
        lab = [f"{c}:{m}" for c in CAUSES for m, _ in modes]
        return _make(asset_id, int(h.days[-1]), post.ravel(), lab)


def _family_of(lf: LifeRecord) -> str:
    for m in MODES_BY_CLASS[ASSET_CLASS[lf.asset_id]]:
        if m.mode == lf.mode:
            return m.signature
    raise ValueError(f"{lf.mode!r} is not a mode of {ASSET_CLASS[lf.asset_id]}")


def _line(x: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[float, float, float]:
    """Intercept, slope and residual sd of ``y ~ x``, with one trimmed refit against outliers."""
    X = np.column_stack([np.ones(x.size), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ coef
    sd = float(res.std())
    if sd > _MIN_SD:
        keep = np.abs(res) <= 3.0 * sd
        if keep.sum() >= 3:
            coef, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
            res = y[keep] - X[keep] @ coef
    spread = float(res.std(ddof=2)) if res.size > 2 else 0.0
    return float(coef[0]), float(coef[1]), max(spread, _MIN_SD)


def _smooth(
    samples: Sequence[float], grid: NDArray[np.float64], bw: float, fallback: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Kernel density of ``samples`` as probability masses on ``grid`` (or the fallback)."""
    if len(samples) == 0:
        p = fallback.astype(float)
    else:
        p = _npdf(grid[:, None], np.asarray(samples, dtype=float)[None, :], bw).sum(axis=1)
    p = p + 1e-6 * p.max()
    return p / p.sum()


# --------------------------------------------------------------------- the correlational model
def _features(h: ObservationHistory, window: int) -> NDArray[np.float64]:
    """Today's reading, window means, and the load today, in the window and over the history."""
    w = h.window(window)
    cls = ASSET_CLASS[h.asset_id]
    onehot = [float(cls == c) for c in ("CRUSHER", "BELT_CONVEYOR", "SLURRY_PUMP")]
    hi_today = h.y[-1, 0] if not np.isnan(h.y[-1, 0]) else float(np.nanmean(w.y[:, 0]))
    hi_win = float(np.nanmean(w.y[:, 0])) if np.any(~np.isnan(w.y[:, 0])) else hi_today
    return np.array(
        [
            hi_today,
            *h.y[-1, 1:4],
            float(h.load[-1]),
            hi_win,
            *w.y[:, 1:4].mean(axis=0),
            float(w.load.mean()),
            float(h.load.mean()),
            *onehot,
        ]
    )


class CorrDiagnoser:
    """Multinomial logistic regression of the joint label on the observations (see above)."""

    name = "corr"

    def __init__(self, *, window: int = 5, l2: float = 1.0) -> None:
        self.window = window
        self.l2 = l2
        self.fitted = False
        self._labels = [(c, f) for c in CAUSES for f in FAMILIES]

    def fit(self, lives: Sequence[LifeRecord]) -> None:
        X, y = [], []
        for lf in lives:
            if not lf.days.size or lf.mode is None:
                continue
            fam = _family_of(lf)
            for a in lf.alerts:
                X.append(_features(lf.history_at(a.day), self.window))
                y.append(self._labels.index((a.cause, fam)))
        if len(X) < 10:
            raise ValueError("too few labelled alerts to fit the correlational diagnoser")
        self.model = SoftmaxRegression(len(self._labels), l2=self.l2).fit(np.array(X), np.array(y))
        self.n_train = len(X)
        self.fitted = True

    def diagnose(self, asset_id: str, day: int, history: ObservationHistory) -> Diagnosis:
        if not self.fitted:
            raise RuntimeError("call fit() first")
        h = _trim(history, day)
        p = self.model.predict_proba(_features(h, self.window))[0]
        modes = class_modes(asset_id)
        post = np.zeros(len(CAUSES) * len(modes))
        lab = []
        for ic, c in enumerate(CAUSES):
            for jm, (m, fam) in enumerate(modes):
                post[ic * len(modes) + jm] = p[self._labels.index((c, fam))]
                lab.append(f"{c}:{m}")
        z = post.sum()
        post = post / z if z > 0.0 else np.full(post.size, 1.0 / post.size)
        return _make(asset_id, int(h.days[-1]), post, lab)


# ------------------------------------------------------------------------------------ the oracle
Lookup = Callable[[str, int], tuple[str, str]]


class OracleDiagnoser:
    """Returns the truth from a bound lookup ``(asset_id, day) -> (mode, cause)``."""

    name = "oracle"

    def __init__(self) -> None:
        self._lookup: Lookup | None = None

    def bind(self, lookup: Lookup) -> None:
        self._lookup = lookup

    def fit(self, lives: Sequence[LifeRecord]) -> None:
        return None

    def diagnose(self, asset_id: str, day: int, history: ObservationHistory) -> Diagnosis:
        if self._lookup is None:
            raise RuntimeError("bind() a truth lookup first: the oracle has no other source")
        mode, cause = self._lookup(asset_id, day)
        lab = labels_for(asset_id)
        post = np.zeros(len(lab))
        post[lab.index(f"{cause}:{mode}")] = 1.0
        return _make(asset_id, day, post, lab)


class AlertTruthTable:
    """A lookup for the oracle built from alert labels: the latest alert on or before the day."""

    def __init__(self, labels: Mapping[str, Sequence[tuple[int, str, str]]] | None = None) -> None:
        self._by_asset: dict[str, list[tuple[int, str, str]]] = {
            k: sorted(v) for k, v in (labels or {}).items()
        }

    def add(self, asset_id: str, day: int, mode: str, cause: str) -> None:
        self._by_asset.setdefault(asset_id, []).append((day, mode, cause))
        self._by_asset[asset_id].sort()

    def __call__(self, asset_id: str, day: int) -> tuple[str, str]:
        rows = [r for r in self._by_asset.get(asset_id, []) if r[0] <= day]
        if not rows:
            raise KeyError(f"no alert for {asset_id} on or before day {day}")
        _, mode, cause = rows[-1]
        return mode, cause
