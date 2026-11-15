"""The seven conformal interval providers V1 to V7 (packet P4b). Everything here is SYNTHETIC.

Interface text: docs/adr/0002-interfaces.md, section 3.6; variants: docs/adr/0001-architecture.md,
section 6.3. All seven share the estimator; they differ in how the quantile of the nonconformity
score ``r = (L - m) / sigma`` is calibrated and updated:

====  ===============  ==========================================================================
V1    split            quantile of the development scores, never updated
V2    ACI-naive        ACI on the resolved labels, a censored label counted as a hit
V3    ACI-inf          ACI on ``HIT``, ``MISS`` and ``LOWER_HIT`` labels; ``UNKNOWN`` is skipped
V4    ACI-aud          ACI on the labels of audit lives only (a life is an audit life w.p. rho)
V5    conformal PID    quantile tracker plus integrator on the V3 label stream
V6    per-life band    max-score band over development lives (``alpha`` is ``alpha_life``)
V7    oracle ACI       ACI on the true twin-world labels (sim only, not deployable)
====  ===============  ==========================================================================

The interval at a checkpoint is ``[m + sigma Q_a(r), m + sigma Q_{1-a}(r)]`` with ``a = alpha / 2``
on each side, ``Q`` the finite-sample conformal quantile of a pool of scores (``+-inf`` when the
pool is too small for the level). Online variants (V2 to V5, V7) keep their pool as a rolling window
of 500 scores, started from a random draw of the development scores; only labels from an *observed
failure* add a score, because a censored life gives a bound on the score and not its value. The
tracker (``alpha_t`` for ACI, an offset for PID) is updated from each label's miss indicator. A
censored life never adds a score, and nothing is special-cased for it: the label statuses of
``perception.labels`` decide what a label is worth, and a policy that replaces before the lower
bound leaves ``UNKNOWN`` labels that V3 to V5 skip and V2 counts as hits.

Life-unit forms (a follow-up, not part of the registered set): ``V2L``, ``V3L`` and ``V7L``
apply V2's, V3's and V7's label rules with one update per resolved life, ``gamma`` unchanged at
0.02, using that life's mean miss indicator. The per-label forms step ``alpha_t`` once per label,
about 40 times for a life whose labels resolve together and are strongly dependent; the life forms
answer whether that is what hurts them. ``V2``, ``V3`` and ``V7`` (``unit="label"``) are untouched.

Prior-status checkpoints (blind window, not degrading, insufficient trend) carry a class-prior
``(m, sigma)``. They have their own pool of development scores and their own tracker, so a shift in
the trend scores does not move the prior intervals and the other way round. V6 takes the same two
classes (0003 A18): four per-life maxima, each at tail ``alpha_life / 4``, whose union bound is the
per-life guarantee. It pooled them before, and one pooled band was as wide as the worst estimated
score, which no prior output could vouch for.

Time order. ``observe`` is called at resolution events and the labels must arrive in non-decreasing
``resolved_day``; an out-of-order label raises ``ValueError``. Providers other than V7 never read
``Label.true_rul``; a test feeds identical streams with and without it.

Calibration data come from ``development_data`` and only from the ``dev`` and ``calib`` seed roles
(0001 section 11); a held-out seed raises.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from closed_loop.audit.seeds import CALIB_SEEDS, DEV_SEEDS
from closed_loop.core.rng import rng_for
from closed_loop.perception.coverage import OpenLoopLife, open_loop_lives
from closed_loop.perception.estimator import (
    STATUS_ESTIMATED,
    EstimatorOutput,
    StreamingEstimator,
)
from closed_loop.perception.labels import (
    Checkpoint,
    Label,
    LabelStatus,
    RulInterval,
    learner_view,
    resolve_labels,
)
from closed_loop.plant.params import PlantParams
from closed_loop.plant.scenario import Scenario

__all__ = [
    "ALPHA_LIFE",
    "AUDIT_RHO",
    "GAMMA",
    "ALL_VARIANTS",
    "LIFE_VARIANTS",
    "VARIANTS",
    "WINDOW",
    "DevelopmentData",
    "IntervalProvider",
    "RulInterval",
    "conformal_quantiles",
    "development_data",
    "is_audit_life",
    "make_provider",
    "replay_open_loop",
    "require_calibration_seeds",
    "score",
]

GAMMA = 0.02  # ACI step (0001 section 6.3)
WINDOW = 500  # scores in the rolling calibration pool
AUDIT_RHO = 0.10  # probability a life is an audit life (V4)
ALPHA_LIFE = 0.05  # miss level of the per-life band (V6)
VARIANTS = ("V1", "V2", "V3", "V4", "V5", "V6", "V7")
# Life-unit forms of V2, V3 and V7 (a follow-up to the registered set, which stays as it is).
LIFE_VARIANTS = ("V2L", "V3L", "V7L")
ALL_VARIANTS = (*VARIANTS, *LIFE_VARIANTS)
_EPS = 1e-9  # keeps ceil and floor of ranks like (n + 1)(1 - a) from tipping on a rounding error

# Conformal PID defaults (Angelopoulos, Candes and Tibshirani 2023, simplified: no scorecaster and
# a tanh instead of their tan saturation). Fixed here, not tuned: the tracker step is a fraction of
# the spread of the development scores, the integrator saturates after about `c_sat` net misses.
ETA_REL = 0.05
K_I = 0.1
C_SAT = 10.0


class IntervalProvider(Protocol):
    name: str  # "V1".."V7"

    def predict(self, est: EstimatorOutput, level: float) -> RulInterval: ...

    def observe(self, labels: Sequence[Label]) -> None: ...  # at resolution events, in time order

    def state(self) -> Mapping[str, float]: ...  # alpha_t, window size, ...: logged


# ----------------------------------------------------------------------------- quantiles


def score(rul: float, m: float, sigma: float) -> float:
    """The nonconformity score ``(L - m) / sigma``."""
    return (rul - m) / sigma


def conformal_quantiles(sorted_scores: NDArray[np.float64], a: float) -> tuple[float, float]:
    """Finite-sample conformal ``(lower, upper)`` quantiles at tail mass ``a`` on each side.

    With ``n`` sorted scores the upper quantile is the ``ceil((n + 1)(1 - a))``-th smallest and the
    lower the ``floor((n + 1) a)``-th, ``+inf`` / ``-inf`` when the rank falls outside the pool, so
    each tail of an exchangeable new score is missed with probability at most ``a``. ``a <= 0``
    gives ``(-inf, inf)``; ``a`` above one half is treated as one half.
    """
    n = len(sorted_scores)
    if n == 0:
        raise ValueError("an empty pool has no quantile")
    if a <= 0.0:
        return -math.inf, math.inf
    a = min(a, 0.5)
    k_hi = math.ceil((n + 1) * (1.0 - a) - _EPS)
    k_lo = math.floor((n + 1) * a + _EPS)
    hi = math.inf if k_hi > n else float(sorted_scores[max(k_hi, 1) - 1])
    lo = -math.inf if k_lo < 1 else float(sorted_scores[min(k_lo, n) - 1])
    return lo, hi


def is_audit_life(seed: int, asset_id: str, life_id: int, rho: float = AUDIT_RHO) -> bool:
    """Whether a life is an audit life: ``rng_for(seed, "audit", asset_id, life_id) < rho``.

    Independent of everything the plant or the policy does. The asset id is part of the key (the
    interface text keys on the life id alone, but life ids count per asset, so two assets would
    share a draw).
    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [0, 1], got {rho!r}")
    return bool(rng_for(seed, "audit", asset_id, life_id).random() < rho)


# ----------------------------------------------------------------------------- one pool + tracker


def _clean(scores: Any, what: str) -> NDArray[np.float64]:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{what} must be a non-empty 1-d array of scores")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{what} holds a non-finite score")
    return arr


class _Stream:
    """A pool of scores with its tracker: one per estimator-status class."""

    def __init__(self, scores: NDArray[np.float64], window: int, rng: np.random.Generator) -> None:
        self.width = max(float(np.quantile(scores, 0.95) - np.quantile(scores, 0.05)), 1e-9)
        pick = scores
        if len(scores) > window:
            pick = scores[rng.choice(len(scores), size=window, replace=False)]
        self._window: deque[float] = deque(pick.tolist(), maxlen=window)
        self._sorted: NDArray[np.float64] | None = None
        self.alpha_t = math.nan  # set by the provider
        self.offset = 0.0
        self.integral = 0.0
        self.n_updates = 0
        self.n_skipped = 0
        self.n_scores_added = 0

    def sorted(self) -> NDArray[np.float64]:
        if self._sorted is None:
            self._sorted = np.sort(np.fromiter(self._window, dtype=np.float64))
        return self._sorted

    def add(self, value: float) -> None:
        self._window.append(value)
        self._sorted = None
        self.n_scores_added += 1

    @property
    def size(self) -> int:
        return len(self._window)


# ----------------------------------------------------------------------------- providers


class _Provider:
    name = ""

    def __init__(self, alpha: float) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        self.alpha = float(alpha)
        self.level = 1.0 - self.alpha
        self._last_resolved = -(10**9)
        self._n_labels = 0

    def predict(self, est: EstimatorOutput, level: float) -> RulInterval:
        if abs(level - self.level) > 1e-9:
            raise ValueError(
                f"{self.name} was built for level {self.level:.6g} (alpha {self.alpha:.6g}), "
                f"not {level!r}: make one provider per level"
            )
        q_lo, q_hi = self._quantiles(est.status != STATUS_ESTIMATED)
        lo = est.m_days + est.sigma_days * q_lo
        hi = est.m_days + est.sigma_days * q_hi
        if lo > hi:  # a narrowing offset that crossed: collapse onto the middle
            lo = hi = 0.5 * (lo + hi)
        return RulInterval(lo, est.m_days, hi, self.level, est.status != STATUS_ESTIMATED)

    def observe(self, labels: Sequence[Label]) -> None:
        for lab in labels:
            self._accept(lab)
            self._update(lab)

    def _accept(self, lab: Label) -> None:
        """Count a label and check that labels arrive in time order."""
        if lab.resolved_day < self._last_resolved:
            raise ValueError(
                f"label resolved on day {lab.resolved_day} arrives after day "
                f"{self._last_resolved}: online updates are in time order"
            )
        self._last_resolved = lab.resolved_day
        self._n_labels += 1

    def state(self) -> Mapping[str, float]:
        return {"alpha": self.alpha, "n_labels": float(self._n_labels)}

    def _quantiles(self, prior: bool) -> tuple[float, float]:
        raise NotImplementedError

    def _update(self, lab: Label) -> None:
        raise NotImplementedError


class _Split(_Provider):
    """V1: the split-conformal quantile of the development pool, never updated."""

    name = "V1"

    def __init__(
        self,
        *,
        alpha: float,
        development_scores: Any,
        prior_scores: Any | None,
        seed: int,
        **_: Any,
    ) -> None:
        super().__init__(alpha)
        est = _clean(development_scores, "development_scores")
        pri = est if prior_scores is None else _clean(prior_scores, "prior_scores")
        self._q = {
            False: conformal_quantiles(np.sort(est), alpha / 2.0),
            True: conformal_quantiles(np.sort(pri), alpha / 2.0),
        }
        self._sizes = (len(est), len(pri))

    def _quantiles(self, prior: bool) -> tuple[float, float]:
        return self._q[prior]

    def _update(self, lab: Label) -> None:
        pass  # never updated

    def state(self) -> Mapping[str, float]:
        return {
            **super().state(),
            "pool_estimated": float(self._sizes[0]),
            "pool_prior": float(self._sizes[1]),
        }


class _LifeBand(_Provider):
    """V6: per-life max-score band, in the two classes every other variant uses (0003 A18).

    ``development_scores`` is ``(n_lives, 4)``: for each development life the largest ``r`` and the
    largest ``-r`` over its trend-estimated checkpoints (columns 0 and 1), then the same over its
    prior-class checkpoints, status ``prior``, ``insufficient`` or ``not_degrading`` (columns 2 and
    3). A life with no checkpoint in a class holds ``-inf`` there; ``+inf`` and NaN are refused.
    The band is ``[m - q_minus sigma, m + q_plus sigma]``, taken from the columns of the class of
    the output (``est.status``), each ``q`` the conformal quantile of its column at tail
    ``alpha / 4``, **every column using all** ``n_lives``.

    Why this is a level-``alpha_life`` band. Let ``A`` be the event that a new life has some
    estimated checkpoint with ``r > q_plus_est``, ``B`` that it has one with ``-r > q_minus_est``,
    and ``C`` and ``D`` the same for the prior class. A life with no checkpoint in a class has
    ``-inf`` there and cannot make the event. The new life's four maxima are exchangeable with the
    development lives' (the same four numbers computed the same way, lives drawn alike), so each
    column's conformal quantile at tail ``a`` is exceeded by the new life's value with probability
    at most ``a`` (``conformal_quantiles``). Hence ``P(A), P(B), P(C), P(D) <= alpha / 4`` and, by
    the union bound, the probability that the new life leaves its band anywhere is at most
    ``P(A) + P(B) + P(C) + P(D) <= alpha`` when lives are exchangeable. The columns are not
    independent and need not be: the union bound does not ask for it. Pooling the two classes into
    one pair, as V6 did before, made the prior-class band as wide as the worst estimated score
    (``q_minus`` 4.67 against 2.16 for the prior class alone at tail 0.0125 on the calibration
    lives), which no prior output could vouch for.
    """

    name = "V6"

    def __init__(
        self,
        *,
        alpha: float,
        development_scores: Any,
        prior_scores: Any | None = None,
        seed: int = 0,
        **_: Any,
    ) -> None:
        super().__init__(alpha)
        arr = np.asarray(development_scores, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 4 or arr.shape[0] == 0:
            raise ValueError("V6 needs development_scores of shape (n_lives, 4)")
        if np.any(np.isnan(arr)) or np.any(np.isposinf(arr)):
            raise ValueError("development_scores holds a NaN or a +inf score (-inf is allowed)")
        tail = alpha / 4.0
        # the largest -r of a life bounds the lower side: r >= -S_minus is covered
        cols = [conformal_quantiles(np.sort(arr[:, k]), tail)[1] for k in range(4)]
        self._hi = {False: cols[0], True: cols[2]}  # keyed by "prior"
        self._lo = {False: -cols[1], True: -cols[3]}
        self._n_lives = arr.shape[0]

    def _quantiles(self, prior: bool) -> tuple[float, float]:
        return self._lo[prior], self._hi[prior]

    def _update(self, lab: Label) -> None:
        pass  # a calibration on development lives, not updated online

    def state(self) -> Mapping[str, float]:
        return {
            **super().state(),
            "n_dev_lives": float(self._n_lives),
            "q_plus_estimated": self._hi[False],
            "q_minus_estimated": -self._lo[False],
            "q_plus_prior": self._hi[True],
            "q_minus_prior": -self._lo[True],
        }


class _Online(_Provider):
    """V2 to V5 and V7: a rolling pool and a tracker driven by label miss indicators.

    ``rule`` picks which labels update and how a label counts: see the table in the module
    docstring. ACI moves ``alpha_t`` by ``gamma (alpha - err)``; PID moves a score-space offset by
    ``eta (err - alpha)`` and adds a saturating integral of ``err - alpha``.

    ``V2L``, ``V3L`` and ``V7L`` are the same rules with ``unit="life"``: the labels of one life
    (one call of ``observe``, per estimator-status class) make ONE update,
    ``alpha_t += gamma (alpha - mean err)``, where ``mean err`` is the mean of that life's miss
    indicators under the variant's label rule (labels the rule skips are left out; a life with
    none left is skipped, so V3L skips a life of ``UNKNOWN`` labels). ``gamma`` is unchanged. The
    scores of exactly observed checkpoints still enter the rolling pool one by one. A life must
    arrive in a single call; a second call naming it raises.
    """

    def __init__(
        self,
        name: str,
        *,
        alpha: float,
        development_scores: Any,
        prior_scores: Any | None,
        seed: int,
        gamma: float = GAMMA,
        window: int = WINDOW,
        eta_rel: float = ETA_REL,
        k_i: float = K_I,
        c_sat: float = C_SAT,
        **_: Any,
    ) -> None:
        super().__init__(alpha)
        if gamma <= 0.0 or window < 2 or eta_rel <= 0.0 or k_i < 0.0 or c_sat <= 0.0:
            raise ValueError(
                "gamma, eta_rel, c_sat must be positive, k_i non-negative, window >= 2"
            )
        base = name.removesuffix("L")  # V2L is V2's rule with the life as the unit of update
        self.name = name
        self.unit = "life" if base != name else "label"
        self._lives_seen: set[tuple[str, int, bool]] = set()
        self.gamma, self.window = gamma, window
        self.eta_rel, self.k_i, self.c_sat = eta_rel, k_i, c_sat
        est = _clean(development_scores, "development_scores")
        pri = est if prior_scores is None else _clean(prior_scores, "prior_scores")
        self._streams = {
            False: _Stream(est, window, rng_for(seed, "window", base, "estimated")),
            True: _Stream(pri, window, rng_for(seed, "window", base, "prior")),
        }
        for s in self._streams.values():
            s.alpha_t = self.alpha
        self.pid = base == "V5"
        self.oracle = base == "V7"
        self.audit_only = base == "V4"
        self.naive = base == "V2"

    # ---- intervals
    def _quantiles(self, prior: bool) -> tuple[float, float]:
        s = self._streams[prior]
        pool = s.sorted()
        if not self.pid:
            return conformal_quantiles(pool, s.alpha_t / 2.0)
        lo, hi = conformal_quantiles(pool, self.alpha / 2.0)
        w = s.offset + self.k_i * s.width * math.tanh(s.integral / self.c_sat)
        return lo - w, hi + w

    # ---- the miss indicator of a label under this variant (None = the label is skipped)
    def _err(self, lab: Label) -> int | None:
        if self.oracle:
            if lab.true_rul is None:
                raise ValueError("V7 is the oracle: it needs the sim-only true_rul on every label")
            return int(not (lab.lo <= lab.true_rul <= lab.hi))
        if self.audit_only and not lab.audit:
            return None
        if self.naive:
            return int(lab.status is LabelStatus.MISS and not lab.censored)
        if lab.status is LabelStatus.UNKNOWN:
            return None
        return int(lab.status is LabelStatus.MISS)

    def _score_of(self, lab: Label) -> float | None:
        """The exact score of a label, if it has one: the oracle's, or an observed failure's."""
        if lab.m is None or lab.sigma is None:
            return None
        if self.oracle:
            assert lab.true_rul is not None  # checked in _err
            return score(lab.true_rul, lab.m, lab.sigma)
        if lab.censored or lab.observed_rul is None:
            return None
        return score(lab.observed_rul, lab.m, lab.sigma)

    def _update(self, lab: Label) -> None:
        s = self._streams[lab.prior]
        err = self._err(lab)
        if err is None:
            s.n_skipped += 1
            return
        s.n_updates += 1
        if self.pid:
            s.offset += self.eta_rel * s.width * (err - self.alpha)
            s.integral += err - self.alpha
        else:
            s.alpha_t += self.gamma * (self.alpha - err)
        value = self._score_of(lab)
        if value is not None:
            s.add(value)

    def observe(self, labels: Sequence[Label]) -> None:
        if self.unit == "label":
            super().observe(labels)
            return
        groups: dict[tuple[str, int, bool], list[Label]] = {}
        for lab in labels:
            self._accept(lab)
            groups.setdefault((lab.asset_id, lab.life_id, lab.prior), []).append(lab)
        for key, group in groups.items():
            if key in self._lives_seen:
                raise ValueError(
                    f"the labels of life {key[:2]} were already observed: a life is one update"
                )
            self._lives_seen.add(key)
            self._update_life(key[2], group)

    def _update_life(self, prior: bool, group: Sequence[Label]) -> None:
        s = self._streams[prior]
        errs = [e for e in (self._err(lab) for lab in group) if e is not None]
        if not errs:
            s.n_skipped += 1
            return
        s.n_updates += 1
        s.alpha_t += self.gamma * (self.alpha - sum(errs) / len(errs))
        for lab in group:
            value = self._score_of(lab)
            if value is not None:
                s.add(value)

    def state(self) -> Mapping[str, float]:
        out: dict[str, float] = dict(super().state())
        if self.unit == "life":
            out["unit_life"] = 1.0
        for key, prior in (("est", False), ("prior", True)):
            s = self._streams[prior]
            out[f"alpha_t_{key}"] = s.alpha_t
            out[f"window_{key}"] = float(s.size)
            out[f"updates_{key}"] = float(s.n_updates)
            out[f"skipped_{key}"] = float(s.n_skipped)
            out[f"scores_added_{key}"] = float(s.n_scores_added)
            if self.pid:
                out[f"offset_{key}"] = s.offset
                out[f"integral_{key}"] = s.integral
        return out


def make_provider(
    name: str, *, alpha: float, development_scores: NDArray[np.float64], seed: int, **kw: Any
) -> IntervalProvider:
    """Build provider ``name`` (``V1`` to ``V7``, or the life-unit ``V2L``, ``V3L``, ``V7L``) at
    miss level ``alpha`` (``alpha_life`` for V6).

    ``development_scores`` is a 1-d array of scores for the trend-estimated checkpoints, or the
    ``(n_lives, 4)`` array of per-life maxima for V6 (estimated, then prior class). Optional
    keywords: ``prior_scores`` (the scores of the prior-status checkpoints; without it the prior
    intervals reuse
    ``development_scores``), ``gamma``, ``window``, ``eta_rel``, ``k_i``, ``c_sat`` (a keyword a
    variant has no use for is ignored). ``seed`` fixes the random draw of the
    initial rolling pool; the audit assignment is ``is_audit_life``.
    """
    if name not in ALL_VARIANTS:
        raise ValueError(f"unknown variant {name!r}; expected one of {ALL_VARIANTS}")
    prior_scores = kw.pop("prior_scores", None)
    common: dict[str, Any] = {
        "alpha": alpha,
        "development_scores": development_scores,
        "prior_scores": prior_scores,
        "seed": seed,
    }
    if name == "V1":
        return _Split(**common, **kw)
    if name == "V6":
        return _LifeBand(**common, **kw)
    return _Online(name, **common, **kw)


# ----------------------------------------------------------------------------- development data


@dataclass(frozen=True)
class DevelopmentData:
    """Nonconformity scores of the estimator on open-loop first lives (SYNTHETIC)."""

    scores: NDArray[np.float64]  # trend-estimated checkpoints
    prior_scores: NDArray[np.float64]  # blind, not-degrading and insufficient checkpoints
    # (n_lives, 4): max r and max -r over a life's estimated checkpoints, then over its
    # prior-class checkpoints; -inf where a life has none in the class (0003 A18)
    life_scores: NDArray[np.float64]
    seeds: tuple[int, ...]

    def for_variant(self, name: str) -> NDArray[np.float64]:
        return self.life_scores if name == "V6" else self.scores

    def provider(self, name: str, *, alpha: float, seed: int, **kw: Any) -> IntervalProvider:
        """``make_provider`` with this data: the trend scores, the prior scores, the life maxima."""
        return make_provider(
            name,
            alpha=alpha,
            development_scores=self.for_variant(name),
            seed=seed,
            prior_scores=self.prior_scores,
            **kw,
        )


def require_calibration_seeds(seeds: Sequence[int]) -> None:
    """Raise ``ValueError`` unless every seed is a ``dev`` (1000-1999) or ``calib`` (2000-2999) one.

    Anything that calibrates a band goes through here, so a search seed or a derived held-out seed
    cannot reach a quantile (0001 section 6.3: held-out lives never calibrate anything).
    """
    allowed = set(DEV_SEEDS) | set(CALIB_SEEDS)
    bad = [s for s in seeds if s not in allowed]
    if bad:
        raise ValueError(
            f"development data come from the dev (1000-1999) and calib (2000-2999) seeds only; "
            f"got {bad[:3]}"
        )


def _extremes(r: NDArray[np.float64]) -> tuple[float, float]:
    """The largest ``r`` and the largest ``-r`` of one class of a life; ``-inf`` for none."""
    if r.size == 0:
        return -math.inf, -math.inf
    return float(np.max(r)), float(np.max(-r))


def development_data(
    params: PlantParams,
    seeds: Sequence[int],
    scenario: Scenario | None = None,
    estimator_factory: Callable[[], StreamingEstimator] | None = None,
) -> DevelopmentData:
    """Scores from the first lives of plant seeds ``seeds``, which must be ``dev`` or ``calib``.

    A seed outside 1000-2999 (a search seed, a held-out seed, anything else) raises: held-out lives
    never calibrate anything (0001 section 6.3).
    """
    require_calibration_seeds(seeds)
    est_scores: list[float] = []
    prior_scores: list[float] = []
    life_max: list[tuple[float, float, float, float]] = []
    for seed in seeds:
        for life in open_loop_lives(params, seed, scenario, estimator_factory):
            if len(life.outputs) == 0:
                continue
            r = np.array(
                [
                    score(float(t), o.m_days, o.sigma_days)
                    for o, t in zip(life.outputs, life.true_rul, strict=True)
                ]
            )
            is_est = np.array([o.status == STATUS_ESTIMATED for o in life.outputs])
            est_scores += r[is_est].tolist()
            prior_scores += r[~is_est].tolist()
            life_max.append(
                (
                    *_extremes(r[is_est]),
                    *_extremes(r[~is_est]),
                )
            )
    if not est_scores or not prior_scores:
        raise ValueError("the seeds gave no estimated or no prior checkpoints: use more seeds")
    return DevelopmentData(
        scores=np.array(est_scores),
        prior_scores=np.array(prior_scores),
        life_scores=np.array(life_max),
        seeds=tuple(int(s) for s in seeds),
    )


# ----------------------------------------------------------------------------- open-loop replay


def replay_open_loop(
    lives: Sequence[OpenLoopLife],
    provider: IntervalProvider,
    level: float,
    *,
    audit_rho: float = AUDIT_RHO,
) -> tuple[list[Checkpoint], list[Label]]:
    """Run ``provider`` over open-loop first lives in time order: predict each day, deliver each
    life's labels on the first day they are visible.

    On day ``d`` the labels with ``resolved_day == d`` go to ``observe`` first (all of one life in
    one call, lives in asset order), then the day's intervals are predicted. A life resolves when it
    fails or, if it is still running, at the end of the episode. Only the oracle V7 is handed the
    sim-only truth; every other provider gets ``learner_view`` copies. All lives must come from one
    plant seed.
    """
    if not lives:
        return [], []
    seeds = {x.seed for x in lives}
    if len(seeds) != 1:
        raise ValueError("replay_open_loop takes the lives of one plant seed")
    seed = lives[0].seed
    horizon = lives[0].horizon_days
    oracle = provider.name in ("V7", "V7L")
    all_cps: dict[str, list[Checkpoint]] = {x.asset_id: [] for x in lives}
    by_day: dict[int, list[OpenLoopLife]] = {}
    for x in lives:
        by_day.setdefault(math.ceil(x.end_day), []).append(x)
    labels_out: list[Label] = []
    for day in range(horizon + 1):
        for x in by_day.get(day, []):
            labs = resolve_labels(
                all_cps[x.asset_id],
                x.end_day,
                x.censored,
                is_audit_life(seed, x.asset_id, 0, audit_rho),
            )
            labels_out += labs
            provider.observe(labs if oracle else [learner_view(lab) for lab in labs])
        if day == horizon:
            break
        for x in lives:
            if day >= len(x.days):
                continue
            out = x.outputs[day]
            iv = provider.predict(out, level)
            all_cps[x.asset_id].append(
                Checkpoint(
                    0,
                    x.asset_id,
                    day,
                    iv.lo,
                    iv.m,
                    iv.hi,
                    out.sigma_days,
                    level,
                    out.status,
                    true_rul=float(x.true_rul[day]),
                )
            )
    return [cp for x in lives for cp in all_cps[x.asset_id]], labels_out
