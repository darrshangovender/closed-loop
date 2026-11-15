"""Counterfactual queries Q1, Q2 and Q4 and their scoring against twin-world truth (packet P7).

Test lives are first lives of the F1+D plant under the confounded operator policy ``pi_obs``,
observed to their end (a failure, or the end of the 150-day horizon). Each life that alerted gets a
reference day ``t0`` (its first alert) and three twin worlds that share every exogenous stream with
it (``twin.twin_truths``): nobody acts; derate from ``t0``; replace at ``t0 + 5``. The models see
only the factual record. The ground truth is the first uncontrolled failure in each twin world.

``Q1`` (M-CF, claim C-CAUSE)
    the counterfactual failure day under ``do(derate 30% from t0)``: absolute error in days, capped
    at 30 and divided by 30. Lower is better.
``Q2``
    the probability of a failure before ``t0 + 30`` under ``do(nothing)`` and under
    ``do(replace at t0 + 5)``: Brier score over both arms, and the calibration slope (the slope of
    a logistic regression of the outcome on the logit of the forecast; 1 is perfect).
``Q3``
    diagnosis under a confounding shift. It is a classification question for a ``Diagnoser``,
    not for an effect model: ``closed_loop.causal.diagnoser`` and ``closed_loop.causal.mdiag``.
``Q4``
    a decision-level proxy for "the planner using each effect model in the DERATE study". A derate
    costs ``derate_cost_days`` days of life (A). The model derates when its predicted gain exceeds
    that cost; the regret is the shortfall, in days of true life, against the better of the two
    actions, capped at 30 and divided by 30. The full planner-in-the-loop version waits for the
    belief planner (packet P8b); this one asks the same question of the effect model alone.

The scoring is paired: every model is scored on the same cases, so ``paired_difference`` can give a
confidence sequence for the per-life difference between two models. Every record is labelled F1+D.
The SCM is handed the structure of the plant, so Q1 and Q2 favour it by construction; the
misspecified SCM and the diagnosis question are the fair part.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from closed_loop.causal.correlational import CorrelationalEffectModel
from closed_loop.causal.records import LifeRecord
from closed_loop.causal.scm import SCMEffectModel
from closed_loop.causal.twin import Query, causal_params, run_world, twin_truths
from closed_loop.core.results import ResultRecord
from closed_loop.core.types import Fidelity
from closed_loop.plant import PlantParams
from closed_loop.stats import betting_cs, paired_cs

SCRIPT = "closed_loop.causal.queries"
RETAINED_LOAD = 0.7  # the plant's derate keeps 1 - derate_fraction of the load


class EffectModel(Protocol):
    """Frozen text of 0002 section 3.9."""

    def fit(self, lives: Sequence[LifeRecord]) -> None: ...

    def counterfactual_failure_day(
        self, life: LifeRecord, derate_from_day: float, load_cap: float
    ) -> float: ...

    def p_fail_before(
        self, life: LifeRecord, replace_day: float | None, horizon_day: float
    ) -> float: ...


def make_effect_model(kind: str) -> EffectModel:
    """``"correlational"``, ``"scm"`` or ``"scm_misspecified"`` (not yet fitted)."""
    if kind == "correlational":
        return CorrelationalEffectModel()
    if kind == "scm":
        return SCMEffectModel()
    if kind == "scm_misspecified":
        return SCMEffectModel(misspecified=True)
    raise ValueError(
        f"unknown effect model {kind!r}; expected correlational, scm or scm_misspecified"
    )


@dataclass(frozen=True)
class QueryConfig:
    """How the three queries are posed (A)."""

    replace_after: float = 5.0  # replace this many days after the reference day
    horizon_after: float = 30.0  # ask about a failure this many days after the reference day
    derate_cost_days: float = 10.0  # Q4: what a derate costs, in days of life
    cap_days: float = 30.0  # errors and regrets are capped here and divided by it
    hi_alert: float = 2.0
    blind_days: float = 23.0


@dataclass(frozen=True, eq=False)
class Case:
    """One test life with its twin-world truths."""

    life: LifeRecord
    t0: int
    horizon: float
    replace_day: float
    truth_nothing: float
    truth_derate: float
    truth_replace: float
    derate_applied: float  # the day the twin's derate took effect (t0, unless a refusal delayed it)


def build_cases(
    seeds: Sequence[int],
    params: PlantParams | None = None,
    cfg: QueryConfig | None = None,
    policy: str = "pi_obs",
) -> tuple[list[Case], int]:
    """Cases from ``seeds`` (four lives each) and how many lives never alerted."""
    p = params if params is not None else causal_params()
    c = cfg if cfg is not None else QueryConfig()
    cases: list[Case] = []
    skipped = 0
    for seed in seeds:
        lives = run_world(seed, p, policy)
        picked = []
        queries: list[Query] = []
        for lf in lives:
            t0 = lf.first_alert_day(c.hi_alert, c.blind_days)
            if t0 is None or t0 + c.replace_after >= p.horizon_days:
                skipped += 1
                continue
            picked.append((lf, t0))
            queries += [
                Query("nothing", lf.asset_id),
                Query("derate", lf.asset_id, float(t0), RETAINED_LOAD),
                Query("replace", lf.asset_id, float(t0) + c.replace_after),
            ]
        if not picked:
            continue
        answers = twin_truths(seed, queries, params=p)
        for i, (lf, t0) in enumerate(picked):
            a_n, a_d, a_r = answers[3 * i : 3 * i + 3]
            cases.append(
                Case(
                    lf,
                    t0,
                    t0 + c.horizon_after,
                    t0 + c.replace_after,
                    a_n.failure_day,
                    a_d.failure_day,
                    a_r.failure_day,
                    float(a_d.applied_day if a_d.applied_day is not None else t0),
                )
            )
    return cases, skipped


# ---------------------------------------------------------------------------------- scoring
@dataclass(frozen=True, eq=False)
class PerLifeScores:
    """Per-case scores of one model on a list of cases (all arrays have one row per case)."""

    q1_err_days: NDArray[np.float64]
    q1_scaled: NDArray[np.float64]  # min(err, cap) / cap, in [0, 1]
    q2_pred: NDArray[np.float64]  # (n, 2): do(nothing), do(replace)
    q2_out: NDArray[np.float64]  # (n, 2): 1 if the truth fails before the horizon
    q2_brier: NDArray[np.float64]  # (n,) mean over the two arms, in [0, 1]
    q4_regret_days: NDArray[np.float64]
    q4_scaled: NDArray[np.float64]
    q4_derated: NDArray[np.bool_]
    n: int
    cfg: QueryConfig


def per_life_scores(
    model: EffectModel, cases: Sequence[Case], cfg: QueryConfig | None = None
) -> PerLifeScores:
    c = cfg if cfg is not None else QueryConfig()
    n = len(cases)
    err = np.empty(n)
    pred = np.empty((n, 2))
    out = np.empty((n, 2))
    regret = np.empty(n)
    derated = np.empty(n, dtype=bool)
    for i, k in enumerate(cases):
        t_d = model.counterfactual_failure_day(k.life, float(k.t0), RETAINED_LOAD)
        t_n = model.counterfactual_failure_day(k.life, float(k.t0), 1.0)
        err[i] = abs(k.truth_derate - t_d)
        pred[i, 0] = model.p_fail_before(k.life, None, k.horizon)
        pred[i, 1] = model.p_fail_before(k.life, k.replace_day, k.horizon)
        out[i] = (float(k.truth_nothing < k.horizon), float(k.truth_replace < k.horizon))
        go = (t_d - t_n) > c.derate_cost_days
        u_derate = k.truth_derate - c.derate_cost_days
        u_nothing = k.truth_nothing
        chosen = u_derate if go else u_nothing
        regret[i] = max(u_derate, u_nothing) - chosen
        derated[i] = go
    brier = ((pred - out) ** 2).mean(axis=1)
    return PerLifeScores(
        err,
        np.minimum(err, c.cap_days) / c.cap_days,
        pred,
        out,
        brier,
        regret,
        np.minimum(regret, c.cap_days) / c.cap_days,
        derated,
        n,
        c,
    )


def calibration_slope(p: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Slope of the logistic regression of ``y`` on ``logit(p)`` (1 for a calibrated forecast).

    Forecasts are clipped to ``[1e-3, 1 - 1e-3]`` and the fit has a tiny ridge, so perfect
    separation gives a large finite slope instead of failing.
    """
    from scipy.optimize import minimize  # type: ignore[import-untyped]  # heavy import

    pc = np.clip(np.asarray(p, dtype=float).ravel(), 1e-3, 1.0 - 1e-3)
    yy = np.asarray(y, dtype=float).ravel()
    if pc.size < 2 or np.ptp(pc) == 0.0:
        return 0.0
    x = np.log(pc / (1.0 - pc))

    def nll(theta: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        z = theta[0] + theta[1] * x
        loglik = yy * z - np.logaddexp(0.0, z)
        pr = 1.0 / (1.0 + np.exp(-z))
        g = np.array([np.sum(pr - yy), np.sum((pr - yy) * x)])
        return -float(loglik.sum()) + 1e-3 * float(theta @ theta), g + 2e-3 * theta

    sol = minimize(nll, np.array([0.0, 1.0]), jac=True, method="L-BFGS-B")
    return float(sol.x[1])


def _record(
    name: str, values: NDArray[np.float64], measured_on: str, extra: Mapping[str, float | int | str]
) -> ResultRecord:
    low, up = betting_cs(np.clip(values, 0.0, 1.0))
    return ResultRecord(
        name=name,
        value=float(np.mean(values)),
        n=int(values.size),
        fidelity=Fidelity.F1D,
        measured_on=measured_on,
        script=SCRIPT,
        ci_lo=float(low[-1]),
        ci_hi=float(up[-1]),
        ci_kind="confseq-0.05",
        extra=dict(extra),
    )


def records_from_scores(
    sc: PerLifeScores, model_name: str, measured_on: str
) -> dict[str, ResultRecord]:
    """Records of one model: ``Q1_mcf``, ``Q2_brier``, ``Q2_calibration_slope``, ``Q4_regret``."""
    tag = f"{model_name}."
    recs = {
        "Q1_mcf": _record(
            tag + "Q1_mcf",
            sc.q1_scaled,
            measured_on,
            {"mean_abs_error_days": float(sc.q1_err_days.mean()), "cap_days": sc.cfg.cap_days},
        ),
        "Q2_brier": _record(
            tag + "Q2_brier",
            sc.q2_brier,
            measured_on,
            {
                "brier_nothing": float(((sc.q2_pred[:, 0] - sc.q2_out[:, 0]) ** 2).mean()),
                "brier_replace": float(((sc.q2_pred[:, 1] - sc.q2_out[:, 1]) ** 2).mean()),
                "base_rate_nothing": float(sc.q2_out[:, 0].mean()),
                "base_rate_replace": float(sc.q2_out[:, 1].mean()),
            },
        ),
        "Q4_regret": _record(
            tag + "Q4_regret",
            sc.q4_scaled,
            measured_on,
            {
                "mean_regret_days": float(sc.q4_regret_days.mean()),
                "share_derated": float(sc.q4_derated.mean()),
                "derate_cost_days": sc.cfg.derate_cost_days,
            },
        ),
    }
    slope = calibration_slope(sc.q2_pred, sc.q2_out)
    recs["Q2_calibration_slope"] = ResultRecord(
        name=tag + "Q2_calibration_slope",
        value=slope,
        n=int(sc.q2_pred.size),
        fidelity=Fidelity.F1D,
        measured_on=measured_on,
        script=SCRIPT,
        extra={"ideal": 1.0},
    )
    return recs


def score(
    model: EffectModel,
    test_seeds: Sequence[int],
    *,
    cases: Sequence[Case] | None = None,
    params: PlantParams | None = None,
    cfg: QueryConfig | None = None,
) -> Mapping[str, ResultRecord]:
    """Q1, Q2 and Q4 records for a fitted ``model`` on the first lives of ``test_seeds``.

    Pass ``cases`` (from ``build_cases``) to score several models on the same twin truths without
    rebuilding them. Q3 is the diagnosis question; see the module docstring.

    C-CAUSE, which Q1 decides, is **favoured by construction**: the SCM is handed the structure of
    the plant (``scm.py``), and the correlational model is not.
    """
    c = cfg if cfg is not None else QueryConfig()
    if cases is None:
        cases, _ = build_cases(test_seeds, params, c)
    if not cases:
        raise ValueError("no test case: none of the test lives alerted")
    sc = per_life_scores(model, cases, c)
    name = getattr(model, "name", type(model).__name__)
    seeds = sorted(set(int(k.life.seed) for k in cases if k.life.seed is not None))
    where = (
        f"{len(cases)} test lives from seeds {seeds[0]}..{seeds[-1]}, F1+D plant, operator policy "
        f"pi_obs, twin-world truth"
    )
    return records_from_scores(sc, name, where)


def paired_difference(
    a: PerLifeScores,
    b: PerLifeScores,
    metric: str,
    *,
    a_name: str,
    b_name: str,
    alpha: float = 0.05,
    measured_on: str = "",
) -> ResultRecord:
    """Confidence sequence for the per-life mean of ``metric(a) - metric(b)`` (negative: a wins).

    ``metric`` is ``"q1"``, ``"q2_brier"`` or ``"q4"``; all are in [0, 1] for every life.
    """
    pick = {"q1": "q1_scaled", "q2_brier": "q2_brier", "q4": "q4_scaled"}
    if metric not in pick:
        raise ValueError(f"metric must be one of {sorted(pick)}")
    if a.n != b.n:
        raise ValueError("the two models were scored on different numbers of cases")
    va, vb = getattr(a, pick[metric]), getattr(b, pick[metric])
    lo, hi = paired_cs(va, vb, 0.0, 1.0, alpha)
    return ResultRecord(
        name=f"{a_name}-{b_name}.{metric}",
        value=float(np.mean(va - vb)),
        n=int(a.n),
        fidelity=Fidelity.F1D,
        measured_on=measured_on or f"{a.n} paired test lives, F1+D plant",
        script=SCRIPT,
        ci_lo=lo,
        ci_hi=hi,
        ci_kind=f"confseq-{alpha:g}",
    )
