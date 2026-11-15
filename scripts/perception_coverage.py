"""First coverage numbers of the perception layer, open loop and closed loop (packets P4a to P4c).

Everything is SYNTHETIC. Nothing here is tuned: the estimator, the intervals and the labels are the
ones in ``closed_loop.perception``, the plant is the F1 plant, and the only policy parameter this
script chooses (the bound rule's ``k``) is a command-line argument whose default is stated below.

What it measures, per variant (V1 to V7) and condition (nominal, shifted):

* the calibration pools come from ``calib`` seeds (2000...), the evaluation from ``dev`` seeds
  (1000...); no held-out seed is derived or touched;
* **open loop**: the first life of each asset, nobody acts (``plant.simulate_lives``);
* **closed loop**: the F1 plant with a bound rule, ``replace when the scheduling interval's lower
  bound is below k days`` (``k`` = 7), on trend-estimated checkpoints only, carried out through the
  real job path (``RAISE_WO`` with the true failure mode, an oracle diagnosis, then the plant's own
  ``JobRunner``). Approval, crew, permits, handovers and spares all apply, so a replacement comes
  one to a few days after the decision, not at once. This is a stand-in for B3 (packet P6a), not B3.
  With ``--audit-k`` (default 1) V4 runs an audit life until its lower bound is below 1 day.

Reported (``Checkpoint``/``Label`` populations as named in ``perception.coverage``):
``C_orc`` (interval, lower bound), ``C_lrn``, label statuses, Manski bounds, per-life any-miss on
realised RUL, ``Gap_A`` and ``Gap_B`` (0001 section 6.2), the two readings of the shield sets (A10)
for the per-step 99% set and the per-life 95% band, and the firing of the coverage monitors.

    python scripts/perception_coverage.py --episodes 100 --out docs/adr/progress/P4-numbers.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from closed_loop.core.results import ResultRecord  # noqa: E402
from closed_loop.core.timegrid import DAY_Q  # noqa: E402
from closed_loop.core.types import Fidelity, Op  # noqa: E402
from closed_loop.perception.conformal import (  # noqa: E402
    ALPHA_LIFE,
    AUDIT_RHO,
    VARIANTS,
    DevelopmentData,
    IntervalProvider,
    development_data,
    is_audit_life,
    replay_open_loop,
)
from closed_loop.perception.contract import (  # noqa: E402
    ContractCounts,
    contract_counts,
    contract_report,
    make_shield_provider,
    nominal_miss_level,
    report_records,
)
from closed_loop.perception.coverage import (  # noqa: E402
    CoverageRun,
    checkpoint_coverage,
    gaps,
    learner_coverage,
    manski_bounds,
    open_loop_lives,
    per_life_counts,
    select,
)
from closed_loop.perception.estimator import STATUS_ESTIMATED, TheilSenEstimator  # noqa: E402
from closed_loop.perception.labels import Checkpoint, Label, learner_view, resolve_labels  # noqa: E402
from closed_loop.perception.monitor import (  # noqa: E402
    LABEL_MODES,
    CoverageMonitor,
    ProxyBand,
    life_fraction,
    proxy_development_scores,
)
from closed_loop.plant import F1Plant, PlantParams, Scenario  # noqa: E402
from closed_loop.plant.jobs import JobRunner, work_order_command  # noqa: E402
from closed_loop.plant.params import ASSET_IDS  # noqa: E402
from closed_loop.stats.eprocess import BoundedEProcess  # noqa: E402
from closed_loop.stats.intervals import wilson  # noqa: E402

SCRIPT = "scripts/perception_coverage.py"
SCHED_ALPHA = 0.10  # scheduling interval: two-sided 90% (0001 section 6.1)
SETS = ("step99", "life95")


@dataclass
class Streams:
    """Checkpoints and labels of one interval source over a set of episodes."""

    checkpoints: list[Checkpoint] = field(default_factory=list)
    labels: list[Label] = field(default_factory=list)
    lives: ContractCounts = field(default_factory=ContractCounts)
    life_miss: int = 0
    life_n: int = 0


@dataclass
class EpisodeOut:
    sched: Streams
    sets: dict[str, Streams]
    fired: dict[str, tuple[bool, float | None]]  # monitor key -> (fired, day)
    planned_jobs: int
    unplanned_failures: int
    audit_lives: int
    sched_state: dict[str, float] = field(default_factory=dict)  # provider state at the end
    # V4 only: monitor `opt` fed by the audit lives' labels, per shield set
    audit: dict[str, dict[str, object]] = field(default_factory=dict)


# Extra keywords for every online provider (``--gamma`` sets ``gamma``). Empty by default: the
# defaults of ``perception.conformal`` are the pre-registered ones.
PROVIDER_KW: dict[str, float] = {}


def _make_providers(
    variant: str, dev: DevelopmentData, seed: int
) -> tuple[IntervalProvider, float, dict[str, tuple[IntervalProvider, float]]]:
    """The scheduling provider (and its level) and the two shield-set providers."""
    alpha = ALPHA_LIFE if variant == "V6" else SCHED_ALPHA
    sched = dev.provider(variant, alpha=alpha, seed=seed, **PROVIDER_KW)
    sets: dict[str, tuple[IntervalProvider, float]] = {}
    for name in SETS:
        prov = make_shield_provider(
            name, variant=variant, development=dev, seed=seed, **PROVIDER_KW
        )
        sets[name] = (prov, 1.0 - (0.01 if name == "step99" else ALPHA_LIFE))
    return sched, 1.0 - alpha, sets


def last_running_day(s: object, life_id: int, default: float) -> float:
    """The last time the component of life ``life_id`` ran: the start of its last stop.

    Read from the clock log of the asset state (a sim internal that stands in for the DCS record of
    when the asset was taken out of service); ``default`` if the component was never stopped.
    """
    log = s.clock.log  # type: ignore[attr-defined]
    stop = None
    for q, lid, _tau, running, _derated, _tf in log:
        if lid == life_id:
            stop = None if running else q
    return default if stop is None else stop / DAY_Q


def closed_loop_episode(
    seed: int,
    condition: str,
    variant: str,
    dev: DevelopmentData,
    band: ProxyBand,
    params: PlantParams,
    *,
    k: float = 7.0,
    audit_k: float = 1.0,
    act: bool = True,
    stop_at_end: bool = True,
) -> EpisodeOut:
    """One F1 episode with the bound rule (``act=False``: the same loop with nobody acting).

    ``stop_at_end``: a replaced life is censored at the last time its component ran (the job's STOP;
    the stopped component does not age, so ``T > stop`` is what the twin world supports); with False
    it is censored at the completion of the replacement, which overstates the censoring time by the
    length of the stop. Labels resolve when the replacement completes either way.

    Every day: close the lives that ended (a failure is seen when it happens, a replacement when the
    new component starts), deliver their labels, then update the estimator, ask every provider for
    its interval, store the checkpoints and, if the rule says so, order a replacement.
    """
    plant = F1Plant(params)
    obs = plant.reset(seed, Scenario(condition=condition))
    runner = JobRunner()
    est = {a: TheilSenEstimator(params) for a in ASSET_IDS}
    sched, sched_level, set_provs = _make_providers(variant, dev, seed)
    kinds: dict[str, tuple[IntervalProvider, float]] = {"sched": (sched, sched_level), **set_provs}
    out_sched = Streams()
    out_sets = {name: Streams() for name in SETS}
    streams: dict[str, Streams] = {"sched": out_sched, **out_sets}
    monitors: dict[str, CoverageMonitor] = {}
    for name in SETS:
        for mode in LABEL_MODES:
            monitors[f"{name}.{mode}"] = CoverageMonitor(mode, nominal_miss_level(name))
    monitors["proxy"] = CoverageMonitor.proxy(band)
    audit_mon: dict[str, CoverageMonitor] = {}  # V4: mode `opt` on audit lives only
    audit_x: dict[str, list[float]] = {name: [] for name in SETS}
    if variant == "V4":
        for name in SETS:
            audit_mon[name] = CoverageMonitor("opt", nominal_miss_level(name))
    open_cps: dict[tuple[str, str], list[Checkpoint]] = defaultdict(list)  # (kind, asset)
    cur_life = dict.fromkeys(ASSET_IDS, 0)
    closed = dict.fromkeys(ASSET_IDS, False)  # the current life was closed by a failure
    seen_fail = dict.fromkeys(ASSET_IDS, 0)
    life_start_q = dict.fromkeys(ASSET_IDS, 0)
    pending: list[tuple[int, int, str, list[Label]]] = []  # resolved day, order, kind, labels
    order = 0
    audit_lives = 0

    def close_life(
        aid: str, life_id: int, end_day: float, censored: bool, resolved_at: float | None = None
    ) -> None:
        nonlocal order
        audit = is_audit_life(seed, aid, life_id, AUDIT_RHO)
        for kind in kinds:
            cps = open_cps.pop((kind, aid), [])
            labs = resolve_labels(cps, end_day, censored, audit, resolved_at)
            if labs:
                pending.append(
                    (math.ceil(end_day if resolved_at is None else resolved_at), order, kind, labs)
                )
                order += 1

    def deliver(day: int) -> None:
        due = sorted((p for p in pending if p[0] <= day), key=lambda p: (p[0], p[1]))
        for p in due:
            pending.remove(p)
        for _, _, kind, labs in due:
            provider = kinds[kind][0]
            oracle = provider.name in ("V7", "V7L")
            provider.observe(labs if oracle else [learner_view(x) for x in labs])
            streams[kind].labels += labs
            if kind in SETS:
                for mode in LABEL_MODES:
                    monitors[f"{kind}.{mode}"].observe([learner_view(x) for x in labs])
                if audit_mon and labs[0].audit:
                    seen = [learner_view(x) for x in labs]
                    audit_mon[kind].observe(seen)
                    audit_x[kind].append(life_fraction(seen, "opt") or 0.0)

    for q in range(params.horizon_q):
        cmds = []
        if q % DAY_Q == 0:
            day = q // DAY_Q
            truth = plant.truth
            des = plant.des
            for aid in ASSET_IDS:
                s = des.by_id[aid]
                # a failure is seen the day it happens
                if len(s.unc_fail_q) > seen_fail[aid] and not closed[aid]:
                    fail_q = s.unc_fail_q[seen_fail[aid]]
                    close_life(aid, cur_life[aid], (fail_q + 1) / DAY_Q, censored=False)
                    closed[aid] = True
                seen_fail[aid] = len(s.unc_fail_q)
                # a replacement starts a new life
                if truth.life_id[aid] != cur_life[aid]:
                    if not closed[aid]:
                        renewed = s.life_start_q / DAY_Q
                        end = (
                            last_running_day(s, cur_life[aid], renewed) if stop_at_end else renewed
                        )
                        close_life(aid, cur_life[aid], end, censored=True, resolved_at=renewed)
                    cur_life[aid] = truth.life_id[aid]
                    closed[aid] = False
                    life_start_q[aid] = s.life_start_q
            deliver(day)
            for aid in ASSET_IDS:
                if closed[aid] or truth.true_rul_days[aid] <= 0.0:
                    continue
                view = obs.assets[aid]
                out = est[aid].update(aid, day, view.hi_last, cur_life[aid])
                trend = est[aid].trend(aid)
                monitors["proxy"].observe_day(aid, day, view.hi_last, trend, cur_life[aid])
                lo_sched = math.inf
                for kind, (provider, level) in kinds.items():
                    iv = provider.predict(out, level)
                    open_cps[(kind, aid)].append(
                        Checkpoint(
                            cur_life[aid],
                            aid,
                            day,
                            iv.lo,
                            iv.m,
                            iv.hi,
                            out.sigma_days,
                            level,
                            out.status,
                            true_rul=truth.true_rul_days[aid],
                            health_bin=truth.health_bin[aid],
                        )
                    )
                    streams[kind].checkpoints.append(open_cps[(kind, aid)][-1])
                    if kind == "sched":
                        lo_sched = iv.lo
                audit = variant == "V4" and is_audit_life(seed, aid, cur_life[aid], AUDIT_RHO)
                limit = audit_k if audit else k
                if (
                    act
                    and out.status == STATUS_ESTIMATED
                    and lo_sched < limit
                    and not view.job_open
                    and view.op is Op.RUN
                ):
                    cmds.append(work_order_command(aid, truth.life_mode[aid], source="bound-rule"))
        for view in obs.assets.values():
            if view.job is not None:
                cmd = runner.next_command(view, view.job, quarter=obs.quarter)
                if cmd is not None:
                    cmds.append(cmd)
        obs = plant.step(cmds).obs
    # the episode ends: lives still running are censored at the end of the horizon
    horizon = float(params.horizon_days)
    for aid in ASSET_IDS:
        if not closed[aid]:
            close_life(aid, cur_life[aid], horizon, censored=True)
    deliver(params.horizon_days + 1)
    stats = plant.stats()
    for st in streams.values():
        k_, n_ = per_life_counts(st.checkpoints)
        st.life_miss, st.life_n = k_, n_
    for name in SETS:
        out_sets[name].lives = contract_counts(out_sets[name].checkpoints)
    audit_lives = sum(
        is_audit_life(seed, aid, lid, AUDIT_RHO)
        for aid in ASSET_IDS
        for lid in range(cur_life[aid] + 1)
    )
    fired = {key: (m.fired, m.fired_day) for key, m in monitors.items()}
    return EpisodeOut(
        out_sched,
        out_sets,
        fired,
        stats.planned_jobs,
        stats.unplanned_failures,
        audit_lives,
        dict(sched.state()),
        {
            name: {
                "fired": m.fired,
                "day": m.fired_day,
                "n_lives": m.n_lives,
                "log_e": m.log_e_value,
                "x": audit_x[name],
            }
            for name, m in audit_mon.items()
        },
    )


def open_loop_episode(
    seed: int, condition: str, variant: str, dev: DevelopmentData, params: PlantParams
) -> tuple[list[Checkpoint], list[Label]]:
    sched, level, _ = _make_providers(variant, dev, seed)
    lives = open_loop_lives(params, seed, Scenario(condition=condition))
    return replay_open_loop(lives, sched, level)


def summarise(
    variant: str,
    condition: str,
    seeds: list[int],
    dev: DevelopmentData,
    band: ProxyBand,
    params: PlantParams,
    k: float,
    audit_k: float,
    stop_at_end: bool = True,
) -> tuple[list[ResultRecord], dict[str, object]]:
    sched = Streams()
    sets = {name: ContractCounts() for name in SETS}
    open_cps: list[Checkpoint] = []
    open_labels: list[Label] = []
    fired: dict[str, int] = defaultdict(int)
    fired_days: dict[str, list[float]] = defaultdict(list)
    jobs = unplanned = audit_n = 0
    alpha_t: dict[str, list[float]] = defaultdict(list)
    audit_eps: dict[str, list[dict[str, object]]] = defaultdict(list)
    open_life_miss = open_life_n = 0
    for seed in seeds:
        ep = closed_loop_episode(
            seed,
            condition,
            variant,
            dev,
            band,
            params,
            k=k,
            audit_k=audit_k,
            stop_at_end=stop_at_end,
        )
        sched.checkpoints += ep.sched.checkpoints
        sched.labels += ep.sched.labels
        sched.life_miss += ep.sched.life_miss
        sched.life_n += ep.sched.life_n
        for name in SETS:
            sets[name] = sets[name] + ep.sets[name].lives
        for key, (f, day) in ep.fired.items():
            fired[key] += int(f)
            if f and day is not None:
                fired_days[key].append(day)
        for key in ("alpha_t_est", "alpha_t_prior"):
            if key in ep.sched_state:
                alpha_t[key].append(ep.sched_state[key])
        for name, rec_ in ep.audit.items():
            audit_eps[name].append(rec_)
        jobs += ep.planned_jobs
        unplanned += ep.unplanned_failures
        audit_n += ep.audit_lives
        cps, labs = open_loop_episode(seed, condition, variant, dev, params)
        open_cps += cps
        open_labels += labs
        k_, n_ = per_life_counts(cps)
        open_life_miss += k_
        open_life_n += n_
    n = len(seeds)
    measured = (
        f"dev seeds {seeds[0]}-{seeds[-1]} ({n} episodes), {condition} condition, variant "
        f"{variant}, calibration on calib seeds, bound rule k={k} through the real job path, "
        f"censored at {'the stop' if stop_at_end else 'the replacement'}"
    )
    recs: list[ResultRecord] = []
    row: dict[str, object] = {
        "variant": variant,
        "condition": condition,
        "episodes": n,
        "k": k,
        "planned_jobs_per_episode": jobs / n,
        "unplanned_failures_per_episode": unplanned / n,
        "lives_closed": sched.life_n,
        "lives_open": open_life_n,
    }

    def rec(
        name: str,
        value: float,
        n_: int,
        extra: dict[str, float | int | str] | None = None,
        ci: tuple[float, float] | None = None,
        fidelity: Fidelity = Fidelity.F1,
    ) -> None:
        if math.isnan(value):
            return
        recs.append(
            ResultRecord(
                name=f"perception.{variant}.{condition}.{name}",
                value=value,
                n=n_,
                fidelity=fidelity,
                measured_on=measured,
                script=SCRIPT,
                ci_lo=None if ci is None else ci[0],
                ci_hi=None if ci is None else ci[1],
                ci_kind="" if ci is None else "wilson-0.95",
                extra=extra or {},
            )
        )

    for pop in ("post_blind", "estimated", "prior"):
        for label, cps in (("open", open_cps), ("closed", sched.checkpoints)):
            sel = select(cps, pop)
            if not sel:
                continue
            c = checkpoint_coverage(cps, pop)
            lo = checkpoint_coverage(cps, pop, lower=True)
            kk = round(c * len(sel))
            row[f"c_orc.{label}.{pop}"] = c
            row[f"c_orc_lo.{label}.{pop}"] = lo
            row[f"n.{label}.{pop}"] = len(sel)
            rec(f"c_orc.{label}.{pop}", c, len(sel), ci=wilson(kk, len(sel)))
            rec(f"c_orc_lo.{label}.{pop}", lo, len(sel))
    for pop in ("post_blind", "estimated"):
        lrn = learner_coverage(sched.labels, pop)
        lo_m, hi_m = manski_bounds(sched.labels, pop)
        for key, val in lrn.items():
            row[f"lrn.{pop}.{key}"] = val
        row[f"manski.{pop}"] = (lo_m, hi_m)
        try:
            g = gaps(CoverageRun(open_cps), CoverageRun(sched.checkpoints, sched.labels), pop)
        except ValueError as exc:
            row[f"gaps.{pop}"] = f"undefined: {exc}"
        else:
            row[f"gaps.{pop}"] = g
            rec(f"gap_a.{pop}", g["gap_a"], int(g["n_labels"]))
            rec(f"gap_b.{pop}", g["gap_b"], int(g["n_closed"]))
    row["life_any_miss.closed"] = sched.life_miss / sched.life_n if sched.life_n else math.nan
    row["life_any_miss.open"] = open_life_miss / open_life_n if open_life_n else math.nan
    if sched.life_n:
        rec(
            "life_any_miss.closed",
            sched.life_miss / sched.life_n,
            sched.life_n,
            ci=wilson(sched.life_miss, sched.life_n),
        )
    if open_life_n:
        rec(
            "life_any_miss.open",
            open_life_miss / open_life_n,
            open_life_n,
            ci=wilson(open_life_miss, open_life_n),
        )
    report = contract_report(sets["step99"], sets["life95"])
    recs += [
        ResultRecord(**{**r.to_dict(), "name": f"perception.{variant}.{condition}.{r.name}"})
        for r in report_records(report, measured_on=measured, script=SCRIPT, fidelity=Fidelity.F1)
    ]
    for name, r in report.items():
        row[f"contract.{name}"] = {
            "lives": r.counts.lives,
            "any_miss_rul": r.any_miss_rul,
            "any_miss_bin": r.any_miss_bin,
            "step_miss_rul": r.step_miss_rul,
            "step_miss_bin": r.step_miss_bin,
            "mean_width": r.mean_width,
        }
    row["monitors_fired"] = {key: f"{fired[key]}/{n}" for key in sorted(fired)}
    row["monitors_median_day"] = {
        key: float(sorted(v)[len(v) // 2]) for key, v in sorted(fired_days.items()) if v
    }
    row["audit_lives"] = audit_n
    row["planned_jobs_total"] = jobs
    row["unplanned_failures_total"] = unplanned
    if audit_eps:
        row["audit_monitor"] = {
            name: _audit_summary(name, eps, n) for name, eps in audit_eps.items()
        }
    for key, vals in alpha_t.items():
        vals.sort()
        row[f"{key}.final"] = {"min": vals[0], "median": vals[len(vals) // 2], "max": vals[-1]}
    return recs, row


def _audit_summary(name: str, eps: list[dict[str, object]], n: int) -> dict[str, object]:
    """Mode `opt` fed by V4's audit lives: per episode, and pooled over the episodes."""
    fired_days = sorted(float(e["day"]) for e in eps if e["fired"])  # type: ignore[arg-type]
    xs = [x for e in eps for x in e["x"]]  # type: ignore[attr-defined]
    pooled = BoundedEProcess(nominal_miss_level(name), 0.01)
    for x in xs:
        pooled.update(x)
    lives = [int(e["n_lives"]) for e in eps]  # type: ignore[call-overload]
    return {
        "p0": nominal_miss_level(name),
        "episodes": n,
        "fired_per_episode": f"{len(fired_days)}/{n}",
        "median_fired_day": fired_days[len(fired_days) // 2] if fired_days else None,
        "audit_lives_total": sum(lives),
        "audit_lives_per_episode": sum(lives) / n,
        "episodes_with_three_or_more_audit_lives": sum(k >= 3 for k in lives),
        "audit_lives_with_a_miss_label": sum(x > 0.0 for x in xs),
        "max_e_value_in_an_episode": max(math.exp(min(float(e["log_e"]), 700.0)) for e in eps),  # type: ignore[arg-type]
        "pooled_over_episodes": {
            "lives": pooled.n,
            "fired": pooled.fired,
            "fired_at_life": pooled.fired_at,
            "e_value": pooled.e_value,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--calib", type=int, default=300, help="calib seeds for the score pools")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--conditions", default="nominal,shifted")
    ap.add_argument("--k", type=float, default=7.0, help="bound rule: replace when lo < k days")
    ap.add_argument("--audit-k", type=float, default=1.0)
    ap.add_argument(
        "--gamma",
        type=float,
        default=0.0,
        help="diagnostic: override the ACI step of the online variants (0 keeps the 0.02 default)",
    )
    ap.add_argument("--first-seed", type=int, default=1000)
    ap.add_argument(
        "--censor-at",
        choices=("stop", "replacement"),
        default="stop",
        help="censor a replaced life when its component stopped, or when the new one started",
    )
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--merge",
        action="store_true",
        help="add the rows and records of this run to an existing --out file, replacing any "
        "for the same variant and condition",
    )
    args = ap.parse_args()
    params = PlantParams()
    if args.gamma > 0.0:
        PROVIDER_KW["gamma"] = args.gamma
    t0 = time.perf_counter()
    cal = list(range(2000, 2000 + args.calib))
    dev = development_data(params, cal)
    band = ProxyBand.fit(proxy_development_scores(params, cal), 0.05)
    print(
        f"calibration: {len(dev.scores)} trend scores, {len(dev.prior_scores)} prior scores, "
        f"{len(dev.life_scores)} lives, proxy quantile {band.quantile:.3f} "
        f"({time.perf_counter() - t0:.0f} s)",
        flush=True,
    )
    seeds = list(range(args.first_seed, args.first_seed + args.episodes))
    all_rows: list[dict[str, object]] = []
    all_records: list[dict[str, object]] = []
    for condition in args.conditions.split(","):
        for variant in args.variants.split(","):
            t1 = time.perf_counter()
            recs, row = summarise(
                variant,
                condition,
                seeds,
                dev,
                band,
                params,
                args.k,
                args.audit_k,
                args.censor_at == "stop",
            )
            all_rows.append(row)
            all_records += [r.to_dict() for r in recs]
            print(f"--- {variant} {condition} ({time.perf_counter() - t1:.0f} s)")
            for key in (
                "c_orc.open.post_blind",
                "c_orc.closed.post_blind",
                "c_orc_lo.closed.post_blind",
                "lrn.post_blind.c_lrn",
                "lrn.post_blind.unknown_rate",
                "life_any_miss.closed",
                "life_any_miss.open",
            ):
                if key in row:
                    print(f"  {key:34s} {row[key]!r}")
            print("  manski post_blind", row.get("manski.post_blind"))
            print("  gaps post_blind  ", row.get("gaps.post_blind"))
            print("  contract         ", row.get("contract.step99"), row.get("contract.life95"))
            print("  monitors fired   ", row["monitors_fired"])
            if "audit_monitor" in row:
                print("  audit monitor    ", row["audit_monitor"])
            print(
                "  alpha_t final    ", row.get("alpha_t_est.final"), row.get("alpha_t_prior.final")
            )
            sys.stdout.flush()
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        if args.merge and path.exists():
            old = json.loads(path.read_text(encoding="utf-8"))
            keys = {(r["variant"], r["condition"]) for r in all_rows}
            prefixes = tuple(f"perception.{v}.{c}." for v, c in keys)
            all_rows = [
                r for r in old["rows"] if (r["variant"], r["condition"]) not in keys
            ] + all_rows
            all_records = [
                r for r in old["records"] if not r["name"].startswith(prefixes)
            ] + all_records
        path.write_text(
            json.dumps({"rows": all_rows, "records": all_records}, indent=1, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
