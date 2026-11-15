"""Claim assembly: the inputs of the claims, the shield counts and the ablation rows, from cells.

Moved out of ``stages.py`` by P12f (0003 A18, F3). This module reads the cells of a run and writes
``ResultRecord``s; it defines no worker task and nothing a simulated episode imports, so it is left
out of ``runner.simulation_code_sha256`` (``runner._NOT_SIMULATION``): a fix to what a verdict
reads, made after the full run, does not change a cell key and does not cost a recompute.
``tests/eval/test_assemble.py`` pins both halves: editing this file leaves the hash alone, editing
``stages.py`` moves it.

The helpers shared with the stages that run cells (``value_record``, ``paired_record``,
``per_seed``, ``claim_cell``, ``CellStore``, ``StageContext``) stay in ``stages.py`` and are
imported from there; ``stages.py`` reaches this module only inside the functions that call it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np

from closed_loop.core.results import ResultRecord
from closed_loop.core.types import Fidelity
from closed_loop.eval.arms import ARMS
from closed_loop.eval.claims import (
    ADAPT_VARIANTS,
    SHIELD_PAIRS,
    claim_name,
)
from closed_loop.eval.counts import (
    ALPHA_DESCRIPTIVE,
    ALPHA_PERC,
    ALPHA_PRIMARY,
)
from closed_loop.eval.episode import EpisodeResult
from closed_loop.eval.metrics import (
    METRICS,
    EpisodeCoverage,
)
from closed_loop.eval.stages import (
    OPEN_LOOP_VARIANTS,
    SCRIPT,
    CellStore,
    StageContext,
    _range,
    _span_text,
    claim_cell,
    paired_record,
    per_seed,
)

__all__ = ["UNDEFINED_GAP_A", "claim_records"]


UNDEFINED_GAP_A: Final[str] = (
    "no label of this cell ended in an observed failure, so the learner has nothing to confirm an "
    "upper bound with: C_lrn and Gap_A are undefined and only the Manski bounds are known. A "
    "policy that prevents failures leaves its learner with no way to confirm an upper bound"
)


def _pooled_gap_a(results: Sequence[EpisodeResult], pop: str) -> dict[str, float | str] | None:
    """Gap_A = C_lrn - C_orc pooled over a cell's episodes, and the Manski bounds.

    A label is a HIT only if its life ended in an observed failure (0003 A18, F1). When the cell has
    no such label (``uncensored`` is 0) ``C_lrn`` and ``Gap_A`` are undefined: the dict then holds
    the label counts and the Manski bounds, ``undefined`` = 1 and the reason, and neither number.
    A cell with uncensored labels is unchanged, even if ``hit`` is 0.
    """
    key = "post_blind" if pop == "pooled" else "estimated"
    n = sum(r.learner.get(f"{key}.n", 0) for r in results)
    hit = sum(r.learner.get(f"{key}.hit", 0) for r in results)
    miss = sum(r.learner.get(f"{key}.miss", 0) for r in results)
    lower_hit = sum(r.learner.get(f"{key}.lower_hit", 0) for r in results)
    unknown = sum(r.learner.get(f"{key}.unknown", 0) for r in results)
    uncensored = sum(r.learner.get(f"{key}.uncensored", 0) for r in results)
    informative = hit + miss + lower_hit
    if pop == "pooled":
        weights = [(r.coverage.c, r.coverage.n) for r in results]
    else:
        weights = [(r.coverage.c_trend, r.coverage.n_trend) for r in results]
    num = sum(c * w for c, w in weights if c is not None)
    den = sum(w for c, w in weights if c is not None)
    if not n or not den:
        return None
    c_orc = num / den
    counts: dict[str, float | str] = {
        "c_orc": c_orc,
        "manski_lo": hit / n,
        "manski_hi": (n - miss) / n,
        "n_labels": float(n),
        "informative": float(informative),
        "uncensored": float(uncensored),
        "hit": float(hit),
        "miss": float(miss),
        "lower_hit": float(lower_hit),
        "unknown": float(unknown),
        "unknown_rate": unknown / n,
        "n_checkpoints": float(den),
    }
    if not uncensored:
        return {**counts, "undefined": 1.0, "reason": UNDEFINED_GAP_A}
    c_lrn = hit / informative
    return {**counts, "c_lrn": c_lrn, "gap_a": c_lrn - c_orc}


def claim_records(sc: StageContext) -> list[ResultRecord]:
    """The inputs of claims C-PERC, C-ADAPT, C-PLAN, C-MON and C-SHIELD, and the descriptive gaps
    and ablations, from the cells in ``sc.store`` (the other claims' stages write their own)."""
    out: list[ResultRecord] = []
    store = sc.store
    primary = sc.primary_variant
    open_cov: Mapping[tuple[str, str], dict[int, EpisodeCoverage]] = sc.shared.get("open_loop", {})

    # ---- Gap_B (C-PERC), Gap_A, per variant, condition and stratum
    for variant in OPEN_LOOP_VARIANTS:
        name = f"B3-{variant}"
        for condition in ("nominal", "shifted"):
            results = store.results(name, condition)
            cov_open = open_cov.get((condition, variant))
            if not results or not cov_open:
                continue
            for stratum, cfield in (("pooled", "c"), ("trend", "c_trend")):
                a = {
                    s: getattr(c, cfield)
                    for s, c in cov_open.items()
                    if getattr(c, cfield) is not None
                }
                b = {
                    r.seed: getattr(r.coverage, cfield)
                    for r in results
                    if getattr(r.coverage, cfield) is not None
                }
                where = (
                    f"Gap_B = oracle coverage of the open-loop first lives minus that of the "
                    f"closed loop under B3 with {variant}, {stratum} stratum, {condition} condition"
                )
                for alpha, rname in (
                    (ALPHA_DESCRIPTIVE, f"gapb.{condition}.{stratum}.{variant}"),
                    (ALPHA_PERC, claim_name("C-PERC", "gap_b", stratum, variant)),
                ):
                    if rname.startswith("claim/") and condition != "nominal":
                        continue  # the verdict is read on the nominal condition
                    rec = paired_record(
                        rname, a, b, lo=0.0, hi=1.0, alpha=alpha, fidelity=Fidelity.F1,
                        measured_on=where,
                        extra={"failed": sum(1 for r in results if r.failure)},
                    )  # fmt: skip
                    if rec is not None:
                        out.append(rec)
                gap = _pooled_gap_a(results, stratum)
                if gap is not None:
                    out.append(
                        ResultRecord(
                            name=f"gapa.{condition}.{stratum}.{variant}",
                            value=float(
                                gap.get("gap_a", 0.0)
                            ),  # 0.0 with ``undefined``: not a number
                            n=int(gap["informative"]),
                            fidelity=Fidelity.F1,
                            measured_on=(
                                f"Gap_A = learner minus oracle coverage, B3 with {variant}, "
                                f"{stratum} stratum, {condition}, {len(results)} episodes "
                                f"({_span_text([r.seed for r in results])}), labels pooled"
                            ),
                            script=SCRIPT,
                            extra=gap,
                        )
                    )

    # ---- C-ADAPT: paired c_e^lo(Vx) - c_e^lo(V1) under shifted, and coverage of Vx
    v1_shifted = store.results("B3-V1", "shifted")
    for variant in ADAPT_VARIANTS:
        name = f"B3-{variant}"
        sh, nom = store.results(name, "shifted"), store.results(name, "nominal")
        # the verdict is the primary variant's, at the claim level; the others are secondary rows,
        # shown at the descriptive level of prereg section 6
        level = ALPHA_PRIMARY if variant == primary else ALPHA_DESCRIPTIVE
        if sh and v1_shifted:
            rec = paired_record(
                claim_name("C-ADAPT", "diff_c_lo", variant),
                per_seed(sh, "m_cov_lo"),
                per_seed(v1_shifted, "m_cov_lo"),
                lo=0.0, hi=1.0, alpha=level, fidelity=Fidelity.F1,
                measured_on=f"c_e^lo of B3 with {variant} minus B3 with V1, shifted condition",
            )  # fmt: skip
            if rec is not None:
                out.append(rec)
        if sh:
            out.append(
                claim_cell(sh, "m_cov_lo", claim_name("C-ADAPT", "cov_lo_shifted", variant), level)
            )
        if nom:
            out.append(
                claim_cell(nom, "m_cov_lo", claim_name("C-ADAPT", "cov_lo_nominal", variant), level)
            )

    # ---- C-PLAN: P0 against B3, P1 against B3 behind the shield; nominal (verdict) and shifted
    for planner, baseline in (("P0", f"B3-{primary}"), ("P1", "B3-S")):
        for condition in ("nominal", "shifted"):
            p_res, b_res = store.results(planner, condition), store.results(baseline, condition)
            if not p_res or not b_res:
                continue
            where = f"{planner} minus {baseline}, {condition} condition"
            # the cost leg is on the spread-scaled sequence where it applies; the safety leg (f_e,
            # the share of episodes with an unplanned failure) is on the betting sequence alone,
            # which stays finite-sample when failures are rare (0003 A16)
            for metric, rname, betting in (
                ("m_cost", "cost_diff", False),
                ("f_e", "failure_diff", True),
            ):
                lo, hi = _range(metric)
                rec = paired_record(
                    claim_name("C-PLAN", rname, planner, condition),
                    per_seed(p_res, metric), per_seed(b_res, metric),
                    lo=lo, hi=hi,
                    alpha=ALPHA_PRIMARY if condition == "nominal" else ALPHA_DESCRIPTIVE,
                    fidelity=Fidelity.F1,
                    measured_on=f"{METRICS[metric].describe}: {where}",
                    extra={"failed": sum(1 for r in (*p_res, *b_res) if r.failure)},
                    betting_only=betting,
                )  # fmt: skip
                if rec is not None:
                    out.append(rec)

    # ---- C-MON: the four modes, passive on P1 (shield on, no acting monitor)
    p1_nominal, p1_shifted = store.results("P1", "nominal"), store.results("P1", "shifted")
    for mode in ("opt", "pess", "inf", "proxy"):
        level = ALPHA_PRIMARY if mode == sc.monitor_mode else ALPHA_DESCRIPTIVE
        if p1_nominal:
            try:
                out.append(
                    claim_cell(
                        p1_nominal,
                        f"m_false_alarm_{mode}",
                        claim_name("C-MON", "false_alarm", mode),
                        level,
                    )
                )
            except ValueError:
                pass
        if p1_shifted:
            try:
                out.append(
                    claim_cell(
                        p1_shifted,
                        f"m_detect_{mode}",
                        claim_name("C-MON", "detection", mode),
                        level,
                    )
                )
            except ValueError:
                pass

    out.extend(_shield_records(store))
    out.extend(_uncovered_records(store))
    out.extend(_ablation_records(store, primary))
    return out


_SHIELDED_ARMS: Final[tuple[str, ...]] = (
    "H-C1", "H-G1", "P1", "P1-nopair", "P2", "P1-corr", "P1-oracle", "P1-step", "B3-S", "S-clause",
)  # fmt: skip
# P1-nopair is P1 as it was before the operator pair stop (0003 A18, R8): a shielded arm, so an
# assumption_held violation in it is a defect like any other.
# S-stale is left out of the defect scan on purpose: its shield is built from a superseded clause
# and scored against the current one, so a violation there is the finding, not a defect.


def _shield_records(store: CellStore) -> list[ResultRecord]:
    out: list[ResultRecord] = []
    for arm_name in ("H-C0", "H-C1", "H-G0", "H-G1", "H-C-I", "H-G-I"):
        res = store.results(arm_name, "nominal")
        if res:
            out.append(claim_cell(res, "m_viol_p13", claim_name("C-SHIELD", "viol_p13", arm_name)))
    for arm_name in _SHIELDED_ARMS:
        res = store.results(arm_name, "nominal")
        if res:
            out.append(claim_cell(res, "m_defect", claim_name("C-SHIELD", "defects", arm_name)))
            out.append(
                claim_cell(
                    res, "m_viol_p13_plant_only", claim_name("C-SHIELD", "plant_only", arm_name)
                )
            )
            k_p45 = sum(
                1
                for r in res
                if r.min_robustness.get("P4", 1) < 0 or r.min_robustness.get("P5", 1) < 0
            )
            contract = sum(
                1
                for r in res
                if any(
                    a.prop in ("P4", "P5") and a.label in ("C1", "A_phys") for a in r.attributions
                )
            )
            out.append(
                ResultRecord(
                    name=claim_name("C-SHIELD", "p45", arm_name),
                    value=float(k_p45),
                    n=len(res),
                    fidelity=Fidelity.F1,
                    measured_on=f"episodes of {arm_name}, nominal: P4-P5 violations, attribution",
                    script=SCRIPT,
                    extra={"k": k_p45, "contract": contract},
                )
            )
    for pair, (shielded, unshielded, interlock) in SHIELD_PAIRS.items():
        s, u, i = (store.results(x, "nominal") for x in (shielded, unshielded, interlock))
        if s and u:
            ms, mu = (
                float(np.mean([r.throughput for r in s])),
                float(np.mean([r.throughput for r in u])),
            )
            out.append(
                ResultRecord(
                    name=claim_name("C-SHIELD", "throughput_ratio", pair),
                    value=ms / mu if mu > 0 else 1.0,  # 0 over 0 is no change; x over 0 is no ratio
                    n=min(len(s), len(u)),
                    fidelity=Fidelity.F1,
                    measured_on=(
                        f"mean throughput of {shielded} over that of {unshielded} (shield on over "
                        "off, same proposer), nominal condition"
                    ),
                    script=SCRIPT,
                    extra={"shielded": ms, "unshielded": mu, "shielded_arm": shielded,
                           "unshielded_arm": unshielded, "unshielded_stopped": int(mu == 0.0)},
                )
            )  # fmt: skip
        if s and i:
            rec = paired_record(
                claim_name("C-SHIELD", "interlock", pair),
                per_seed(i, "m_viol_p13"), per_seed(s, "m_viol_p13"),
                lo=0.0, hi=1.0, alpha=ALPHA_PRIMARY, fidelity=Fidelity.F1,
                measured_on=f"M-VIOL (P1-P3) of {interlock} minus {shielded}, nominal condition",
                extra={
                    "throughput_shield": float(np.mean([r.throughput for r in s])),
                    "throughput_interlock": float(np.mean([r.throughput for r in i])),
                    "override_shield": _mean_or_nan(per_seed(s, "override_rate")),
                    "override_interlock": _mean_or_nan(per_seed(i, "override_rate")),
                },
            )  # fmt: skip
            if rec is not None:
                out.append(rec)
    return out


UNCOVERED_MEANING: Final[str] = (
    "decisions at which the abstract state had no allowed command, so the runtime fell back to the "
    "first command in the table's priority order (a pump pair: STOP|STOP, which keeps both pumps "
    "stopped each quarter). The proof's premise is that an issued command is in the allowed set; "
    "this counts the decisions where it was not. It happens only after the model's assumptions "
    "are already violated. The fallback is a known limit of the shield and is published as one"
)


def _uncovered_records(store: CellStore) -> list[ResultRecord]:
    """``shield.uncovered.<arm>.<condition>`` for every shielded cell (G7): the episodes with any
    uncovered decision out of the cell's episodes, and the total number of decisions."""
    out: list[ResultRecord] = []
    for arm_name, condition in store.cells():
        arm = ARMS.get(arm_name)
        res = store.results(arm_name, condition)
        if arm is None or not arm.shield or not res:
            continue
        total = sum(r.uncovered for r in res)
        k = sum(1 for r in res if r.uncovered > 0)
        out.append(
            ResultRecord(
                name=f"shield.uncovered.{arm_name}.{condition}",
                value=float(total),
                n=len(res),
                fidelity=Fidelity.F1,
                measured_on=f"episodes of {arm_name}, {condition} (value: all decisions; k: the "
                f"episodes with any); {UNCOVERED_MEANING}",
                script=SCRIPT,
                extra={"k": k, "total": total},
            )
        )
    return out


def _mean_or_nan(values: Mapping[int, float]) -> float:
    return float(np.mean(list(values.values()))) if values else 0.0


_ABLATIONS: Final[tuple[tuple[str, str, str], ...]] = (
    # (label, arm, reference); "{P}" is the primary variant
    ("perception variant: V1 against the primary variant", "B3-V1", "B3-{P}"),
    ("planner against rule", "P0", "B3-{P}"),
    ("shield on against off (planner)", "P1", "P0"),
    ("shield on against off (rule)", "B3-S", "B3-{P}"),
    ("monitor on against off", "P2", "P1"),
    ("screening on against off", "H-G2", "H-G0"),
    ("specification: stale against clause", "S-stale", "S-clause"),
    ("INSPECT disabled", "P0-noinspect", "P0"),
    ("planner model mismatch 10%", "P0-mis10", "P0"),
    ("planner model mismatch 25%", "P0-mis25", "P0"),
    ("correlational against structural diagnoser", "P1-corr", "P1"),
    ("oracle against structural diagnoser", "P1-oracle", "P1"),
    ("per-step 99% set against per-life band", "P1-step", "P1"),
    ("interlock against shield (planner)", "P1-I", "P1"),
    ("operator pair stop off against on (the pump deadlock, 0003 A18 R8)", "P1-nopair", "P1"),
)
_ABLATION_METRICS: Final[tuple[str, ...]] = ("m_cost", "u_e", "throughput", "m_viol_p15")


def _ablation_records(store: CellStore, primary: str) -> list[ResultRecord]:
    out: list[ResultRecord] = []
    for idx, (label, a_name, b_name) in enumerate(_ABLATIONS):
        b_name = b_name.format(P=primary)
        for condition in ("nominal", "shifted"):
            a_res, b_res = store.results(a_name, condition), store.results(b_name, condition)
            if not a_res or not b_res:
                continue
            for metric in _ABLATION_METRICS:
                lo, hi = _range(metric)
                rec = paired_record(
                    f"ablation.{idx:02d}.{condition}.{metric}",
                    per_seed(a_res, metric), per_seed(b_res, metric),
                    lo=lo, hi=hi, alpha=ALPHA_DESCRIPTIVE, fidelity=Fidelity.F1,
                    measured_on=f"{METRICS[metric].describe}: {a_name} minus {b_name}, {condition}",
                    extra={
                        "label": label, "arm": a_name, "reference": b_name,
                        "condition": condition, "metric": metric,
                    },
                )  # fmt: skip
                if rec is not None:
                    out.append(rec)
    return out
