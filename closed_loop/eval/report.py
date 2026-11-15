"""The report: ``results.json``, ``RESULTS.md`` and ``docs/PROOF-SCOPE.md`` (packet P12b).

``results.json`` is the list of ``ResultRecord``s of a run, in the order the stages produced them.
``RESULTS.md`` and ``PROOF-SCOPE.md`` are **pure functions of that list**: every number in them is
read from a record, nothing is typed in, and ``check_outputs`` regenerates both and compares them,
byte for byte, with the files on disk. That is the test that a hand-edited number is caught: edit
``RESULTS.md`` and it differs from what ``results.json`` generates; edit ``results.json`` and the
regenerated tables differ from ``RESULTS.md``, the verdicts rebuilt from the records differ from the
stored ones, and the digest in the ledger no longer matches.

The text avoids absolutes. The only unqualified statements are definitions; a result of "0 of n"
is printed with its upper bound, because a finite number of episodes bounds a rate from above and
does not make it zero (prereg section 7). A proof is stated with the assumptions it needs.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Final

from closed_loop.core.results import ResultRecord
from closed_loop.eval.claims import (
    Verdict,
    verdicts_from_records,
    verdicts_from_stored,
)
from closed_loop.eval.counts import ALPHA_DESCRIPTIVE

__all__ = [
    "FORBIDDEN",
    "UNCOVERED_TEXT",
    "absolutes_in",
    "check_outputs",
    "headline_of",
    "records_from_json",
    "records_to_json",
    "uncovered_rows",
    "render_proof_scope",
    "render_results",
    "results_sha256",
    "SECONDARY_NOTE",
    "split_verdicts",
    "write_outputs",
]

# Words that a generated sentence may not use unless a proof backs it (the owner's rule).
FORBIDDEN: Final[re.Pattern[str]] = re.compile(
    r"\b(never|always|guaranteed|guarantees)\b|100%", re.IGNORECASE
)
# Free text that comes from outside the report (an exception message in a cause, the summary line
# of a test run, the provenance of a tuned value) sits in these ``extra`` fields. It is data: the
# guard against absolutes reads the report's own sentences, so a word in an exception message does
# not stop a run from publishing.
_DATA_KEYS: Final[tuple[str, ...]] = ("cause", "failures", "notes", "tuned", "counts", "evidence")
QUICK_BANNER: Final[str] = "QUICK RUN, NOT A RESULT"
DASH: Final[str] = "-"
SECONDARY_NOTE: Final[str] = (
    "Secondary rows, shown at the descriptive level of 0.05. They decide nothing, whatever their "
    "label says: a win here is not a registered win."
)


# ----------------------------------------------------------------------------- json


def records_to_json(records: Sequence[ResultRecord]) -> str:
    """The canonical text of ``results.json``: a list of records, LF line ends, final newline."""
    payload: list[dict[str, Any]] = []
    for rec in records:
        if rec.value in (float("inf"), float("-inf")):
            raise ValueError(f"record {rec.name!r} has an infinite value and cannot be written")
        d = rec.to_dict()
        d["extra"] = {k: _json_safe(v) for k, v in d["extra"].items()}
        payload.append(d)
    return json.dumps(payload, indent=1, ensure_ascii=False, allow_nan=False) + "\n"


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return str(value)  # "nan" or "inf": text, so that the file stays valid JSON
    return value


def records_from_json(text: str) -> list[ResultRecord]:
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("results.json must hold a list of records")
    return [ResultRecord.from_dict(d) for d in data]


def results_sha256(text: str) -> str:
    """SHA-256 of ``results.json`` as text with LF line ends (what the ledger records)."""
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------- helpers


class Index:
    """Records by name, with the formatting every table uses."""

    def __init__(self, records: Sequence[ResultRecord]) -> None:
        self.records = list(records)
        self.by_name = {r.name: r for r in records}

    def get(self, name: str) -> ResultRecord | None:
        return self.by_name.get(name)

    def prefixed(self, prefix: str) -> list[ResultRecord]:
        return [r for r in self.records if r.name.startswith(prefix)]

    def param(self, key: str, default: str = "") -> str:
        rec = self.by_name.get("run/params")
        return str(rec.extra.get(key, default)) if rec is not None else default


def num(x: float | None, places: int = 4) -> str:
    return DASH if x is None else f"{x:.{places}f}"


def value_ci(rec: ResultRecord | None) -> str:
    if rec is None:
        return DASH
    ci = (
        "" if rec.ci_lo is None or rec.ci_hi is None else f" ({num(rec.ci_lo)} to {num(rec.ci_hi)})"
    )
    return f"{num(rec.value)}{ci}"


def k_over_n(rec: ResultRecord | None) -> str:
    if rec is None:
        return DASH
    k = rec.extra.get("k")
    return f"{int(k)}/{rec.n}" if k is not None else str(rec.n)


def esc(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> list[str]:
    body = [list(map(esc, row)) for row in rows]
    if not body:
        return ["(no rows)", ""]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in body]
    out.append("")
    return out


def on_text(rec: ResultRecord | None) -> str:
    return DASH if rec is None else rec.measured_on


def fid(rec: ResultRecord | None) -> str:
    return DASH if rec is None else rec.fidelity.value


# ----------------------------------------------------------------------------- RESULTS.md


def _banner(ix: Index) -> list[str]:
    out: list[str] = []
    if ix.param("mode") == "quick":
        out += [
            f"> **{QUICK_BANNER}.** Development seeds, counts divided by 25 (minimum 4), no "
            "ledger entry. Nothing below is a result; it shows the pipeline runs.",
            "",
        ]
    comp = ix.get("run/completeness")
    if comp is not None and comp.value < 1.0:
        not_built = str(comp.extra.get("not_built", "")) or "none"
        skipped = str(comp.extra.get("skipped", "")) or "none"
        out += [
            "> **INCOMPLETE RUN.** This run does not claim completeness. "
            f"Stages not built: {not_built}. Stages skipped: {skipped}. "
            f"Stages that ran only in part: {comp.extra.get('partial', 'none') or 'none'}.",
            "",
        ]
    return out


def split_verdicts(verdicts: Iterable[Verdict]) -> tuple[list[Verdict], list[Verdict]]:
    """The registered claims and the secondary rows (descriptive level, deciding nothing)."""
    listed = list(verdicts)
    registered = [v for v in listed if v.alpha < ALPHA_DESCRIPTIVE]
    return registered, [v for v in listed if v.alpha >= ALPHA_DESCRIPTIVE]


def _verdict_section(verdicts: Sequence[Verdict]) -> list[str]:
    out = ["## Verdicts", ""]
    out += [
        "Each claim is judged at the level the pre-registration fixes (0.05 over the nine primary "
        "claims; the two C-PERC verdicts at 0.05 over 18). A loss and an inconclusive result are "
        "published with the same prominence as a win.",
        "",
    ]
    registered, secondary = split_verdicts(verdicts)
    for group in (registered, secondary):
        if group is secondary and group:
            out += [SECONDARY_NOTE, ""]
        if group:
            rows = [(v.claim, v.text, f"{v.alpha:.5f}", v.headline) for v in group]
            out += table(["Claim", "Verdict", "Level", "What the numbers say"], rows)
    by_rule: dict[str, list[str]] = {}
    for v in verdicts:
        by_rule.setdefault(v.rule, []).append(v.claim)
    for rule, claims in by_rule.items():
        out.append(f"Rule for {', '.join(claims)}: {rule}.")
    out.append("")
    for v in verdicts:
        if v.notes:
            out.append(f"**{v.claim}** ({v.text})")
            out += [f"- {note}" for note in v.notes]
            out.append("")
    return out


def _stage_section(ix: Index) -> list[str]:
    rows = []
    for rec in ix.prefixed("run/stage/"):
        rows.append(
            (
                rec.name.removeprefix("run/stage/"),
                rec.extra.get("rows", ""),
                rec.extra.get("status", ""),
                rec.n,
                rec.extra.get("notes", ""),
            )
        )
    return ["## Stages", "", *table(["Stage", "Budget row", "Status", "Records", "Notes"], rows)]


def _cells(ix: Index) -> list[tuple[str, str, ResultRecord]]:
    out = []
    for rec in ix.records:
        if rec.name.endswith(".cell") and "arm" in rec.extra:
            out.append((str(rec.extra["arm"]), str(rec.extra["condition"]), rec))
    return out


def _gap_a_text(ga: ResultRecord | None) -> str:
    """Gap_A as the table prints it; a cell with no observed failure prints its reason and the
    Manski bounds, never a number (0003 A18, F1)."""
    if ga is None:
        return DASH
    if ga.extra.get("undefined"):
        lo, hi = float(ga.extra["manski_lo"]), float(ga.extra["manski_hi"])
        return (
            f"undefined: no life ended in an observed failure ({int(ga.extra['uncensored'])} "
            f"uncensored labels); Manski bounds {num(lo, 3)} to {num(hi, 3)}"
        )
    return num(ga.value)


def _perception_section(ix: Index) -> list[str]:
    out = ["## Perception: open loop against closed loop", ""]
    out += [
        "Oracle coverage of the scheduling interval (nominal 0.90) on post-blind checkpoints. "
        "Open loop: the first lives of each seed with nobody acting. Closed loop: the bound rule "
        "B3 acting on the interval. `Gap_B` is the open-loop value minus the closed-loop value, "
        "paired on the seed; the pooled stratum is every post-blind checkpoint, the trend stratum "
        "the checkpoints with a trend estimate (the policy selects these, so the open and closed "
        "populations differ). `Gap_A` is learner minus oracle coverage in the closed loop. It is "
        "undefined for a cell in which no life ended in an observed failure: a policy that "
        "prevents failures leaves its learner with no way to confirm an upper bound, and the cell "
        "shows the Manski bounds instead of a number.",
        "",
    ]
    variants = []
    for rec in ix.prefixed("open."):
        v = rec.name.split(".")[1]
        if v not in variants:
            variants.append(v)
    rows = []
    for v in variants:
        for cond in ("nominal", "shifted"):
            o = ix.get(f"open.{v}.{cond}.m_cov")
            c = ix.get(f"B3-{v}.{cond}.m_cov")
            gb = ix.get(f"gapb.{cond}.pooled.{v}")
            gt = ix.get(f"gapb.{cond}.trend.{v}")
            ga = ix.get(f"gapa.{cond}.pooled.{v}")
            if o is None and c is None:
                continue
            rows.append(
                (v, cond, value_ci(o), value_ci(c), value_ci(gb), value_ci(gt),
                 _gap_a_text(ga), k_over_n(c), fid(c), on_text(c))
            )  # fmt: skip
    out += table(
        ["Variant", "Condition", "Open loop", "Closed loop", "Gap_B pooled", "Gap_B trend",
         "Gap_A pooled", "n (closed)", "Fidelity", "Closed loop measured on"],
        rows,
    )  # fmt: skip
    return out


def _metric_of(ix: Index, arm: str, cond: str) -> Callable[[str], ResultRecord | None]:
    """``g(metric)``: the record of a cell's metric."""
    return lambda m: ix.get(f"{arm}.{cond}.{m}")


def _arm_rows(ix: Index, names: Sequence[str]) -> list[Sequence[object]]:
    rows: list[Sequence[object]] = []
    for arm, cond, cell in _cells(ix):
        if names and arm not in names:
            continue
        g = _metric_of(ix, arm, cond)
        rows.append(
            (arm, cond, value_ci(g("m_cost")), value_ci(g("u_e")), value_ci(g("throughput")),
             value_ci(g("m_cov")), value_ci(g("wrong_mode_jobs")), cell.extra.get("failed", 0),
             k_over_n(cell), fid(cell), on_text(cell))
        )  # fmt: skip
    return rows


def _cost_section(ix: Index) -> list[str]:
    out = ["## Closed-loop cells: cost, failures, throughput, coverage", ""]
    out += [
        "One row per cell. Values are means over the episodes with a confidence sequence at 0.05 "
        "(descriptive); `M-COST` is cost capped at 200 and divided by 200, `u_e` is unplanned "
        "failures per episode. Values are shown to four decimals; `results.json` holds them in "
        "full. A failed or timed-out episode is counted at its worst value and listed below.",
        "",
    ]
    out += table(
        ["Arm", "Condition", "M-COST", "u_e", "Throughput", "M-COV c_e", "Wrong-mode jobs",
         "Failed", "n", "Fidelity", "Measured on"],
        _arm_rows(ix, ()),
    )  # fmt: skip
    return out


def _safety_section(ix: Index) -> list[str]:
    out = ["## Violations and what gets through", ""]
    out += [
        "`M-VIOL` is the share of episodes whose minimum robustness of a property is negative "
        "(P1-P3, then P4-P5), shown as `k/n`. Each violation is attributed (0003 A5): "
        "`assumption_held` (a defect in a shielded arm), or the first broken assumption "
        "(`C1`, `A_phys`, `A_exec`), a contract failure that the proof does not cover. "
        "`Disagree` counts episodes where the plant's own events and the specification differ.",
        "",
    ]
    rows = []
    for arm, cond, cell in _cells(ix):
        g = _metric_of(ix, arm, cond)
        if g("m_viol_p15") is None:
            continue
        zero = g("m_viol_p13")
        zero_text = ""
        if zero is not None and zero.extra.get("k") == 0:
            zero_text = f"upper bound {num(float(zero.extra['zero_event_upper']))}"
        rows.append(
            (arm, cond, k_over_n(g("m_viol_p13")), zero_text, k_over_n(g("m_viol_p45")),
             k_over_n(g("m_defect")), k_over_n(g("m_attr_c1")), k_over_n(g("m_attr_a_phys")),
             k_over_n(g("m_attr_a_exec")), k_over_n(g("m_disagree")), value_ci(g("override_rate")),
             fid(cell), on_text(cell))
        )  # fmt: skip
    out += table(
        ["Arm", "Condition", "P1-P3", "Zero-event bound", "P4-P5", "assumption_held", "C1",
         "A_phys", "A_exec", "Disagree", "Overrides per 1,000", "Fidelity", "Measured on"],
        rows,
    )  # fmt: skip
    out += _uncovered_block(ix)
    return out


UNCOVERED_TEXT = (
    "The shield's proof assumes that every command it issues is in the set its table allows. At a "
    "decision where that set is empty the runtime has nothing safe to issue and falls back to the "
    "first command in the table's priority order; for the pump pair that is `STOP|STOP`, so both "
    "pumps stay stopped each quarter and a repair cannot progress in that episode. An empty set "
    "arises only after the abstract model's assumptions are already violated. The fallback is a "
    "known limit of the shield and is left as it is; this table counts how often it was used, as "
    "`k/n` episodes with any such decision and the total number of decisions. A cell with `0/n` "
    "did not need it."
)


def uncovered_rows(ix: Index) -> list[tuple[str, ...]]:
    """One row per shielded cell: arm, condition, episodes with any, decisions in all."""
    rows: list[tuple[str, ...]] = []
    for rec in ix.prefixed("shield.uncovered."):
        arm, _, cond = rec.name.removeprefix("shield.uncovered.").rpartition(".")
        rows.append((arm, cond, k_over_n(rec), str(int(rec.extra.get("total", 0))), fid(rec)))
    return rows


def _uncovered_block(ix: Index) -> list[str]:
    rows = uncovered_rows(ix)
    if not rows:
        return []
    out = ["", "### Shield decisions with no allowed command", "", UNCOVERED_TEXT, ""]
    out += table(["Arm", "Condition", "Episodes with any", "Decisions", "Fidelity"], rows)
    return out


def _value_or_dash(rec: ResultRecord | None) -> str:
    return DASH if rec is None else num(rec.value)


def _contract_section(ix: Index) -> list[str]:
    out = ["## The perception contract: per-life any-miss, both readings", ""]
    out += [
        "The share of lives with at least one checkpoint outside the set the shield was given, "
        "scored on the realised RUL (the estimator's truth) and on the health bin (the "
        "abstraction's `h`); a gap between them is a finding (0003 A10). `step` is the share of "
        "checkpoints outside the set. Wilson intervals at 95%.",
        "",
    ]
    rows = []
    for rec in ix.prefixed("contract."):
        if not rec.name.endswith(".any_miss_rul"):
            continue
        base = rec.name.removesuffix(".any_miss_rul")
        arm_cond = base.removeprefix("contract.")
        arm, _, cond = arm_cond.rpartition(".")
        g = ix.get
        rows.append(
            (arm, cond, rec.extra.get("source", ""), value_ci(rec), k_over_n(rec),
             value_ci(g(f"{base}.any_miss_bin")), k_over_n(g(f"{base}.any_miss_bin")),
             _value_or_dash(g(f"{base}.step_miss_bin")),
             _value_or_dash(g(f"{base}.mean_width_bins")),
             fid(rec), on_text(rec))
        )  # fmt: skip
    out += table(
        ["Arm", "Condition", "Set", "Any-miss, RUL", "k/n", "Any-miss, bin", "k/n",
         "Step miss, bin", "Mean width (bins)", "Fidelity", "Measured on"],
        rows,
    )  # fmt: skip
    return out


def _latency_section(ix: Index) -> list[str]:
    out = ["## Online monitors in the loop: quarters from a violation to its first alarm", ""]
    out += [
        "For the arms with the hand-written interlock, whose triggers read the streaming monitors "
        "(0003 A4). `not noticed` counts violations whose online robustness stayed positive.",
        "",
    ]
    arms: dict[tuple[str, str, str], dict[str, ResultRecord]] = {}
    for rec in ix.prefixed("latency."):
        arm, cond, prop, label = rec.name.removeprefix("latency.").rsplit(".", 3)[-4:]
        arms.setdefault((arm, cond, prop), {})[label] = rec
    labels = ("0", "1", "2-4", "5+", "never")  # the keys of metrics.LAG_BINS
    shown = {"never": "not noticed"}
    rows = []
    for (arm, cond, prop), bins in arms.items():
        any_rec = next(iter(bins.values()))
        rows.append(
            (arm, cond, prop, any_rec.n, *(int(bins[b].value) if b in bins else 0 for b in labels),
             fid(any_rec), on_text(any_rec))
        )  # fmt: skip
    out += table(
        [
            "Arm",
            "Condition",
            "Property",
            "Violations",
            *(shown.get(b, b) for b in labels),
            "Fidelity",
            "Measured on",
        ],
        rows,
    )
    return out


def _ablation_section(ix: Index) -> list[str]:
    out = ["## Ablations", ""]
    out += [
        "Paired differences (first arm minus the reference) on the same seeds, with confidence "
        "sequences at 0.05. They are descriptive: no decision rule is attached.",
        "",
    ]
    rows = []
    for rec in ix.prefixed("ablation."):
        ex = rec.extra
        rows.append(
            (ex.get("label", ""), f"{ex.get('arm', '')} minus {ex.get('reference', '')}",
             ex.get("condition", ""), ex.get("metric", ""), value_ci(rec), rec.n, fid(rec),
             on_text(rec))
        )  # fmt: skip
    out += table(
        ["Ablation", "Arms", "Condition", "Metric", "Difference", "n", "Fidelity", "Measured on"],
        rows,
    )
    return out


def _gas_section(ix: Index) -> list[str]:
    """The two secondary rows of 0003 A18 F5, per arm, from the trace; no decision rule."""
    out = ["## The gas clock: the job stand and the late tests (secondary)", ""]
    out += [
        "The abstraction reads the gas clock before the command and the plant and the monitor read "
        "it after; both are kept, and these two rows publish what that costs in the plant. "
        "`Crusher quarters in WORK per job` is the mean over episodes of the quarters CR-501 "
        "stands in WORK divided by the crusher jobs that completed (an episode with none is left "
        "out; `n` is the episodes counted). `Late gas tests` is, per episode, the rows where "
        "someone is inside at both the previous quarter and this one, the recorded gas age a "
        "quarter ago was at least the re-test period and is below it now: a test that came a "
        "quarter late with someone inside. It should be 0 in a shielded arm. Neither row decides "
        "a claim.",
        "",
    ]
    rows = []
    for arm, cond, cell in _cells(ix):
        w, late = ix.get(f"{arm}.{cond}.m_crusher_work_q"), ix.get(f"{arm}.{cond}.m_late_gas_tests")
        if w is None and late is None:
            continue
        rows.append(
            (arm, cond, value_ci(w), w.n if w is not None else DASH, value_ci(late),
             late.n if late is not None else DASH, fid(cell))
        )  # fmt: skip
    out += table(
        ["Arm", "Condition", "Crusher quarters in WORK per job", "n", "Late gas tests", "n",
         "Fidelity"],
        rows,
    )  # fmt: skip
    return out


def _layer_section(ix: Index) -> list[str]:
    out = ["## Every layer, open loop against closed loop", ""]
    out += [
        "The open-loop figure is what the layer achieves on its own test; the closed-loop figure "
        "is what it achieves inside the loop, where its decisions change what it sees next. A gap "
        "is a finding, not an error. The two figures of a row measure different things and are "
        "not subtracted.",
        "",
    ]
    primary = ix.param("primary_variant", "V3")
    rows: list[Sequence[object]] = []

    def add(
        layer: str,
        open_what: str,
        opened: ResultRecord | None,
        closed_what: str,
        closed: ResultRecord | None,
    ) -> None:
        if opened is None and closed is None:
            return
        rows.append(
            (
                layer,
                open_what,
                value_ci(opened),
                k_over_n(opened),
                fid(opened),
                closed_what,
                value_ci(closed),
                k_over_n(closed),
                fid(closed),
                on_text(opened),
                on_text(closed),
            )
        )

    for variant in ("V1", primary):
        add(
            "Perception",
            f"oracle coverage of the 90% interval, {variant}, no policy",
            ix.get(f"open.{variant}.nominal.m_cov"),
            f"oracle coverage, B3 with {variant}",
            ix.get(f"B3-{variant}.nominal.m_cov"),
        )
    for setting in ("reversed", "unreversed"):
        for diagnoser, name, arm in (
            ("scm", "structural", "P1"),
            ("corr", "correlational", "P1-corr"),
        ):
            add(
                "Diagnosis",
                f"M-DIAG accuracy, {name} diagnoser, {setting} correlation",
                ix.get(f"M-DIAG.{diagnoser}.{setting}"),
                f"wrong-mode jobs per episode, {arm}",
                ix.get(f"{arm}.nominal.wrong_mode_jobs"),
            )
    add(
        "Planning",
        "beliefs with planner value within 5% of exact DP",
        ix.get("exact_dp.value_within_5pct"),
        "M-COST, P0 (planner, no shield)",
        ix.get("P0.nominal.m_cost"),
    )
    add(
        "Planning",
        "exact-DP best action agreed with, decisive beliefs",
        ix.get("exact_dp.action_agreement"),
        f"M-COST, B3 with {primary} (the rule)",
        ix.get(f"B3-{primary}.nominal.m_cost"),
    )
    for game in ("crusher", "pumps", "conveyor"):
        add(
            "Shield",
            f"checker findings, {game} game, width 4",
            ix.get(f"F0.{game}.w4.findings"),
            "M-VIOL P1-P3, H-C1 (Careless behind the shield)",
            ix.get("H-C1.nominal.m_viol_p13"),
        )
    add(
        "Shield",
        "ticks of Careless traces the games cannot produce",
        ix.get("conformance.unmatched"),
        "M-VIOL P1-P5, H-C1",
        ix.get("H-C1.nominal.m_viol_p15"),
    )
    out += table(
        [
            "Layer",
            "Open loop quantity",
            "Open loop",
            "k/n",
            "Fidelity",
            "Closed loop quantity",
            "Closed loop",
            "k/n",
            "Fidelity",
            "Open loop measured on",
            "Closed loop measured on",
        ],
        rows,
    )
    return out


def _monitor_section(ix: Index) -> list[str]:
    out = ["## Monitors", ""]
    rows = []
    for mode in ("opt", "pess", "inf", "proxy"):
        fa = ix.get(f"claim/C-MON/false_alarm/{mode}")
        det = ix.get(f"claim/C-MON/detection/{mode}")
        if fa is None and det is None:
            continue
        rows.append(
            (mode, k_over_n(fa), num(fa.ci_lo) if fa else DASH, num(fa.ci_hi) if fa else DASH,
             k_over_n(det), num(det.ci_lo) if det else DASH, num(det.ci_hi) if det else DASH,
             fid(fa), on_text(fa))
        )  # fmt: skip
    out += [
        "False alarms on unshifted episodes and detection within 30 days of the shift, from the "
        "planner arm P1 with the shield on (the monitors run passively). The chosen mode is "
        f"`{ix.param('monitor_mode')}`, at the claim level; the other three are secondary rows at "
        "the descriptive level of 0.05. A win needs the upper bound of the false-alarm rate at "
        "most 0.02 and the lower bound of detection at least 0.5; a loss needs the lower bound "
        "of the false-alarm rate above 0.02 or the upper bound of detection below 0.1.",
        "",
        *table(["Mode", "False alarms", "Lower bound", "Upper bound", "Detected", "Lower bound",
                "Upper bound", "Fidelity", "Measured on"], rows),
    ]  # fmt: skip
    return out


def _causal_section(ix: Index) -> list[str]:
    out = ["## Causal study", ""]
    rows = []
    for rec in ix.records:
        if rec.name.startswith(("X-", "M-DIAG.")):
            rows.append((rec.name, value_ci(rec), k_over_n(rec), fid(rec), on_text(rec)))
    out += table(["Record", "Value", "n", "Fidelity", "Measured on"], rows)
    return out


def _conformance_section(ix: Index) -> list[str]:
    out = ["## Conformance and exact dynamic programming", ""]
    rows = []
    for rec in ix.records:
        if rec.name.startswith(("conformance.", "exact_dp.")):
            rows.append((rec.name, value_ci(rec), k_over_n(rec), fid(rec), on_text(rec)))
    out += table(["Record", "Value", "k/n", "Fidelity", "Measured on"], rows)
    return out


def _failure_rows(ix: Index) -> list[tuple[object, ...]]:
    """Every episode or run that did not produce a result, whatever stage ran it: the F1 cells, the
    F2 arms (the first failure of an arm names its seed and reason), the search runs, and the F2
    replays that errored or timed out. Each row says where, how many and which (P12g, G3: the
    section used to read the F1 cells only and said "none" over a failed F2 episode)."""
    rows: list[tuple[object, ...]] = []
    for arm, cond, cell in _cells(ix):
        if int(cell.extra.get("failed", 0)):
            rows.append(
                (
                    "F1 cell",
                    f"{arm}, {cond}",
                    cell.extra.get("failed", 0),
                    cell.extra.get("failures", ""),
                )
            )
    for arm_rec in ix.prefixed("claim/C-TOGETHER/arm/"):
        tried = int(arm_rec.extra.get("tried", 0))
        done = int(arm_rec.extra.get("completed", 0))
        if done < tried:
            name = arm_rec.name.rsplit("/", 1)[1]
            rows.append(("F2 arm", name, tried - done, arm_rec.extra.get("cause", "")))
    for run_rec in ix.records:
        if run_rec.name.startswith("search.") and run_rec.name.endswith(".find_rate"):
            if int(run_rec.extra.get("failed", 0)):
                where = run_rec.name.removeprefix("search.").removesuffix(".find_rate")
                rows.append(("search runs", where, run_rec.extra["failed"], run_rec.measured_on))
    for kind in ("replay_error", "replay_timeout"):
        replay = ix.get(f"f2.replay.reason.{kind}")
        if replay is not None and int(replay.extra.get("k", 0)):
            rows.append(("F2 replays", kind, replay.extra["k"], replay.measured_on))
    return rows


def _failures_section(ix: Index) -> list[str]:
    rows = _failure_rows(ix)
    out = ["## Failed and timed-out episodes", ""]
    if rows:
        out += table(["Stage", "Cell", "Failed", "Which (seed and reason)"], rows)
    else:
        out += [
            "None: every episode of every cell, every F2 episode and every search run produced "
            "a result.",
            "",
        ]
    return out


def _provisional_section(ix: Index) -> list[str]:
    names: dict[str, set[str]] = {}
    sources: set[str] = set()
    for arm, cond, _cell in _cells(ix):
        rec = ix.get(f"{arm}.{cond}.throughput")
        if rec is not None and rec.extra.get("provisional"):
            for n in str(rec.extra["provisional"]).split(","):
                names.setdefault(n, set()).add(arm)
        if rec is not None and rec.extra.get("tuning"):
            sources.add(str(rec.extra["tuning"]))
    out = ["## Tuned and provisional values", ""]
    out.append(f"Tuned values read from: {', '.join(sorted(sources)) or DASH}.")
    out.append("")
    if names:
        out += table(
            ["Value", "Provisional for arms"],
            [(n, ", ".join(sorted(a))) for n, a in sorted(names.items())],
        )
    else:
        out += ["No value used by any arm is provisional.", ""]
    return out


def _history_section(ix: Index) -> list[str]:
    heads = ix.prefixed("run/headline/")
    out = ["## Runs", ""]
    runs = ix.get("run/history")
    out.append(
        f"Full runs recorded in the ledger before this one: "
        f"{int(runs.value) if runs is not None else 0}. The first run's numbers stay in the "
        "report next to a rerun's (prereg section 10)."
    )
    out.append("")
    if heads:
        keys = sorted({r.name.split("/", 3)[3] for r in heads})
        order = sorted({int(r.name.split("/")[2]) for r in heads})
        rows = []
        for key in keys:
            cells = []
            for i in order:
                rec = ix.get(f"run/headline/{i}/{key}")
                cells.append(DASH if rec is None else f"{rec.extra.get('text', num(rec.value))}")
            rows.append((key, *cells))
        out += table(["Headline", *(f"run {i + 1}" for i in order)], rows)
    return out


_LIMITS: Final[tuple[str, ...]] = (
    "Everything here is synthetic data from a simulated plant, built by one author.",
    "The F1 plant shares its structure with the estimator and the planner's world model.",
    "F1 timings and several plant parameters are assumptions, not measurements of a real plant.",
    "The abstraction is a proof about a model; the conformance gap above is a measured, "
    "non-zero distance between that model and the simulated plant on sampled traces.",
    "F2 samples are small, so F2 can separate only very large effects.",
    "The falsification search scores spec robustness (`evaluate_theta`), so it cannot find an "
    "event that only the plant records; plant-only events are counted in M-VIOL and shown apart in "
    "C-SHIELD, but no search looks for them.",
)


def _ledger_lines(ix: Index) -> list[str]:
    """What the ledger says about this run (a full run only): its number, the held-out seeds handed
    out so far, and whether the code and the tuned values that ran are the frozen ones."""
    facts = ix.get("run/ledger")
    if facts is None:
        return []
    ex = facts.extra
    code = "is the frozen code" if ex.get("code_equals_frozen") else "is NOT the frozen code"
    tuned = "is the frozen file" if ex.get("tuned_equals_frozen") else "is NOT the frozen file"
    return [
        f"Ledger: this is full run {ex.get('run_number', DASH)}; held-out seeds handed out so far "
        f"(counting every use, search seeds included): {ex.get('heldout_evaluations', DASH)}. "
        f"The running code {code}; `tuned.json` {tuned}.",
        "",
    ]


def render_results(records: Sequence[ResultRecord]) -> str:
    """The text of ``RESULTS.md``, from the records and nothing else."""
    ix = Index(records)
    verdicts = verdicts_from_stored(records)
    out = ["# Results (SYNTHETIC)", ""]
    out += _banner(ix)
    out += [
        "Generated by `benchmarks/run.py` from `benchmarks/results.json`; nothing here is typed by "
        "hand, and `python benchmarks/run.py --check` regenerates this file and compares it.",
        "",
        f"Mode: {ix.param('mode')}. Seeds: {ix.param('seed_role')} "
        f"({ix.param('n_seeds')} drawn). Pre-registration hash: "
        f"{ix.param('prereg_sha256') or DASH}. "
        f"Primary perception variant: {ix.param('primary_variant')}. Monitor mode chosen: "
        f"{ix.param('monitor_mode')}.",
        "",
    ]
    out += _ledger_lines(ix)
    out += _verdict_section(verdicts)
    out += _stage_section(ix)
    out += _layer_section(ix)
    out += _perception_section(ix)
    out += _cost_section(ix)
    out += _safety_section(ix)
    out += _contract_section(ix)
    out += _latency_section(ix)
    out += _monitor_section(ix)
    out += _ablation_section(ix)
    out += _gas_section(ix)
    out += _causal_section(ix)
    out += _conformance_section(ix)
    out += _failures_section(ix)
    out += _provisional_section(ix)
    out += _history_section(ix)
    out += ["## Limitations carried into every report", ""]
    out += [f"- {x}" for x in _LIMITS]
    out += ["", "Proof scope: see `docs/PROOF-SCOPE.md`.", ""]
    return "\n".join(out).rstrip("\n") + "\n"


# ----------------------------------------------------------------------------- PROOF-SCOPE.md


def _games(ix: Index) -> list[str]:
    seen: list[str] = []
    for rec in ix.prefixed("F0."):
        parts = rec.name.split(".")
        if len(parts) >= 3 and parts[2] == "states_raw" and parts[1] not in seen:
            seen.append(parts[1])
    return seen


def _component_list(ix: Index) -> str:
    games = [g for g in _games(ix) if not g.endswith("_stale")]
    return ", ".join(games) if games else "none: the F0 stage did not run"


def _widths_checked(ix: Index) -> str:
    widths = sorted(
        {
            int(rec.name.split(".")[2][1:])
            for rec in ix.prefixed("F0.")
            if len(rec.name.split(".")) >= 4 and rec.name.split(".")[2][:1] == "w"
            and rec.name.split(".")[2][1:].isdigit()
        }
    )  # fmt: skip
    return "(none)" if not widths else ", ".join(f"{w}" for w in widths) + " bins at most"


def render_proof_scope(records: Sequence[ResultRecord]) -> str:
    """The text of ``docs/PROOF-SCOPE.md``: 0001 section 4.6 with the measured counts."""
    ix = Index(records)
    out = ["# What the proof says (SYNTHETIC)", ""]
    if ix.param("mode") == "quick":
        out += [
            f"> **{QUICK_BANNER}.** Counts below come from a short run on development seeds.",
            "",
        ]
    out += [
        "Generated by `benchmarks/run.py` from `benchmarks/results.json`. The numbers are counts "
        "of a finite abstract model, found by exhaustive enumeration, and the conformance figures "
        "are measured on sampled traces of the simulated plant.",
        "",
        "## Statement",
        "",
        "Under A_iso, A_exec, A_phys, A_map and C1, if every command issued to the components "
        f"that have a game in this run ({_component_list(ix)}) is in the shield's allowed set, "
        "then no state that violates P1 to P5 is reachable in the abstract models, from their "
        "initial states, "
        "for any environment behaviour the models allow. It is checked by explicit enumeration of "
        "the reachable product of game state, hidden health bin and perception set. It is a "
        "statement about the abstract models. It is not a statement about the F1 or F2 "
        "simulators.",
        "",
        "## Counts",
        "",
    ]
    games = _games(ix)
    rows = []
    for g in games:
        raw = ix.get(f"F0.{g}.states_raw")
        win = ix.get(f"F0.{g}.states_winning")
        obs = ix.get(f"F0.{g}.states_observable")
        mean = ix.get(f"F0.{g}.mean_safe_commands")
        diff = ix.get(f"F0.{g}.differential_mismatches")
        w4 = ix.get(f"F0.{g}.w4.findings")
        reach = ix.get(f"F0.{g}.w4.reachable_states")
        prod = ix.get(f"F0.{g}.w4.reachable_product")
        rows.append(
            (g, k_over_n(win), k_over_n(obs), num(mean.value, 2) if mean else DASH,
             int(raw.extra.get("commands", 0)) if raw else DASH, k_over_n(reach),
             prod.n if prod else DASH, k_over_n(w4), k_over_n(diff))
        )  # fmt: skip
    out += table(
        ["Game", "Winning of raw states", "Observable winning of observable", "Mean safe commands",
         "Commands", "Reachable of raw states (width 4)", "Reachable product states (width 4)",
         "Checker findings (width 4)", "Differential mismatches"],
        rows,
    )  # fmt: skip
    out += [
        "The raw count can flatter the proof: a raw state that no run of the shielded product "
        "reaches proves nothing about the plant. The reachable count is printed beside it for "
        "that reason.",
        "",
        f"Perception sets of width {_widths_checked(ix)} were checked; the table of each width and "
        "the number of table entries that admit no command are in `results.json` "
        "(`F0.<game>.w<width>.*`).",
        "",
    ]
    sab = ix.get("F0.sabotage_suite")
    if sab is not None:
        out += [
            f"Sabotage suite (each mutation of a model or table must turn the checker or the "
            f"differential red): {k_over_n(sab)} tests passed. {sab.extra.get('summary', '')}",
            "",
        ]
    stale = ix.get("F0.crusher_stale.states_winning")
    if stale is not None:
        out += [
            f"The stale-specification game (re-test period of two hours) has {k_over_n(stale)} "
            "states in its winning region; the arm that uses it is scored against the current "
            "specification.",
            "",
        ]
    out += [
        "## Assumptions the statement needs",
        "",
        *table(
            ["ID", "Assumption", "How it is tested"],
            [
                (
                    "A_iso",
                    "isolation holds once proven (no energy release while isolated)",
                    "not testable here; stated",
                ),
                (
                    "A_exec",
                    "STOP and EXIT are executed; every other command may be refused",
                    "plant code and unit test; conformance trace check",
                ),
                (
                    "A_phys",
                    "a health bin rises by at most one per tick; an uncontrolled failure "
                    "occurs only from the top bin while running",
                    "measured on sampled traces (below)",
                ),
                (
                    "A_dec",
                    "components interact only through the properties of the pump pair, so "
                    "the game decomposes",
                    "argued; a unit test on the atoms of the specification",
                ),
                (
                    "C1",
                    "the perception set holds the true health bin at every tick",
                    "monitored by an e-process whose unit is the life (below)",
                ),
                (
                    "A_map",
                    "the abstraction map and the predicates are right",
                    "conformance test; an independent reimplementation of the relation",
                ),
            ],
        ),
        "C1 holds at every tick, which is an event of a whole life. The set the shield uses by "
        "default is the per-life 95% band (`life95`); a per-step 99% set is an ablation and does "
        "not deliver C1 as stated. The sequential unit of the monitor modes that read labels "
        "(`opt`, `pess` and `inf`) is the life, and their test assumes the lives are independent "
        "given the past (a conditional-mean assumption); checkpoints within a life are dependent, "
        "so a per-checkpoint test would not satisfy the condition Ville's inequality needs. The "
        "unit of mode `proxy` is the day (the miss indicator of a one-day-ahead band), it needs no "
        "label and it is not a test of C1; its false-alarm rate is measured on the unshifted "
        "episodes in `RESULTS.md`, and the test does not bound it.",
        "",
        "## Measured gap between the models and the plant",
        "",
    ]
    conf_rows = []
    for name in ("conformance.unmatched", "conformance.unmatched_when_safe",
                 "conformance.missed_violations", "conformance.a_phys"):  # fmt: skip
        rec = ix.get(name)
        if rec is not None:
            conf_rows.append((name, k_over_n(rec), value_ci(rec), fid(rec), on_text(rec)))
    out += table(
        ["Quantity", "k/n", "Value (95% Wilson interval)", "Fidelity", "Measured on"], conf_rows
    )
    out += [
        "These are measured rates on sampled traces, not a proof. A count of zero is reported "
        "with its denominator.",
        "",
        "## What the statement does not cover",
        "",
        "- Violations the abstraction cannot express: a mistaken zero-energy test, an unmodelled "
        "hazard class, two simultaneous entries.",
        "- Anything while C1 fails, except under the fallback: when the monitor fires, the sets "
        "become the whole range and C1 holds by construction, and the statement then stands "
        "conditional on the other assumptions.",
        "- Concrete-plant behaviour with no abstract counterpart (the measured gap above), and "
        "the 15-minute quantisation.",
        "- A wrong specification: the specification is what this repository wrote, not what a "
        "regulator wrote. The permit suspension on an emergency alarm (clause 6.2 of the permit "
        "document) is a clause the specification does not cover.",
        "- The game is stricter than the properties: some commands are violations in the game that "
        "have no formula among P1 to P6, so the shield refuses more than the properties require.",
        "- The renewal of a component on completing a job is an environment choice in the "
        "models (a wrong-mode job completes and renews nothing).",
        "- The pump availability definition: a locked-out pump is not available.",
        "- A confined-space task of N quarters becomes N cycles of test, enter, exit; the "
        "throughput this costs is measured against the interlock in `RESULTS.md`.",
        "",
    ]
    return "\n".join(out).rstrip("\n") + "\n"


# ----------------------------------------------------------------------------- headline, I/O


def headline_of(records: Sequence[ResultRecord]) -> dict[str, Any]:
    """The values the ledger entry records: every verdict, completeness, and the digest of the
    records. Numbers are copied from records unchanged."""
    out: dict[str, Any] = {}
    for rec in records:
        if rec.name.startswith("verdict/"):
            label = str(rec.extra["label"])
            qual = str(rec.extra.get("qualifier", ""))
            out[rec.name] = label + (f", {qual}" if qual else "")
            # the numbers behind the label, so that a rerun can be compared with the first run on
            # more than a word
            out[rec.name.replace("verdict/", "numbers/", 1)] = str(rec.extra["headline"])
    comp = next((r for r in records if r.name == "run/completeness"), None)
    if comp is not None:
        out["complete"] = bool(comp.value)
    return out


def absolutes_in(body: str, records: Sequence[ResultRecord]) -> list[str]:
    """The absolutes ("never", "always", "100%", ...) in ``body`` that are the report's own words:
    the free text of ``_DATA_KEYS`` is taken out before the search."""
    text = body
    data = {
        str(v)
        for r in records
        for k, v in r.extra.items()
        if k in _DATA_KEYS and isinstance(v, str) and len(v) >= 4
    }
    for d in sorted(data, key=len, reverse=True):
        text = text.replace(d, " ").replace(esc(d), " ")
    return [m.group(0) for m in FORBIDDEN.finditer(text)]


def write_outputs(
    records: Sequence[ResultRecord], results_json: Path, results_md: Path, proof_scope_md: Path
) -> str:
    """Write the three files (UTF-8, LF) and return the SHA-256 of ``results.json``."""
    text = records_to_json(records)
    documents = (
        (results_json, text),
        (results_md, render_results(records)),
        (proof_scope_md, render_proof_scope(records)),
    )
    for path, body in documents[1:]:
        found = absolutes_in(body, records)
        if found:  # refuse before writing anything: a run must not publish what --check rejects
            raise ValueError(f"{path.name} would contain the absolute {found[0]!r}")
    for path, body in documents:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body.encode("utf-8"))
    return results_sha256(text)


def check_outputs(
    results_json: Path,
    results_md: Path,
    proof_scope_md: Path,
    *,
    expected_sha256: str | None = None,
) -> list[str]:
    """Problems found in the published files; an empty list when they agree.

    Checks that ``results.json`` parses, that both documents are exactly what it generates, that
    the verdicts rebuilt from its claim-level records equal the stored ones, that no generated text
    uses an absolute, and, if given, that its digest is the one the ledger recorded.
    """
    problems: list[str] = []
    try:
        text = results_json.read_bytes().decode("utf-8")
        records = records_from_json(text)
    except (OSError, ValueError, KeyError) as exc:
        return [f"results.json cannot be read: {exc}"]
    digest = results_sha256(text)
    if expected_sha256 is not None and digest != expected_sha256:
        problems.append(
            f"results.json digest {digest[:16]}... differs from the ledger's "
            f"{expected_sha256[:16]}...: "
            "the file was changed after the run"
        )
    for label, path, generate in (
        ("RESULTS.md", results_md, render_results),
        ("PROOF-SCOPE.md", proof_scope_md, render_proof_scope),
    ):
        try:
            on_disk = path.read_bytes().decode("utf-8")
        except OSError as exc:
            problems.append(f"{label} cannot be read: {exc}")
            continue
        expected = generate(records)
        if on_disk != expected:
            problems.append(
                f"{label} is not what results.json generates "
                f"({_first_difference(on_disk, expected)})"
            )
        for word in absolutes_in(expected, records):
            problems.append(f"{label} would contain the absolute {word!r}")
    stored = {v.claim: v for v in verdicts_from_stored(records)}
    fresh = {v.claim: v for v in verdicts_from_records(records)}
    if set(stored) != set(fresh):
        problems.append(
            f"verdicts stored {sorted(set(stored) ^ set(fresh))} differ from those the records give"
        )
    for claim in sorted(set(stored) & set(fresh)):
        a, b = stored[claim], fresh[claim]
        if (a.label, a.qualifier, a.headline) != (b.label, b.qualifier, b.headline):
            problems.append(
                f"verdict {claim} was edited: stored {a.text!r}, the records give {b.text!r}"
            )
    return problems


def _first_difference(a: str, b: str) -> str:
    la, lb = a.splitlines(), b.splitlines()
    for i, (x, y) in enumerate(zip(la, lb, strict=False)):
        if x != y:
            return f"first difference at line {i + 1}"
    return f"line counts differ: {len(la)} against {len(lb)}"
