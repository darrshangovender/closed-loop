"""Generate ``docs/WHAT-GETS-THROUGH.md`` from ``results.json`` and the cached search runs.

    python scripts/render_what_gets_through.py --dry-run              the quick output, no writes
    python scripts/render_what_gets_through.py --workers 3 --lock L   after the full run

Pre-registration section 8 fixes the content: every violation found by search beyond the
assumptions, shrunk to a minimal scenario and labelled with the broken assumption; the conformance
gap; the screener miss list; unconfirmed F2 replays with reason codes; the known gaps of the
abstraction. Each item comes from a record, from the repository's own tables, or from a re-run, and
none is typed here.

``results.json`` does not hold shrunk scenarios: a cached ``SearchOutcome`` keeps only the first
find of a run, ``(theta, properties)``. So the script reads the search cells with the runner's own
cache reader (``CellCache``; it needs no code hash, it reads the header the cell carries), takes
the first find of every run, **evaluates it again** on the shielded target with the tuned values the
cell recorded, **shrinks** it (``search.falsify.build_finding``) and hands the findings to
``search.falsify.what_gets_through``. A find that does not reproduce is listed as such and is not
shrunk. The cost is printed: one evaluation to reproduce a find, then the shrink's own calls and
one ablation per surviving dimension; a bound on the shrink is ``--max-calls``.

Cache. Outside ``--dry-run`` the finished finds are kept by the runner's cell cache under
``benchmarks/.cache/wgt/`` (one cell per regime and algorithm), so an interrupted run resumes. With
``--dry-run`` nothing is written: not the cache, not the document. ``--lock PATH`` takes the
heavy-run lock for the duration, as ``scripts/tune.py`` does.

The search tables come from the search seeds, not from held-out ones; the other tables read the
records of the run, which are on held-out seeds, and each says what it was measured on.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from closed_loop.core.results import ResultRecord  # noqa: E402
from closed_loop.eval.report import (  # noqa: E402
    QUICK_BANNER,
    UNCOVERED_TEXT,
    Index,
    k_over_n,
    num,
    records_from_json,
    render_proof_scope,
    table,
    uncovered_rows,
    value_ci,
)
from closed_loop.eval.runner import CellCache, arm_fingerprint, read_cell_header  # noqa: E402
from closed_loop.eval.stages import (  # noqa: E402
    SearchFind,
    SearchOutcome,
    search_hand_off,
)
from closed_loop.eval.tuned import TunedValues, use_tuned  # noqa: E402
from closed_loop.plant.notes import MALICIOUS_NOTES, screener_misses  # noqa: E402
from closed_loop.search import falsify  # noqa: E402
from closed_loop.stats.intervals import clopper_pearson  # noqa: E402

DEFAULT_RESULTS: Final[Path] = ROOT / "benchmarks" / "results.json"
DEFAULT_CELLS: Final[Path] = ROOT / "benchmarks" / ".cache" / "full"
DEFAULT_OUT: Final[Path] = ROOT / "docs" / "WHAT-GETS-THROUGH.md"
WGT_CACHE: Final[Path] = ROOT / "benchmarks" / ".cache" / "wgt"
SCRIPT: Final[str] = "scripts/render_what_gets_through.py"
ARMS_SHOWN: Final[tuple[str, ...]] = ("P0", "P1", "P1-I", "B3-V2L", "B3-S", "H-C0", "H-C1")
PLANT_ONLY_ARMS: Final[tuple[str, ...]] = (
    "H-C0", "H-G0", "P0", "H-C1", "H-G1", "P1", "P1-nopair", "B3-S", "H-C-I", "H-G-I", "P1-I",
)  # fmt: skip
PLANT_ONLY_GROUPS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("shield off", ("H-C0", "H-G0", "P0")),
    ("shield", ("H-C1", "H-G1", "P1", "P1-nopair", "B3-S")),
    ("interlock", ("H-C-I", "H-G-I", "P1-I")),
)


# ----------------------------------------------------------------------------- the search cells


@dataclass(frozen=True)
class SearchCell:
    """One cached search cell: a regime and an optimiser, with the runs the cache holds."""

    path: Path
    regime: str
    algo: str
    budget: int
    code: str
    tuned: dict[str, Any]
    outcomes: Mapping[int, SearchOutcome]
    failed: tuple[int, ...]


def read_search_cells(directory: Path, code: str | None = None) -> list[SearchCell]:
    """The search cells of a cache directory, one per (regime, algorithm).

    When a directory holds several cells for the same pair (the code changed between runs) the one
    for ``code`` is taken, or the most recently written. The cell is read with ``CellCache.load``,
    the reader the runner itself uses.
    """
    chosen: dict[tuple[str, str], tuple[float, SearchCell]] = {}
    for path in sorted(directory.glob("*.cell")):
        try:
            header = read_cell_header(path)
        except (ValueError, OSError, json.JSONDecodeError):
            continue
        task = header.get("task", {})
        if task.get("kind") != "search":
            continue
        env_code = str(header.get("env", {}).get("code", ""))
        if code is not None and env_code != code:
            continue
        stored = CellCache(
            directory, path.stem, {k: v for k, v in header.items() if k != "key"}
        ).load()
        if not stored:
            raise ValueError(f"{path.name}: the cell could not be read back with CellCache")
        outcomes = {s: o.value for s, o in stored.items() if hasattr(o, "value")}
        cell = SearchCell(
            path=path,
            regime=str(task["regime"]),
            algo=str(task["algo"]),
            budget=int(task["budget"]),
            code=env_code,
            tuned=dict(task["tuned"]),
            outcomes=outcomes,
            failed=tuple(sorted(s for s, o in stored.items() if not hasattr(o, "value"))),
        )
        key = (cell.regime, cell.algo)
        mtime = path.stat().st_mtime
        if key not in chosen or mtime > chosen[key][0]:
            chosen[key] = (mtime, cell)
    return [chosen[k][1] for k in sorted(chosen)]


def finds_of(cells: Sequence[SearchCell]) -> list[SearchFind]:
    """The first find of every run that found anything, in the order ``search_hand_off`` fixes."""
    runs = {(c.regime, c.algo): c.outcomes for c in cells}
    seeds = sorted({s for c in cells for s in c.outcomes})
    return search_hand_off(runs, seeds, limit=10**9)


def run_summaries(cells: Sequence[SearchCell]) -> list[dict[str, Any]]:
    """One line per run, found or not, in the shape of ``what_gets_through``'s ``runs``."""
    out = []
    for c in cells:
        for seed in sorted(c.outcomes):
            o = c.outcomes[seed]
            out.append(
                {
                    "arm": falsify.target_arm().name,
                    "regime": f"R_{c.regime}",
                    "algo": c.algo,
                    "seed": seed,
                    "evals": o.evals,
                    "found": o.first_violation_eval is not None,
                    "first_violation_eval": o.first_violation_eval,
                    "best_objective": None,
                    "violating_evaluations": o.violating,
                    "wall_s_per_eval": None,
                    "cpu_s_per_eval": None,
                }
            )
    return out


# ------------------------------------------------------------------- re-evaluate and shrink


@dataclass(frozen=True)
class ShrinkTask:
    """Re-evaluate and shrink the first find of each run of one cell (the seed is the run's seed).

    Picklable, so the runner can send it to a worker and key its cache on ``fingerprint``.
    """

    regime: str
    algo: str
    finds: tuple[tuple[int, tuple[float, ...], tuple[str, ...], int], ...]
    tuned: TunedValues
    max_calls: int | None = None
    tol: float = 0.02

    def label(self) -> str:
        return f"wgt/R_{self.regime}/{self.algo}"

    def fingerprint(self) -> dict[str, Any]:
        blob = json.dumps(self.finds, sort_keys=True, default=str).encode("utf-8")
        return {
            "kind": "wgt-shrink",
            "regime": self.regime,
            "algo": self.algo,
            "finds": hashlib.sha256(blob).hexdigest(),
            "target": arm_fingerprint(falsify.target_arm()),
            "tuned": dataclasses.asdict(self.tuned),
            "max_calls": self.max_calls,
            "tol": self.tol,
        }

    def __call__(self, seed: int) -> dict[str, Any]:
        _seed, theta, props, first_eval = next(f for f in self.finds if f[0] == seed)
        with use_tuned(self.tuned):
            return reproduce_and_shrink(
                seed,
                self.regime,
                self.algo,
                np.asarray(theta, dtype=np.float64),
                props,
                first_eval,
                self.max_calls,
                self.tol,
            )


def reproduce_and_shrink(
    seed: int,
    regime: str,
    algo: str,
    theta: NDArray[np.float64],
    props: Sequence[str],
    first_eval: int,
    max_calls: int | None,
    tol: float,
    evaluate: falsify.Runner | None = None,
) -> dict[str, Any]:
    """One find: evaluate it again, and if it still violates, shrink it and label what is left.

    Returns a JSON-able dict with the number of evaluations spent (``evals``: the re-evaluation and
    every evaluation the shrink and its ablations ran), whether the find ``reproduced`` and, if it
    did, the entries ``what_gets_through`` makes of the finding. ``evaluate`` replaces
    ``falsify.evaluate_theta`` in tests.
    """
    from closed_loop.search.random_search import SearchRun

    arm = falsify.target_arm()
    calls = [0]
    inner = evaluate if evaluate is not None else falsify.evaluate_theta

    def counting(a: Any, t: Any, r: str, s: int, c: Any) -> falsify.Evaluation:
        calls[0] += 1
        return inner(a, t, r, s, c)

    ev = counting(arm, theta, regime, seed, None)
    same = ev.objective < 0.0 and bool(set(ev.violated) & set(props))
    if not same:
        return {
            "seed": seed, "regime": regime, "algo": algo, "reproduced": False,
            "evals": calls[0], "recorded": list(props), "got": list(ev.violated),
            "entries": [],
        }  # fmt: skip
    ev = dataclasses.replace(ev, index=first_eval)
    run = SearchRun(
        algo=algo, regime=f"R_{regime}", seed=seed, best_f=ev.objective,
        first_violation_eval=first_eval, evals=first_eval, theta_best=ev.theta,
        history=np.array([ev.objective]),
    )  # fmt: skip
    result = falsify.FalsifyResult(
        run=run, arm=arm.name, shielded=arm.shield, regime=regime, violations=(ev,), seconds=0.0
    )
    finding = falsify.build_finding(
        result, ev, arm=arm, tol=tol, max_calls=max_calls, runner=counting
    )
    doc = falsify.what_gets_through([finding])
    return {
        "seed": seed, "regime": regime, "algo": algo, "reproduced": True, "evals": calls[0],
        "recorded": list(props), "got": list(ev.violated), "entries": doc["entries"],
        "shrink_calls": finding.shrink_calls, "truncated": finding.shrink_truncated,
    }  # fmt: skip


def shrink_all(
    cells: Sequence[SearchCell],
    runner: Any,
    *,
    limit: int | None,
    max_calls: int | None,
    log: Callable[[str], None],
) -> list[dict[str, Any]]:
    """Run ``ShrinkTask`` over every cell through ``runner`` (its cache gives the resume)."""
    finds = finds_of(cells)
    if limit is not None:
        finds = finds[:limit]
    out: list[dict[str, Any]] = []
    for cell in cells:
        mine = tuple(
            (f.seed, f.theta, f.props, cell.outcomes[f.seed].first_violation_eval or 0)
            for f in finds
            if (f.regime, f.algo) == (cell.regime, cell.algo)
        )
        if not mine:
            continue
        task = ShrinkTask(cell.regime, cell.algo, mine, TunedValues(**cell.tuned), max_calls)
        log(f"{task.label()}: {len(mine)} finds")
        run = runner.run_seeds(
            task,
            [m[0] for m in mine],
            batch=max(1, len(mine)),
            n_max=len(mine),
            standin=lambda failure: {
                "reproduced": False,
                "evals": 0,
                "failed": failure.kind,
                "entries": [],
            },  # fmt: skip
        )
        out.extend(run.values)
    return out


# ----------------------------------------------------------------------------- the document


def merge(
    results: Sequence[Mapping[str, Any]], runs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """The structured list: the entries of every reproduced find, grouped by minimal scenario."""
    docs = [{"entries": r["entries"], "runs": []} for r in results if r.get("reproduced")]
    docs.append({"entries": [], "runs": list(runs)})
    return falsify.merge_documents(docs)


def cost_of(results: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Evaluations spent: the re-evaluations (one per find) and the shrink and ablation calls."""
    finds = len(results)
    total = sum(int(r.get("evals", 0)) for r in results)
    return {
        "finds": finds,
        "reproduced": sum(1 for r in results if r.get("reproduced")),
        "evaluations": total,
        "reevaluations": finds,
        "shrink_and_ablation": total - finds,
    }


def _bullets_under(text: str, heading: str) -> list[str]:
    """The ``- `` lines under the heading ``## <heading>`` of a rendered document."""
    out: list[str] = []
    inside = False
    for line in text.splitlines():
        if line.startswith("## "):
            inside = line[3:].strip() == heading
            continue
        if inside and line.startswith("- "):
            out.append(line[2:].strip())
    return out


def _dimension_text(d: Mapping[str, Any]) -> str:
    flags = []
    if d.get("beyond_r_in"):
        flags.append(f"beyond R_in: {d['beyond_r_in']}")
    nec = d.get("necessary")
    flags.append("necessary" if nec else "not necessary" if nec is False else "not ablated")
    return f"{d['name']} = {d['decoded']} ({', '.join(flags)})"


def _event_text(e: Mapping[str, Any]) -> str:
    where = f" on {e['asset']}" if e.get("asset") else ""
    return f"{e['property']} at row {e['row']}{where}, {e['label']}"


def render_document(
    records: Sequence[ResultRecord],
    doc: Mapping[str, Any],
    cost: Mapping[str, int],
    results: Sequence[Mapping[str, Any]],
    cells: Sequence[SearchCell],
    *,
    dry_run: bool = False,
) -> str:
    ix = Index(records)
    out = ["# What still gets through (SYNTHETIC)", ""]
    out += [
        f"Generated by `{SCRIPT}` from `benchmarks/results.json` and the cached search runs; "
        "nothing here is typed by hand. The search tables come from the search seeds "
        "(`docs/PREREGISTRATION.md` sections 3 and 8), not from held-out ones. The other tables "
        "read the records of the run, which are on held-out seeds, and each says what it was "
        "measured on.",
        "",
    ]
    if ix.param("mode") == "quick":
        out += [f"> **{QUICK_BANNER}.** Nothing below is a result.", ""]
    if dry_run:
        out += ["> **DRY RUN.** Nothing was written to the cache or to `docs/`.", ""]
    out += _search_section(ix, doc, cost, results, cells)
    out += _conformance_section(ix)
    out += _screener_section()
    out += _f2_section(ix)
    out += _gaps_section(records)
    out += _dev_section(ix, doc)
    return "\n".join(out).rstrip("\n") + "\n"


def _search_section(
    ix: Index,
    doc: Mapping[str, Any],
    cost: Mapping[str, int],
    results: Sequence[Mapping[str, Any]],
    cells: Sequence[SearchCell],
) -> list[str]:
    out = ["## Violations found by search beyond the assumptions", ""]
    out += [
        "The attacker is the falsification search (CMA-ES and random search) against the shielded "
        "target: the bound rule with Careless and Gullible behind the shield. `R_in` stays inside "
        "the assumptions; `R_out` leaves them. Each first find was evaluated again, shrunk to a "
        "smaller scenario that still violates, and labelled with the assumption it broke "
        "(`A_exec`, `A_phys`, `C1`) or `assumption_held`. A label of `assumption_held` in `R_in` "
        "is a defect; in `R_out` it is a gap the three assumptions do not name.",
        "",
    ]
    rows: list[Sequence[object]] = []
    for regime in ("in", "out"):
        for algo in ("cma", "random"):
            rec = ix.get(f"search.R_{regime}.{algo}.find_rate")
            first = ix.get(f"search.R_{regime}.{algo}.evals_to_first")
            if rec is not None:
                rows.append((f"R_{regime}", algo, k_over_n(rec), value_ci(first), rec.measured_on))
    out += table(
        [
            "Regime",
            "Algorithm",
            "Runs that found a violation",
            "Evaluations to the first (mean)",
            "Measured on",
        ],
        rows,
    )
    in_viol = ix.get("claim/C-SEARCH/r_in_violations")
    if in_viol is not None:
        out += [
            f"Defects in `R_in` (an `assumption_held` violation of P1 to P5 by the shielded "
            f"target): {k_over_n(in_viol)} runs. {in_viol.measured_on}",
            "",
        ]
    entries = list(doc.get("entries", []))
    if not entries:
        out += ["No find was shrunk: no cached run found a violation that reproduced.", ""]
    else:
        out += ["### Shrunk scenarios", ""]
        scenario_rows: list[Sequence[object]] = []
        for e in entries:
            fb = ", ".join(
                f"{b['algo']} seed {b['seed']} evaluation {b['evaluation']}" for b in e["found_by"]
            )
            scenario_rows.append(
                (
                    e["id"], e["regime"], e["kind"], e["broken_assumption"],
                    ", ".join(e["properties"]),
                    "; ".join(_dimension_text(d) for d in e["dimensions"]) or "none left",
                    "; ".join(_event_text(x) for x in e["first_events"]) or "none",
                    fb, f"{e['shrink_calls']}" + (" (cut short)" if e["shrink_truncated"] else ""),
                )
            )  # fmt: skip
        out += table(
            ["Id", "Regime", "Kind", "Broken assumption", "Properties",
             "Dimensions that survived the shrink", "First events", "Found by", "Shrink calls"],
            scenario_rows,
        )  # fmt: skip
    lost = [r for r in results if not r.get("reproduced")]
    if lost:
        out += ["### Finds that did not reproduce", ""]
        out += [
            f"- R_{r['regime']} {r['algo']} seed {r['seed']}: recorded {', '.join(r['recorded'])}; "
            f"the evaluation gave {', '.join(r.get('got', [])) or 'no violation'}"
            f"{' (' + r['failed'] + ')' if r.get('failed') else ''}"
            for r in lost
        ] + ["", "A find that does not reproduce is not shrunk and is not in the table above.", ""]
    codes = sorted({c.code[:12] for c in cells})
    out += [
        "### What it cost to build this list",
        "",
        f"{cost['finds']} first find{'' if cost['finds'] == 1 else 's'}, "
        f"{cost['reproduced']} reproduced. "
        f"{cost['evaluations']} evaluations in all: {cost['reevaluations']} to reproduce the finds "
        f"and {cost['shrink_and_ablation']} for the shrinks and the one ablation per surviving "
        f"dimension. The search cells read were written by code {', '.join(codes) or 'unknown'}.",
        "",
    ]
    return out


def _conformance_section(ix: Index) -> list[str]:
    out = ["## The conformance gap", ""]
    out += [
        "The abstraction is a proof about a model; the plant is not the model. These are the "
        "measured distances on sampled traces, with the count of zero reported against its "
        "denominator.",
        "",
    ]
    names = (
        "conformance.unmatched", "conformance.unmatched_when_safe", "conformance.missed_violations",
        "conformance.a_phys",
    )  # fmt: skip
    rows = [
        (r.name, k_over_n(r), value_ci(r), r.ci_kind, r.measured_on)
        for n in names
        if (r := ix.get(n))
    ]
    out += table(["Quantity", "k/n", "Rate (interval)", "Interval", "Measured on"], rows)
    by_var = [
        (r.name.removeprefix("conformance.unmatched."), k_over_n(r))
        for r in ix.prefixed("conformance.unmatched.")
        if r.name.count(".") == 3
    ]
    if by_var:
        out += table(["Unmatched ticks by variable", "k/n"], by_var)
    return out


def _screener_section() -> list[str]:
    misses = screener_misses()
    out = ["## The screener miss list", ""]
    out += [
        "Malicious notes that the cached injection screener lets through, which a screened "
        "proposer still obeys. The notes are synthetic, the list is the repository's own "
        f"(`closed_loop.plant.notes.screener_misses`): {len(misses)} of {len(MALICIOUS_NOTES)} "
        "malicious notes.",
        "",
    ]
    rows = [
        (n.note_id, n.language, n.style, n.induces.value if n.induces else "-", n.asset_id or "-")
        for n in misses
    ]
    out += table(["Note", "Language", "Style", "Command it induces", "Asset"], rows)
    return out


def _f2_section(ix: Index) -> list[str]:
    out = ["## F2 replays that did not confirm the F1 violation", ""]
    confirmed = ix.get("claim/C-TOGETHER/confirmed")
    reasons = [r for r in ix.prefixed("f2.replay.reason.") if r.name.count(".") == 3]
    if confirmed is None and not reasons:
        return out + ["The F2 stage did not run in this results file.", ""]
    if confirmed is not None:
        # the pre-registration reports the fraction with its interval (section 7), and the
        # record carries the count only: the exact interval is computed here from that count
        k = int(confirmed.extra.get("k", confirmed.value))
        lo, hi = clopper_pearson(k, confirmed.n, 0.05)
        out += [
            f"Replays that confirmed: {k_over_n(confirmed)}, exact 95% interval {lo:.4f} to "
            f"{hi:.4f} (Clopper-Pearson, from the count). {confirmed.measured_on}",
            "",
        ]
    rows = [(r.name.removeprefix("f2.replay.reason."), k_over_n(r)) for r in reasons]
    out += table(["Reason code", "Replays (k/n)"], rows)
    dropped = ix.prefixed("f2.dropped_pieces.")
    if dropped:
        out += [
            "Telemetry pieces dropped by the guard of `AssetTelemetry.history()` (`0003` A20). "
            "A piece that would restart a life after the failure day fixed at the life's first "
            "start is dropped and its repair carried to the next piece, so the pipeline sees no "
            "more rows of that life.",
            "",
        ]
        out += table(
            ["Arm", "Episodes with a dropped piece (k/n)", "Pieces dropped"],
            [
                (r.name.removeprefix("f2.dropped_pieces."), f"{r.extra.get('k', 0)}/{r.n}",
                 str(r.extra.get("total", 0)))
                for r in dropped
            ],
        )  # fmt: skip
    return out


def _gaps_section(records: Sequence[ResultRecord]) -> list[str]:
    out = ["## What the abstraction does not model", ""]
    out += [
        "From the proof scope the same `results.json` produces "
        "(`closed_loop.eval.report.render_proof_scope`):",
        "",
    ]
    bullets = _bullets_under(render_proof_scope(records), "What the statement does not cover")
    out += [f"- {b}" for b in bullets] + [""]
    return out


def _arm_rows(ix: Index, metric: str, arms: Sequence[str]) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for arm in arms:
        rec = ix.get(f"{arm}.nominal.{metric}")
        if rec is not None:
            rows.append((arm, k_over_n(rec), value_ci(rec), rec.measured_on))
    return rows


def _dev_section(ix: Index, doc: Mapping[str, Any]) -> list[str]:
    out = ["## What development showed, measured in this run", ""]
    out += [
        "Each item below was first seen on development seeds and is read here from the records "
        "of this run.",
        "",
    ]
    # START with the work open: plant events the specification does not see
    out += [
        "### START with the work open: an event the plant records and the specification misses",
        "",
    ]
    out += [
        "Energising an asset while work is open is a plant event with no formula behind it. Its "
        "share of episodes (`m_viol_p13_plant_only`), arm by arm and then by what stands between "
        "the proposer and the plant:",
        "",
    ]
    rows = _arm_rows(ix, "m_viol_p13_plant_only", PLANT_ONLY_ARMS)
    out += table(["Arm", "Episodes (k/n)", "Rate (interval)", "Measured on"], rows)
    group_rows = []
    for label, arms in PLANT_ONLY_GROUPS:
        recs = [r for a in arms if (r := ix.get(f"{a}.nominal.m_viol_p13_plant_only"))]
        if recs:
            k = sum(int(r.extra.get("k", 0)) for r in recs)
            n = sum(r.n for r in recs)
            group_rows.append(
                (
                    label,
                    ", ".join(a for a in arms if ix.get(f"{a}.nominal.m_viol_p13_plant_only")),
                    f"{k}/{n}",
                )
            )
    out += table(
        ["Group", "Arms", "Episodes with the event (k/n, pooled over the arms)"], group_rows
    )
    # failures against violations
    out += ["### Failures that are not violations", ""]
    out += [
        "A conveyor failure is not a violation of P1 to P5, and a pump failure is not one unless "
        "P4 goes with it (both pumps unavailable). Unplanned failures per episode (`u_e`) beside "
        "the share of episodes with a P4 or P5 violation:",
        "",
    ]
    rows2: list[Sequence[object]] = []
    for arm in ("B1", "B2", "B3-V2L", "B3-S", "P0", "P1", "P1-nopair"):
        u, v = ix.get(f"{arm}.nominal.u_e"), ix.get(f"{arm}.nominal.m_viol_p45")
        if u is not None and v is not None:
            rows2.append((arm, num(u.value), k_over_n(v), v.measured_on))
    out += table(
        ["Arm", "Unplanned failures per episode", "P4-P5 violations (k/n)", "Measured on"], rows2
    )
    # decisions with no allowed command (P12g, G7)
    uncovered = uncovered_rows(ix)
    if uncovered:
        out += ["### Shield decisions with no allowed command", "", UNCOVERED_TEXT, ""]
        out += table(["Arm", "Condition", "Episodes with any", "Decisions", "Fidelity"], uncovered)
    # wrong repair
    out += ["### A wrong repair that restarts the estimator on a failing part", ""]
    out += [
        "A completed work order is a renewal as far as perception can tell, so a repair of the "
        "wrong mode restarts the estimator on a part that is still failing, which can break "
        "contract assumption C1. Wrong-mode jobs per episode beside the share of episodes with a "
        "violation attributed to C1:",
        "",
    ]
    rows3: list[Sequence[object]] = []
    for arm in ("B3-V2L", "B3-S", "P0", "P1", "P1-nopair"):
        w, c1 = ix.get(f"{arm}.nominal.wrong_mode_jobs"), ix.get(f"{arm}.nominal.m_attr_c1")
        if w is not None and c1 is not None:
            rows3.append((arm, num(w.value), k_over_n(c1), c1.measured_on))
    out += table(
        ["Arm", "Wrong-mode jobs per episode", "C1 attributions (k/n)", "Measured on"], rows3
    )
    # gas clock
    out += ["### The gas clock, in both directions", ""]
    out += [
        "The abstraction reads the gas clock before the command and the plant reads it after "
        "(`0003` A19). The two secondary rows: quarters the crusher stands in WORK per crusher job "
        "that completes, and late gas tests (a test a quarter late with someone inside; none is "
        "expected behind the shield or the interlock). No decision rule is attached.",
        "",
    ]
    rows4: list[Sequence[object]] = []
    for arm in ARMS_SHOWN:
        w, late = (
            ix.get(f"{arm}.nominal.m_crusher_work_q"),
            ix.get(f"{arm}.nominal.m_late_gas_tests"),
        )
        if w is not None or late is not None:
            rows4.append(
                (arm, value_ci(w), w.n if w else "-", value_ci(late), late.n if late else "-")
            )
    out += table(
        [
            "Arm",
            "Crusher quarters in WORK per job (interval)",
            "n",
            "Late gas tests per episode (interval)",
            "n",
        ],
        rows4,
    )
    return out


# ----------------------------------------------------------------------------- command line


def parse(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--cells", type=Path, default=DEFAULT_CELLS, help="the cache of search cells")
    ap.add_argument("--code", default=None, help="read only cells written by this code hash")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cache", type=Path, default=WGT_CACHE, help="this script's own cache")
    ap.add_argument("--dry-run", action="store_true", help="write nothing: no cache, no document")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--lock", metavar="PATH", help="take the heavy-run lock")
    ap.add_argument("--limit-finds", type=int, default=None, help="shrink only the first N finds")
    ap.add_argument("--max-calls", type=int, default=None, help="cap the calls of each shrink")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse(argv)
    records = records_from_json(args.results.read_text(encoding="utf-8"))
    cells = read_search_cells(args.cells, args.code)
    if not cells:
        print(f"error: no search cell in {args.cells}", file=sys.stderr)
        return 2
    for c in cells:
        print(f"cell R_{c.regime} {c.algo}: {len(c.outcomes)} runs, budget {c.budget}, "
              f"code {c.code[:12]}, {len(c.failed)} failed")  # fmt: skip
    started = time.perf_counter()
    from closed_loop.eval.context import default_context
    from closed_loop.eval.runner import Runner

    def go() -> list[dict[str, Any]]:
        with Runner(
            workers=args.workers,
            context=default_context(),
            cache_dir=None if args.dry_run else args.cache,
            tuned=TunedValues(**cells[0].tuned),
            log=print,
        ) as runner:
            return shrink_all(
                cells, runner, limit=args.limit_finds, max_calls=args.max_calls, log=print
            )

    if args.lock and not args.dry_run:
        from closed_loop.audit.guard import HeavyRunLock

        with HeavyRunLock(Path(args.lock), "closed-loop", SCRIPT):
            results = go()
    else:
        results = go()
    cost = cost_of(results)
    doc = merge(results, run_summaries(cells))
    text = render_document(records, doc, cost, results, cells, dry_run=args.dry_run)
    print(f"evaluations: {cost}; {time.perf_counter() - started:.0f} s")
    if args.dry_run:
        sys.stdout.write(text)
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
