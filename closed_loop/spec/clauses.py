"""Clause-backed specification parameters, from a checked-in cache or from plant-copilot.

Interface text: docs/adr/0002-interfaces.md, section 3.4; cache rule: docs/adr/0003-review-
amendments.md, A2; method: docs/adr/0001-architecture.md, section 5.

Three questions are put to plant-copilot at an as-of date. For each, the claim that cites the
expected clause (English only) gives the verbatim text, document id, clause number and revision, and
a pinned regular expression extracts the number:

* ``retest``: CORP-HSE-002 clause 4.3, "re-tested every N minutes" (Rev 3) or "at least every N
  hours" (Rev 2), and "break in work longer than N minutes" (Rev 3 only).
* ``permit``: CORP-HSE-004 clause 4.1, "valid for a maximum of N hours".
* ``alarm``: CORP-HSE-002 clause 4.4, "leave the space immediately", read as one quarter (the
  resolution of the plant); a wording with "within N minutes" is also understood.

Minutes are converted to whole quarters rounding **down**, which is the stricter reading (a 20
minute period becomes one quarter); a period shorter than one quarter is refused. A revision
without a break clause (Rev 2) sets ``break_q = retest_q``, which adds no constraint beyond the
re-test period.

``spec/clause_cache.json`` (SYNTHETIC, checked in) stores the sources for each as-of date in use, so
F0 and F1 need no sibling. This is the only module outside ``closed_loop/f2/`` that may import a
sibling, and it imports ``plant_copilot`` lazily, inside ``fetch_sources``. An ``f2``-marked test
recomputes the cache through the real library and fails on any difference. To refresh it after a
deliberate change in plant-copilot, run ``python -m closed_loop.spec.clauses --write``.
"""

from __future__ import annotations

import importlib
import json
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from closed_loop.core.timegrid import QUARTER_MIN
from closed_loop.spec.properties import SpecParams

__all__ = [
    "CACHE_PATH",
    "CACHE_DATES",
    "ClauseError",
    "ClauseSource",
    "QUESTIONS",
    "build_cache",
    "fetch_sources",
    "load_cache",
    "spec_for",
    "spec_from_cache",
    "spec_from_copilot",
    "spec_from_sources",
]

CACHE_PATH = Path(__file__).with_name("clause_cache.json")
# The as-of dates the benchmark uses: the simulated "today" and a date before CORP-HSE-002 Rev 3.
CACHE_DATES: tuple[date, ...] = (date(2026, 1, 1), date(2026, 10, 1))
CACHE_SOURCE_NOTE = "SYNTHETIC: clauses from the bundled plant-copilot corpus, no real procedure"

# key -> (question put to plant-copilot, expected document id, expected clause number)
QUESTIONS: dict[str, tuple[str, str, str]] = {
    "retest": (
        "How often must the atmosphere in a confined space be re-tested?",
        "CORP-HSE-002",
        "4.3",
    ),
    "permit": ("How long is a work permit valid?", "CORP-HSE-004", "4.1"),
    "alarm": (
        "What must entrants do when a gas alarm sounds in a confined space?",
        "CORP-HSE-002",
        "4.4",
    ),
}

_DURATION = r"(\d+)\s+(minutes?|hours?)"
_RETEST_RE = re.compile(rf"re-?tested\s+(?:at\s+least\s+)?every\s+{_DURATION}", re.IGNORECASE)
_BREAK_RE = re.compile(rf"break\s+in\s+work\s+longer\s+than\s+{_DURATION}", re.IGNORECASE)
_PERMIT_RE = re.compile(r"valid\s+for\s+a\s+maximum\s+of\s+(\d+(?:\.\d+)?)\s+hours?", re.IGNORECASE)
_ALARM_WITHIN_RE = re.compile(rf"within\s+{_DURATION}", re.IGNORECASE)
_ALARM_NOW_RE = re.compile(r"\bimmediately\b", re.IGNORECASE)


class ClauseError(Exception):
    """A clause, a revision or a number could not be found, or the cache does not cover a date."""


@dataclass(frozen=True)
class ClauseSource:
    """One verbatim clause as plant-copilot returned it."""

    key: str
    question: str
    doc_id: str
    clause: str
    revision: int
    language: str
    text: str
    ignored: tuple[str, ...] = ()  # reasons other claims were set aside (for example, Arabic)

    def provenance(self) -> str:
        return (
            f"{self.doc_id} clause {self.clause} Rev {self.revision} ({self.language}): {self.text}"
        )


# ----------------------------------------------------------------------------- parsing


def _minutes(count: str, unit: str) -> int:
    return int(count) * (60 if unit.lower().startswith("hour") else 1)


def _quarters(minutes: int, what: str, source: ClauseSource) -> int:
    q = minutes // QUARTER_MIN
    if q < 1:
        raise ClauseError(
            f"{what}: {minutes} minutes is shorter than one quarter ({QUARTER_MIN} min) in "
            f"{source.doc_id} clause {source.clause}"
        )
    return q


def _need(sources: Mapping[str, ClauseSource], key: str) -> ClauseSource:
    if key not in sources:
        raise ClauseError(f"no source for {key!r}; have {sorted(sources)}")
    src = sources[key]
    _, doc_id, clause = QUESTIONS[key]
    if (src.doc_id, src.clause) != (doc_id, clause):
        raise ClauseError(
            f"{key!r}: expected {doc_id} clause {clause}, got {src.doc_id} {src.clause}"
        )
    if src.revision < 1:
        raise ClauseError(f"{key!r}: missing revision for {src.doc_id} clause {src.clause}")
    return src


def spec_from_sources(sources: Mapping[str, ClauseSource], as_of: date | None = None) -> SpecParams:
    """Extract the numbers from the verbatim clauses. Raises ``ClauseError`` when one is missing."""
    retest = _need(sources, "retest")
    match = _RETEST_RE.search(retest.text)
    if match is None:
        raise ClauseError(f"no re-test period in {retest.provenance()!r}")
    retest_q = _quarters(_minutes(*match.groups()), "re-test period", retest)

    provenance = {"retest_q": retest.provenance()}
    brk = _BREAK_RE.search(retest.text)
    if brk is None:
        break_q = retest_q
        provenance["break_q"] = (
            f"no break-in-work wording in {retest.doc_id} clause {retest.clause} Rev "
            f"{retest.revision}; break_q set to retest_q, which adds no constraint"
        )
    else:
        break_q = _quarters(_minutes(*brk.groups()), "break in work", retest)
        provenance["break_q"] = retest.provenance()

    permit = _need(sources, "permit")
    hours = _PERMIT_RE.search(permit.text)
    if hours is None:
        raise ClauseError(f"no permit validity in {permit.provenance()!r}")
    permit_max_h = float(hours.group(1))
    provenance["permit_max_h"] = permit.provenance()

    alarm = _need(sources, "alarm")
    within = _ALARM_WITHIN_RE.search(alarm.text)
    if within is not None:
        alarm_exit_q = max(1, _minutes(*within.groups()) // QUARTER_MIN)
        provenance["alarm_exit_q"] = alarm.provenance()
    elif _ALARM_NOW_RE.search(alarm.text):
        alarm_exit_q = 1
        provenance["alarm_exit_q"] = (
            alarm.provenance() + " [read as one quarter, the resolution of the plant]"
        )
    else:
        raise ClauseError(f"no exit time in {alarm.provenance()!r}")

    for src in (retest, permit, alarm):
        if src.ignored:
            provenance[f"ignored:{src.key}"] = "; ".join(src.ignored)
    if as_of is not None:
        provenance["as_of"] = as_of.isoformat()
    return SpecParams(
        retest_q=retest_q,
        break_q=break_q,
        permit_max_h=permit_max_h,
        alarm_exit_q=alarm_exit_q,
        provenance=provenance,
    )


def select_claim(
    key: str, claims: Sequence[Any], question: str | None = None
) -> ClauseSource | None:
    """First English claim citing the clause expected for ``key``; None if there is none.

    Claims are plant-copilot ``Claim`` objects (``text`` and ``citations``, each citation with
    ``doc_id``, ``clause``, ``revision``, ``language``). A claim for the expected clause in another
    language is set aside, and the reason is recorded on the source that is returned.
    """
    asked, doc_id, clause = QUESTIONS[key]
    ignored: list[str] = []
    for claim in claims:
        for cit in claim.citations:
            if (cit.doc_id, cit.clause) != (doc_id, clause):
                continue
            if cit.language != "en":
                ignored.append(
                    f"{cit.doc_id} clause {cit.clause} Rev {cit.revision} ({cit.language}) set "
                    "aside: the extractor reads the English master only"
                )
                continue
            return ClauseSource(
                key=key,
                question=question or asked,
                doc_id=cit.doc_id,
                clause=cit.clause,
                revision=int(cit.revision),
                language=cit.language,
                text=claim.text,
                ignored=tuple(ignored),
            )
    return None


# ----------------------------------------------------------------------------- live path


def fetch_sources(as_of: date, copilot: Any | None = None) -> dict[str, ClauseSource]:
    """Ask the real plant-copilot the three questions as of ``as_of`` (imports it lazily)."""
    # importlib keeps the sibling out of module scope, and out of mypy's reach when it is absent
    try:
        copilot_module = importlib.import_module("plant_copilot.copilot")
        models_module = importlib.import_module("plant_copilot.models")
    except ImportError as exc:
        raise ClauseError(
            "plant_copilot is not installed, so clauses cannot be read live; use the cache "
            f"(spec_from_cache) for the dates it holds: {[d.isoformat() for d in CACHE_DATES]}"
        ) from exc
    if copilot is None:
        copilot = copilot_module.PlantCopilot(as_of=as_of)
    user = models_module.User.of("closed-loop-spec", "hse_officer", "phos-01")
    out: dict[str, ClauseSource] = {}
    for key, (question, doc_id, clause) in QUESTIONS.items():
        answer = copilot.ask(question, user)
        if not answer.answered:
            raise ClauseError(
                f"{key!r}: plant-copilot refused the question ({answer.refusal_reason})"
            )
        source = select_claim(key, answer.claims)
        if source is None:
            raise ClauseError(f"{key!r}: no English claim cites {doc_id} clause {clause}")
        out[key] = source
    return out


def spec_from_copilot(as_of: date) -> SpecParams:
    """Parameters read live from plant-copilot as of ``as_of``.

    Raises ``ClauseError`` if a clause or revision is missing or plant-copilot is not installed.
    """
    return spec_from_sources(fetch_sources(as_of), as_of)


# ----------------------------------------------------------------------------- cache


class _SourceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    key: str
    question: str
    doc_id: str
    clause: str
    revision: int
    language: str
    text: str
    ignored: tuple[str, ...] = ()


class _CacheModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    synthetic: Literal[True]
    note: str
    entries: dict[str, dict[str, _SourceModel]]


def load_cache(path: Path | None = None) -> dict[date, dict[str, ClauseSource]]:
    """Read and validate the cache file. Raises ``ClauseError`` if it is missing or malformed."""
    target = path or CACHE_PATH
    try:
        model = _CacheModel.model_validate(json.loads(target.read_text(encoding="utf-8")))
    except (OSError, ValueError, ValidationError) as exc:
        raise ClauseError(f"clause cache {target} is missing or malformed: {exc}") from exc
    out: dict[date, dict[str, ClauseSource]] = {}
    for iso, by_key in model.entries.items():
        try:
            day = date.fromisoformat(iso)
        except ValueError as exc:
            raise ClauseError(f"clause cache {target}: {iso!r} is not an ISO date") from exc
        out[day] = {k: ClauseSource(**s.model_dump()) for k, s in by_key.items()}
    return out


def spec_from_cache(as_of: date, path: Path | None = None) -> SpecParams:
    """Parameters for ``as_of`` from the checked-in cache; needs no sibling repository."""
    cache = load_cache(path)
    if as_of not in cache:
        raise ClauseError(
            f"the clause cache has no entry for {as_of.isoformat()}; it holds "
            f"{sorted(d.isoformat() for d in cache)}. Add the date with "
            "`python -m closed_loop.spec.clauses --write` (needs plant-copilot)."
        )
    return spec_from_sources(cache[as_of], as_of)


def spec_for(as_of: date) -> SpecParams:
    """The cache when it covers ``as_of``, else a live read; the entry point for F0 and F1."""
    if as_of in load_cache():
        return spec_from_cache(as_of)
    return spec_from_copilot(as_of)


def build_cache(dates: Iterable[date] = CACHE_DATES) -> dict[str, Any]:
    """Recompute the whole cache document through the real plant-copilot."""
    entries: dict[str, Any] = {}
    for day in sorted(dates):
        sources = fetch_sources(day)
        entries[day.isoformat()] = {k: asdict(s) for k, s in sources.items()}
    return {"synthetic": True, "note": CACHE_SOURCE_NOTE, "entries": entries}


def dump_cache(document: Mapping[str, Any]) -> str:
    """The exact text written to ``clause_cache.json`` (sorted keys, LF endings)."""
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _main(argv: Sequence[str]) -> int:
    if len(argv) != 1 or argv[0] not in ("--write", "--check"):
        print("usage: python -m closed_loop.spec.clauses (--write | --check)")
        return 2
    text = dump_cache(build_cache())
    if argv[0] == "--write":
        with CACHE_PATH.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        print(f"wrote {CACHE_PATH}")
        return 0
    current = CACHE_PATH.read_text(encoding="utf-8")
    print("cache matches plant-copilot" if current == text else "cache DIFFERS from plant-copilot")
    return 0 if current == text else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    raise SystemExit(_main(sys.argv[1:]))
