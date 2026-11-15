"""P2: clause-backed parameters. Everything runs with no sibling installed except the f2 test."""

import json
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from closed_loop.spec import clauses
from closed_loop.spec.clauses import (
    CACHE_DATES,
    CACHE_PATH,
    ClauseError,
    ClauseSource,
    dump_cache,
    load_cache,
    select_claim,
    spec_for,
    spec_from_cache,
    spec_from_copilot,
    spec_from_sources,
)

REPO = Path(__file__).resolve().parents[2]
REV3_TEXT = (
    "The atmosphere shall be monitored continuously while persons are inside, and re-tested every "
    "30 minutes and after any break in work longer than 15 minutes."
)
REV2_TEXT = (
    "The atmosphere shall be re-tested at least every 2 hours while persons are inside the space."
)
PERMIT_TEXT = "A permit is valid for a maximum of 12 hours and shall not extend."
ALARM_TEXT = "If any gas alarm sounds, all entrants leave immediately."


def source(key: str, text: str, revision: int = 3, **kw) -> ClauseSource:
    question, doc_id, clause = clauses.QUESTIONS[key]
    return ClauseSource(
        key=key, question=question, doc_id=doc_id, clause=clause, revision=revision,
        language="en", text=text, **kw,
    )  # fmt: skip


def sources(
    retest: str = REV3_TEXT, permit: str = PERMIT_TEXT, alarm: str = ALARM_TEXT
) -> dict[str, ClauseSource]:
    return {
        "retest": source("retest", retest),
        "permit": source("permit", permit),
        "alarm": source("alarm", alarm),
    }


class TestCacheDates:
    def test_the_sim_date_gives_the_rev_3_rule_with_its_clause(self) -> None:
        p = spec_from_cache(date(2026, 10, 1))
        assert (p.retest_q, p.break_q, p.permit_max_h, p.alarm_exit_q) == (2, 1, 12.0, 1)
        assert p.provenance["retest_q"] == "CORP-HSE-002 clause 4.3 Rev 3 (en): " + REV3_TEXT
        assert p.provenance["permit_max_h"].startswith("CORP-HSE-004 clause 4.1 Rev 6 (en): ")
        assert "CORP-HSE-002 clause 4.4 Rev 3" in p.provenance["alarm_exit_q"]
        assert "one quarter" in p.provenance["alarm_exit_q"]
        assert p.provenance["as_of"] == "2026-10-01"

    def test_the_stale_date_gives_the_two_hour_rule_of_rev_2(self) -> None:
        p = spec_from_cache(date(2026, 1, 1))
        assert p.retest_q == 8 and p.break_q == 8  # 120 minutes; no break clause in Rev 2
        assert p.provenance["retest_q"] == "CORP-HSE-002 clause 4.3 Rev 2 (en): " + REV2_TEXT
        assert "no break-in-work wording" in p.provenance["break_q"]
        assert "Rev 2" in p.provenance["break_q"]
        assert p.permit_max_h == 12.0 and p.alarm_exit_q == 1

    def test_the_cache_holds_exactly_the_dates_in_use(self) -> None:
        assert set(load_cache()) == set(CACHE_DATES) == {date(2026, 1, 1), date(2026, 10, 1)}

    def test_a_date_not_in_the_cache_is_refused_with_the_available_dates(self) -> None:
        with pytest.raises(ClauseError, match=r"no entry for 2026-06-15.*2026-01-01.*2026-10-01"):
            spec_from_cache(date(2026, 6, 15))

    def test_spec_for_uses_the_cache_and_never_touches_the_sibling(self, monkeypatch) -> None:
        def boom(*args, **kwargs):
            raise AssertionError("the sibling must not be read for a cached date")

        monkeypatch.setattr(clauses, "fetch_sources", boom)
        assert spec_for(date(2026, 10, 1)) == spec_from_cache(date(2026, 10, 1))

    def test_spec_for_falls_back_to_a_live_read_for_other_dates(self, monkeypatch) -> None:
        monkeypatch.setattr(clauses, "fetch_sources", lambda as_of, copilot=None: sources())
        assert spec_for(date(2026, 6, 15)).retest_q == 2

    def test_cache_text_is_what_dump_cache_writes(self) -> None:
        """Byte for byte, so `--write` after a refresh produces a minimal diff."""
        document = {
            "synthetic": True,
            "note": clauses.CACHE_SOURCE_NOTE,
            "entries": {
                d.isoformat(): {k: asdict(s) for k, s in by_key.items()}
                for d, by_key in sorted(load_cache().items())
            },
        }
        assert CACHE_PATH.read_text(encoding="utf-8") == dump_cache(document)
        assert b"\r" not in CACHE_PATH.read_bytes()

    def test_cache_is_labelled_synthetic(self) -> None:
        raw = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        assert raw["synthetic"] is True and "SYNTHETIC" in raw["note"]


class TestWithNoSiblingInstalled:
    def _run(self, code: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
            cwd=REPO, check=False,
        )  # fmt: skip

    def test_cache_path_works_and_importing_does_not_load_the_sibling(self) -> None:
        code = (
            "import sys\n"
            "sys.modules['plant_copilot'] = None\n"  # `import plant_copilot` now raises
            "from datetime import date\n"
            "from closed_loop.spec.clauses import spec_for\n"
            "p = spec_for(date(2026, 10, 1))\n"
            "assert p.retest_q == 2, p\n"
            "assert not [m for m in sys.modules if m.startswith('plant_copilot.')]\n"
            "print('ok')\n"
        )
        done = self._run(code)
        assert done.returncode == 0 and done.stdout.strip() == "ok", done.stderr

    def test_live_read_without_the_sibling_says_so(self) -> None:
        code = (
            "import sys\n"
            "sys.modules['plant_copilot'] = None\n"
            "sys.modules['plant_copilot.copilot'] = None\n"
            "sys.modules['plant_copilot.models'] = None\n"
            "from datetime import date\n"
            "from closed_loop.spec.clauses import ClauseError, spec_from_copilot\n"
            "try:\n"
            "    spec_from_copilot(date(2026, 10, 1))\n"
            "except ClauseError as e:\n"
            "    print('not installed' in str(e), 'spec_from_cache' in str(e))\n"
        )
        done = self._run(code)
        assert done.stdout.split() == ["True", "True"], done.stderr

    def test_no_other_spec_stats_search_or_audit_module_names_a_sibling(self) -> None:
        offenders = []
        for path in (REPO / "closed_loop").rglob("*.py"):
            rel = path.relative_to(REPO / "closed_loop")
            if rel.parts[0] not in {"spec", "stats", "search", "audit"}:
                continue  # other packets police their own files
            if rel.as_posix() == "spec/clauses.py":
                continue
            text = path.read_text(encoding="utf-8")
            for name in ("asset_health", "maintenance_agent", "plant_copilot"):
                if f"import {name}" in text or f"from {name}" in text or f'"{name}' in text:
                    offenders.append((rel.as_posix(), name))
        assert offenders == []

    def test_the_sibling_is_imported_inside_a_function_not_at_module_level(self) -> None:
        text = (REPO / "closed_loop" / "spec" / "clauses.py").read_text(encoding="utf-8")
        for line in text.splitlines():
            assert not line.startswith(("import plant_copilot", "from plant_copilot")), line
        assert 'importlib.import_module("plant_copilot.copilot")' in text


class TestExtraction:
    def test_minutes_become_whole_quarters_rounding_down(self) -> None:
        cases = {
            "15 minutes": 1, "20 minutes": 1, "30 minutes": 2, "45 minutes": 3,
            "1 hour": 4, "2 hours": 8,
        }  # fmt: skip
        for phrase, quarters in cases.items():
            p = spec_from_sources(sources(retest=f"It shall be re-tested every {phrase}."))
            assert p.retest_q == quarters, phrase

    def test_a_period_shorter_than_a_quarter_is_refused(self) -> None:
        with pytest.raises(ClauseError, match="shorter than one quarter"):
            spec_from_sources(sources(retest="It shall be re-tested every 10 minutes."))

    @pytest.mark.parametrize(
        "text",
        [
            "The atmosphere shall be re-tested regularly while persons are inside.",
            "The atmosphere shall be re-tested every few minutes.",
            "",
            "re-tested every minutes",
        ],
    )
    def test_a_retest_clause_without_a_number_fails_loudly(self, text: str) -> None:
        with pytest.raises(ClauseError, match="no re-test period"):
            spec_from_sources(sources(retest=text))

    def test_a_permit_clause_without_a_number_fails_loudly(self) -> None:
        with pytest.raises(ClauseError, match="no permit validity"):
            spec_from_sources(sources(permit="A permit is valid until the job is done."))

    def test_an_alarm_clause_with_no_exit_time_fails_loudly(self) -> None:
        with pytest.raises(ClauseError, match="no exit time"):
            spec_from_sources(sources(alarm="Entrants shall consider leaving the space."))

    def test_an_alarm_clause_with_a_time_limit(self) -> None:
        p = spec_from_sources(sources(alarm="All entrants shall leave within 30 minutes."))
        assert p.alarm_exit_q == 2
        p = spec_from_sources(sources(alarm="All entrants shall leave within 5 minutes."))
        assert p.alarm_exit_q == 1  # never less than one quarter

    def test_a_break_clause_is_read_when_present(self) -> None:
        text = (
            "It shall be re-tested every 60 minutes and after any break in work longer than "
            "30 minutes."
        )
        p = spec_from_sources(sources(retest=text))
        assert (p.retest_q, p.break_q) == (4, 2)
        assert p.provenance["break_q"] == p.provenance["retest_q"]

    def test_fractional_permit_hours(self) -> None:
        p = spec_from_sources(sources(permit="A permit is valid for a maximum of 7.5 hours."))
        assert p.permit_max_h == 7.5

    def test_missing_pieces_are_named(self) -> None:
        good = sources()
        with pytest.raises(ClauseError, match="no source for 'alarm'"):
            spec_from_sources({k: v for k, v in good.items() if k != "alarm"})
        with pytest.raises(ClauseError, match="expected CORP-HSE-002 clause 4.3"):
            spec_from_sources({**good, "retest": replace(good["retest"], clause="4.1")})
        with pytest.raises(ClauseError, match="expected CORP-HSE-004"):
            spec_from_sources({**good, "permit": replace(good["permit"], doc_id="CORP-HSE-002")})
        with pytest.raises(ClauseError, match="missing revision"):
            spec_from_sources({**good, "retest": replace(good["retest"], revision=0)})

    def test_ignored_claims_are_recorded_in_the_provenance(self) -> None:
        good = sources()
        good["retest"] = replace(good["retest"], ignored=("an Arabic claim set aside",))
        p = spec_from_sources(good)
        assert p.provenance["ignored:retest"] == "an Arabic claim set aside"


def claim(text: str, doc_id: str, clause: str, revision: int, language: str) -> SimpleNamespace:
    cit = SimpleNamespace(doc_id=doc_id, clause=clause, revision=revision, language=language)
    return SimpleNamespace(text=text, citations=[cit])


class TestSelectClaim:
    def test_first_english_claim_for_the_expected_clause_wins(self) -> None:
        claims = [
            claim("about 4.1", "CORP-HSE-002", "4.1", 3, "en"),
            claim(REV3_TEXT, "CORP-HSE-002", "4.3", 3, "en"),
            claim("another 4.3", "CORP-HSE-002", "4.3", 3, "en"),
        ]
        got = select_claim("retest", claims)
        assert got is not None and got.text == REV3_TEXT and got.revision == 3
        assert (got.doc_id, got.clause, got.language, got.ignored) == (
            "CORP-HSE-002", "4.3", "en", (),
        )  # fmt: skip

    def test_arabic_text_is_ignored_with_a_recorded_reason(self) -> None:
        claims = [
            claim("\u0646\u0635 \u0639\u0631\u0628\u064a", "CORP-HSE-002", "4.3", 2, "ar"),
            claim(REV3_TEXT, "CORP-HSE-002", "4.3", 3, "en"),
        ]
        got = select_claim("retest", claims)
        assert got is not None and got.language == "en" and got.text == REV3_TEXT
        assert len(got.ignored) == 1
        assert "(ar)" in got.ignored[0] and "English master only" in got.ignored[0]

    def test_only_an_arabic_claim_gives_nothing(self) -> None:
        arabic = claim("\u0646\u0635", "CORP-HSE-002", "4.3", 2, "ar")
        assert select_claim("retest", [arabic]) is None

    def test_no_claim_for_the_clause_gives_nothing(self) -> None:
        assert select_claim("permit", [claim(REV3_TEXT, "CORP-HSE-002", "4.3", 3, "en")]) is None
        assert select_claim("retest", []) is None

    def test_a_claim_with_several_citations(self) -> None:
        multi = SimpleNamespace(
            text=REV3_TEXT,
            citations=[
                SimpleNamespace(doc_id="X", clause="1", revision=1, language="en"),
                SimpleNamespace(doc_id="CORP-HSE-002", clause="4.3", revision=3, language="en"),
            ],
        )
        got = select_claim("retest", [multi])
        assert got is not None and got.clause == "4.3"


class TestFetchWithAFakeCopilot:
    def _copilot(self, claims_by_question: dict[str, list]):
        class Fake:
            def ask(self, question, user):
                claims = claims_by_question[question]
                return SimpleNamespace(
                    answered=bool(claims), claims=claims, refusal_reason="nothing found"
                )

        return Fake()

    def test_refusal_is_reported(self) -> None:
        pytest.importorskip("plant_copilot")
        copilot = self._copilot({q: [] for q, _, _ in clauses.QUESTIONS.values()})
        with pytest.raises(ClauseError, match="refused the question"):
            clauses.fetch_sources(date(2026, 10, 1), copilot=copilot)

    def test_an_answer_that_does_not_cite_the_clause_is_an_error(self) -> None:
        pytest.importorskip("plant_copilot")
        wrong = [claim("x", "OTHER", "1.1", 1, "en")]
        copilot = self._copilot({q: wrong for q, _, _ in clauses.QUESTIONS.values()})
        with pytest.raises(ClauseError, match="no English claim cites"):
            clauses.fetch_sources(date(2026, 10, 1), copilot=copilot)


class TestMalformedCache:
    def _write(self, tmp_path: Path, document: object) -> Path:
        path = tmp_path / "cache.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def _good(self) -> dict:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ClauseError, match="missing or malformed"):
            load_cache(tmp_path / "absent.json")

    def test_not_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ClauseError, match="missing or malformed"):
            load_cache(path)

    def test_must_be_labelled_synthetic(self, tmp_path: Path) -> None:
        doc = self._good()
        doc["synthetic"] = False
        with pytest.raises(ClauseError, match="malformed"):
            load_cache(self._write(tmp_path, doc))

    def test_unknown_fields_are_refused(self, tmp_path: Path) -> None:
        doc = self._good()
        doc["entries"]["2026-10-01"]["retest"]["surprise"] = 1
        with pytest.raises(ClauseError, match="malformed"):
            load_cache(self._write(tmp_path, doc))

    def test_missing_field_is_refused(self, tmp_path: Path) -> None:
        doc = self._good()
        del doc["entries"]["2026-10-01"]["retest"]["revision"]
        with pytest.raises(ClauseError, match="malformed"):
            load_cache(self._write(tmp_path, doc))

    def test_a_bad_date_key(self, tmp_path: Path) -> None:
        doc = self._good()
        doc["entries"]["01/10/2026"] = doc["entries"].pop("2026-10-01")
        with pytest.raises(ClauseError, match="not an ISO date"):
            load_cache(self._write(tmp_path, doc))

    def test_a_cache_edit_changes_the_spec(self, tmp_path: Path) -> None:
        doc = self._good()
        text = doc["entries"]["2026-10-01"]["retest"]["text"]
        doc["entries"]["2026-10-01"]["retest"]["text"] = text.replace("30 minutes", "45 minutes")
        assert spec_from_cache(date(2026, 10, 1), self._write(tmp_path, doc)).retest_q == 3


@pytest.mark.f2
def test_cache_equals_a_recomputation_through_the_real_plant_copilot() -> None:
    """Fails on any difference between the checked-in cache and the live library (0003 A2).

    Compares the text byte for byte, then the public live entry point against the cache.
    """
    live = clauses.build_cache()
    assert dump_cache(live) == CACHE_PATH.read_text(encoding="utf-8")
    cached = load_cache()
    assert spec_from_copilot(date(2026, 10, 1)) == spec_from_sources(
        cached[date(2026, 10, 1)], date(2026, 10, 1)
    )
    assert spec_from_cache(date(2026, 1, 1)).retest_q == 8
