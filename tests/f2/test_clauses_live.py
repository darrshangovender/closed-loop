"""P11a: the clause specification read live from the real plant-copilot.

``spec/clauses.py`` was written in packet P2 (with its cache and its offline tests, including the
failure of the regular expression on a text without a number and the recorded reason for ignoring
Arabic text). What is added here is the "done when" of P11a: the live read at the two dates.
"""

from __future__ import annotations

from datetime import date

import pytest

from closed_loop.spec.clauses import ClauseError, spec_for, spec_from_cache, spec_from_copilot

pytestmark = pytest.mark.f2


@pytest.fixture(scope="module")
def current():
    return spec_from_copilot(date(2026, 10, 1))


@pytest.fixture(scope="module")
def stale():
    return spec_from_copilot(date(2026, 1, 1))


class TestLiveRead:
    def test_the_sim_date_gives_the_30_minute_rule_of_rev_3(self, current) -> None:
        assert current.retest_q == 2
        text = current.provenance["retest_q"]
        assert "CORP-HSE-002 clause 4.3 Rev 3 (en)" in text
        assert "30 minutes" in text

    def test_a_date_before_rev_3_gives_the_two_hour_rule_of_rev_2(self, stale) -> None:
        assert stale.retest_q == 8
        assert "CORP-HSE-002 clause 4.3 Rev 2 (en)" in stale.provenance["retest_q"]
        assert "2 hours" in stale.provenance["retest_q"]

    def test_the_other_parameters_carry_their_clauses(self, current) -> None:
        assert (current.break_q, current.permit_max_h, current.alarm_exit_q) == (1, 12.0, 1)
        assert "CORP-HSE-004 clause 4.1" in current.provenance["permit_max_h"]
        assert "CORP-HSE-002 clause 4.4" in current.provenance["alarm_exit_q"]
        assert current.provenance["as_of"] == "2026-10-01"

    def test_the_live_read_equals_the_cache_for_both_dates(self, current, stale) -> None:
        assert spec_from_cache(date(2026, 10, 1)) == current
        assert spec_from_cache(date(2026, 1, 1)) == stale

    def test_spec_for_serves_the_cached_dates_without_a_live_read(self, current) -> None:
        assert spec_for(date(2026, 10, 1)) == current


class TestRefusals:
    def test_a_date_with_no_corpus_revision_fails_loudly(self) -> None:
        # before any revision of CORP-HSE-002 existed there is no clause to read
        with pytest.raises(ClauseError):
            spec_from_copilot(date(2000, 1, 1))
