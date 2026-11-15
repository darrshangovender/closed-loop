"""The README and its Results block (packet P13a).

Two kinds of test. The first kind runs the checks on the real ``README.md`` against the real
``benchmarks/results.json`` and **skips** while either is the stub or missing. The second kind
runs the same checks on a small made-up README and records and proves that each check would fail on
the thing it is there for (a wrong number, a hand-edited block, a missing section, a denied name).

The deny-list is not in the repository: ``CLOSED_LOOP_DENY_LIST`` names a file outside it, and the
test skips when the variable is not set. No name from that list is written in this tree.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from closed_loop.eval.report import records_from_json
from tests.docs import readme_checks as rc
from tests.docs.fixtures import GOOD_README, small_records

README = rc.ROOT / "README.md"
RESULTS = rc.ROOT / "benchmarks" / "results.json"
QUICK_RESULTS = rc.ROOT / "benchmarks" / ".cache" / "quick-output" / "results.json"

# Numbers that may stand in the prose without being a record, each with its reason. The prose may
# state a registered quantity; it may not state a result that is not in results.json.
ALLOWED_NUMBERS = {
    "0.05": "the registered level alpha (pre-registration section 2)",
    "0.90": "the nominal level of the scheduling interval (section 2)",
    "0.9": "the same nominal level, written short",
    "0.95": "the nominal per-life level, 1 minus alpha_life (section 2)",
    "0.03": "the registered band of C-PERC and the coverage tolerance (section 7)",
    "0.02": "the registered C-PLAN margin and the C-MON false-alarm bound (section 7)",
    "0.1": "the registered C-PLAN safety margin (section 7)",
    "0.5": "the registered detection bound of C-MON and the vacuity ratio (section 7)",
    "18": "the 0.05 / 18 level of the two C-PERC verdicts (section 6)",
    "9": "the nine primary claims (section 6)",
    "30": "days: the detection window of C-MON and the look-back of the attribution (section 7)",
    "90": "days in an episode (section 2)",
    "3.11": "the Python versions of the CI matrix",
    "3.12": "the Python versions of the CI matrix",
    "2026": "the year of the licence, the citation file and the freeze",
    "/^000[1-3]$/": "the numbers of the three decision records, 0001 to 0003",
    "4.50": "the worst shape in the simulation of pre-registration section 6, quoted from there",
}
# Absolutes that may stand in the prose, each with its reason. A key is a piece of the sentence.
ALLOWED_ABSOLUTES = {
    'never "safe"': "quotes the registered wording of C-SHIELD (pre-registration section 7)",
}


def _real_inputs() -> tuple[str, list]:
    if not README.exists():
        pytest.skip("README.md is missing")
    text = README.read_text(encoding="utf-8").replace("\r\n", "\n")
    if rc.is_stub(text):
        pytest.skip("README.md is still the scaffold")
    if not RESULTS.exists():
        pytest.skip("benchmarks/results.json is missing: no full run has been published")
    records = records_from_json(RESULTS.read_text(encoding="utf-8"))
    if not any(r.name == "run/params" for r in records):
        pytest.skip("benchmarks/results.json is a stub")
    return text, records


def _block_span(text: str) -> tuple[int, int] | None:
    script = rc.load_script("render_readme_results")
    try:
        return script.block_span(text)
    except script.MarkerError:
        return None


# ----------------------------------------------------------------------------- the real README


class TestTheRealReadme:
    def test_the_results_block_is_what_the_script_generates_from_results_json(self) -> None:
        text, records = _real_inputs()
        assert rc.block_problems(text, records) == []

    def test_every_number_in_the_prose_is_a_record_or_on_the_allow_list(self) -> None:
        text, records = _real_inputs()
        prose = rc.prose_of(text, _block_span(text))
        assert rc.unknown_numbers(prose, records, ALLOWED_NUMBERS) == []

    def test_the_readme_meets_the_owners_standards(self) -> None:
        text, _records = _real_inputs()
        assert rc.standards_problems(text) == []

    def test_no_absolutes_in_the_prose(self) -> None:
        text, _records = _real_inputs()
        prose = rc.prose_of(text, _block_span(text))
        assert rc.absolute_hits(prose, ALLOWED_ABSOLUTES) == []


class TestTheAllowListsAreHonest:
    def test_every_allowed_number_and_absolute_gives_its_reason(self) -> None:
        assert rc.empty_reasons(ALLOWED_NUMBERS) == []
        assert rc.empty_reasons(ALLOWED_ABSOLUTES) == []

    def test_the_lists_are_short(self) -> None:
        assert len(ALLOWED_NUMBERS) <= 20 and len(ALLOWED_ABSOLUTES) <= 5


class TestTheDenyList:
    def test_no_name_on_the_deny_list_is_anywhere_in_the_tree(self) -> None:
        path = os.environ.get("CLOSED_LOOP_DENY_LIST")
        if not path:
            pytest.skip("CLOSED_LOOP_DENY_LIST is not set")
        names = rc.read_deny_list(Path(path))
        assert names, "the deny-list file holds no name"
        hits = rc.deny_hits(rc.ROOT, names)
        assert hits == [], f"{len(hits)} lines hold a denied name (path, line): {hits[:20]}"


# ----------------------------------------------------------------------------- the checks, proved


def _readme_with_block() -> tuple[str, list]:
    script = rc.load_script("render_readme_results")
    records = small_records()
    block = script.render_block(records)
    return GOOD_README.replace("{block}", block), records


class TestTheResultsBlockCheck:
    def test_a_block_made_from_the_records_passes(self) -> None:
        text, records = _readme_with_block()
        assert rc.block_problems(text, records) == []

    def test_one_changed_digit_in_the_block_fails(self) -> None:
        text, records = _readme_with_block()
        start = text.index("<!-- RESULTS:BEGIN")
        assert "0.0431" in text[start:]
        bad = text[:start] + text[start:].replace("0.0431", "0.0432", 1)
        problems = rc.block_problems(bad, records)
        assert problems and "differs" in problems[0]

    def test_a_block_from_other_records_fails(self) -> None:
        text, records = _readme_with_block()
        other = [r for r in records if r.name != "claim/C-PLAN/cost_diff/P1/nominal"]
        assert rc.block_problems(text, other) != []

    def test_a_missing_marker_is_reported(self) -> None:
        text, records = _readme_with_block()
        problems = rc.block_problems(text.replace("<!-- RESULTS:END -->", ""), records)
        assert problems and "exactly one" in problems[0]

    def test_text_added_inside_the_block_fails(self) -> None:
        text, records = _readme_with_block()
        bad = text.replace("### Verdicts", "### Verdicts\n\nA hand-written sentence.")
        assert rc.block_problems(bad, records) != []


class TestTheNumberCheck:
    def prose(self) -> tuple[str, list]:
        text, records = _readme_with_block()
        return rc.prose_of(text, _block_span(text)), records

    def test_the_good_readme_prose_passes_with_the_allow_list(self) -> None:
        prose, records = self.prose()
        assert rc.unknown_numbers(prose, records, ALLOWED_NUMBERS) == []

    def test_without_the_allow_list_the_registered_level_is_unknown(self) -> None:
        prose, records = self.prose()
        assert rc.unknown_numbers(prose, records, {}) == ["0.05"]

    def test_a_number_that_is_not_a_record_fails(self) -> None:
        prose, records = self.prose()
        bad = prose.replace("0.0431", "0.0521")
        assert "0.0521" in rc.unknown_numbers(bad, records, ALLOWED_NUMBERS)

    def test_a_count_that_is_not_a_record_fails(self) -> None:
        prose, records = self.prose()
        assert "641" in rc.unknown_numbers(prose.replace("640", "641"), records, ALLOWED_NUMBERS)

    def test_a_number_in_code_the_block_or_a_link_is_not_read(self) -> None:
        text, records = _readme_with_block()
        text += "\n```python\nx = 12345\n```\nSee [the tests](https://example.invalid/987654).\n"
        text += "A `77777` literal.\n"
        prose = rc.prose_of(text, _block_span(text))
        assert rc.unknown_numbers(prose, records, ALLOWED_NUMBERS) == []

    def test_identifiers_with_digits_are_not_numbers(self) -> None:
        records = small_records()
        prose = "Arms P1, H-C1 and B3-V2L, claims C-PERC and the V2L variant, F2 and R_in."
        assert rc.unknown_numbers(prose, records, {}) == []

    def test_a_percentage_is_checked_against_the_record_times_a_hundred(self) -> None:
        records = small_records()
        assert rc.unknown_numbers("The throughput was 93.45% of nominal.", records, {}) == []
        assert rc.unknown_numbers("The throughput was 94.45% of nominal.", records, {}) == [
            "94.45%"
        ]

    def test_a_pattern_entry_of_the_allow_list_is_honoured(self) -> None:
        records = small_records()
        assert rc.unknown_numbers("See 0002 and 0009.", records, {"/^000[1-3]$/": "ADRs"}) == [
            "0009"
        ]

    def test_thousands_commas_are_read_as_one_number(self) -> None:
        records = small_records()
        assert rc.number_tokens("4,776 tests and 1,2") == [
            ("4776", False),
            ("1", False),
            ("2", False),
        ]
        assert rc.unknown_numbers("It had 640 episodes.", records, {}) == []

    def test_an_entry_without_a_reason_is_flagged(self) -> None:
        assert rc.empty_reasons({"0.05": "alpha", "0.3": " "}) == ["0.3"]


class TestTheStandardsCheck:
    def test_the_good_readme_passes(self) -> None:
        text, _ = _readme_with_block()
        assert rc.standards_problems(text) == []

    def test_a_second_h1(self) -> None:
        text, _ = _readme_with_block()
        assert any("H1" in p for p in rc.standards_problems(text + "\n# Another\n"))

    def test_a_fifth_or_a_third_badge(self) -> None:
        text, _ = _readme_with_block()
        extra = "[![x](https://example.invalid/b.svg)](https://example.invalid)\n"
        assert any(
            "4 badges" in p
            for p in rc.standards_problems(
                text.replace("\n> A closed", f"\n{extra}\n> A closed", 1)
            )
        )
        first_line = next(ln for ln in text.splitlines() if ln.startswith("[![License"))
        assert any(
            "4 badges" in p for p in rc.standards_problems(text.replace(first_line + "\n", "", 1))
        )

    def test_ci_must_be_the_first_badge(self) -> None:
        text, _ = _readme_with_block()
        lines = text.splitlines()
        i = next(k for k, ln in enumerate(lines) if ln.startswith("[![tests]"))
        lines[i], lines[i + 1] = lines[i + 1], lines[i]
        assert any("first badge" in p for p in rc.standards_problems("\n".join(lines)))

    @pytest.mark.parametrize("section", list(rc.REQUIRED_SECTIONS))
    def test_a_missing_required_section(self, section: str) -> None:
        text, _ = _readme_with_block()
        bad = re.sub(rf"^## {section}.*$", "## Something else", text, flags=re.MULTILINE)
        assert any(section in p for p in rc.standards_problems(bad))

    def test_sections_out_of_order(self) -> None:
        text, _ = _readme_with_block()
        bad = (
            text.replace("## Limitations", "## TMP")
            .replace("## Design decisions", "## Limitations")
            .replace("## TMP", "## Design decisions")
        )
        assert any("out of order" in p for p in rc.standards_problems(bad))

    def test_no_mermaid_diagram(self) -> None:
        text, _ = _readme_with_block()
        assert any(
            "mermaid" in p for p in rc.standards_problems(text.replace("```mermaid", "```text"))
        )

    def test_the_blockquote_must_follow_the_badges_and_be_one_line(self) -> None:
        text, _ = _readme_with_block()
        assert any(
            "blockquote" in p for p in rc.standards_problems(text.replace("> A closed", "A closed"))
        )
        two = text.replace(
            "a pre-registered run.\n", "a pre-registered run.\n> A second quote line.\n"
        )
        assert any("blockquote" in p for p in rc.standards_problems(two))

    def test_the_footer_must_be_the_last_line_and_exact(self) -> None:
        text, _ = _readme_with_block()
        assert any(
            "footer" in p for p in rc.standards_problems(text.replace("Agulhas Code", "Agulhas"))
        )
        assert any("footer" in p for p in rc.standards_problems(text + "\nA late paragraph.\n"))

    def test_a_scaffold_is_recognised_as_a_stub(self) -> None:
        assert rc.is_stub("# closed-loop\n\n> Scaffold only.\n")
        assert rc.is_stub("# closed-loop\n\nText with no section.\n")
        assert not rc.is_stub(GOOD_README)


class TestTheAbsolutesCheck:
    def test_each_flagged_word_is_found(self) -> None:
        for word in ("never", "always", "100%", "guarantee", "guarantees", "guaranteed", "every"):
            hits = rc.absolute_hits(f"The shield {word} fails.", {})
            assert hits and hits[0][0] == word.lower(), word

    def test_ordinary_words_are_left_alone(self) -> None:
        assert (
            rc.absolute_hits("The shield held in 640 episodes, and everything ran offline.", {})
            == []
        )

    def test_a_reasoned_entry_lets_one_sentence_through_and_only_that_one(self) -> None:
        prose = 'It reports "0 of n", never "safe". The shield always works.'
        allow = {'never "safe"': "quotes the registered wording"}
        assert rc.absolute_hits(prose, allow) == [("always", "The shield always works.")]

    def test_the_results_block_is_not_prose(self) -> None:
        text, _ = _readme_with_block()
        text = text.replace("### Verdicts", "### Verdicts\n\nEvery claim is shown.", 1)
        prose = rc.prose_of(text, _block_span(text))
        assert rc.absolute_hits(prose, {}) == []


class TestTheDenyCheck:
    def test_a_denied_name_in_a_text_file_is_found_by_path_and_line_only(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "a.md").write_text("fine\nfor Zorblax Mining only\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("# nothing\n", encoding="utf-8")
        hits = rc.deny_hits(tmp_path, ["zorblax mining"])
        assert hits == [("docs/a.md", 2)]
        assert "zorblax" not in repr(hits).lower()

    def test_caches_and_binaries_are_skipped(self, tmp_path: Path) -> None:
        (tmp_path / ".cache").mkdir()
        (tmp_path / ".cache" / "x.txt").write_text("Zorblax", encoding="utf-8")
        (tmp_path / "blob.bin").write_bytes(b"\xff\xfeZorblax\x00")
        assert rc.deny_hits(tmp_path, ["Zorblax"]) == []

    def test_comments_and_blank_lines_of_the_list_are_not_names(self, tmp_path: Path) -> None:
        f = tmp_path / "deny.txt"
        f.write_text("# a comment\n\nZorblax\n  \nQuuxCo\n", encoding="utf-8")
        assert rc.read_deny_list(f) == ["Zorblax", "QuuxCo"]


# ----------------------------------------------------------------------------- the quick output


@pytest.fixture(scope="module")
def quick() -> list:
    if not QUICK_RESULTS.exists():
        pytest.skip("no quick output under benchmarks/.cache/quick-output/")
    return records_from_json(QUICK_RESULTS.read_text(encoding="utf-8"))


class TestOnTheQuickOutput:
    """The checks on real records: the output of the quick run, which is not a result but has the
    shape of one (every table filled, the real formatting of every number)."""

    def test_the_block_of_the_quick_run_passes_the_block_check_and_the_standards(
        self, quick: list
    ) -> None:
        script = rc.load_script("render_readme_results")
        block = script.render_block(quick)
        text = GOOD_README.replace("{block}", block)
        assert rc.block_problems(text, quick) == []
        assert rc.standards_problems(text) == []
        assert "QUICK RUN, NOT A RESULT" in block

    def test_every_number_the_generator_prints_is_among_the_records(self, quick: list) -> None:
        script = rc.load_script("render_readme_results")
        block = script.render_block(quick)
        body = re.sub(r"<!--.*?-->", " ", block, flags=re.DOTALL)
        assert rc.unknown_numbers(body, quick, {}) == []

    def test_a_changed_digit_in_the_quick_block_is_caught(self, quick: list) -> None:
        script = rc.load_script("render_readme_results")
        block = script.render_block(quick)
        digit = next(m for m in re.finditer(r"\| (\d\.\d{4}) \|", block))
        wrong = block[: digit.start(1)] + "9.9999" + block[digit.end(1) :]
        text = GOOD_README.replace("{block}", wrong)
        assert rc.block_problems(text, quick) != []

    def test_the_prose_check_reads_the_quick_records_too(self, quick: list) -> None:
        index = rc.NumberIndex(quick)
        some = next(r for r in quick if r.name.endswith(".m_cost") and r.n > 1)
        assert index.matches(f"{some.value:.4f}")
        assert not index.matches("123456.789")
