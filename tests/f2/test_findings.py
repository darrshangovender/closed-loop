"""P11c: the findings table is checked data.

Every row has a mechanised check, every cited line exists in the installed sibling and holds the
text the row relies on, and the wording stays free of absolutes and of references to our internal
documents. The siblings are read as files, never imported here.
"""

from __future__ import annotations

import re
from importlib.util import find_spec
from pathlib import Path

import pytest

from closed_loop.f2.findings import (
    FINDINGS,
    STATUS,
    Location,
    render_document,
    render_measured,
    render_table,
)
from closed_loop.f2.mismatch import CHECKS

REPOSITORIES = Path(__file__).resolve().parents[3]
IDS = [f"M{i}" for i in range(1, 22)] + [f"S{i}" for i in range(1, 13)]
LOCATIONS = [(f.id, i, loc) for f in FINDINGS for i, loc in enumerate(f.locations)]
FORBIDDEN = re.compile(r"\b(never|always|guaranteed)\b|100\s?%", re.IGNORECASE)
INTERNAL = re.compile(
    r"\b(ADR|packet|progress note|P1[0-3][a-c]?|Mnesarchus|Boethus)\b", re.IGNORECASE
)


def installed_file(loc: Location) -> Path | None:
    """The file a location names: in the installed package, or in the repository beside ours."""
    if loc.in_package:
        package = loc.path.split("/")[0]
        spec = find_spec(package)
        if spec is None or not spec.submodule_search_locations:
            return None
        return Path(next(iter(spec.submodule_search_locations))).parent / loc.path
    path = REPOSITORIES / loc.repo / loc.path
    return path if path.exists() else None


class TestTheRows:
    def test_there_are_33_findings_with_the_ids_of_21_mismatches_and_12_defects(self) -> None:
        assert [f.id for f in FINDINGS] == IDS
        assert sum(f.kind == "mismatch" for f in FINDINGS) == 21
        assert sum(f.kind == "defect" for f in FINDINGS) == 12

    def test_every_row_is_recorded_not_fixed_with_a_known_severity_and_check(self) -> None:
        for f in FINDINGS:
            assert f.status == STATUS == "recorded, not fixed"
            assert f.severity in ("low", "medium", "high"), f.id
            assert f.check in ("runs the siblings", "reads their source"), f.id

    def test_every_row_has_a_mechanised_check_and_every_check_a_row(self) -> None:
        assert set(CHECKS) == {f.id for f in FINDINGS}

    def test_a_row_says_something_and_points_somewhere(self) -> None:
        for f in FINDINGS:
            assert f.title.endswith(".") and len(f.text) >= 80, f.id
            assert f.locations, f.id

    def test_the_high_severity_rows_are_the_ones_that_break_a_hand_off(self) -> None:
        high = {f.id for f in FINDINGS if f.severity == "high"}
        assert {"M4", "M7", "S1", "S6", "S11"} <= high

    def test_enough_locations_pin_the_text_they_rely_on(self) -> None:
        pinned = sum(1 for _, _, loc in LOCATIONS if loc.needle is not None)
        assert pinned >= 40


@pytest.mark.parametrize(
    ("finding", "index", "loc"), LOCATIONS, ids=[f"{a}-{b}" for a, b, _ in LOCATIONS]
)
def test_the_cited_line_exists_and_holds_the_text(finding: str, index: int, loc: Location) -> None:
    path = installed_file(loc)
    if path is None:
        pytest.skip(f"{loc.repo} source not found beside this repository for {loc.path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert loc.last <= len(lines), (
        f"{loc.path} has {len(lines)} lines, row {finding} cites {loc.last}"
    )
    if loc.needle is not None:
        cited = "\n".join(lines[loc.first - 1 : loc.last])
        assert loc.needle in cited, f"{finding}: {loc.needle!r} is not at {loc.path}:{loc.lines}"


class TestTheWording:
    def test_no_absolutes_and_no_internal_references_in_the_document(self) -> None:
        text = render_document(render_measured(None, None))
        assert not FORBIDDEN.search(text), FORBIDDEN.search(text)
        assert not INTERNAL.search(text), INTERNAL.search(text)

    def test_the_document_says_it_is_synthetic_and_that_nothing_was_fixed(self) -> None:
        text = render_document(render_measured(None, None))
        assert "SYNTHETIC" in text and "None has been fixed" in text

    def test_the_table_has_one_row_per_finding(self) -> None:
        rows = [line for line in render_table().splitlines() if re.match(r"\| [MS]\d+ \|", line)]
        assert [r.split("|")[1].strip() for r in rows] == IDS
        assert all(r.count("recorded, not fixed") == 1 for r in rows)

    def test_a_generated_section_without_data_says_how_to_generate_it(self) -> None:
        text = render_measured(None, None)
        assert "Not generated" in text and "f2_measure.py" in text

    def test_the_measured_section_copies_the_numbers_it_is_given(self) -> None:
        episodes = {
            "seed": 1023,
            "context": "default",
            "budget_s_per_episode": 30.0,
            "composed_estimate_s": 14.0,
            "episodes": [
                {
                    "arm": "F2-B3",
                    "wall_s": 12.34,
                    "cpu_s": 11.0,
                    "run_asset_calls": 4,
                    "run_asset_calls_by_the_caching_rule": 4,
                    "analyse_run_calls": 7,
                    "alerts_raised": 5,
                    "drafts_naming_a_mode": 1,
                    "drafts_inspection_first": 4,
                    "alerts_without_a_draft": 0,
                    "repairs_issued": 1,
                    "wrong_compliant_signed": 0,
                    "unplanned_failures": 2,
                }
            ],
        }
        text = render_measured(episodes, None, [("M1", True, "seen")])
        assert "| F2-B3 | 12.34 | 11.0 | 4 | 4 | 7 | 5 | 1 | 4 | 0 | 1 | 0 | 2 |" in text
        assert "| M1 | yes | seen |" in text
