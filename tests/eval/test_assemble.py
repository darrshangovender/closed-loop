"""0003 A18, F3: claim assembly is not simulation, so a fix to it does not move a cell key.

The hash is taken over a copy of the real package (not a toy tree), so the test is about the files
as they are: editing ``assemble.py`` leaves ``simulation_code_sha256`` alone, editing ``stages.py``
moves it.
"""

from __future__ import annotations

import ast
import dataclasses
import shutil
from pathlib import Path

import pytest

from closed_loop.eval import assemble, runner, stages
from closed_loop.eval.episode import EpisodeResult
from closed_loop.eval.runner import simulation_code_sha256

ROOT = Path(runner.__file__).resolve().parents[2]


def sha(root: Path) -> str:
    return simulation_code_sha256.__wrapped__(str(root))  # type: ignore[attr-defined]


@pytest.fixture
def package_copy(tmp_path: Path) -> Path:
    shutil.copytree(
        ROOT / "closed_loop",
        tmp_path / "closed_loop",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    return tmp_path


class TestAssemblyIsNotSimulation:
    def test_editing_assemble_does_not_move_the_hash(self, package_copy: Path) -> None:
        before = sha(package_copy)
        target = package_copy / "closed_loop/eval/assemble.py"
        target.write_bytes(target.read_bytes() + b"\n# a one-line fix to a verdict input\n")
        assert sha(package_copy) == before

    def test_editing_stages_moves_it(self, package_copy: Path) -> None:
        before = sha(package_copy)
        target = package_copy / "closed_loop/eval/stages.py"
        target.write_bytes(target.read_bytes() + b"\n# a change to a worker task\n")
        assert sha(package_copy) != before

    def test_the_module_is_on_the_list_and_the_list_still_leaves_the_workers_out(self) -> None:
        assert "closed_loop/eval/assemble.py" in runner._NOT_SIMULATION
        assert "closed_loop/eval/stages.py" not in runner._NOT_SIMULATION
        assert "closed_loop/eval/runner.py" not in runner._NOT_SIMULATION


class TestWhatMovedAndWhatDidNot:
    def test_the_assembly_lives_in_assemble_and_not_in_stages(self) -> None:
        for name in (
            "claim_records",
            "_pooled_gap_a",
            "_shield_records",
            "_ablation_records",
            "_ABLATIONS",
            "_SHIELDED_ARMS",
        ):
            assert hasattr(assemble, name), name
            assert not hasattr(stages, name), name

    def test_assemble_defines_no_class_and_so_no_worker_task(self) -> None:
        tree = ast.parse(Path(assemble.__file__).read_text(encoding="utf-8"))
        assert not [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]

    def test_stages_reaches_assemble_only_inside_a_function(self) -> None:
        """A module-level import would make the simulation hash depend on a file it leaves out."""
        tree = ast.parse(Path(stages.__file__).read_text(encoding="utf-8"))
        top = [
            n
            for n in tree.body
            if isinstance(n, ast.ImportFrom | ast.Import) and "assemble" in ast.dump(n)
        ]
        assert top == []
        inner = [
            n
            for f in ast.walk(tree)
            if isinstance(f, ast.FunctionDef)
            for n in ast.walk(f)
            if isinstance(n, ast.ImportFrom) and n.module == "closed_loop.eval.assemble"
        ]
        assert inner, "run_pipeline must import the assembly where it calls it"


def episode(learner: dict[str, int], c: float | None = 0.9, n: int = 50) -> EpisodeResult:
    from closed_loop.core.types import Fidelity
    from closed_loop.eval.metrics import EpisodeCoverage

    return EpisodeResult(
        arm="B3-V1", seed=1, condition="nominal", fidelity=Fidelity.F1, violations={},
        min_robustness={}, throughput=1.0, cost=0.0, unplanned_failures=0, planned_jobs=0,
        shield_overrides=0, shield_admitted=0, monitor_fired_day=None, labels=(), wall_s=0.0,
        coverage=EpisodeCoverage(c, c, n, c, c, n, 0, n), learner=learner,
    )  # fmt: skip


def counts(hit: int, miss: int, lower_hit: int, unknown: int, uncensored: int) -> dict[str, int]:
    total = hit + miss + lower_hit + unknown
    out = {}
    for pop in ("post_blind", "estimated"):
        out |= {
            f"{pop}.n": total, f"{pop}.hit": hit, f"{pop}.miss": miss,
            f"{pop}.lower_hit": lower_hit, f"{pop}.unknown": unknown,
            f"{pop}.uncensored": uncensored,
        }  # fmt: skip
    return out


class TestGapAWithNoObservedFailure:
    """0003 A18, F1. Under B3 at nominal nothing fails, so the labels of a life that never failed
    cannot be HITs and ``C_lrn`` is 0 by construction."""

    def test_a_cell_with_no_uncensored_label_has_no_gap_a_only_counts_and_bounds(self) -> None:
        # the review's cell: 3,983 UNKNOWN, 3,976 LOWER_HIT, 13 MISS, 0 HIT, none uncensored
        cell = [episode(counts(0, 13, 3976, 3983, 0))]
        gap = assemble._pooled_gap_a(cell, "pooled")
        assert gap is not None
        assert gap["undefined"] == 1.0 and "c_lrn" not in gap and "gap_a" not in gap
        assert gap["manski_lo"] == 0.0 and gap["manski_hi"] == pytest.approx((7972 - 13) / 7972)
        assert gap["n_labels"] == 7972 and gap["uncensored"] == 0
        assert (gap["hit"], gap["miss"], gap["lower_hit"], gap["unknown"]) == (0, 13, 3976, 3983)
        assert "no way to confirm an upper bound" in str(gap["reason"])

    def test_a_cell_with_uncensored_labels_is_unchanged_even_if_hit_is_zero(self) -> None:
        cell = [episode(counts(0, 3, 10, 7, 3))]
        gap = assemble._pooled_gap_a(cell, "pooled")
        assert gap is not None and "undefined" not in gap
        assert gap["c_lrn"] == 0.0 and gap["gap_a"] == pytest.approx(0.0 - 0.9)
        cell = [episode(counts(20, 4, 2, 4, 24))]
        gap = assemble._pooled_gap_a(cell, "pooled")
        assert gap is not None and gap["c_lrn"] == pytest.approx(20 / 26)

    def test_the_uncensored_count_is_pooled_over_episodes(self) -> None:
        cell = [episode(counts(0, 0, 5, 5, 0)), episode(counts(1, 0, 5, 5, 1))]
        gap = assemble._pooled_gap_a(cell, "pooled")
        assert gap is not None and "undefined" not in gap and gap["uncensored"] == 1

    def test_labels_that_say_nothing_still_leave_a_record_with_the_bounds(self) -> None:
        """All UNKNOWN: nothing is informative and nothing is uncensored; the record stays, with
        the Manski bounds [0, 1], flagged undefined."""
        gap = assemble._pooled_gap_a([episode(counts(0, 0, 0, 40, 0))], "pooled")
        assert gap is not None and gap["undefined"] == 1.0 and gap["informative"] == 0.0
        assert (gap["manski_lo"], gap["manski_hi"]) == (0.0, 1.0)

    def test_no_labels_at_all_is_no_record(self) -> None:
        assert assemble._pooled_gap_a([episode(counts(0, 0, 0, 0, 0))], "pooled") is None

    def test_the_report_prints_the_reason_and_the_bounds_not_a_number(self) -> None:
        from closed_loop.core.results import ResultRecord
        from closed_loop.core.types import Fidelity
        from closed_loop.eval.report import _gap_a_text

        gap = assemble._pooled_gap_a([episode(counts(0, 13, 3976, 3983, 0))], "pooled")
        assert gap is not None
        rec = ResultRecord(
            name="gapa.nominal.pooled.V1",
            value=float(gap.get("gap_a", 0.0)),
            n=int(gap["informative"]),
            fidelity=Fidelity.F1, measured_on="a cell", script="tests", extra=gap,
        )  # fmt: skip
        text = _gap_a_text(rec)
        assert text.startswith("undefined: no life ended in an observed failure")
        assert "Manski bounds 0.000 to 0.998" in text
        assert "-0.92" not in text and "0.0000" not in text
        defined = assemble._pooled_gap_a([episode(counts(20, 4, 2, 4, 24))], "pooled")
        assert defined is not None
        rec2 = ResultRecord(
            name="gapa.nominal.pooled.V1", value=defined["gap_a"], n=26, fidelity=Fidelity.F1,
            measured_on="a cell", script="tests", extra=defined,
        )  # fmt: skip
        assert _gap_a_text(rec2) == f"{defined['gap_a']:.4f}"
        assert _gap_a_text(None) == "-"


class FakeStore:
    """The two methods of ``CellStore`` that ``_uncovered_records`` reads."""

    def __init__(self, cells: dict[tuple[str, str], list[EpisodeResult]]) -> None:
        self._cells = cells

    def cells(self) -> list[tuple[str, str]]:
        return list(self._cells)

    def results(self, arm: str, condition: str) -> list[EpisodeResult]:
        return self._cells.get((arm, condition), [])


def with_uncovered(n: int) -> EpisodeResult:
    return dataclasses.replace(episode({}), uncovered=n)


class TestUncoveredRecords:
    """P12g, G7: ``shield.uncovered.<arm>.<condition>`` for every shielded cell."""

    def test_a_shielded_cell_gets_episodes_with_any_of_n_and_the_total(self) -> None:
        store = FakeStore({("P1", "nominal"): [with_uncovered(n) for n in (0, 2, 0, 1)]})
        (rec,) = assemble._uncovered_records(store)  # type: ignore[arg-type]
        assert rec.name == "shield.uncovered.P1.nominal"
        assert (rec.value, rec.n, rec.extra) == (3.0, 4, {"k": 2, "total": 3})
        assert "episodes with any" in rec.measured_on and "STOP|STOP" in rec.measured_on

    def test_a_cell_that_never_needed_the_fallback_is_a_record_of_zero_not_a_gap(self) -> None:
        store = FakeStore({("H-C1", "shifted"): [with_uncovered(0)] * 3})
        (rec,) = assemble._uncovered_records(store)  # type: ignore[arg-type]
        assert (rec.value, rec.n, rec.extra) == (0.0, 3, {"k": 0, "total": 0})

    def test_an_arm_without_the_shield_has_none_and_neither_has_an_empty_cell(self) -> None:
        store = FakeStore(
            {
                ("B0", "nominal"): [with_uncovered(0)],  # no shield
                ("P1-I", "nominal"): [with_uncovered(0)],  # the interlock, not the shield
                ("P1", "shifted"): [],
            }
        )
        assert assemble._uncovered_records(store) == []  # type: ignore[arg-type]

    def test_both_conditions_of_one_arm_are_two_records(self) -> None:
        store = FakeStore(
            {("P1", "nominal"): [with_uncovered(1)], ("P1", "shifted"): [with_uncovered(0)]}
        )
        names = [r.name for r in assemble._uncovered_records(store)]  # type: ignore[arg-type]
        assert names == ["shield.uncovered.P1.nominal", "shield.uncovered.P1.shifted"]

    def test_every_shielded_arm_is_covered_the_stale_clause_one_included(self) -> None:
        from closed_loop.eval.arms import ARMS

        shielded = [a for a, arm in ARMS.items() if arm.shield]
        assert "S-stale" in shielded and "P1" in shielded and "B0" not in shielded
        store = FakeStore({(a, "nominal"): [with_uncovered(1)] for a in ARMS})
        records = assemble._uncovered_records(store)  # type: ignore[arg-type]
        assert {r.name.split(".")[2] for r in records} == set(shielded)

    def test_the_claim_records_include_them(self) -> None:
        src = Path(assemble.__file__).read_text(encoding="utf-8")
        assert "out.extend(_uncovered_records(store))" in src

    def test_the_count_survives_slimming_and_the_cache(self, tmp_path: Path) -> None:
        from closed_loop.eval.runner import CellCache, Done, slim_result

        r = with_uncovered(4)
        assert slim_result(r).uncovered == 4
        cache = CellCache(tmp_path, "k", {"x": 1})
        cache.append(Done(1, slim_result(r)))
        loaded = CellCache(tmp_path, "k", {"x": 1}).load()
        assert loaded[1].value.uncovered == 4  # type: ignore[union-attr]


class TestPlantOnlyEventsAreOwnedBySomeone:
    """0003 A18, F4. The shield verdict no longer calls a plant-only event a contract failure."""

    def test_p1_nopair_is_in_the_defect_scan(self) -> None:
        assert "P1-nopair" in assemble._SHIELDED_ARMS
        assert "S-stale" not in assemble._SHIELDED_ARMS
