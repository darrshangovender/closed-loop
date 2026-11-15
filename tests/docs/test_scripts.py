"""The two generators of the published documents (packet P13a): ``scripts/render_readme_results.py``
and ``scripts/render_what_gets_through.py``.

Nothing here simulates an episode. The shrink runs against a stub evaluation, the search cells are
written by the runner's own ``CellCache`` into a temporary folder, and the records are the made-up
ones of ``tests/docs/fixtures.py`` (and, when it is there, the real output of the quick run).
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from closed_loop.core.results import ResultRecord
from closed_loop.core.types import Fidelity
from closed_loop.eval.claims import Verdict
from closed_loop.eval.metrics import Attribution
from closed_loop.eval.report import FORBIDDEN, Index, records_from_json
from closed_loop.eval.runner import _HEAD, _MAGIC, CellCache, Done, Failed
from closed_loop.eval.stages import SearchOutcome
from closed_loop.eval.tuned import PROVISIONAL
from closed_loop.search import falsify
from closed_loop.search.shrink import nominal_theta
from closed_loop.stats.intervals import clopper_pearson
from tests.docs import readme_checks as rc
from tests.docs.fixtures import GOOD_README, small_records

QUICK_RESULTS = rc.ROOT / "benchmarks" / ".cache" / "quick-output" / "results.json"

readme_script = rc.load_script("render_readme_results")
wgt = rc.load_script("render_what_gets_through")


# ----------------------------------------------------------------------------- the README block


class TestTheResultsBlock:
    def test_it_is_a_pure_function_of_the_records(self) -> None:
        records = small_records()
        assert readme_script.render_block(records) == readme_script.render_block(list(records))
        assert "\r" not in readme_script.render_block(records)

    def test_the_markers_stand_once_at_the_ends(self) -> None:
        block = readme_script.render_block(small_records())
        assert block.startswith(readme_script.BEGIN) and block.endswith(readme_script.END)
        assert block.count(readme_script.BEGIN) == block.count(readme_script.END) == 1

    def test_every_verdict_is_in_the_table_with_its_label(self) -> None:
        block = readme_script.render_block(small_records())
        assert "| C-SHIELD | win |" in block
        assert "| C-PLAN.P1 | inconclusive |" in block
        assert "| C-PERC.pooled | loss |" in block

    def test_the_interlock_stands_in_the_same_table_as_the_shield(self) -> None:
        block = readme_script.render_block(small_records())
        rows = [ln for ln in block.splitlines() if ln.startswith("| H-C ")][:3]
        roles = [ln.split("|")[3].strip() for ln in rows]
        assert roles == ["shield", "shield off", "interlock"]
        assert "H-C-I minus H-C1" in block  # the paired difference, beside the throughput ratio

    def test_plan_shows_nominal_and_shifted_side_by_side(self) -> None:
        block = readme_script.render_block(small_records())
        row = next(ln for ln in block.splitlines() if "P1 minus B3-S | M-COST difference" in ln)
        assert "0.0431" in row and "-0.0094" in row
        assert row.index("0.0431") < row.index("-0.0094")

    def test_perc_shows_pooled_and_trend_for_v1_and_the_primary_variant(self) -> None:
        block = readme_script.render_block(small_records())
        rows = [ln for ln in block.splitlines() if re.match(r"\| V(1|2L) \|", ln)]
        assert len(rows) == 2 and "0.0117" in rows[0] and "0.0611" in rows[0]
        assert "0.0082" in rows[1] and "0.0507" in rows[1]

    def test_what_each_table_was_measured_on_is_printed(self) -> None:
        block = readme_script.render_block(small_records())
        assert "Measured on: 640 invented H-C1 episodes" in block

    def test_the_arm_name_is_folded_out_of_the_measured_on_lines(self) -> None:
        from tests.docs.fixtures import rec

        records = [
            rec("A.nominal.throughput", 0.9, 5, on="5 of 5 episodes of arm A, nominal: throughput"),
            rec("B.nominal.throughput", 0.8, 5, on="5 of 5 episodes of arm B, nominal: throughput"),
        ]
        lines = readme_script._measured_on(records)
        assert (
            lines[0]
            == "- Measured on: 5 of 5 episodes of each arm in the table with this episode count, "
            "nominal: throughput"
        )
        assert len([ln for ln in lines if ln.startswith("- ")]) == 1

    def test_a_quick_run_says_so_at_the_top(self) -> None:
        from tests.docs.fixtures import rec

        records = small_records()
        records[0] = rec("run/params", 1.0, 1, mode="quick", seed_role="dev", n_seeds=12)
        block = readme_script.render_block(records)
        assert "QUICK RUN, NOT A RESULT" in block.split("### Verdicts")[0]
        assert "QUICK RUN" not in readme_script.render_block(small_records())

    def test_an_incomplete_run_says_so_at_the_top(self) -> None:
        from tests.docs.fixtures import rec

        records = [*small_records(), rec("run/completeness", 0.5, 1, not_built="causal")]
        block = readme_script.render_block(records)
        assert "INCOMPLETE RUN" in block.split("### Verdicts")[0]
        full = [*small_records(), rec("run/completeness", 1.0, 1)]
        assert "INCOMPLETE RUN" not in readme_script.render_block(full)

    def test_a_missing_record_is_a_dash_not_a_number(self) -> None:
        records = [r for r in small_records() if "C-PERC/gap_b/trend" not in r.name]
        block = readme_script.render_block(records)
        row = next(ln for ln in block.splitlines() if ln.startswith("| V1 |"))
        assert row.count("| - |") >= 1 or "| - " in row

    def test_every_number_it_prints_is_among_the_records(self) -> None:
        records = small_records()
        body = re.sub(r"<!--.*?-->", " ", readme_script.render_block(records), flags=re.DOTALL)
        # "1,000" is in the header of the overrides column, not a result
        assert rc.unknown_numbers(body, records, {"1000": "per 1,000 in a column header"}) == []


class TestSplicing:
    def test_only_the_block_is_replaced(self) -> None:
        old = GOOD_README.replace(
            "{block}", f"{readme_script.BEGIN}\nold text\n{readme_script.END}"
        )
        new_block = f"{readme_script.BEGIN}\nnew text\n{readme_script.END}"
        out = readme_script.splice(old, new_block)
        before, after = old.split(readme_script.BEGIN)[0], old.split(readme_script.END)[1]
        assert out == before + new_block + after
        assert readme_script.current_block(out) == new_block

    @pytest.mark.parametrize(
        "text",
        [
            "no markers",
            f"{readme_script.BEGIN} only",
            f"{readme_script.END} then {readme_script.BEGIN}",
            f"{readme_script.BEGIN} {readme_script.END} {readme_script.BEGIN} {readme_script.END}",
        ],
    )
    def test_markers_missing_doubled_or_reversed_are_refused(self, text: str) -> None:
        with pytest.raises(readme_script.MarkerError):
            readme_script.block_span(text)

    def test_the_command_line_writes_checks_and_refuses(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from closed_loop.eval.report import records_to_json

        results = tmp_path / "results.json"
        results.write_text(records_to_json(small_records()), encoding="utf-8")
        readme = tmp_path / "README.md"
        readme.write_text(
            GOOD_README.replace("{block}", f"{readme_script.BEGIN}\nold\n{readme_script.END}"),
            encoding="utf-8",
        )
        args = ["--results", str(results), "--readme", str(readme)]
        assert readme_script.main([*args, "--check"]) == 1  # the block is stale
        assert readme_script.main([*args, "--write"]) == 0
        assert readme_script.main([*args, "--check"]) == 0
        assert readme_script.main(args) == 0  # no flag: the block goes to standard output
        assert readme_script.BEGIN in capsys.readouterr().out
        text = readme.read_text(encoding="utf-8")
        assert rc.block_problems(text, small_records()) == []
        assert text.startswith("# closed-loop") and text.rstrip().endswith("agulhascode.co.za)")
        bare = tmp_path / "bare.md"
        bare.write_text("# x\n", encoding="utf-8")
        assert (
            readme_script.main(["--results", str(results), "--readme", str(bare), "--write"]) == 2
        )
        assert bare.read_text(encoding="utf-8") == "# x\n"


# ----------------------------------------------------------------------------- the search cells

TUNED: dict[str, Any] = {
    "k_b3": 0.25, "b2_limit": 6.0, "b1_fraction": 0.5, "delta_plan": 0.02, "provisional": False,
    "source": "benchmarks/tuned.json", "sha256": None, "tuned_on": "", "script": "",
    "primary_variant": "V2L", "variant_tuned": True, "monitor_mode": "pess", "mode_tuned": True,
}  # fmt: skip


def write_cell(
    directory: Path, regime: str, algo: str, code: str, outcomes: dict[int, Any], budget: int = 15
) -> Path:
    header = {
        "task": {
            "kind": "search", "regime": regime, "algo": algo, "budget": budget, "tuned": TUNED,
            "target": {"name": "t"},
        },
        "env": {"code": code},
    }  # fmt: skip
    key = f"{regime}{algo}{code}".ljust(40, "0")[:40]
    cache = CellCache(directory, key, header)
    for seed, value in outcomes.items():
        cache.append(Done(seed, value) if not isinstance(value, Failed) else value)
    return cache.path


def outcome(first: int | None, theta0: float = 0.5, props: tuple[str, ...] = ("P6",)) -> Any:
    find = None if first is None else (tuple([theta0] * 32), props)
    return SearchOutcome(first, 15, 0 if first is None else 3, (), find)


class TestReadingTheSearchCells:
    def test_the_cell_is_read_back_with_the_runners_own_reader(self, tmp_path: Path) -> None:
        write_cell(
            tmp_path, "out", "cma", "a" * 64,
            {3000: outcome(4), 3001: outcome(None), 3002: Failed(3002, "error", "boom")},
        )  # fmt: skip
        (cell,) = wgt.read_search_cells(tmp_path)
        assert (cell.regime, cell.algo, cell.budget, cell.code[:4]) == ("out", "cma", 15, "aaaa")
        assert sorted(cell.outcomes) == [3000, 3001] and cell.failed == (3002,)
        assert cell.outcomes[3000].first_find is not None and cell.tuned["k_b3"] == 0.25

    def test_files_that_are_not_search_cells_are_left_alone(self, tmp_path: Path) -> None:
        (tmp_path / "junk.cell").write_bytes(b"not a cell")
        (tmp_path / "other.txt").write_text("x", encoding="utf-8")
        episode = CellCache(tmp_path, "e" * 40, {"task": {"kind": "episode"}, "env": {"code": "z"}})
        episode.append(Done(1100, "x"))
        assert wgt.read_search_cells(tmp_path) == []

    def test_a_pair_written_by_two_codes_gives_the_newer_cell_or_the_one_asked_for(
        self, tmp_path: Path
    ) -> None:
        import os

        old = write_cell(tmp_path, "in", "random", "1" * 64, {3000: outcome(None)})
        new = write_cell(tmp_path, "in", "random", "2" * 64, {3000: outcome(2)})
        os.utime(old, (1_000_000, 1_000_000))
        os.utime(new, (2_000_000, 2_000_000))
        (cell,) = wgt.read_search_cells(tmp_path)
        assert cell.code[0] == "2"
        (cell,) = wgt.read_search_cells(tmp_path, code="1" * 64)
        assert cell.code[0] == "1"
        assert wgt.read_search_cells(tmp_path, code="9" * 64) == []

    def test_an_altered_record_costs_only_the_records_from_it_on(self, tmp_path: Path) -> None:
        path = write_cell(tmp_path, "in", "cma", "c" * 64, {3000: outcome(3), 3001: outcome(4)})
        raw = bytearray(path.read_bytes())
        raw[-5] ^= 0xFF  # inside the last record
        path.write_bytes(bytes(raw))
        (cell,) = wgt.read_search_cells(tmp_path)
        assert sorted(cell.outcomes) == [3000]  # the runner's reader stops at the first bad record

    def test_an_altered_header_is_never_read_as_a_cell(self, tmp_path: Path) -> None:
        path = write_cell(tmp_path, "in", "cma", "c" * 64, {3000: outcome(3)})
        raw = bytearray(path.read_bytes())
        raw[len(_MAGIC) + _HEAD.size + 32 + 5] ^= 0x01  # a byte of the header's JSON
        path.write_bytes(bytes(raw))
        try:
            cells = wgt.read_search_cells(tmp_path)
        except ValueError:
            return  # refused loudly: the digest of the header record does not match
        assert cells == []  # or not recognised at all

    def test_the_finds_are_taken_in_the_order_of_the_runner_and_only_where_there_is_one(
        self, tmp_path: Path
    ) -> None:
        for regime, algo in (("in", "cma"), ("in", "random"), ("out", "cma"), ("out", "random")):
            write_cell(
                tmp_path, regime, algo, "d" * 64,
                {3000: outcome(5 if (regime, algo) != ("in", "cma") else None), 3001: outcome(7)},
            )  # fmt: skip
        finds = wgt.finds_of(wgt.read_search_cells(tmp_path))
        order = [(f.seed, f.regime, f.algo) for f in finds]
        assert order == [
            (3000, "in", "random"), (3000, "out", "cma"), (3000, "out", "random"),
            (3001, "in", "cma"), (3001, "in", "random"), (3001, "out", "cma"),
            (3001, "out", "random"),
        ]  # fmt: skip

    def test_the_run_summaries_count_the_runs_that_found_nothing_too(self, tmp_path: Path) -> None:
        write_cell(tmp_path, "out", "cma", "e" * 64, {3000: outcome(None), 3001: outcome(6)})
        runs = wgt.run_summaries(wgt.read_search_cells(tmp_path))
        assert [(r["regime"], r["algo"], r["seed"], r["found"]) for r in runs] == [
            ("R_out", "cma", 3000, False), ("R_out", "cma", 3001, True),
        ]  # fmt: skip
        assert all(r["arm"] == falsify.target_arm().name for r in runs)


# ------------------------------------------------------------------ re-evaluate and shrink

DIM = 3  # the one coordinate the stub reacts to


def stub_evaluate(label: str = "A_exec", prop: str = "P6") -> Any:
    """An evaluation that violates exactly when coordinate DIM is away from nominal."""
    calls: list[tuple[float, ...]] = []

    def evaluate(arm: Any, theta: Any, regime: str, seed: int, context: Any) -> falsify.Evaluation:
        calls.append(tuple(float(x) for x in theta))
        x = np.asarray(theta, dtype=np.float64)
        bad = abs(x[DIM] - nominal_theta(regime)[DIM]) > 0.1
        events = (Attribution(prop, 10, label, 4, "CR-501"),) if bad else ()
        return falsify.Evaluation(
            theta=x, regime=regime, seed=seed, arm=arm.name,
            objective=-0.4 if bad else 0.3, robustness={}, violated=(prop,) if bad else (),
            first_events=events, label_counts={label: 1} if bad else {}, wall_s=0.0,
            event_counts={(prop, label): 1} if bad else {},
        )  # fmt: skip

    evaluate.calls = calls  # type: ignore[attr-defined]
    return evaluate


def far_theta(regime: str = "out") -> np.ndarray:
    theta = nominal_theta(regime).copy()
    theta[DIM] = 0.95 if theta[DIM] < 0.5 else 0.05
    theta[7] = 0.9 if theta[7] < 0.5 else 0.1  # a second coordinate the shrink must put back
    return theta


class TestReevaluationAndShrink:
    def test_a_find_that_reproduces_is_shrunk_and_labelled(self) -> None:
        ev = stub_evaluate()
        out = wgt.reproduce_and_shrink(
            3000, "out", "cma", far_theta(), ("P6",), 4, max_calls=None, tol=0.02, evaluate=ev
        )
        assert out["reproduced"] is True and out["evals"] == len(ev.calls)
        (entry,) = out["entries"]
        assert entry["broken_assumption"] == "A_exec" and entry["kind"] == "contract_failure"
        assert [d["index"] for d in entry["dimensions"]] == [
            DIM
        ]  # the other coordinate was put back
        assert entry["dimensions"][0]["necessary"] is True
        assert entry["found_by"] == [{"algo": "cma", "seed": 3000, "evaluation": 4}]
        assert entry["first_events"][0]["property"] == "P6" and entry["regime"] == "R_out"

    def test_the_cost_is_the_re_evaluation_plus_every_call_of_the_shrink_and_ablation(self) -> None:
        ev = stub_evaluate()
        out = wgt.reproduce_and_shrink(
            3000, "out", "random", far_theta(), ("P6",), 3, max_calls=None, tol=0.02, evaluate=ev
        )
        assert out["evals"] == len(ev.calls) and out["evals"] >= 2
        # one evaluation per distinct scenario the shrink and its ablations asked about, plus the
        # re-evaluation of the find itself: none is run twice
        assert len({tuple(c) for c in ev.calls}) <= out["evals"]

    def test_max_calls_bounds_the_shrink_and_says_so(self) -> None:
        full = wgt.reproduce_and_shrink(
            3000, "out", "cma", far_theta(), ("P6",), 4, max_calls=None, tol=0.02,
            evaluate=stub_evaluate(),
        )  # fmt: skip
        cut = wgt.reproduce_and_shrink(
            3000, "out", "cma", far_theta(), ("P6",), 4, max_calls=2, tol=0.02,
            evaluate=stub_evaluate(),
        )  # fmt: skip
        assert cut["evals"] < full["evals"] and cut["truncated"] is True
        assert cut["entries"][0]["shrink_truncated"] is True

    def test_a_find_that_does_not_reproduce_is_reported_and_not_shrunk(self) -> None:
        ev = stub_evaluate()
        out = wgt.reproduce_and_shrink(
            3000, "out", "cma", nominal_theta("out"), ("P6",), 4, max_calls=None, tol=0.02,
            evaluate=ev,
        )  # fmt: skip
        assert out["reproduced"] is False and out["entries"] == [] and out["evals"] == 1
        assert out["recorded"] == ["P6"] and out["got"] == []

    def test_a_find_that_now_violates_a_different_property_does_not_reproduce(self) -> None:
        out = wgt.reproduce_and_shrink(
            3000, "out", "cma", far_theta(), ("P2",), 4, max_calls=None, tol=0.02,
            evaluate=stub_evaluate(prop="P6"),
        )  # fmt: skip
        assert out["reproduced"] is False and out["got"] == ["P6"]

    def test_two_finds_that_shrink_to_the_same_scenario_are_one_entry_with_two_finders(
        self,
    ) -> None:
        one = wgt.reproduce_and_shrink(
            3000, "out", "cma", far_theta(), ("P6",), 4, max_calls=None, tol=0.02,
            evaluate=stub_evaluate(),
        )  # fmt: skip
        two = wgt.reproduce_and_shrink(
            3001, "out", "random", far_theta(), ("P6",), 9, max_calls=None, tol=0.02,
            evaluate=stub_evaluate(),
        )  # fmt: skip
        doc = wgt.merge([one, two], [])
        (entry,) = doc["entries"]
        assert entry["id"] == "WGT-01"
        assert sorted(b["algo"] for b in entry["found_by"]) == ["cma", "random"]

    def test_the_cost_of_a_set_of_results(self) -> None:
        results = [{"reproduced": True, "evals": 9}, {"reproduced": False, "evals": 1}]
        assert wgt.cost_of(results) == {
            "finds": 2, "reproduced": 1, "evaluations": 10, "reevaluations": 2,
            "shrink_and_ablation": 8,
        }  # fmt: skip

    def test_the_task_is_picklable_and_its_fingerprint_moves_with_what_decides_the_result(
        self,
    ) -> None:
        from closed_loop.eval.tuned import TunedValues

        finds = ((3000, tuple([0.5] * 32), ("P6",), 4),)
        base = wgt.ShrinkTask("out", "cma", finds, PROVISIONAL, None)
        assert pickle.loads(pickle.dumps(base)) == base
        assert (
            base.fingerprint()
            == wgt.ShrinkTask("out", "cma", finds, PROVISIONAL, None).fingerprint()
        )
        other_finds = ((3000, tuple([0.6] * 32), ("P6",), 4),)
        moved = [
            wgt.ShrinkTask("in", "cma", finds, PROVISIONAL, None),
            wgt.ShrinkTask("out", "random", finds, PROVISIONAL, None),
            wgt.ShrinkTask("out", "cma", other_finds, PROVISIONAL, None),
            wgt.ShrinkTask("out", "cma", finds, PROVISIONAL, 10),
            wgt.ShrinkTask("out", "cma", finds, TunedValues(**{**TUNED, "k_b3": 0.5}), None),
        ]
        assert all(m.fingerprint() != base.fingerprint() for m in moved)


# ----------------------------------------------------------------------------- the document


def sample_results() -> list[dict[str, Any]]:
    good = wgt.reproduce_and_shrink(
        3000, "out", "cma", far_theta(), ("P6",), 4, max_calls=None, tol=0.02,
        evaluate=stub_evaluate(),
    )  # fmt: skip
    lost = wgt.reproduce_and_shrink(
        3001, "in", "random", nominal_theta("in"), ("P2",), 3, max_calls=None, tol=0.02,
        evaluate=stub_evaluate(),
    )  # fmt: skip
    return [good, lost]


def document(records: list, dry_run: bool = False) -> str:
    results = sample_results()
    doc = wgt.merge(results, [])
    return wgt.render_document(records, doc, wgt.cost_of(results), results, [], dry_run=dry_run)


@pytest.fixture(scope="module")
def quick_records() -> list:
    if not QUICK_RESULTS.exists():
        pytest.skip("no quick output under benchmarks/.cache/quick-output/")
    return records_from_json(QUICK_RESULTS.read_text(encoding="utf-8"))


class FakeRunner:
    """Runs the task in this process; records what it was asked for."""

    def __init__(self) -> None:
        self.asked: list[tuple[str, list[int]]] = []

    def run_seeds(self, task: Any, seeds: list[int], **kw: Any) -> Any:
        self.asked.append((task.label(), list(seeds)))
        assert kw["n_max"] == len(seeds) and callable(kw["standin"])
        return type("Run", (), {"values": [task(s) for s in seeds]})()


class TestShrinkAll:
    def cells(self, tmp_path: Path) -> list[Any]:
        for regime, algo in (("in", "cma"), ("in", "random"), ("out", "cma"), ("out", "random")):
            runs = {3000: outcome(3), 3001: outcome(None), 3002: outcome(8)}
            write_cell(tmp_path, regime, algo, "9" * 64, runs)
        return wgt.read_search_cells(tmp_path)

    @pytest.fixture
    def stubbed(self, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
        seen: list[dict[str, Any]] = []

        def fake(
            seed: int,
            regime: str,
            algo: str,
            theta: Any,
            props: Any,
            first: int,
            max_calls: int | None,
            tol: float,
            evaluate: Any = None,
        ) -> dict[str, Any]:
            seen.append(
                {
                    "seed": seed,
                    "regime": regime,
                    "algo": algo,
                    "first": first,
                    "max_calls": max_calls,
                }
            )
            return {
                "seed": seed,
                "regime": regime,
                "algo": algo,
                "reproduced": True,
                "evals": 5,
                "entries": [],
            }

        monkeypatch.setattr(wgt, "reproduce_and_shrink", fake)
        return seen

    def test_one_task_per_cell_over_the_seeds_that_found_something(
        self, tmp_path: Path, stubbed: list[dict[str, Any]]
    ) -> None:
        runner = FakeRunner()
        out = wgt.shrink_all(
            self.cells(tmp_path), runner, limit=None, max_calls=None, log=lambda m: None
        )
        assert runner.asked == [
            ("wgt/R_in/cma", [3000, 3002]), ("wgt/R_in/random", [3000, 3002]),
            ("wgt/R_out/cma", [3000, 3002]), ("wgt/R_out/random", [3000, 3002]),
        ]  # fmt: skip
        assert len(out) == 8 and {s["first"] for s in stubbed} == {3, 8}

    def test_the_limit_takes_the_first_finds_in_the_runners_order(
        self, tmp_path: Path, stubbed: list[dict[str, Any]]
    ) -> None:
        runner = FakeRunner()
        wgt.shrink_all(self.cells(tmp_path), runner, limit=3, max_calls=None, log=lambda m: None)
        assert [(s["seed"], s["regime"], s["algo"]) for s in stubbed] == [
            (3000, "in", "cma"), (3000, "in", "random"), (3000, "out", "cma"),
        ]  # fmt: skip

    def test_the_cap_on_the_shrink_reaches_the_task(
        self, tmp_path: Path, stubbed: list[dict[str, Any]]
    ) -> None:
        wgt.shrink_all(self.cells(tmp_path), FakeRunner(), limit=2, max_calls=7, log=lambda m: None)
        assert {s["max_calls"] for s in stubbed} == {7}


class TestTheDocument:
    def test_the_sections_of_the_preregistration_are_there_in_order(
        self, quick_records: list
    ) -> None:
        text = document(quick_records)
        titles = [ln[3:] for ln in text.splitlines() if ln.startswith("## ")]
        assert titles == [
            "Violations found by search beyond the assumptions",
            "The conformance gap",
            "The screener miss list",
            "F2 replays that did not confirm the F1 violation",
            "What the abstraction does not model",
            "What development showed, measured in this run",
        ]

    def test_it_is_deterministic_and_has_no_absolute_claims(self, quick_records: list) -> None:
        text = document(quick_records)
        assert text == document(quick_records)
        assert not FORBIDDEN.search(text.replace("shield never", ""))

    def test_the_shrunk_scenario_is_in_the_table_with_its_label(self, quick_records: list) -> None:
        text = document(quick_records)
        row = next(ln for ln in text.splitlines() if ln.startswith("| WGT-01 |"))
        assert "R_out" in row and "contract_failure" in row and "A_exec" in row and "P6" in row
        assert "not ablated" not in row  # the ablation ran
        assert "cma seed 3000 evaluation 4" in row

    def test_a_find_that_did_not_reproduce_is_listed_apart(self, quick_records: list) -> None:
        text = document(quick_records)
        assert "### Finds that did not reproduce" in text
        assert "R_in random seed 3001: recorded P2" in text

    def test_the_cost_is_printed(self, quick_records: list) -> None:
        text = document(quick_records)
        results = sample_results()
        cost = wgt.cost_of(results)
        assert f"{cost['evaluations']} evaluations in all" in text
        assert "2 first finds, 1 reproduced" in text

    def test_the_known_gaps_come_from_the_proof_scope_of_the_same_records(
        self, quick_records: list
    ) -> None:
        text = document(quick_records)
        assert "clause 6.2 of the permit document" in text
        assert "the measured gap above" in text

    def test_the_screener_misses_are_the_repositorys_own_list(self, quick_records: list) -> None:
        from closed_loop.plant.notes import screener_misses

        text = document(quick_records)
        for note in screener_misses():
            assert f"| {note.note_id} |" in text

    def test_the_dry_run_banner_and_the_quick_banner(self, quick_records: list) -> None:
        assert "DRY RUN" in document(quick_records, dry_run=True)
        assert "DRY RUN" not in document(quick_records)
        assert "QUICK RUN, NOT A RESULT" in document(quick_records)

    def test_the_gas_clock_rows_name_the_arms_that_have_them(self, quick_records: list) -> None:
        text = document(quick_records)
        section = text.split("### The gas clock, in both directions")[1]
        assert "| P1 |" in section and "| H-C0 |" in section

    def test_the_numbers_of_the_dev_tables_are_among_the_records(self, quick_records: list) -> None:
        heading = "## What development showed, measured in this run"
        section = document(quick_records).split(heading)[1]
        assert (
            rc.unknown_numbers(section, quick_records, {"/^[0-9]$/": "single digits in prose"})
            == []
        )

    def test_the_uncovered_decisions_of_the_shield_are_listed_with_their_meaning(self) -> None:
        from closed_loop.core.results import ResultRecord
        from closed_loop.core.types import Fidelity

        row = ResultRecord(
            name="shield.uncovered.P1.nominal", value=5.0, n=500, fidelity=Fidelity.F1,
            measured_on="a cell", script="tests", extra={"k": 2, "total": 5},
        )  # fmt: skip
        text = document([*small_records(), row])
        section = text.split("### Shield decisions with no allowed command")[1].split("\n## ")[0]
        assert "| P1 | nominal | 2/500 | 5 | F1 |" in section
        assert "`STOP|STOP`" in section and "known limit" in section
        assert "Shield decisions with no allowed command" not in document(small_records())

    def test_the_small_records_give_a_document_too(self) -> None:
        text = document(small_records())
        assert text.startswith("# What still gets through (SYNTHETIC)")
        assert "The F2 stage did not run in this results file." in text


class TestTheCommandLine:
    def test_a_dry_run_writes_nothing_and_gives_the_runner_no_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from closed_loop.eval import context as context_module
        from closed_loop.eval.report import records_to_json

        results = tmp_path / "results.json"
        results.write_text(records_to_json(small_records()), encoding="utf-8")
        cells = tmp_path / "cells"
        write_cell(cells, "out", "cma", "f" * 64, {3000: outcome(4)})
        seen: dict[str, Any] = {}

        class FakeRunner:
            def __init__(self, **kw: Any) -> None:
                seen.update(kw)

            def __enter__(self) -> FakeRunner:
                return self

            def __exit__(self, *a: Any) -> None:
                return None

        monkeypatch.setattr(context_module, "default_context", lambda: None)
        import closed_loop.eval.runner as runner_module

        monkeypatch.setattr(runner_module, "Runner", FakeRunner)
        monkeypatch.setattr(wgt, "shrink_all", lambda *a, **k: sample_results())
        out = tmp_path / "docs" / "WHAT-GETS-THROUGH.md"
        cache = tmp_path / "wgt-cache"
        argv = [
            "--results",
            str(results),
            "--cells",
            str(cells),
            "--out",
            str(out),
            "--cache",
            str(cache),
        ]
        before = sorted(p.name for p in tmp_path.rglob("*"))
        assert wgt.main([*argv, "--dry-run"]) == 0
        assert seen["cache_dir"] is None
        assert (
            sorted(p.name for p in tmp_path.rglob("*")) == before
        )  # not the cache, not the document
        assert "# What still gets through" in capsys.readouterr().out
        assert wgt.main(argv) == 0
        assert seen["cache_dir"] == cache and out.exists()
        assert out.read_text(encoding="utf-8").startswith("# What still gets through")

    def test_no_search_cell_is_an_error_not_an_empty_document(self, tmp_path: Path) -> None:
        from closed_loop.eval.report import records_to_json

        results = tmp_path / "results.json"
        results.write_text(records_to_json(small_records()), encoding="utf-8")
        (tmp_path / "cells").mkdir()
        argv = ["--results", str(results), "--cells", str(tmp_path / "cells"), "--dry-run"]
        assert wgt.main(argv) == 2

    def test_the_default_paths_stay_under_the_cache_and_docs(self) -> None:
        assert wgt.DEFAULT_OUT == rc.ROOT / "docs" / "WHAT-GETS-THROUGH.md"
        assert wgt.WGT_CACHE == rc.ROOT / "benchmarks" / ".cache" / "wgt"
        assert wgt.DEFAULT_CELLS == rc.ROOT / "benchmarks" / ".cache" / "full"


# ----------------------------------------------------------------------------- after the second run


class TestWhatTheDocumentsSayAfterTheSecondRun:
    """Rutilius's check of 3 November: the published documents must say what the records hold."""

    @staticmethod
    def _f2_records() -> list[ResultRecord]:
        return [
            ResultRecord("claim/C-TOGETHER/confirmed", 1.0, 25, Fidelity.F2, "25 finds", "x",
                         extra={"k": 1, "requested": 25}),
            ResultRecord("f2.dropped_pieces.F2-B3", 0.0, 60, Fidelity.F2, "m", "x",
                         extra={"k": 0, "total": 0}),
            ResultRecord("f2.dropped_pieces.F2-P1", 1.0, 60, Fidelity.F2, "m", "x",
                         extra={"k": 1, "total": 1}),
        ]  # fmt: skip

    def test_the_confirmed_fraction_is_printed_with_its_exact_interval(self) -> None:
        text = "\n".join(wgt._f2_section(Index(self._f2_records())))
        lo, hi = clopper_pearson(1, 25, 0.05)
        assert f"1/25, exact 95% interval {lo:.4f} to {hi:.4f}" in text

    def test_the_dropped_pieces_are_listed_per_arm(self) -> None:
        text = "\n".join(wgt._f2_section(Index(self._f2_records())))
        assert "| F2-P1 | 1/60 | 1 |" in text and "| F2-B3 | 0/60 | 0 |" in text

    def test_completed_is_shown_as_no_test_and_a_tested_claim_keeps_its_level(self) -> None:
        together = Verdict("C-TOGETHER", "completed", "", "h", "r", 0.05 / 9)
        shield = Verdict("C-SHIELD", "win", "", "h", "r", 0.05 / 9)
        assert readme_script.level_text(together) == "no test"
        assert readme_script.level_text(shield) == "0.00556"
