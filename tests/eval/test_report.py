"""``eval.report``: results.json, RESULTS.md and PROOF-SCOPE.md are functions of the records alone,
and ``check_outputs`` finds a hand-edited number."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest

from closed_loop.core.results import ResultRecord
from closed_loop.core.types import Fidelity
from closed_loop.eval import report
from closed_loop.eval.claims import (
    Est,
    claim_name,
    verdict_records,
    verdicts_from_records,
)
from closed_loop.eval.report import (
    FORBIDDEN,
    check_outputs,
    headline_of,
    records_from_json,
    records_to_json,
    render_proof_scope,
    render_results,
    results_sha256,
    write_outputs,
)


def rec(
    name: str,
    value: float,
    n: int = 100,
    lo: float | None = None,
    hi: float | None = None,
    fid: Fidelity = Fidelity.F1,
    on: str = "a hand-built record of the tests",
    **extra: float | int | str,
) -> ResultRecord:
    return ResultRecord(
        name=name, value=value, n=n, fidelity=fid, measured_on=on, script="tests", ci_lo=lo,
        ci_hi=hi, ci_kind="confseq-0.05" if lo is not None else "", extra=extra,
    )  # fmt: skip


def est(name: str, v: float, lo: float, hi: float, n: int = 500, **extra: float | int | str):
    return rec(name, v, n, lo, hi, **extra)


def base_records(mode: str = "full", complete: bool = True) -> list[ResultRecord]:
    out: list[ResultRecord] = [
        rec(
            "run/params",
            500,
            500,
            mode=mode,
            seed_role="heldout" if mode == "full" else "dev",
            n_seeds=500,
            prereg_sha256="ab" * 32,
            primary_variant="V3",
            monitor_mode="proxy",
            n_max_hostile=500,
        ),  # fmt: skip
        rec(
            "run/completeness",
            1.0 if complete else 0.0,
            9,
            not_built="" if complete else "search,f2",
            partial="",
            skipped="",
        ),  # fmt: skip
        rec("run/history", 0.0, 0),
        rec("run/stage/f0", 0.0, 12, rows="1", status="done", notes=""),
        rec(
            "run/stage/search",
            0.0,
            0,
            rows="7",
            status="not built",
            notes="NOT BUILT: needs closed_loop.search.falsify",
        ),  # fmt: skip
    ]
    # a few cells
    for arm, cond, cost in (
        ("B3-V1", "nominal", 0.30),
        ("P0", "nominal", 0.12),
        ("H-C1", "nominal", 0.10),
    ):
        out.append(rec(f"{arm}.{cond}.cell", 500, 500, arm=arm, condition=cond, n_max=500,
                       stopped_by="n_max", failed=0, computed=500, cached=0,
                       **({"failures": ""} if False else {})))  # fmt: skip
        out.append(rec(f"{arm}.{cond}.m_cost", cost, 500, cost - 0.01, cost + 0.01, failed=0))
        out.append(rec(f"{arm}.{cond}.throughput", 0.9, 500, 0.89, 0.91, failed=0,
                       provisional="k_b3", tuning="provisional defaults"))  # fmt: skip
        out.append(rec(f"{arm}.{cond}.u_e", 0.1, 500, 0.0, 0.2, failed=0))
    out.append(rec("H-C1.nominal.m_viol_p13", 0.0, 500, 0.0, 0.009, k=0, zero_event_upper=0.009))
    out.append(rec("H-C1.nominal.m_viol_p15", 0.0, 500, 0.0, 0.009, k=0))
    out.append(rec("F0.crusher.states_raw", 1728, 1728, fid=Fidelity.F0, commands=14))
    out.append(rec("F0.crusher.states_winning", 1693, 1728, fid=Fidelity.F0, k=1693))
    out.append(rec("F0.crusher.states_observable", 432, 432, fid=Fidelity.F0, k=432))
    out.append(rec("F0.crusher.mean_safe_commands", 6.21, 1693, fid=Fidelity.F0, commands=14))
    out.append(rec("F0.crusher.differential_mismatches", 0, 24192, fid=Fidelity.F0, k=0))
    out.append(rec("F0.crusher.w4.findings", 0, 1520, fid=Fidelity.F0, k=0))
    out.append(rec("F0.crusher.w4.reachable_states", 1700, 1728, fid=Fidelity.F0, k=1700))
    out.append(rec("F0.crusher.w4.reachable_product", 1520, 1520, fid=Fidelity.F0))
    out.append(rec("conformance.unmatched", 0.01, 1000, 0.005, 0.018, k=10))
    out.append(rec("M-DIAG.scm.reversed", 0.9, 900, 0.88, 0.92))
    out += [
        est(claim_name("C-CAUSE", "q1", "scm-corr"), -0.08, -0.12, -0.05, 300),
        est(claim_name("C-DIAG", "reversed", "scm-corr"), 0.1, 0.05, 0.15, 900),
    ]
    verdicts = verdicts_from_records(out)
    return [*out, *verdict_records(verdicts)]


@pytest.fixture
def files(tmp_path: Path) -> dict[str, Path]:
    return {
        "json": tmp_path / "benchmarks" / "results.json",
        "md": tmp_path / "benchmarks" / "RESULTS.md",
        "scope": tmp_path / "docs" / "PROOF-SCOPE.md",
    }


def written(files: dict[str, Path], records: list[ResultRecord]) -> str:
    return write_outputs(records, files["json"], files["md"], files["scope"])


def check(files: dict[str, Path], digest: str | None = None) -> list[str]:
    return check_outputs(files["json"], files["md"], files["scope"], expected_sha256=digest)


class TestJson:
    def test_the_records_round_trip_exactly(self) -> None:
        records = base_records()
        again = records_from_json(records_to_json(records))
        assert again == records

    def test_the_text_is_canonical(self) -> None:
        text = records_to_json(base_records())
        assert text.endswith("\n") and "\r" not in text
        assert json.loads(text)[0]["name"] == "run/params"
        assert records_to_json(base_records()) == text  # deterministic

    def test_an_infinite_value_is_refused_with_its_name(self) -> None:
        with pytest.raises(ValueError, match="bad.ratio"):
            records_to_json([rec("bad.ratio", math.inf)])

    def test_nan_in_the_extras_becomes_text_and_the_file_stays_valid(self) -> None:
        text = records_to_json([rec("x", 1.0, a=math.nan, b=math.inf)])
        assert json.loads(text)[0]["extra"] == {"a": "nan", "b": "inf"}

    def test_the_digest_ignores_line_endings(self) -> None:
        text = records_to_json(base_records())
        assert results_sha256(text) == results_sha256(text.replace("\n", "\r\n"))

    def test_a_file_that_is_not_a_list_is_refused(self) -> None:
        with pytest.raises(ValueError, match="list"):
            records_from_json('{"a": 1}')


class TestGenerated:
    def test_results_md_is_exactly_what_results_json_generates(
        self, files: dict[str, Path]
    ) -> None:
        records = base_records()
        written(files, records)
        from_disk = records_from_json(files["json"].read_text(encoding="utf-8"))
        assert files["md"].read_text(encoding="utf-8") == render_results(from_disk)
        assert files["scope"].read_text(encoding="utf-8") == render_proof_scope(from_disk)
        assert check(files) == []

    def test_the_files_are_utf8_with_lf(self, files: dict[str, Path]) -> None:
        written(files, base_records())
        for p in files.values():
            data = p.read_bytes()
            assert b"\r" not in data and data.endswith(b"\n")
            data.decode("utf-8")

    def test_a_number_in_a_table_is_the_record_it_came_from(self) -> None:
        text = render_results(base_records())
        assert "0.1200 (0.1100 to 0.1300)" in text  # P0's M-COST, value and interval
        changed = [r if r.name != "P0.nominal.m_cost" else rec(r.name, 0.5, 500, 0.49, 0.51)
                   for r in base_records()]  # fmt: skip
        assert render_results(changed) != text and "0.5000 (0.4900 to 0.5100)" in render_results(
            changed
        )

    def test_the_same_records_give_the_same_text(self) -> None:
        assert render_results(base_records()) == render_results(base_records())
        assert render_proof_scope(base_records()) == render_proof_scope(base_records())

    def test_a_row_carries_k_over_n_fidelity_and_what_it_was_measured_on(self) -> None:
        text = render_results(base_records())
        row = next(
            line for line in text.splitlines() if line.startswith("| H-C1 | nominal | 0/500")
        )
        assert (
            "F1" in row and "a violation of any of P1-P5" not in row
        )  # (the row has its own text)
        layer = next(ln for ln in text.splitlines() if "checker findings, crusher game" in ln)
        assert "0/1520" in layer and "F0" in layer and "a hand-built record of the tests" in layer

    def test_the_verdict_table_lists_every_claim_with_its_level(self) -> None:
        text = render_results(base_records())
        for claim in ("C-CAUSE", "C-DIAG", "C-SHIELD", "C-SEARCH", "C-TOGETHER", "C-PERC.pooled"):
            assert f"| {claim} |" in text
        assert "0.00556" in text and "0.00278" in text  # 0.05 / 9 and 0.05 / 18

    def test_a_quick_run_is_labelled_on_both_documents(self) -> None:
        records = base_records(mode="quick")
        assert "QUICK RUN, NOT A RESULT" in render_results(records)
        assert "QUICK RUN, NOT A RESULT" in render_proof_scope(records)
        assert "QUICK RUN" not in render_results(base_records())

    def test_an_incomplete_run_says_so_loudly_and_names_the_missing_stages(self) -> None:
        text = render_results(base_records(complete=False))
        assert "INCOMPLETE RUN" in text and "does not claim completeness" in text
        assert "Stages not built: search,f2" in text
        assert "| search | 7 | not built |" in text and "NOT BUILT" in text
        assert "INCOMPLETE" not in render_results(base_records(complete=True))

    def test_a_claim_that_could_not_be_judged_is_not_run_in_the_table(self) -> None:
        text = render_results(base_records())
        row = next(line for line in text.splitlines() if line.startswith("| C-SEARCH |"))
        assert "not run" in row

    def test_provisional_values_are_listed_with_the_arms_that_use_them(self) -> None:
        text = render_results(base_records())
        assert "| k_b3 | B3-V1, H-C1, P0 |" in text and "provisional defaults" in text

    def test_the_proof_scope_prints_raw_and_reachable_counts_side_by_side(self) -> None:
        text = render_proof_scope(base_records())
        assert "1693/1728" in text and "432/432" in text and "1700/1728" in text
        assert "0/1520" in text and "0/24192" in text and "6.21" in text
        assert "C1 holds at every tick, which is an event of a whole life" in text
        assert "conditional-mean assumption" in text

    def test_the_proof_scope_names_only_the_games_and_widths_that_were_run(self) -> None:
        # the F0 counts are read from the records (the code); the text fixes none of them
        text = render_proof_scope(base_records())
        assert "(crusher)" in text and "pumps" not in text.split("## Counts")[0]
        assert "width 4 bins at most were checked" in text
        more = [
            *base_records(),
            rec("F0.pumps.states_raw", 576, 576, fid=Fidelity.F0, commands=64),
            rec("F0.pumps.w1.findings", 0, 10, fid=Fidelity.F0, k=0),
            rec("F0.crusher.w2.findings", 0, 10, fid=Fidelity.F0, k=0),
        ]
        later = render_proof_scope(more)
        assert "(crusher, pumps)" in later and "width 1, 2, 4 bins at most were checked" in later

    def test_the_counts_come_from_the_records_whatever_they_are(self) -> None:
        winning = rec("F0.crusher.states_winning", 1700, 1800, fid=Fidelity.F0, k=1700)
        changed = [winning if r.name == winning.name else r for r in base_records()]
        text = render_proof_scope(changed)
        assert "1700/1800" in text and "1693/1728" not in text

    def test_the_proof_scope_states_what_the_proof_does_not_cover(self) -> None:
        text = render_proof_scope(base_records())
        for phrase in ("A wrong specification", "clause 6.2", "stricter than the properties",
                       "measured rates on sampled traces, not a proof"):  # fmt: skip
            assert phrase in text

    def test_the_text_uses_no_absolute(self) -> None:
        for records in (base_records(), base_records("quick", complete=False)):
            for text in (render_results(records), render_proof_scope(records)):
                assert not FORBIDDEN.search(text), FORBIDDEN.search(text)

    def test_every_page_says_synthetic(self) -> None:
        assert "SYNTHETIC" in render_results(base_records())
        assert "SYNTHETIC" in render_proof_scope(base_records())

    def test_the_forbidden_words_are_the_owners_list(self) -> None:
        for word in ("never", "always", "100%", "guaranteed"):
            assert FORBIDDEN.search(f"it is {word} true")
        assert not FORBIDDEN.search("neverland is a place")  # a word, not a prefix of one


class TestTheFailureSectionCoversEveryStage:
    """P12g, G3. The first full run lost an F2-P1 episode, C-TOGETHER named it, and this section
    said that every episode of every cell produced a result, because it read the F1 cells only."""

    NONE = "None: every episode of every cell, every F2 episode and every search run"

    def section(self, extra: list[ResultRecord]) -> str:
        text = render_results([*base_records(), *extra])
        return text.split("## Failed and timed-out episodes")[1].split("\n## ")[0]

    def test_a_clean_run_says_none_for_all_three_kinds(self) -> None:
        assert self.NONE in self.section([])

    def test_a_failed_f2_episode_is_named_with_its_seed_and_reason(self) -> None:
        arm = rec(
            "claim/C-TOGETHER/arm/F2-P1", 59, 60, fid=Fidelity.F2, completed=59, tried=60,
            cause="seed 9426043320444826810: error: RenderError: the piece would start after "
            "the life has failed",
        )  # fmt: skip
        section = self.section([arm])
        assert self.NONE not in section
        assert "| F2 arm | F2-P1 | 1 |" in section
        assert "seed 9426043320444826810" in section and "RenderError" in section

    def test_an_f2_arm_that_completed_all_its_episodes_is_not_listed(self) -> None:
        arm = rec("claim/C-TOGETHER/arm/F2-B3", 60, 60, fid=Fidelity.F2, completed=60, tried=60)
        assert self.NONE in self.section([arm])

    def test_failed_search_runs_and_f2_replays_are_listed(self) -> None:
        found = rec("search.R_out.cma.find_rate", 0.5, 10, k=4, failed=2, on="10 runs, R_out, cma")
        replay = rec(
            "f2.replay.reason.replay_timeout", 1, 25, fid=Fidelity.F2, k=3, on="25 replays"
        )
        error = rec("f2.replay.reason.replay_error", 0, 25, fid=Fidelity.F2, k=0)
        section = self.section([found, replay, error])
        assert "| search runs | R_out.cma | 2 |" in section
        assert "| F2 replays | replay_timeout | 3 |" in section
        assert "replay_error" not in section

    def test_an_f1_cell_with_failures_is_still_listed(self) -> None:
        cell = rec("B3-V1.shifted.cell", 500, 500, arm="B3-V1", condition="shifted", failed=2,
                   failures="seed 1103: timeout: no result after 900 s")  # fmt: skip
        section = self.section([cell])
        assert "| F1 cell | B3-V1, shifted | 2 | seed 1103: timeout" in section


class TestUncoveredDecisionsArePublished:
    """P12g, G7: the shield's fallback when no command is allowed is a published limit."""

    def uncovered(self, arm: str, cond: str, k: int, n: int, total: int) -> ResultRecord:
        return rec(f"shield.uncovered.{arm}.{cond}", float(total), n, k=k, total=total)

    def section(self, records: list[ResultRecord]) -> str:
        text = render_results([*base_records(), *records])
        assert "## Violations and what gets through" in text
        return text.split("### Shield decisions with no allowed command")[1].split("\n## ")[0]

    def test_the_cells_are_listed_with_episodes_with_any_and_the_total(self) -> None:
        section = self.section(
            [
                self.uncovered("P1", "nominal", 2, 500, 7),
                self.uncovered("H-C1", "shifted", 0, 40, 0),
            ]
        )
        assert "| P1 | nominal | 2/500 | 7 | F1 |" in section
        assert "| H-C1 | shifted | 0/40 | 0 | F1 |" in section

    def test_it_says_what_an_uncovered_decision_is_and_that_the_fallback_stays(self) -> None:
        section = self.section([self.uncovered("P1", "nominal", 1, 10, 1)])
        assert "`STOP|STOP`" in section and "a repair cannot progress" in section
        assert "only after the abstract model's assumptions are already violated" in section
        assert "known limit" in section and "left as it is" in section

    def test_with_no_such_record_there_is_no_section(self) -> None:
        assert "Shield decisions with no allowed command" not in render_results(base_records())

    def test_an_arm_name_with_a_dash_and_a_condition_are_split_at_the_last_dot(self) -> None:
        from closed_loop.eval.report import Index, uncovered_rows

        rows = uncovered_rows(Index([self.uncovered("P1-nopair", "shifted", 1, 9, 3)]))
        assert rows == [("P1-nopair", "shifted", "1/9", "3", "F1")]


class TestTheProofScopeSaysWhatTheMonitorsUnitIs:
    def test_the_label_modes_are_per_life_and_proxy_is_per_day_and_measured(self) -> None:
        text = render_proof_scope(base_records())
        assert "(`opt`, `pess` and `inf`) is the life" in text
        assert "mode `proxy` is the day" in text and "it is not a test of C1" in text
        assert "false-alarm rate is measured" in text and "does not bound it" in text
        assert not report.FORBIDDEN.search(text)

    def test_the_monitor_table_names_the_chosen_mode_and_both_bounds_of_each_rate(self) -> None:
        text = report.render_results(base_records())
        assert "The chosen mode is `proxy`" in text or "chosen mode is `" in text
        assert "descriptive level of 0.05" in text


class TestLedgerFactsAndHeadline:
    def ledger_record(self, **extra: object) -> ResultRecord:
        base: dict[str, object] = {
            "run_number": 2, "heldout_evaluations": 1410, "code_equals_frozen": 1,
            "tuned_equals_frozen": 1,
        }  # fmt: skip
        return rec("run/ledger", 1410.0, 2, **{**base, **extra})

    def test_results_md_prints_the_run_number_the_seed_count_and_whether_the_code_is_frozen(
        self,
    ) -> None:
        text = report.render_results([*base_records(), self.ledger_record()])
        assert "this is full run 2" in text and "seeds handed out so far" in text
        assert "1410" in text and "The running code is the frozen code" in text

    def test_a_changed_code_or_tuned_file_is_said_in_capitals(self) -> None:
        text = report.render_results(
            [*base_records(), self.ledger_record(code_equals_frozen=0, tuned_equals_frozen=0)]
        )
        assert "is NOT the frozen code" in text and "is NOT the frozen file" in text

    def test_a_quick_run_prints_no_ledger_line(self) -> None:
        assert "Ledger:" not in report.render_results(base_records())

    def test_the_headline_carries_the_numbers_behind_each_label(self) -> None:
        records = base_records()
        head = report.headline_of(records)
        stored = {r.name: r for r in records if r.name.startswith("verdict/")}
        assert stored
        for name, record in stored.items():
            assert head[name.replace("verdict/", "numbers/", 1)] == record.extra["headline"]
        assert any(
            any(ch.isdigit() for ch in str(v)) for k, v in head.items() if k.startswith("numbers/")
        )  # at least one verdict has a number to carry


class TestWriteRefusesAnAbsolute:
    def test_nothing_is_written_when_the_reports_own_text_contains_one(
        self, files: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # a sentence of the report's own (here a limitation) that says "never" is refused
        monkeypatch.setattr(report, "_LIMITS", (*report._LIMITS, "The shield never fails."))
        with pytest.raises(ValueError, match="would contain the absolute 'never'"):
            written(files, base_records())
        assert not any(p.exists() for p in files.values())

    def test_an_exception_message_in_the_data_does_not_stop_a_run_from_publishing(
        self, files: dict[str, Path]
    ) -> None:
        # the failure text of a cell is data from outside the report (an exception, a test run's
        # summary): the word in it is not the report's claim
        tainted = [
            r
            if r.name != "P0.nominal.cell"
            else rec(
                r.name, 5, 5, arm="P0", condition="nominal", failed=1,
                failures="seed 3: error: KeyError: the value was always missing",
            )
            for r in base_records()
        ]  # fmt: skip
        digest = written(files, tainted)
        assert "always missing" in files["md"].read_text(encoding="utf-8")
        assert check(files, digest) == []  # and --check agrees

    def test_the_stage_notes_are_data_too(self, files: dict[str, Path]) -> None:
        tainted = [
            dataclasses.replace(r, extra={**r.extra, "notes": "pytest said: never mind"})
            if r.name.startswith("run/stage/")
            else r
            for r in base_records()
        ]
        written(files, tainted)  # no refusal

    def test_absolutes_in_reads_only_text_outside_the_data_fields(self) -> None:
        records = [
            rec("x", 1, 1, cause="it never works", other="always"),
        ]
        body = "A sentence. The cause was: it never works. Another sentence."
        assert report.absolutes_in(body, records) == []  # the cause is data
        assert report.absolutes_in(body + " It always works.", records) == ["always"]
        assert report.absolutes_in("Guaranteed 100% of the time", []) == ["Guaranteed", "100%"]


class TestCheckFindsEdits:
    def test_untouched_files_pass_and_the_digest_matches(self, files: dict[str, Path]) -> None:
        digest = written(files, base_records())
        assert check(files, digest) == []

    def test_a_hand_edited_number_in_results_md_fails(self, files: dict[str, Path]) -> None:
        written(files, base_records())
        text = files["md"].read_text(encoding="utf-8")
        assert "0.1200 (0.1100 to 0.1300)" in text
        files["md"].write_text(text.replace("0.1200 (0.1100", "0.0900 (0.1100"), encoding="utf-8")
        problems = check(files)
        assert any("RESULTS.md is not what results.json generates" in p for p in problems)

    def test_a_hand_edited_number_in_results_json_fails(self, files: dict[str, Path]) -> None:
        digest = written(files, base_records())
        data = json.loads(files["json"].read_text(encoding="utf-8"))
        target = next(d for d in data if d["name"] == "P0.nominal.m_cost")
        target["value"] = 0.05
        files["json"].write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        problems = check(files, digest)
        assert any("not what results.json generates" in p for p in problems)  # RESULTS.md is stale
        assert any("differs from the ledger's" in p for p in problems)  # and so is the digest

    def test_results_json_and_results_md_edited_together_still_fail_on_the_digest(
        self, files: dict[str, Path]
    ) -> None:
        digest = written(files, base_records())
        edited = [r if r.name != "P0.nominal.m_cost" else rec(r.name, 0.05, 500, 0.04, 0.06)
                  for r in base_records()]  # fmt: skip
        written(files, edited)  # a consistent forgery: the documents match the edited records
        problems = check(files, digest)
        assert problems and all("differs from the ledger's" in p for p in problems)

    def test_an_edited_verdict_fails_even_if_the_documents_are_regenerated(
        self, files: dict[str, Path]
    ) -> None:
        records = base_records()
        forged = []
        for r in records:
            if r.name == "verdict/C-CAUSE":
                r = rec(r.name, -1.0, 0, **{**dict(r.extra), "label": "loss"})
            forged.append(r)
        written(files, forged)
        problems = check(files)
        assert any("verdict C-CAUSE was edited" in p for p in problems)

    def test_an_edited_claim_input_changes_the_rebuilt_verdict(
        self, files: dict[str, Path]
    ) -> None:
        records = base_records()
        forged = [
            r if r.name != claim_name("C-DIAG", "reversed", "scm-corr")
            else est(r.name, -0.1, -0.15, -0.05, 900)
            for r in records
        ]  # fmt: skip
        written(files, forged)
        assert any("verdict C-DIAG was edited" in p for p in check(files))

    def test_an_edited_proof_scope_fails(self, files: dict[str, Path]) -> None:
        written(files, base_records())
        text = files["scope"].read_text(encoding="utf-8")
        files["scope"].write_text(text.replace("1693/1728", "1700/1728"), encoding="utf-8")
        assert any("PROOF-SCOPE.md is not what" in p for p in check(files))

    def test_a_missing_document_fails(self, files: dict[str, Path]) -> None:
        written(files, base_records())
        files["scope"].unlink()
        assert any("PROOF-SCOPE.md cannot be read" in p for p in check(files))

    def test_an_unreadable_results_json_fails(self, files: dict[str, Path]) -> None:
        written(files, base_records())
        files["json"].write_text("not json", encoding="utf-8")
        assert any("cannot be read" in p for p in check(files))

    def test_crlf_line_ends_in_a_document_fail(self, files: dict[str, Path]) -> None:
        written(files, base_records())
        files["md"].write_bytes(files["md"].read_bytes().replace(b"\n", b"\r\n"))
        assert check(files) != []  # the published bytes are LF, as .gitattributes says


class TestHeadline:
    def test_it_holds_every_verdict_and_the_completeness_flag(self) -> None:
        head = headline_of(base_records(complete=False))
        assert head["complete"] is False
        assert head["verdict/C-CAUSE"] == "win" and head["verdict/C-SEARCH"] == "not run"
        assert "verdict/C-SHIELD" in head

    def test_a_qualifier_is_part_of_the_text(self) -> None:
        v = verdict_records([])
        assert v == []
        records = base_records()
        stored = [r for r in records if r.name == "verdict/C-CAUSE"][0]
        with_qualifier = rec(
            stored.name, 1.0, 0, **{**dict(stored.extra), "qualifier": "vacuous: x"}
        )
        assert headline_of([with_qualifier])["verdict/C-CAUSE"] == "win, vacuous: x"


def test_est_is_importable_from_claims() -> None:
    assert Est(0.1, 0.0, 0.2, 5).n == 5
