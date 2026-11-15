"""P12b: what ``benchmarks/run.py`` does around the experiments: the freeze renames the draft, an
incomplete or paused or interrupted full run is recorded, ``--check`` verifies the published
results, and the body writes its files where it should. Nothing here simulates a plant."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from closed_loop.audit.prereg import prereg_sha256
from closed_loop.eval.report import records_from_json, write_outputs
from closed_loop.eval.runner import CommitChargeStop
from tests.audit.test_run import PREREG, invoke, ledger_of
from tests.eval.test_report import base_records

_LOADED: dict[str, Any] = {}


def load_generator(name: str) -> Any:
    """``scripts/<name>.py`` imported once under a name of its own. ``tests.docs`` loads the same
    scripts under another name; loading one file twice under one name replaces the classes the
    other tests already hold, and their pickling test then fails."""
    if name not in _LOADED:
        path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"_p12g_{name}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _LOADED[name] = module
    return _LOADED[name]


PUBLISHED = ("benchmarks/results.json", "benchmarks/RESULTS.md", "docs/PROOF-SCOPE.md")
WGT = "docs/WHAT-GETS-THROUGH.md"
BEGIN, END = "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->"


def readme_block(repo: Path) -> str:
    """The Results block ``scripts/render_readme_results.py`` makes from the repository's own
    ``results.json``."""
    script = load_generator("render_readme_results")
    text = (repo / "benchmarks" / "results.json").read_text(encoding="utf-8")
    return str(script.render_block(records_from_json(text)))


def write_the_other_documents(repo: Path, *, readme: bool = True) -> None:
    """The two documents that ``--check`` also wants: WHAT-GETS-THROUGH and a README whose block
    is current."""
    (repo / "docs").mkdir(exist_ok=True)
    (repo / WGT).write_text("# What gets through (test copy)\n", encoding="utf-8")
    if readme:
        (repo / "README.md").write_text(
            f"# Title\n\nBefore.\n\n{readme_block(repo)}\n\nAfter.\n", encoding="utf-8"
        )


class TestFreezeRenamesTheDraft:
    """Tested against a temporary copy, never the real draft."""

    def draft_only(self, repo: Path) -> Path:
        draft = repo / "docs" / "PREREGISTRATION.draft.md"
        (repo / "docs" / "PREREGISTRATION.md").rename(draft)
        return draft

    def test_the_draft_becomes_the_frozen_document_and_its_hash_is_recorded(
        self, run_module: Any, repo: Path
    ) -> None:
        draft = self.draft_only(repo)
        before = prereg_sha256(draft)
        code, text = invoke(run_module, repo, "--freeze")
        frozen = repo / "docs" / "PREREGISTRATION.md"
        assert code == 0 and frozen.is_file() and not draft.exists()
        assert "renamed PREREGISTRATION.draft.md to PREREGISTRATION.md" in text
        (entry,) = ledger_of(repo).entries()
        assert entry["kind"] == "freeze" and entry["prereg_sha256"] == before
        assert prereg_sha256(frozen) == before  # the bytes were not touched

    def test_a_frozen_document_that_is_already_there_is_not_replaced_by_a_draft(
        self, run_module: Any, repo: Path
    ) -> None:
        draft = repo / "docs" / "PREREGISTRATION.draft.md"
        draft.write_bytes(b"a different draft\n")
        code, _ = invoke(run_module, repo, "--freeze")
        assert code == 0 and draft.read_bytes() == b"a different draft\n"
        assert (repo / "docs" / "PREREGISTRATION.md").read_bytes() == PREREG.encode()

    def test_a_refused_freeze_leaves_the_draft_where_it_was(
        self, run_module: Any, repo: Path
    ) -> None:
        invoke(run_module, repo, "--freeze")
        draft = repo / "docs" / "PREREGISTRATION.draft.md"
        draft.write_bytes(b"a second draft\n")
        (repo / "docs" / "PREREGISTRATION.md").unlink()
        code, text = invoke(run_module, repo, "--freeze")
        assert code == 1 and "already has entries" in text
        assert draft.exists() and not (repo / "docs" / "PREREGISTRATION.md").exists()

    def test_a_failed_ledger_write_puts_the_draft_back(
        self, run_module: Any, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from closed_loop.audit.ledger import Ledger, LedgerError

        draft = self.draft_only(repo)

        def broken(self: Ledger, kind: str, **fields: Any) -> dict[str, Any]:
            raise LedgerError("disk full")

        monkeypatch.setattr(Ledger, "append", broken)
        code, text = invoke(run_module, repo, "--freeze")
        assert code == 4 and "disk full" in text
        assert draft.exists() and not (repo / "docs" / "PREREGISTRATION.md").exists()

    def test_no_draft_and_no_document_is_not_run(self, run_module: Any, repo: Path) -> None:
        (repo / "docs" / "PREREGISTRATION.md").unlink()
        code, text = invoke(run_module, repo, "--freeze")
        assert code == 4 and "does not exist" in text


class TestFullRunOutcomes:
    def frozen(self, run_module: Any, repo: Path) -> None:
        invoke(run_module, repo, "--freeze")

    def test_a_run_that_is_not_complete_is_recorded_and_says_so(
        self, run_module: Any, repo: Path
    ) -> None:
        self.frozen(run_module, repo)
        code, text = invoke(run_module, repo, body=lambda ctx: {"complete": False, "x": 1})
        assert code == 5 and "INCOMPLETE" in text and "does not claim completeness" in text
        last = ledger_of(repo).entries()[-1]
        assert last["kind"] == "full" and last["headline"]["complete"] is False

    def test_the_ledger_entry_of_a_full_run_records_the_libraries_it_ran_with(
        self, run_module: Any, repo: Path
    ) -> None:
        import platform
        from importlib.metadata import version

        self.frozen(run_module, repo)
        invoke(run_module, repo, body=lambda ctx: {"complete": True})
        last = ledger_of(repo).entries()[-1]
        assert last["versions"] == {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
        }

    def test_a_complete_run_exits_zero(self, run_module: Any, repo: Path) -> None:
        self.frozen(run_module, repo)
        code, text = invoke(run_module, repo, body=lambda ctx: {"complete": True})
        assert code == 0 and "INCOMPLETE" not in text

    def test_a_run_stopped_by_the_commit_charge_is_a_pause_that_resumes_later(
        self, run_module: Any, repo: Path
    ) -> None:
        self.frozen(run_module, repo)

        def body(ctx: Any) -> dict[str, Any]:
            raise CommitChargeStop(91.0, 88.0, "batch 3 of B3-V1/nominal")

        code, text = invoke(run_module, repo, body=body)
        assert code == 3 and "paused" in text and "91.0%" in text
        last = ledger_of(repo).entries()[-1]
        assert last["kind"] == "full-failed" and last["error"].startswith("paused: commit charge")
        assert "resume" in last["error"] or "run again" in last["error"]

    def test_an_interrupted_run_is_recorded_and_the_interrupt_goes_on(
        self, run_module: Any, repo: Path
    ) -> None:
        self.frozen(run_module, repo)

        def body(ctx: Any) -> dict[str, Any]:
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            invoke(run_module, repo, body=body)
        last = ledger_of(repo).entries()[-1]
        assert last["kind"] == "full-failed" and last["error"] == "interrupted"

    def test_a_quick_run_stopped_by_the_commit_charge_exits_3_and_records_nothing(
        self, run_module: Any, repo: Path
    ) -> None:
        def body(ctx: Any) -> dict[str, Any]:
            raise CommitChargeStop(95.0, 88.0, "batch 2")

        code, text = invoke(run_module, repo, "--quick", body=body)
        assert code == 3 and "paused" in text
        assert not (repo / "benchmarks" / "ledger.jsonl").exists()

    def test_a_second_run_sees_the_headlines_of_the_first(
        self, run_module: Any, repo: Path
    ) -> None:
        self.frozen(run_module, repo)
        seen: list[Any] = []
        invoke(run_module, repo, body=lambda ctx: seen.append(ctx.previous_headlines) or {"a": 1})
        invoke(run_module, repo, body=lambda ctx: seen.append(ctx.previous_headlines) or {"a": 2})
        assert seen[0] == () and seen[1] == ({"a": 1},)

    def test_a_full_run_cannot_pick_its_stages(self, run_module: Any, repo: Path) -> None:
        self.frozen(run_module, repo)
        called: list[int] = []
        code, text = invoke(
            run_module, repo, "--stages", "f0", body=lambda ctx: called.append(1) or {}
        )
        assert code == 2 and "--stages is for --quick" in text and called == []

    def test_a_quick_run_can(self, run_module: Any, repo: Path) -> None:
        got: list[Any] = []
        invoke(
            run_module, repo, "--quick", "--stages", "f0,policy", body=lambda c: got.append(c) or {}
        )
        assert got[0].stages == ("f0", "policy") and got[0].mode == "quick"


class TestWorkersAndGates:
    def test_the_worker_count_is_capped_at_three(self, run_module: Any, repo: Path) -> None:
        got: list[int] = []
        for flag in ("9", "3", "2", "0"):
            invoke(
                run_module,
                repo,
                "--quick",
                "--workers",
                flag,
                body=lambda c: got.append(c.workers) or {},
            )
        assert got == [3, 3, 2, 1]

    def test_the_default_and_the_environment(
        self, run_module: Any, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        got: list[int] = []
        invoke(run_module, repo, "--quick", body=lambda c: got.append(c.workers) or {})
        monkeypatch.setenv("CLOSED_LOOP_WORKERS", "2")
        invoke(run_module, repo, "--quick", body=lambda c: got.append(c.workers) or {})
        monkeypatch.setenv("CLOSED_LOOP_WORKERS", "50")
        invoke(run_module, repo, "--quick", body=lambda c: got.append(c.workers) or {})
        assert got == [3, 2, 3]

    def test_the_body_gets_the_commit_reader_and_the_limit_the_run_used(
        self, run_module: Any, repo: Path
    ) -> None:
        got: list[Any] = []

        def body(ctx: Any) -> dict[str, Any]:
            got.append((ctx.commit_reader(), ctx.limit))
            return {}

        invoke(run_module, repo, "--quick", "--limit", "91", commit_reader=lambda: 60.0, body=body)
        assert got == [(60.0, 91.0)]

    def test_no_lock_is_taken_when_the_variable_is_unset(
        self, run_module: Any, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CLOSED_LOOP_LOCK", raising=False)
        before = set(tmp_path.glob("*.lock"))
        code, _ = invoke(run_module, repo, "--quick", body=lambda ctx: {})
        assert code == 0 and set(tmp_path.glob("*.lock")) == before


class TestCheckOfPublishedResults:
    def publish(
        self, repo: Path, records: Any = None, *, recorded: bool = True, others: bool = True
    ) -> str:
        paths = [repo / p for p in PUBLISHED]
        digest = write_outputs(records or base_records(), *paths)
        if others:
            write_the_other_documents(repo)
        if recorded:
            ledger_of(repo).append("full", prereg_sha256="x", headline={"results_sha256": digest})
        return digest

    def test_nothing_published_is_said_plainly(self, run_module: Any, repo: Path) -> None:
        invoke(run_module, repo, "--freeze")
        code, text = invoke(run_module, repo, "--check")
        assert code == 0 and "results: none published yet" in text

    def test_published_results_that_match_their_generator_pass(
        self, run_module: Any, repo: Path
    ) -> None:
        invoke(run_module, repo, "--freeze")
        self.publish(repo)
        code, text = invoke(run_module, repo, "--check")
        assert code == 0 and "results: OK (1 full runs recorded" in text

    def test_a_hand_edited_number_in_results_md_fails_the_check(
        self, run_module: Any, repo: Path
    ) -> None:
        invoke(run_module, repo, "--freeze")
        self.publish(repo)
        md = repo / "benchmarks" / "RESULTS.md"
        md.write_text(md.read_text(encoding="utf-8").replace("0.1200", "0.0100"), encoding="utf-8")
        code, text = invoke(run_module, repo, "--check")
        assert code == 1 and "results: FAILED" in text and "RESULTS.md is not what" in text

    def test_a_hand_edited_number_in_results_json_fails_the_check(
        self, run_module: Any, repo: Path
    ) -> None:
        invoke(run_module, repo, "--freeze")
        self.publish(repo)
        path = repo / "benchmarks" / "results.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        next(d for d in data if d["name"] == "P0.nominal.m_cost")["value"] = 0.0
        path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        code, text = invoke(run_module, repo, "--check")
        assert code == 1 and "results: FAILED" in text

    def test_a_consistent_forgery_is_caught_by_the_digest_in_the_ledger(
        self, run_module: Any, repo: Path
    ) -> None:
        from closed_loop.core.results import ResultRecord

        invoke(run_module, repo, "--freeze")
        records = base_records()
        self.publish(repo, records)
        forged = [
            r if r.name != "P0.nominal.m_cost" else ResultRecord(
                r.name, 0.01, r.n, r.fidelity, r.measured_on, r.script, 0.0, 0.02,
                r.ci_kind, r.extra,
            )
            for r in records
        ]  # fmt: skip
        write_outputs(forged, *[repo / p for p in PUBLISHED])  # json, md and scope all agree
        code, text = invoke(run_module, repo, "--check")
        assert code == 1 and "differs from the ledger's" in text

    def test_results_that_the_ledger_never_heard_of_fail(self, run_module: Any, repo: Path) -> None:
        invoke(run_module, repo, "--freeze")
        self.publish(repo, recorded=False)
        code, text = invoke(run_module, repo, "--check")
        assert code == 1 and "the ledger has no full run" in text

    def test_a_full_run_whose_results_were_deleted_fails(self, run_module: Any, repo: Path) -> None:
        invoke(run_module, repo, "--freeze")
        self.publish(repo)
        (repo / "docs" / "PROOF-SCOPE.md").unlink()
        code, text = invoke(run_module, repo, "--check")
        assert code == 1 and "PROOF-SCOPE.md is missing" in text

    def test_the_check_needs_no_simulation_and_no_heavy_import(self) -> None:
        import subprocess
        import sys

        code = (
            "import sys; sys.path.insert(0, '.');"
            "import benchmarks.run, closed_loop.eval.report;"
            "bad = [m for m in ('closed_loop.eval.episode', 'closed_loop.eval.arms',"
            " 'closed_loop.plant') if m in sys.modules]; print(bad)"
        )
        repo_root = Path(__file__).resolve().parents[2]
        done = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, cwd=repo_root,
            timeout=120, check=False,
        )  # fmt: skip
        assert done.returncode == 0 and done.stdout.strip() == "[]", done.stdout + done.stderr


class TestTheBody:
    def fake_run_benchmark(self, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []

        def fake(**kw: Any) -> Any:
            calls.append(kw)
            records = base_records(kw["mode"])
            return types.SimpleNamespace(
                records=tuple(records),
                complete=kw["mode"] == "full",
                pipeline=types.SimpleNamespace(not_built=("search", "f2")),
            )

        import closed_loop.eval.stages as stages

        monkeypatch.setattr(stages, "run_benchmark", fake)
        return calls

    def test_a_full_run_writes_the_published_files_and_returns_the_digest(
        self, run_module: Any, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self.fake_run_benchmark(monkeypatch)
        invoke(run_module, repo, "--freeze")

        def publish(paths: Any, workers: int, out: Any) -> int:  # stands in for the two scripts
            paths.what_gets_through.write_text("# what gets through\n", encoding="utf-8")
            return 0

        code, text = invoke(run_module, repo, publish=publish)
        assert code == 0 and calls[0]["mode"] == "full" and calls[0]["prereg_sha"]
        assert calls[0]["guard"] is not None and calls[0]["workers"] == 3
        for p in PUBLISHED:
            assert (repo / p).is_file(), p
        last = ledger_of(repo).entries()[-1]
        assert last["kind"] == "full" and len(last["headline"]["results_sha256"]) == 64
        assert last["headline"]["complete"] is True and last["headline"]["n_records"] > 0
        assert invoke(run_module, repo, "--check")[0] == 0  # what it wrote, it can check

    def test_each_full_run_is_archived_so_that_the_first_survives_a_rerun(
        self, run_module: Any, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.fake_run_benchmark(monkeypatch)
        invoke(run_module, repo, "--freeze")
        invoke(run_module, repo)
        runs = repo / "benchmarks" / "runs"
        first = (runs / "results-run-1.json").read_bytes()
        assert first == (repo / "benchmarks" / "results.json").read_bytes()
        assert (runs / "results-run-1.log").is_file()  # the run log: what the records leave out
        invoke(run_module, repo)  # a rerun
        assert (runs / "results-run-1.json").read_bytes() == first  # the first run is still there
        assert (runs / "results-run-2.json").is_file() and (runs / "results-run-2.log").is_file()

    def test_a_run_never_replaces_an_archive_that_is_already_there(
        self, run_module: Any, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.fake_run_benchmark(monkeypatch)
        invoke(run_module, repo, "--freeze")
        runs = repo / "benchmarks" / "runs"
        runs.mkdir(parents=True)
        (runs / "results-run-1.json").write_text("the record of an earlier run", encoding="utf-8")
        invoke(run_module, repo)
        assert (runs / "results-run-1.json").read_text(encoding="utf-8") == (
            "the record of an earlier run"
        )
        assert (runs / "results-run-1-again.json").is_file()

    def test_a_full_run_hands_the_ledger_facts_to_the_benchmark(
        self, run_module: Any, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self.fake_run_benchmark(monkeypatch)
        invoke(run_module, repo, "--freeze")
        invoke(run_module, repo)
        facts = calls[0]["ledger_facts"]
        assert facts["run_number"] == 1 and facts["code_equals_frozen"] is True
        assert facts["tuned_equals_frozen"] is True
        assert len(facts["running_code_sha256"]) == 64
        invoke(run_module, repo)
        assert calls[1]["ledger_facts"]["run_number"] == 2

    def test_a_quick_run_has_no_ledger_facts_and_no_archive(
        self, run_module: Any, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self.fake_run_benchmark(monkeypatch)
        invoke(run_module, repo, "--quick")
        assert calls[0]["ledger_facts"] is None
        assert not (repo / "benchmarks" / "runs").exists()

    def test_a_quick_run_writes_under_the_cache_and_never_over_the_published_files(
        self, run_module: Any, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self.fake_run_benchmark(monkeypatch)
        code, text = invoke(run_module, repo, "--quick", "--workers", "2")
        assert code == 0 and calls[0]["mode"] == "quick" and calls[0]["guard"] is None
        assert calls[0]["prereg_sha"] is None and calls[0]["workers"] == 2
        quick = repo / "benchmarks" / ".cache" / "quick-output"
        assert (quick / "results.json").is_file() and (quick / "RESULTS.md").is_file()
        assert (quick / "PROOF-SCOPE.md").is_file()
        for p in PUBLISHED:
            assert not (repo / p).exists(), p
        assert "QUICK RUN, NOT A RESULT" in (quick / "RESULTS.md").read_text(encoding="utf-8")

    def test_an_incomplete_stage_set_is_printed_by_the_body(
        self, run_module: Any, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.fake_run_benchmark(monkeypatch)
        # a full run whose pipeline reports itself incomplete
        import closed_loop.eval.stages as stages

        monkeypatch.setattr(
            stages,
            "run_benchmark",
            lambda **kw: types.SimpleNamespace(
                records=tuple(base_records(kw["mode"], complete=False)),
                complete=False,
                pipeline=types.SimpleNamespace(not_built=("search", "f2")),
            ),
        )
        invoke(run_module, repo, "--freeze")
        code, text = invoke(run_module, repo)
        assert "INCOMPLETE RUN: this run does not claim completeness. Not built: search, f2" in text
        assert code == 5
