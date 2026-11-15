"""P12g, G4: ``python benchmarks/run.py`` regenerates every published number, literally.

After a full run has recorded its ledger entry, ``run.py`` writes ``docs/WHAT-GETS-THROUGH.md`` and
the README Results block itself; ``--check`` reports the first as missing and the second as stale.
The scripts are not started here (the first one re-evaluates and shrinks hundreds of finds): the
step runner is replaced by a recorder, except where the README script, which needs no simulation,
is run for real.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

import pytest

from closed_loop.eval.report import write_outputs
from tests.audit import test_run_body
from tests.audit.test_run import invoke, ledger_of
from tests.audit.test_run_body import (
    BEGIN,
    END,
    PUBLISHED,
    WGT,
    load_generator,
    readme_block,
)
from tests.eval.test_report import base_records


class Steps:
    """Stands in for ``stream_command``: records each command, answers with the next exit code."""

    def __init__(self, *codes: int) -> None:
        self.commands: list[list[str]] = []
        self.codes = list(codes)

    def __call__(self, command: list[str], out: Any) -> int:
        self.commands.append(command)
        return self.codes.pop(0) if self.codes else 0


def flags(command: list[str]) -> dict[str, str]:
    """The ``--flag value`` pairs of a command line (a flag with no value maps to '')."""
    out: dict[str, str] = {}
    for i, word in enumerate(command):
        if word.startswith("--"):
            nxt = command[i + 1] if i + 1 < len(command) else ""
            out[word] = "" if nxt.startswith("--") else nxt
    return out


def with_results(repo: Path) -> None:
    write_outputs(base_records(), *[repo / p for p in PUBLISHED])


def with_readme(repo: Path, block: str = f"{BEGIN}\nold\n{END}") -> Path:
    path = repo / "README.md"
    path.write_text(f"# Title\n\nBefore.\n\n{block}\n\nAfter.\n", encoding="utf-8")
    return path


def publish_with(run_module: Any, repo: Path, steps: Any, workers: int = 2) -> tuple[int, str]:
    out = io.StringIO()
    paths = run_module.paths_for(repo)
    return run_module.publish_documents(paths, workers, out, stream=steps), out.getvalue()


class TestThePublishedCommands:
    def test_both_generators_are_run_with_this_runs_files_and_workers(
        self, run_module: Any, repo: Path
    ) -> None:
        with_results(repo)
        with_readme(repo)
        steps = Steps()
        code, text = publish_with(run_module, repo, steps, workers=2)
        assert code == 0 and len(steps.commands) == 2
        wgt, readme = steps.commands
        assert wgt[0] == sys.executable and wgt[1].endswith("render_what_gets_through.py")
        f = flags(wgt)
        assert f["--workers"] == "2"
        assert f["--results"] == str(repo / "benchmarks" / "results.json")
        assert f["--out"] == str(repo / WGT)
        assert f["--cells"] == str(repo / "benchmarks" / ".cache" / "full")
        assert f["--cache"] == str(repo / "benchmarks" / ".cache" / "wgt")
        assert "--lock" not in f, "this process holds the lock; the script must not take it again"
        assert readme[1].endswith("render_readme_results.py") and "--write" in readme
        assert flags(readme)["--readme"] == str(repo / "README.md")
        assert flags(readme)["--results"] == str(repo / "benchmarks" / "results.json")
        assert "publish: wrote docs/WHAT-GETS-THROUGH.md" in text
        assert "publish: wrote the Results block of README.md" in text

    def test_the_flags_are_ones_the_script_accepts(self, run_module: Any, repo: Path) -> None:
        with_results(repo)
        with_readme(repo)
        steps = Steps()
        publish_with(run_module, repo, steps, workers=3)
        script = load_generator("render_what_gets_through")
        args = script.parse(steps.commands[0][2:])  # the script's own parser, not a copy of it
        assert args.workers == 3 and args.out == repo / WGT and args.lock is None
        assert args.results == repo / "benchmarks" / "results.json"

    def test_a_readme_without_markers_is_said_and_left_alone(
        self, run_module: Any, repo: Path
    ) -> None:
        with_results(repo)
        path = repo / "README.md"
        path.write_text("# Title\n\nNo block here.\n", encoding="utf-8")
        steps = Steps()
        code, text = publish_with(run_module, repo, steps)
        assert code == 0 and len(steps.commands) == 1
        assert "README.md has no Results block markers" in text and BEGIN in text
        assert path.read_text(encoding="utf-8") == "# Title\n\nNo block here.\n"

    def test_a_repository_without_a_readme_publishes_the_one_document(
        self, run_module: Any, repo: Path
    ) -> None:
        with_results(repo)
        steps = Steps()
        code, _ = publish_with(run_module, repo, steps)
        assert code == 0 and len(steps.commands) == 1

    def test_the_markers_are_those_of_the_script(self, run_module: Any) -> None:
        script = load_generator("render_readme_results")
        assert (run_module.README_BEGIN, run_module.README_END) == (script.BEGIN, script.END)


class TestAFailureIsNamed:
    def test_a_failed_first_step_is_named_and_the_second_still_runs(
        self, run_module: Any, repo: Path
    ) -> None:
        with_results(repo)
        with_readme(repo)
        steps = Steps(3, 0)
        code, text = publish_with(run_module, repo, steps)
        assert code == 1 and len(steps.commands) == 2
        assert "publish: FAILED (docs/WHAT-GETS-THROUGH.md: exit 3)" in text
        assert "the ledger entry stands" in text
        assert "publish: wrote the Results block of README.md" in text

    def test_a_failed_second_step_is_named(self, run_module: Any, repo: Path) -> None:
        with_results(repo)
        with_readme(repo)
        code, text = publish_with(run_module, repo, Steps(0, 2))
        assert code == 1
        assert "publish: FAILED (the Results block of README.md: exit 2)" in text
        assert "publish: wrote docs/WHAT-GETS-THROUGH.md" in text

    def test_both_failing_are_both_named(self, run_module: Any, repo: Path) -> None:
        with_results(repo)
        with_readme(repo)
        code, text = publish_with(run_module, repo, Steps(1, 1))
        assert (
            code == 1
            and "WHAT-GETS-THROUGH.md: exit 1; the Results block of README.md: exit 1" in text
        )

    def test_a_script_that_cannot_start_is_a_failure_not_a_traceback(
        self, run_module: Any, repo: Path
    ) -> None:
        with_results(repo)

        def broken(command: list[str], out: Any) -> int:
            raise FileNotFoundError("no interpreter")

        code, text = publish_with(run_module, repo, broken)
        assert code == 1 and "could not start the step (FileNotFoundError" in text


class TestTheReadmeScriptForReal:
    def test_the_block_is_replaced_and_the_rest_of_the_readme_is_not_touched(
        self, run_module: Any, repo: Path
    ) -> None:
        with_results(repo)
        path = with_readme(repo)
        real = run_module.stream_command

        def only_the_readme(command: list[str], out: Any) -> int:
            return real(command, out) if command[1].endswith("render_readme_results.py") else 0

        code, text = publish_with(run_module, repo, only_the_readme)
        assert code == 0, text
        assert path.read_text(encoding="utf-8") == (
            f"# Title\n\nBefore.\n\n{readme_block(repo)}\n\nAfter.\n"
        )

    def test_a_step_that_exits_non_zero_is_reported_with_its_own_output(
        self, run_module: Any, repo: Path
    ) -> None:
        with_results(repo)
        with_readme(repo, block=f"{BEGIN}\nold\n")  # the END marker is missing
        real = run_module.stream_command

        def only_the_readme(command: list[str], out: Any) -> int:
            return real(command, out) if command[1].endswith("render_readme_results.py") else 0

        code, text = publish_with(run_module, repo, only_the_readme)
        assert code == 1 and "needs exactly one" in text and "exit 2" in text


class TestStreamCommand:
    def test_output_is_copied_line_by_line_and_the_exit_code_is_returned(
        self, run_module: Any
    ) -> None:
        out = io.StringIO()
        program = "import sys; print('one'); print('two', file=sys.stderr); sys.exit(7)"
        code = run_module.stream_command([sys.executable, "-c", program], out)
        assert code == 7
        assert sorted(out.getvalue().splitlines()) == ["  | one", "  | two"]

    def test_a_program_that_is_not_there_raises_oserror(
        self, run_module: Any, tmp_path: Path
    ) -> None:
        with pytest.raises(OSError):
            run_module.stream_command([str(tmp_path / "no-such-program.exe")], io.StringIO())


class TestWhereTheRunPublishes:
    def frozen(self, run_module: Any, repo: Path) -> None:
        invoke(run_module, repo, "--freeze")

    def test_it_runs_after_the_ledger_entry_with_the_workers_of_the_run(
        self, run_module: Any, repo: Path
    ) -> None:
        self.frozen(run_module, repo)
        seen: list[Any] = []

        def publish(paths: Any, workers: int, out: Any) -> int:
            seen.append((ledger_of(repo).entries()[-1]["kind"], workers, paths.root))
            return 0

        code, _ = invoke(run_module, repo, "--workers", "2", body=lambda c: {}, publish=publish)
        assert code == 0 and seen == [("full", 2, repo)]

    def test_a_quick_run_publishes_nothing(self, run_module: Any, repo: Path) -> None:
        seen: list[int] = []
        invoke(
            run_module, repo, "--quick", body=lambda c: {}, publish=lambda *a: seen.append(1) or 0
        )
        assert seen == []

    def test_a_run_that_is_refused_or_failed_publishes_nothing(
        self, run_module: Any, repo: Path
    ) -> None:
        seen: list[int] = []

        def publish(*a: Any) -> int:
            seen.append(1)
            return 0

        invoke(run_module, repo, body=lambda c: {}, publish=publish)  # no freeze entry
        self.frozen(run_module, repo)

        def raises(ctx: Any) -> dict[str, Any]:
            raise RuntimeError("worker died")

        invoke(run_module, repo, body=raises, publish=publish)
        invoke(run_module, repo, commit_reader=lambda: 95.0, body=lambda c: {}, publish=publish)
        assert seen == []

    def test_a_failed_publication_exits_1_and_the_ledger_entry_stands(
        self, run_module: Any, repo: Path
    ) -> None:
        self.frozen(run_module, repo)
        code, _ = invoke(run_module, repo, body=lambda c: {"complete": True}, publish=lambda *a: 1)
        assert code == 1
        last = ledger_of(repo).entries()[-1]
        assert last["kind"] == "full" and last["headline"] == {"complete": True}
        ledger_of(repo).verify()

    def test_an_incomplete_run_still_publishes_and_still_exits_5(
        self, run_module: Any, repo: Path
    ) -> None:
        self.frozen(run_module, repo)
        seen: list[int] = []
        code, text = invoke(
            run_module, repo, body=lambda c: {"complete": False},
            publish=lambda *a: seen.append(1) or 0,
        )  # fmt: skip
        assert code == 5 and seen == [1] and "INCOMPLETE" in text

    def test_without_a_publisher_given_the_real_one_runs_both_scripts(
        self, run_module: Any, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with_results(repo)
        with_readme(repo)
        self.frozen(run_module, repo)
        steps = Steps()
        monkeypatch.setattr(run_module, "stream_command", steps)
        out = io.StringIO()
        code = run_module.main(
            ["--root", str(repo), "--workers", "3"],
            out=out,
            commit_reader=lambda: 40.0,
            body=lambda ctx: {"complete": True},
        )
        assert code == 0, out.getvalue()
        assert [c[1].rsplit("render_", 1)[1] for c in steps.commands] == [
            "what_gets_through.py",
            "readme_results.py",
        ]
        assert flags(steps.commands[0])["--workers"] == "3"

    def test_the_default_publisher_is_the_real_one_in_main_and_in_cmd_run(
        self, run_module: Any
    ) -> None:
        import inspect

        for fn in (run_module.main, run_module.cmd_run):
            default = inspect.signature(fn).parameters["publish"].default
            assert default is run_module.publish_documents, fn.__name__


class TestTheCheckSeesTheOtherTwoDocuments:
    def published(self, run_module: Any, repo: Path, **kw: Any) -> None:
        invoke(run_module, repo, "--freeze")
        test_run_body.TestCheckOfPublishedResults().publish(repo, **kw)

    def test_all_three_present_and_current_pass_and_say_so(
        self, run_module: Any, repo: Path
    ) -> None:
        self.published(run_module, repo)
        code, text = invoke(run_module, repo, "--check")
        assert code == 0
        assert "WHAT-GETS-THROUGH.md is there" in text and "README Results block is current" in text

    def test_a_missing_what_gets_through_fails(self, run_module: Any, repo: Path) -> None:
        self.published(run_module, repo)
        (repo / WGT).unlink()
        code, text = invoke(run_module, repo, "--check")
        assert code == 1 and "results: FAILED (docs/WHAT-GETS-THROUGH.md is missing" in text

    def test_a_stale_readme_block_fails(self, run_module: Any, repo: Path) -> None:
        self.published(run_module, repo)
        path = repo / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8").replace(END, f"A hand-typed line.\n{END}"),
            encoding="utf-8",
        )
        code, text = invoke(run_module, repo, "--check")
        assert code == 1 and "the README Results block is stale" in text

    def test_a_readme_block_from_other_results_is_stale(self, run_module: Any, repo: Path) -> None:
        self.published(run_module, repo)
        path = repo / "README.md"
        block = readme_block(repo)
        line = block.splitlines()[4]  # the "Run mode: ..." line, a statement about this run
        assert line.startswith("Run mode")
        path.write_text(f"# T\n\n{block.replace(line, 'Run mode: another.')}\n", encoding="utf-8")
        code, text = invoke(run_module, repo, "--check")
        assert code == 1 and "the README Results block is stale" in text

    def test_a_readme_with_one_marker_cannot_be_checked_and_fails(
        self, run_module: Any, repo: Path
    ) -> None:
        self.published(run_module, repo)
        (repo / "README.md").write_text(f"# T\n\n{BEGIN}\nbroken\n", encoding="utf-8")
        code, text = invoke(run_module, repo, "--check")
        assert code == 1 and "the README Results block could not be checked" in text

    def test_a_readme_with_no_block_is_not_a_failure(self, run_module: Any, repo: Path) -> None:
        self.published(run_module, repo)
        (repo / "README.md").write_text("# T\n\nNo block.\n", encoding="utf-8")
        code, text = invoke(run_module, repo, "--check")
        assert code == 0 and "the README has no Results block" in text

    def test_no_readme_at_all_is_not_a_failure(self, run_module: Any, repo: Path) -> None:
        self.published(run_module, repo)
        (repo / "README.md").unlink()
        assert invoke(run_module, repo, "--check")[0] == 0

    def test_nothing_published_still_needs_neither(self, run_module: Any, repo: Path) -> None:
        invoke(run_module, repo, "--freeze")
        code, text = invoke(run_module, repo, "--check")
        assert code == 0 and "none published yet" in text
