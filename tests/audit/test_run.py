"""P10: the benchmarks/run.py skeleton (freeze, check, quick and full preflight)."""

import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

from closed_loop.audit.ledger import Ledger
from closed_loop.audit.seeds import heldout_seed

REPO = Path(__file__).resolve().parents[2]
RUN_PY = REPO / "benchmarks" / "run.py"
PREREG = "# Pre-registration (test copy)\n\nClaim C1.\n"
TUNED = {
    "k_b3": 7.0, "b2_limit": 5.5, "b1_fraction": 0.5, "delta_plan": 0.02,
    "primary_variant": "V3", "monitor_mode": "proxy",
    "tuned_on": "dev seeds 1100-1139 (a test copy)", "script": "scripts/tune.py",
}  # fmt: skip


@pytest.fixture(scope="module")
def run_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("closed_loop_bench_run", RUN_PY)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up here
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A miniature repository: a package, the real run.py, a pre-registration."""
    (tmp_path / "closed_loop").mkdir()
    (tmp_path / "closed_loop" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "benchmarks").mkdir()
    shutil.copy(RUN_PY, tmp_path / "benchmarks" / "run.py")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "PREREGISTRATION.md").write_bytes(PREREG.encode())
    (tmp_path / "benchmarks" / "tuned.json").write_text(json.dumps(TUNED), encoding="utf-8")
    return tmp_path


class TheRealBenchmarkMustNotRunInATest(BaseException):  # noqa: N818 - must not be caught
    """Raised if a test reaches the real ``run_benchmark``. With a gate removed (as a mutant removes
    one) a test that expects a refusal would otherwise start the benchmark, and a real benchmark is
    hours of simulation. It is a BaseException so that the run's own handlers (which record a failed
    run and go on) do not swallow it."""


def no_publication(paths, workers, out) -> int:
    """The publisher of these tests: it writes no document and starts no script. The real one is
    tested in ``tests/audit/test_run_body.py`` (P12g)."""
    return 0


def invoke(module: ModuleType, repo: Path, *argv: str, **kwargs):
    out = io.StringIO()
    code = module.main(
        ["--root", str(repo), *argv],
        out=out,
        commit_reader=kwargs.pop("commit_reader", lambda: 40.0),
        publish=kwargs.pop("publish", no_publication),
        **kwargs,
    )
    return code, out.getvalue()


def ledger_of(repo: Path) -> Ledger:
    return Ledger(repo / "benchmarks" / "ledger.jsonl")


class TestFreeze:
    def test_freeze_writes_the_first_entry_with_the_hashes(self, run_module, repo: Path) -> None:
        code, text = invoke(run_module, repo, "--freeze")
        assert code == 0 and "froze pre-registration" in text and "ledger head" in text
        (entry,) = ledger_of(repo).entries()
        assert entry["kind"] == "freeze" and entry["seq"] == 0
        assert entry["prereg_sha256"] and entry["code_sha256"]
        assert entry["tuned_sha256"] and entry["siblings_lock_sha256"] is None
        assert entry["seed_roles"] == {
            "dev": [1000, 1999],
            "calib": [2000, 2999],
            "search-dev": [3000, 3999],
        }  # fmt: skip  (the published search seeds are derived, so they have no range)

    def test_a_second_freeze_is_refused(self, run_module, repo: Path) -> None:
        invoke(run_module, repo, "--freeze")
        code, text = invoke(run_module, repo, "--freeze")
        assert code == 1 and "already has entries" in text
        assert len(ledger_of(repo).entries()) == 1

    def test_freeze_without_a_preregistration(self, run_module, repo: Path) -> None:
        (repo / "docs" / "PREREGISTRATION.md").unlink()
        code, text = invoke(run_module, repo, "--freeze")
        assert code == 4 and "does not exist" in text
        assert not (repo / "benchmarks" / "ledger.jsonl").exists()

    def test_freeze_records_tuned_and_sibling_hashes_when_present(
        self, run_module, repo: Path
    ) -> None:
        (repo / "integration").mkdir()
        (repo / "integration" / "siblings.lock").write_text("a=1\n")
        invoke(run_module, repo, "--freeze")
        entry = ledger_of(repo).entries()[0]
        assert entry["tuned_sha256"] and entry["siblings_lock_sha256"]


class TestTunedGate:
    """0003 A15: a freeze needs a real tuned.json, and a run needs the one that was frozen."""

    def write(self, repo: Path, **changes: object) -> None:
        doc = {k: v for k, v in {**TUNED, **changes}.items() if v is not None}
        (repo / "benchmarks" / "tuned.json").write_text(json.dumps(doc), encoding="utf-8")

    def refused(self, run_module, repo: Path) -> str:
        code, text = invoke(run_module, repo, "--freeze")
        assert code == 1 and text.count("refusing to freeze") >= 1
        assert not (repo / "benchmarks" / "ledger.jsonl").exists()  # nothing recorded
        return text

    def test_no_tuned_file_is_refused(self, run_module, repo: Path) -> None:
        (repo / "benchmarks" / "tuned.json").unlink()
        assert "does not exist" in self.refused(run_module, repo)

    def test_a_file_that_names_no_primary_variant_is_refused(self, run_module, repo: Path) -> None:
        self.write(repo, primary_variant=None)
        assert "does not name primary_variant" in self.refused(run_module, repo)

    def test_a_file_that_names_no_monitor_mode_is_refused(self, run_module, repo: Path) -> None:
        self.write(repo, monitor_mode=None)
        assert "does not name monitor_mode" in self.refused(run_module, repo)

    def test_a_malformed_file_is_refused(self, run_module, repo: Path) -> None:
        (repo / "benchmarks" / "tuned.json").write_text('{"gamma": 0.02}\n', encoding="utf-8")
        assert "malformed" in self.refused(run_module, repo)

    def test_a_refused_freeze_leaves_the_draft_where_it_was(self, run_module, repo: Path) -> None:
        (repo / "docs" / "PREREGISTRATION.md").rename(repo / "docs" / "PREREGISTRATION.draft.md")
        self.write(repo, monitor_mode=None)
        self.refused(run_module, repo)
        assert (repo / "docs" / "PREREGISTRATION.draft.md").is_file()
        assert not (repo / "docs" / "PREREGISTRATION.md").exists()

    def test_a_file_that_names_both_is_frozen_with_its_hash(self, run_module, repo: Path) -> None:
        code, _ = invoke(run_module, repo, "--freeze")
        assert code == 0
        (entry,) = ledger_of(repo).entries()
        from closed_loop.audit.prereg import file_sha256

        assert entry["tuned_sha256"] == file_sha256(repo / "benchmarks" / "tuned.json")

    def test_a_full_run_refuses_a_tuned_file_that_changed_after_the_freeze(
        self, run_module, repo: Path
    ) -> None:
        invoke(run_module, repo, "--freeze")
        self.write(repo, k_b3=9.0)
        code, text = invoke(run_module, repo)
        assert code == 1 and "tuned.json differs from the freeze entry" in text
        assert len(ledger_of(repo).entries()) == 1  # no full-failed entry, nothing was started

    def test_the_check_fails_on_a_changed_tuned_file(self, run_module, repo: Path) -> None:
        invoke(run_module, repo, "--freeze")
        code, text = invoke(run_module, repo, "--check")
        assert code == 0 and "tuned: CHANGED" not in text
        self.write(repo, monitor_mode="inf")
        code, text = invoke(run_module, repo, "--check")
        assert code == 1 and "tuned: CHANGED since the freeze" in text


class TestCodeHash:
    def test_the_spec_caches_are_part_of_the_code_hash(self, run_module, repo: Path) -> None:
        spec = repo / "closed_loop" / "spec"
        spec.mkdir()
        (spec / "clause_cache.json").write_text('{"a": 1}\n', encoding="utf-8")
        (spec / "note_verdicts.json").write_text('{"b": 1}\n', encoding="utf-8")
        paths = run_module.paths_for(repo)
        before = run_module.current_hashes(paths)["code_sha256"]
        (spec / "clause_cache.json").write_text('{"a": 2}\n', encoding="utf-8")
        mid = run_module.current_hashes(paths)["code_sha256"]
        (spec / "note_verdicts.json").write_text('{"b": 2}\n', encoding="utf-8")
        after = run_module.current_hashes(paths)["code_sha256"]
        assert len({before, mid, after}) == 3

    def test_a_repository_without_the_caches_still_hashes(self, run_module, repo: Path) -> None:
        assert len(run_module.current_hashes(run_module.paths_for(repo))["code_sha256"]) == 64


class TestCheck:
    def test_a_draft_that_is_not_frozen_is_reported_not_failed(
        self, run_module, repo: Path
    ) -> None:
        code, text = invoke(run_module, repo, "--check")
        assert code == 0 and "ledger: OK (0 entries" in text and "NOT FROZEN" in text

    def test_matching_hashes(self, run_module, repo: Path) -> None:
        invoke(run_module, repo, "--freeze")
        code, text = invoke(run_module, repo, "--check")
        assert code == 0 and "prereg: MATCH" in text and "code: UNCHANGED" in text

    def test_crlf_checkout_still_matches(self, run_module, repo: Path) -> None:
        invoke(run_module, repo, "--freeze")
        path = repo / "docs" / "PREREGISTRATION.md"
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
        code, text = invoke(run_module, repo, "--check")
        assert code == 0 and "prereg: MATCH" in text

    def test_an_edited_preregistration_fails(self, run_module, repo: Path) -> None:
        invoke(run_module, repo, "--freeze")
        (repo / "docs" / "PREREGISTRATION.md").write_bytes(PREREG.replace("C1", "C2").encode())
        code, text = invoke(run_module, repo, "--check")
        assert code == 1 and "prereg: MISMATCH" in text

    def test_a_deleted_preregistration_fails_once_frozen(self, run_module, repo: Path) -> None:
        invoke(run_module, repo, "--freeze")
        (repo / "docs" / "PREREGISTRATION.md").unlink()
        code, text = invoke(run_module, repo, "--check")
        assert code == 1 and "missing" in text

    def test_changed_code_is_reported_but_does_not_fail(self, run_module, repo: Path) -> None:
        invoke(run_module, repo, "--freeze")
        (repo / "closed_loop" / "mod.py").write_text("x = 2\n", encoding="utf-8")
        code, text = invoke(run_module, repo, "--check")
        assert code == 0 and "code: CHANGED since the freeze" in text

    def test_a_broken_ledger_fails(self, run_module, repo: Path) -> None:
        invoke(run_module, repo, "--freeze")
        path = repo / "benchmarks" / "ledger.jsonl"
        raw = bytearray(path.read_bytes())
        raw[25] ^= 1
        path.write_bytes(bytes(raw))
        code, text = invoke(run_module, repo, "--check")
        assert code == 1 and "ledger: BROKEN" in text

    def test_a_truncated_ledger_fails_when_the_head_is_supplied(
        self, run_module, repo: Path
    ) -> None:
        invoke(run_module, repo, "--freeze")
        ledger_of(repo).append("full", prereg_sha256="x")
        published_head = ledger_of(repo).head()  # what the release notes would carry
        path = repo / "benchmarks" / "ledger.jsonl"
        path.write_bytes(b"".join(path.read_bytes().splitlines(keepends=True)[:1]))
        # the shortened file is a valid chain on its own, so the plain check cannot see the loss
        assert invoke(run_module, repo, "--check")[0] == 0
        code, text = invoke(run_module, repo, "--check", "--head", published_head)
        assert code == 1 and "ledger: BROKEN" in text and "expected" in text
        assert invoke(run_module, repo, "--check", "--head", ledger_of(repo).head())[0] == 0

    def test_reports_the_heldout_counter(self, run_module, repo: Path) -> None:
        invoke(run_module, repo, "--freeze")
        ledger_of(repo).append("heldout-use", n=7)
        assert "held-out seeds handed out: 7" in invoke(run_module, repo, "--check")[1]

    def test_check_works_even_above_the_commit_limit(self, run_module, repo: Path) -> None:
        """--check is cheap; only runs are gated on memory."""
        code, _ = invoke(run_module, repo, "--check", commit_reader=lambda: 99.0)
        assert code == 0


class TestFullRun:
    def test_refused_without_a_freeze_entry(self, run_module, repo: Path) -> None:
        called = []
        code, text = invoke(run_module, repo, body=lambda ctx: called.append(ctx) or {})
        assert code == 1 and "no freeze entry" in text and called == []

    def test_refused_when_the_preregistration_changed(self, run_module, repo: Path) -> None:
        invoke(run_module, repo, "--freeze")
        (repo / "docs" / "PREREGISTRATION.md").write_bytes(b"edited\n")
        called = []
        code, text = invoke(run_module, repo, "--full", body=lambda ctx: called.append(1) or {})
        assert code == 1 and "differs from the freeze entry" in text and called == []

    def test_refused_when_the_ledger_is_broken(self, run_module, repo: Path) -> None:
        invoke(run_module, repo, "--freeze")
        path = repo / "benchmarks" / "ledger.jsonl"
        raw = bytearray(path.read_bytes())
        raw[25] ^= 1
        path.write_bytes(bytes(raw))
        code, text = invoke(run_module, repo, body=lambda ctx: {})
        assert code == 1 and "ledger is broken" in text

    def test_the_default_body_is_the_benchmark(
        self, run_module, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # P12b replaced the skeleton's NotImplementedError by the stages: with no body injected,
        # a full run calls run_experiments (here a stand-in, so the test needs no simulation)
        invoke(run_module, repo, "--freeze")
        seen = []
        monkeypatch.setattr(run_module, "run_experiments", lambda ctx: seen.append(ctx.mode) or {})
        code, _ = invoke(run_module, repo)
        assert code == 0 and seen == ["full"]
        assert [e["kind"] for e in ledger_of(repo).entries()] == ["freeze", "full"]

    def test_a_successful_run_appends_a_full_entry_with_the_headline_hash(
        self, run_module, repo: Path
    ) -> None:
        invoke(run_module, repo, "--freeze")
        seen = {}

        def body(ctx):
            seen["mode"], seen["frozen"] = ctx.mode, ctx.guard.is_frozen()
            seen["seeds"] = ctx.guard.seeds(3)
            return {"C1": 0.25, "n": 10}

        code, text = invoke(run_module, repo, "--full", body=body)
        assert code == 0 and "full run recorded" in text
        entries = ledger_of(repo).entries()
        assert [e["kind"] for e in entries] == ["freeze", "heldout-use", "full"]
        full = entries[-1]
        assert full["headline"] == {"C1": 0.25, "n": 10} and len(full["headline_sha256"]) == 64
        assert full["prereg_sha256"] == full["prereg_sha256_after"] == entries[0]["prereg_sha256"]
        assert seen["mode"] == "full" and seen["frozen"]
        assert seen["seeds"] == [heldout_seed(entries[0]["prereg_sha256"], i) for i in range(3)]
        assert ledger_of(repo).heldout_evaluations() == 3

    def test_default_with_no_flag_is_the_full_run(self, run_module, repo: Path) -> None:
        invoke(run_module, repo, "--freeze")
        modes = []
        invoke(run_module, repo, body=lambda ctx: modes.append(ctx.mode) or {})
        assert modes == ["full"]

    def test_a_run_that_raises_is_recorded_as_full_failed(self, run_module, repo: Path) -> None:
        invoke(run_module, repo, "--freeze")

        def body(ctx):
            raise RuntimeError("worker died")

        code, text = invoke(run_module, repo, body=body)
        assert code == 1 and "recorded" in text
        last = ledger_of(repo).entries()[-1]
        assert last["kind"] == "full-failed" and "RuntimeError: worker died" in last["error"]

    def test_code_changing_during_the_run_is_flagged_and_recorded(
        self, run_module, repo: Path
    ) -> None:
        invoke(run_module, repo, "--freeze")

        def body(ctx):
            (repo / "closed_loop" / "mod.py").write_text("x = 99\n", encoding="utf-8")
            return {}

        code, text = invoke(run_module, repo, body=body)
        assert code == 1 and "code changed during the run" in text
        last = ledger_of(repo).entries()[-1]
        assert last["kind"] == "full" and last["code_sha256"] != last["code_sha256_after"]

    def test_preregistration_changing_during_the_run_is_flagged(
        self, run_module, repo: Path
    ) -> None:
        invoke(run_module, repo, "--freeze")

        def body(ctx):
            (repo / "docs" / "PREREGISTRATION.md").write_bytes(b"sneaky edit\n")
            return {}

        code, text = invoke(run_module, repo, body=body)
        assert code == 1 and "pre-registration changed during the run" in text


class TestQuickRun:
    def test_quick_prints_the_banner_and_writes_nothing(self, run_module, repo: Path) -> None:
        got = {}

        def body(ctx):
            got["mode"], got["guard"], got["sha"] = ctx.mode, ctx.guard, ctx.prereg_sha
            return {"x": 1}

        code, text = invoke(run_module, repo, "--quick", body=body)
        assert code == 0 and text.count("QUICK RUN, NOT A RESULT") == 2
        assert got == {"mode": "quick", "guard": None, "sha": None}  # no held-out seeds at all
        assert not (repo / "benchmarks" / "ledger.jsonl").exists()

    def test_quick_does_not_need_a_freeze_and_leaves_a_frozen_ledger_alone(
        self, run_module, repo: Path
    ) -> None:
        invoke(run_module, repo, "--freeze")
        before = (repo / "benchmarks" / "ledger.jsonl").read_bytes()
        assert invoke(run_module, repo, "--quick", body=lambda ctx: {})[0] == 0
        assert (repo / "benchmarks" / "ledger.jsonl").read_bytes() == before

    def test_the_default_quick_body_is_the_benchmark_too(
        self, run_module, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = []
        monkeypatch.setattr(run_module, "run_experiments", lambda ctx: seen.append(ctx.mode) or {})
        code, text = invoke(run_module, repo, "--quick")
        assert code == 0 and seen == ["quick"] and text.count("QUICK RUN, NOT A RESULT") == 2
        assert not (repo / "benchmarks" / "ledger.jsonl").exists()


class TestGates:
    def test_refuses_to_start_above_88_percent_commit_charge(self, run_module, repo: Path) -> None:
        invoke(run_module, repo, "--freeze")
        called = []
        for flag in ("--quick", "--full"):
            code, text = invoke(
                run_module, repo, flag, commit_reader=lambda: 88.5,
                body=lambda ctx: called.append(1) or {},
            )  # fmt: skip
            assert code == 3 and "refused" in text and "88.5%" in text
        assert called == []

    def test_the_limit_can_be_changed(self, run_module, repo: Path) -> None:
        code, _ = invoke(
            run_module, repo, "--quick", "--limit", "95", commit_reader=lambda: 90.0,
            body=lambda ctx: {},
        )  # fmt: skip
        assert code == 0

    def test_a_held_lock_refuses_and_is_left_alone(
        self, run_module, repo: Path, tmp_path: Path
    ) -> None:
        lock = tmp_path / "heavy-run.lock"
        lock.write_text("owner=someone\n")
        called = []
        code, text = invoke(
            run_module,
            repo,
            "--quick",
            "--lock",
            str(lock),
            body=lambda ctx: called.append(1) or {},
        )
        assert code == 3 and "is held" in text and called == []
        assert lock.read_text() == "owner=someone\n"

    def test_our_lock_is_taken_during_the_run_and_removed_after(
        self, run_module, repo: Path, tmp_path: Path
    ) -> None:
        lock = tmp_path / "heavy-run.lock"
        during = []
        code, _ = invoke(
            run_module, repo, "--quick", "--lock", str(lock),
            body=lambda ctx: during.append(lock.exists()) or {},
        )  # fmt: skip
        assert code == 0 and during == [True] and not lock.exists()

    def test_the_lock_path_can_come_from_the_environment(
        self, run_module, repo: Path, tmp_path: Path, monkeypatch
    ) -> None:
        lock = tmp_path / "env.lock"
        monkeypatch.setenv("CLOSED_LOOP_LOCK", str(lock))
        during = []
        invoke(run_module, repo, "--quick", body=lambda ctx: during.append(lock.exists()) or {})
        assert during == [True] and not lock.exists()

    def test_the_lock_is_removed_after_a_failed_run(
        self, run_module, repo: Path, tmp_path: Path
    ) -> None:
        invoke(run_module, repo, "--freeze")
        lock = tmp_path / "heavy-run.lock"

        def body(ctx):
            raise RuntimeError("boom")

        code, _ = invoke(run_module, repo, "--lock", str(lock), body=body)
        assert code == 1 and not lock.exists()

    def test_an_old_lock_is_reported_as_stale_and_still_refuses(
        self, run_module, repo: Path, tmp_path: Path
    ) -> None:
        lock = tmp_path / "heavy-run.lock"
        lock.write_text("x")
        old = time.time() - 5 * 3600
        os.utime(lock, (old, old))
        code, text = invoke(run_module, repo, "--quick", "--lock", str(lock), body=lambda c: {})
        assert code == 3 and "stale" in text and lock.exists()


class TestCommandLine:
    @pytest.mark.parametrize("flags", [["--quick", "--full"], ["--check", "--freeze"], ["--nope"]])
    def test_usage_errors_exit_2(self, run_module, repo: Path, flags: list[str]) -> None:
        with pytest.raises(SystemExit) as info:
            run_module.main(["--root", str(repo), *flags], out=io.StringIO())
        assert info.value.code == 2

    def test_runs_as_a_script_on_the_real_repository(self) -> None:
        if (REPO / "benchmarks" / "ledger.jsonl").exists():
            pytest.skip("the repository has a real ledger now; --check is its own gate")
        done = subprocess.run(
            [sys.executable, str(RUN_PY), "--check"],
            capture_output=True, text=True, timeout=180, cwd=REPO, check=False,
        )  # fmt: skip
        assert done.returncode == 0, done.stdout + done.stderr
        assert "ledger: OK" in done.stdout

    def test_help_lists_the_exit_codes(self) -> None:
        done = subprocess.run(
            [sys.executable, str(RUN_PY), "--help"],
            capture_output=True, text=True, timeout=180, cwd=REPO, check=False,
        )  # fmt: skip
        assert done.returncode == 0 and "exit codes" in done.stdout
        for flag in ("--quick", "--full", "--check", "--freeze", "--lock"):
            assert flag in done.stdout
