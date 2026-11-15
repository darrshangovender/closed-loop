"""P12g, G5: no path of this machine in anything a run writes or keys on.

The first full run wrote the absolute path of ``benchmarks/tuned.json`` 1,913 times into
``results.json`` and the absolute paths of traceback frames into a verdict and the ledger. The
absolute path was also part of every cached cell's key (the tuned values are in a task's
fingerprint), so a clean checkout at another path would not have found those cells.
"""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

import pytest

from closed_loop.core.types import Fidelity
from closed_loop.eval.arms import ARMS
from closed_loop.eval.context import PreparedContext
from closed_loop.eval.episode import run_episode, scenario_for
from closed_loop.eval.paths import ROOT, relative_path, scrub
from closed_loop.eval.runner import EpisodeTask, _execute, cell_key, fingerprint
from closed_loop.eval.tuned import TUNED_PATH, load_tuned

HOME = Path.home()
QUICK = ROOT / "benchmarks" / ".cache" / "quick-output"


def machine_paths() -> list[str]:
    """Every way this machine's repository folder and home folder are written in text."""
    out: set[str] = set()
    for folder in (ROOT, HOME):
        native = str(folder)
        out |= {native, folder.as_posix(), native.replace("\\", "\\\\")}
    return sorted(out, key=len, reverse=True)


def leaks(text: str) -> list[str]:
    low = text.lower()
    return [p for p in machine_paths() if p.lower() in low]


class TestRelativePath:
    def test_a_file_in_the_repository_is_relative_with_forward_slashes(self) -> None:
        assert relative_path(TUNED_PATH) == "benchmarks/tuned.json"
        assert relative_path(str(ROOT / "closed_loop" / "eval" / "paths.py")) == (
            "closed_loop/eval/paths.py"
        )

    def test_a_file_outside_it_is_its_base_name(self, tmp_path: Path) -> None:
        assert relative_path(tmp_path / "somewhere" / "tuned.json") == "tuned.json"

    def test_the_root_of_this_checkout_is_not_a_constant_of_the_machine(self) -> None:
        assert ROOT == Path(__file__).resolve().parents[2]


class TestScrub:
    def test_a_traceback_frame_keeps_the_relative_path_and_the_line(self) -> None:
        frame = f'File "{ROOT / "closed_loop" / "eval" / "x.py"}", line 12, in f'
        assert scrub(frame) == 'File "closed_loop/eval/x.py", line 12, in f'

    def test_a_frame_outside_the_repository_is_its_base_name(self) -> None:
        frame = f'File "{HOME / "lib" / "site-packages" / "pkg" / "mod.py"}", line 3, in g'
        assert scrub(frame) == 'File "mod.py", line 3, in g'

    def test_the_repository_folder_in_a_message_is_removed_and_the_home_folder_is_a_tilde(
        self,
    ) -> None:
        msg = f"{ROOT / 'benchmarks' / 'tuned.json'} is malformed; see {HOME / 'notes.txt'}"
        out = scrub(msg)
        assert out.startswith("benchmarks") and "is malformed; see ~" in out
        assert leaks(out) == []

    @pytest.mark.parametrize("style", ["native", "posix", "doubled", "upper"])
    def test_every_way_of_writing_the_folder_is_caught(self, style: str) -> None:
        native = str(ROOT / "a.py")
        text = {
            "native": native,
            "posix": (ROOT / "a.py").as_posix(),
            "doubled": native.replace("\\", "\\\\"),
            "upper": native.upper(),
        }[style]
        want = "at A.PY now" if style == "upper" else "at a.py now"
        assert scrub(f"at {text} now") == want

    def test_text_without_a_path_is_unchanged_and_scrubbing_twice_changes_nothing(self) -> None:
        plain = "ValueError: the piece would start after the life has failed | line 4"
        assert scrub(plain) == plain
        dirty = f'RenderError: x | File "{ROOT / "q.py"}", line 1 in f {HOME}'
        assert scrub(scrub(dirty)) == scrub(dirty)


class TestTheTunedSource:
    def test_the_real_file_is_named_by_its_place_in_the_repository(self) -> None:
        assert load_tuned().source == "benchmarks/tuned.json"

    def test_two_copies_in_two_places_are_equal_values(self, tmp_path: Path) -> None:
        a, b = tmp_path / "one" / "tuned.json", tmp_path / "two" / "deeper" / "tuned.json"
        for target in (a, b):
            target.parent.mkdir(parents=True)
            shutil.copyfile(TUNED_PATH, target)
        ta, tb = load_tuned(a), load_tuned(b)
        assert ta == tb and dataclasses.asdict(ta) == dataclasses.asdict(tb)
        assert ta.source == "tuned.json" and leaks(repr(ta)) == []


class TestTheKeysDoNotHoldTheMachine:
    def key_of(self, tuned_file: Path) -> tuple[str, dict]:
        task = EpisodeTask(ARMS["B0"], scenario_for("nominal"), Fidelity.F1, load_tuned(tuned_file))
        return cell_key(task, {"code": "0" * 64})

    def test_a_checkout_at_another_path_has_the_same_cell_keys(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two checkouts, each with its own ``benchmarks/tuned.json``, give the real key."""
        from closed_loop.eval import paths

        real_key, _ = self.key_of(TUNED_PATH)
        keys = set()
        for name in ("clone-a", "deeper/clone-b"):
            root = tmp_path / name
            (root / "benchmarks").mkdir(parents=True)
            shutil.copyfile(TUNED_PATH, root / "benchmarks" / "tuned.json")
            monkeypatch.setattr(paths, "ROOT", root.resolve())
            key, header = self.key_of(root / "benchmarks" / "tuned.json")
            assert leaks(str(header)) == []
            keys.add(key)
        assert keys == {real_key}, "the absolute path of the tuned file reached the cell key"

    def test_a_tuned_file_outside_any_checkout_is_named_by_its_base_name_in_the_key(
        self, tmp_path: Path
    ) -> None:
        keys = set()
        for name in ("x", "y/z"):
            target = tmp_path / name / "tuned.json"
            target.parent.mkdir(parents=True)
            shutil.copyfile(TUNED_PATH, target)
            key, header = self.key_of(target)
            assert leaks(str(header)) == []
            keys.add(key)
        assert len(keys) == 1

    def test_a_different_tuned_value_still_changes_the_key(self, tmp_path: Path) -> None:
        env = {"code": "0" * 64}
        base = load_tuned()
        other = dataclasses.replace(base, k_b3=base.k_b3 + 1.0)
        keys = {
            cell_key(EpisodeTask(ARMS["B0"], scenario_for("nominal"), Fidelity.F1, t), env)[0]
            for t in (base, other)
        }
        assert len(keys) == 2


class TestAnEpisodeDoesNotCarryTheMachine:
    def test_two_runs_under_two_locations_of_tuned_json_give_equal_results(
        self, small_context: PreparedContext, tmp_path: Path
    ) -> None:
        from closed_loop.eval.tuned import use_tuned

        results = []
        for name in ("one", "two"):
            target = tmp_path / name / "tuned.json"
            target.parent.mkdir()
            shutil.copyfile(TUNED_PATH, target)
            with use_tuned(load_tuned(target)):
                results.append(
                    run_episode(ARMS["B0"], 1100, scenario_for("nominal"), context=small_context)
                )
        a, b = results
        assert fingerprint(a) == fingerprint(b)  # everything but the stopwatch
        assert a.tuning == b.tuning and a.tuning.startswith("tuned.json sha256:")
        assert leaks(a.tuning) == []

    def test_the_real_file_gives_a_relative_tuning_string(
        self, small_context: PreparedContext
    ) -> None:
        r = run_episode(ARMS["B0"], 1100, scenario_for("nominal"), context=small_context)
        assert r.tuning.startswith("benchmarks/tuned.json sha256:") and leaks(r.tuning) == []


class _Boom:
    def label(self) -> str:
        return "boom"

    def fingerprint(self) -> dict[str, str]:
        return {"kind": "boom"}

    def __call__(self, seed: int) -> None:
        raise ValueError(f"cannot read {ROOT / 'benchmarks' / 'tuned.json'} for {HOME / 'x'}")


class TestTheErrorText:
    def test_it_keeps_the_type_the_message_and_the_frames_without_the_machine(self) -> None:
        kind, detail = _execute(_Boom(), 1, None)
        assert kind == "error"
        assert detail.startswith("ValueError: cannot read benchmarks")
        assert "for ~" in detail and "in __call__" in detail
        assert 'File "tests/eval/test_paths.py"' in detail
        assert leaks(detail) == []


# ----------------------------------------------------------------------------- what --quick writes


QUICK_FILES = ("results.json", "RESULTS.md", "PROOF-SCOPE.md")


class TestWhatQuickWrites:
    @pytest.mark.parametrize("name", QUICK_FILES)
    def test_the_output_of_the_quick_run_holds_no_path_of_this_machine(self, name: str) -> None:
        path = QUICK / name
        if not path.exists():
            pytest.skip("no quick output under benchmarks/.cache/quick-output/")
        found = leaks(path.read_text(encoding="utf-8"))
        assert found == [], f"{name} holds {found}: run the quick run again with the fix"

    def test_the_scan_finds_a_leak_in_either_spelling_and_in_either_case(self) -> None:
        assert leaks(f'"tuning": "{ROOT / "benchmarks" / "tuned.json"}"') != []
        assert leaks("at " + (HOME / "x").as_posix().upper()) != []
        assert leaks(str(HOME).replace("\\", "\\\\") + "x") != []
        assert leaks("benchmarks/tuned.json sha256:abc") == []
