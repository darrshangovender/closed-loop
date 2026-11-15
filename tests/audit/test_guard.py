"""P10: the held-out guard, the commit-charge gate and the heavy-run lock."""

import hashlib
import os
import sys
import time
from pathlib import Path

import pytest

from closed_loop.audit.guard import (
    COMMIT_LIMIT_PCT,
    GuardError,
    HeavyRunLock,
    HeldoutGuard,
    commit_charge_percent,
    require_headroom,
)
from closed_loop.audit.ledger import Ledger
from closed_loop.audit.seeds import f2_seed, heldout_seed, search_seed

SHA = hashlib.sha256(b"frozen prereg").hexdigest()
OTHER = hashlib.sha256(b"edited prereg").hexdigest()


def frozen_ledger(path: Path, sha: str = SHA) -> Ledger:
    ledger = Ledger(path)
    ledger.append("freeze", prereg_sha256=sha, code_sha256="b" * 64)
    return ledger


class TestHeldoutGuard:
    def test_raises_before_a_freeze_entry_exists_and_records_nothing(self, tmp_path: Path) -> None:
        ledger = Ledger(tmp_path / "l.jsonl")
        guard = HeldoutGuard(ledger, SHA)
        with pytest.raises(GuardError, match="no freeze entry"):
            guard.seeds(5)
        assert ledger.entries() == [] and ledger.heldout_evaluations() == 0

    def test_works_after_the_freeze_and_returns_the_derived_seeds(self, tmp_path: Path) -> None:
        guard = HeldoutGuard(frozen_ledger(tmp_path / "l.jsonl"), SHA)
        assert guard.seeds(4) == [heldout_seed(SHA, i) for i in range(4)]
        assert guard.seeds(3, start=10) == [heldout_seed(SHA, i) for i in (10, 11, 12)]

    def test_every_use_increments_the_counter_by_n_including_repeats(self, tmp_path: Path) -> None:
        ledger = frozen_ledger(tmp_path / "l.jsonl")
        guard = HeldoutGuard(ledger, SHA)
        assert ledger.heldout_evaluations() == 0
        guard.seeds(10)
        assert ledger.heldout_evaluations() == 10
        guard.seeds(10)  # the same ten seeds again: overuse must show
        assert ledger.heldout_evaluations() == 20
        guard.seeds(1, start=500)
        assert ledger.heldout_evaluations() == 21
        uses = [e for e in ledger.entries() if e["kind"] == "heldout-use"]
        assert [(u["n"], u["start"]) for u in uses] == [(10, 0), (10, 0), (1, 500)]
        assert all(u["prereg_sha256"] == SHA for u in uses)

    def test_search_seeds_are_the_derived_search_seeds_and_are_counted_like_the_rest(
        self, tmp_path: Path
    ) -> None:
        ledger = frozen_ledger(tmp_path / "l.jsonl")
        guard = HeldoutGuard(ledger, SHA)
        assert guard.search_seeds(10) == [search_seed(SHA, i) for i in range(10)]
        assert ledger.heldout_evaluations() == 10
        guard.seeds(5)
        guard.search_seeds(10)  # asking again counts again
        assert ledger.heldout_evaluations() == 25
        uses = [e for e in ledger.entries() if e["kind"] == "heldout-use"]
        assert [(u["n"], u.get("role")) for u in uses] == [
            (10, "search"),
            (5, None),
            (10, "search"),
        ]

    def test_f2_seeds_are_the_derived_f2_seeds_and_are_counted_like_the_rest(
        self, tmp_path: Path
    ) -> None:
        # 0003 A16: the same gate and the same count as search_seeds, with role "f2"
        ledger = frozen_ledger(tmp_path / "l.jsonl")
        guard = HeldoutGuard(ledger, SHA)
        assert guard.f2_seeds(60) == [f2_seed(SHA, i) for i in range(60)]
        assert ledger.heldout_evaluations() == 60
        assert guard.f2_seeds(5, start=60) == [f2_seed(SHA, i) for i in range(60, 65)]
        guard.f2_seeds(60)  # asking again counts again
        assert ledger.heldout_evaluations() == 125
        guard.search_seeds(10)
        uses = [e for e in ledger.entries() if e["kind"] == "heldout-use"]
        assert [(u["n"], u["start"], u.get("role")) for u in uses] == [
            (60, 0, "f2"),
            (5, 60, "f2"),
            (60, 0, "f2"),
            (10, 0, "search"),
        ]
        assert all(u["prereg_sha256"] == SHA for u in uses)
        assert not set(guard.f2_seeds(20)) & set(guard.seeds(20))  # unrelated to the F1 seeds

    def test_f2_seeds_need_the_freeze_and_record_nothing_before_it(self, tmp_path: Path) -> None:
        ledger = Ledger(tmp_path / "l.jsonl")
        with pytest.raises(GuardError, match="no freeze entry"):
            HeldoutGuard(ledger, SHA).f2_seeds(60)
        assert ledger.entries() == [] and ledger.heldout_evaluations() == 0
        ledger = frozen_ledger(tmp_path / "other.jsonl", sha=OTHER)
        with pytest.raises(GuardError, match="no freeze entry"):  # a freeze for another document
            HeldoutGuard(ledger, SHA).f2_seeds(60)
        assert ledger.heldout_evaluations() == 0

    def test_f2_seeds_check_their_arguments(self, tmp_path: Path) -> None:
        ledger = frozen_ledger(tmp_path / "l.jsonl")
        guard = HeldoutGuard(ledger, SHA)
        with pytest.raises(ValueError, match="n must be"):
            guard.f2_seeds(0)
        with pytest.raises(ValueError, match="start must be"):
            guard.f2_seeds(1, start=-1)
        assert ledger.heldout_evaluations() == 0

    def test_search_seeds_need_the_freeze_and_record_nothing_before_it(
        self, tmp_path: Path
    ) -> None:
        ledger = Ledger(tmp_path / "l.jsonl")
        with pytest.raises(GuardError, match="no freeze entry"):
            HeldoutGuard(ledger, SHA).search_seeds(10)
        assert ledger.entries() == [] and ledger.heldout_evaluations() == 0
        ledger = frozen_ledger(tmp_path / "other.jsonl", sha=OTHER)
        with pytest.raises(GuardError, match="no freeze entry"):  # a freeze for another document
            HeldoutGuard(ledger, SHA).search_seeds(10)

    def test_search_seeds_check_their_arguments(self, tmp_path: Path) -> None:
        guard = HeldoutGuard(frozen_ledger(tmp_path / "l.jsonl"), SHA)
        with pytest.raises(ValueError, match="n must be"):
            guard.search_seeds(0)
        with pytest.raises(ValueError, match="start must be"):
            guard.search_seeds(1, start=-1)

    def test_the_use_is_on_record_before_the_seeds_are_returned(self, tmp_path: Path) -> None:
        ledger = frozen_ledger(tmp_path / "l.jsonl")
        guard = HeldoutGuard(ledger, SHA)
        seen = []
        original = ledger.append

        def spy(kind: str, **fields):
            entry = original(kind, **fields)
            seen.append(kind)
            return entry

        ledger.append = spy  # type: ignore[method-assign]
        guard.seeds(2)
        assert seen == ["heldout-use"]

    def test_a_freeze_for_another_document_does_not_open_the_guard(self, tmp_path: Path) -> None:
        """The prereg was edited after the freeze: the new hash has no seeds until it is frozen."""
        ledger = frozen_ledger(tmp_path / "l.jsonl", SHA)
        with pytest.raises(GuardError, match="no freeze entry"):
            HeldoutGuard(ledger, OTHER).seeds(1)
        assert ledger.heldout_evaluations() == 0

    def test_a_freeze_entry_is_what_counts_not_any_entry_with_the_hash(
        self, tmp_path: Path
    ) -> None:
        ledger = Ledger(tmp_path / "l.jsonl")
        ledger.append("full", prereg_sha256=SHA)
        ledger.append("full-failed", prereg_sha256=SHA)
        with pytest.raises(GuardError, match="no freeze entry"):
            HeldoutGuard(ledger, SHA).seeds(1)

    def test_a_broken_ledger_closes_the_guard(self, tmp_path: Path) -> None:
        path = tmp_path / "l.jsonl"
        ledger = frozen_ledger(path)
        raw = bytearray(path.read_bytes())
        raw[30] ^= 1
        path.write_bytes(bytes(raw))
        with pytest.raises(GuardError, match="does not verify"):
            HeldoutGuard(ledger, SHA).seeds(1)

    def test_is_frozen(self, tmp_path: Path) -> None:
        ledger = Ledger(tmp_path / "l.jsonl")
        guard = HeldoutGuard(ledger, SHA)
        assert not guard.is_frozen()
        ledger.append("freeze", prereg_sha256=SHA)
        assert guard.is_frozen() and not HeldoutGuard(ledger, OTHER).is_frozen()

    @pytest.mark.parametrize("n, start", [(0, 0), (-1, 0), (1, -1), (1.5, 0), (True, 0), (1, True)])
    def test_bad_arguments(self, tmp_path: Path, n, start) -> None:
        guard = HeldoutGuard(frozen_ledger(tmp_path / "l.jsonl"), SHA)
        with pytest.raises(ValueError):
            guard.seeds(n, start)
        assert guard.ledger.heldout_evaluations() == 0


class TestCommitCharge:
    def test_refuses_above_88_percent(self) -> None:
        with pytest.raises(GuardError, match="89.0%.*88%"):
            require_headroom(lambda: 89.0)

    def test_refuses_at_exactly_the_limit(self) -> None:
        with pytest.raises(GuardError):
            require_headroom(lambda: 88.0)

    def test_passes_below_and_returns_the_reading(self) -> None:
        assert require_headroom(lambda: 87.9) == 87.9
        assert require_headroom(lambda: 0.0) == 0.0

    def test_the_limit_is_88_by_default_and_can_be_set(self) -> None:
        assert COMMIT_LIMIT_PCT == 88.0
        assert require_headroom(lambda: 91.0, limit=95.0) == 91.0
        with pytest.raises(GuardError):
            require_headroom(lambda: 60.0, limit=50.0)

    def test_an_unreadable_charge_passes_unless_strict(self) -> None:
        assert require_headroom(lambda: None) is None
        with pytest.raises(GuardError, match="cannot be read"):
            require_headroom(lambda: None, strict=True)

    def test_the_real_reader_returns_a_percentage_or_none(self) -> None:
        value = commit_charge_percent()
        if sys.platform == "win32":
            assert value is not None and 0.0 < value < 100.0
        else:
            assert value is None


class TestHeavyRunLock:
    def test_creates_and_removes_its_own_lock(self, tmp_path: Path) -> None:
        path = tmp_path / "heavy-run.lock"
        with HeavyRunLock(path, "mnesarchus", "test") as lock:
            text = path.read_text(encoding="utf-8")
            assert text.startswith("owner=mnesarchus task=test start=") and lock.stale_note is None
        assert not path.exists()

    def test_removes_its_lock_when_the_run_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "heavy-run.lock"
        with pytest.raises(RuntimeError), HeavyRunLock(path, "o", "t"):
            raise RuntimeError("boom")
        assert not path.exists()

    def test_refuses_while_a_recent_lock_exists_and_leaves_it(self, tmp_path: Path) -> None:
        path = tmp_path / "heavy-run.lock"
        path.write_text("owner=someone-else\n")
        with pytest.raises(GuardError, match="is held"), HeavyRunLock(path, "o", "t"):
            pass
        assert path.read_text() == "owner=someone-else\n"

    def test_reports_an_old_lock_as_stale_and_leaves_it(self, tmp_path: Path) -> None:
        path = tmp_path / "heavy-run.lock"
        path.write_text("owner=someone-else\n")
        old = time.time() - 3 * 3600
        os.utime(path, (old, old))
        lock = HeavyRunLock(path, "o", "t")
        with pytest.raises(GuardError, match="stale"), lock:
            pass
        assert path.exists() and lock.stale_note is not None and "3.0 h" in lock.stale_note

    def test_the_two_hour_boundary(self, tmp_path: Path) -> None:
        path = tmp_path / "heavy-run.lock"
        path.write_text("x")
        now = time.time()
        os.utime(path, (now - 7000, now - 7000))
        with pytest.raises(GuardError, match="is held"), HeavyRunLock(path, "o", "t"):
            pass
        os.utime(path, (now - 7300, now - 7300))
        with pytest.raises(GuardError, match="stale"), HeavyRunLock(path, "o", "t"):
            pass

    def test_a_lock_we_did_not_create_is_never_deleted(self, tmp_path: Path) -> None:
        path = tmp_path / "heavy-run.lock"
        path.write_text("x")
        lock = HeavyRunLock(path, "o", "t")
        with pytest.raises(GuardError):
            lock.__enter__()
        lock.__exit__(None, None, None)
        assert path.exists()
