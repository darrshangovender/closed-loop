"""P10: the hash-chained ledger. Tampering is tested byte by byte."""

import json
import os
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from closed_loop.audit.ledger import ALLOWED_KINDS, GENESIS_HASH, Ledger, LedgerError

REPO = Path(__file__).resolve().parents[2]


def fixed_clock(start: int = 1_700_000_000):
    state = {"t": start}

    def clock() -> datetime:
        state["t"] += 60
        return datetime.fromtimestamp(state["t"], UTC)

    return clock


def build(path: Path, n: int = 3) -> Ledger:
    ledger = Ledger(path, clock=fixed_clock())
    ledger.append("freeze", prereg_sha256="a" * 64, code_sha256="b" * 64)
    for i in range(1, n):
        ledger.append("full", run=i, headline={"m": 0.5 * i})
    return ledger


class TestAppendAndVerify:
    def test_create_append_verify(self, tmp_path: Path) -> None:
        path = tmp_path / "ledger.jsonl"
        ledger = Ledger(path, clock=fixed_clock())
        assert ledger.head() == GENESIS_HASH and ledger.heldout_evaluations() == 0
        ledger.verify()  # a missing file is an empty ledger
        first = ledger.append("freeze", prereg_sha256="a" * 64)
        second = ledger.append("full", note="x")
        ledger.verify()
        assert [e["seq"] for e in ledger.entries()] == [0, 1]
        assert first["prev_hash"] == GENESIS_HASH and second["prev_hash"] == first["hash"]
        assert ledger.head() == second["hash"]

    def test_entry_shape_and_clock(self, tmp_path: Path) -> None:
        entry = build(tmp_path / "l.jsonl", 1).entries()[0]
        assert set(entry) == {
            "seq", "utc", "kind", "prereg_sha256", "code_sha256", "heldout_evaluations",
            "prev_hash", "hash",
        }  # fmt: skip
        assert entry["utc"] == "2023-11-14T22:14:20Z" and entry["kind"] == "freeze"

    def test_file_is_ascii_lf_canonical_one_object_per_line(self, tmp_path: Path) -> None:
        path = tmp_path / "l.jsonl"
        ledger = build(path)
        ledger.append("full", text="مرحبا café")
        raw = path.read_bytes()
        assert raw.isascii() and b"\r" not in raw and raw.endswith(b"\n")
        lines = raw.decode("ascii").splitlines()
        assert len(lines) == 4
        assert json.loads(lines[-1])["text"] == "مرحبا café"
        for line in lines:  # canonical: sorted keys, no spaces between tokens
            parsed = json.loads(line)
            assert json.dumps(parsed, sort_keys=True, separators=(",", ":")) == line

    def test_reopening_continues_the_chain(self, tmp_path: Path) -> None:
        path = tmp_path / "l.jsonl"
        build(path)
        again = Ledger(path, clock=fixed_clock(1_800_000_000))
        entry = again.append("full-failed", error="boom")
        assert entry["seq"] == 3
        again.verify()

    def test_head_changes_with_every_append(self, tmp_path: Path) -> None:
        ledger = build(tmp_path / "l.jsonl", 1)
        heads = {ledger.head()}
        for i in range(5):
            ledger.append("full", i=i)
            heads.add(ledger.head())
        assert len(heads) == 6

    def test_same_content_different_position_different_hash(self, tmp_path: Path) -> None:
        a, b = (
            Ledger(tmp_path / "a.jsonl", clock=fixed_clock()),
            Ledger(tmp_path / "b.jsonl", clock=fixed_clock()),
        )
        a.append("full", x=1)
        a.append("full", x=1)
        b.append("full", x=1)
        e = a.entries()
        assert e[0]["hash"] != e[1]["hash"] and e[0]["hash"] == b.entries()[0]["hash"]


class TestAppendRefusals:
    def test_unknown_kind(self, tmp_path: Path) -> None:
        with pytest.raises(LedgerError, match="kind must be one of"):
            Ledger(tmp_path / "l.jsonl").append("frozen")
        assert set(ALLOWED_KINDS) == {"freeze", "full", "full-failed", "heldout-use"}

    @pytest.mark.parametrize("name", ["seq", "utc", "heldout_evaluations", "prev_hash", "hash"])
    def test_reserved_field_names(self, tmp_path: Path, name: str) -> None:
        with pytest.raises(LedgerError, match="reserved"):
            Ledger(tmp_path / "l.jsonl").append("full", **{name: 1})

    def test_values_that_are_not_json(self, tmp_path: Path) -> None:
        ledger = Ledger(tmp_path / "l.jsonl")
        with pytest.raises(LedgerError, match="cannot be written as JSON"):
            ledger.append("full", x=float("nan"))
        with pytest.raises(LedgerError, match="cannot be written as JSON"):
            ledger.append("full", x=object())
        assert not (tmp_path / "l.jsonl").exists()  # nothing half-written

    def test_a_heldout_use_needs_an_integer_n(self, tmp_path: Path) -> None:
        ledger = Ledger(tmp_path / "l.jsonl")
        for bad in ({}, {"n": 1.5}, {"n": True}, {"n": "3"}):
            with pytest.raises(LedgerError, match="integer n"):
                ledger.append("heldout-use", **bad)

    def test_appending_to_a_broken_ledger_is_refused_and_leaves_it_alone(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "l.jsonl"
        build(path)
        raw = bytearray(path.read_bytes())
        raw[40] ^= 1
        path.write_bytes(bytes(raw))
        with pytest.raises(LedgerError):
            Ledger(path).append("full")
        assert path.read_bytes() == bytes(raw)


class TestTamperDetection:
    def test_flipping_any_bit_of_any_byte_of_any_line_fails_verify(self, tmp_path: Path) -> None:
        path = tmp_path / "l.jsonl"
        build(path, 2)
        original = path.read_bytes()
        survivors = []
        for offset in range(len(original)):
            for bit in (0x01, 0x20):
                raw = bytearray(original)
                raw[offset] ^= bit
                path.write_bytes(bytes(raw))
                try:
                    Ledger(path).verify()
                except LedgerError:
                    continue
                survivors.append((offset, bit))
        path.write_bytes(original)
        assert survivors == []
        Ledger(path).verify()

    def test_inserting_or_removing_any_single_byte_fails_verify(self, tmp_path: Path) -> None:
        path = tmp_path / "l.jsonl"
        build(path, 2)
        original = path.read_bytes()
        survivors = []
        for offset in range(len(original)):
            for variant in (original[:offset] + original[offset + 1 :],
                            original[:offset] + b" " + original[offset:]):  # fmt: skip
                path.write_bytes(variant)
                try:
                    Ledger(path).verify()
                except LedgerError:
                    continue
                survivors.append(offset)
        assert survivors == []

    def test_deleting_the_last_line_is_detected_only_when_the_head_is_supplied(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "l.jsonl"
        ledger = build(path)
        head = ledger.head()
        lines = path.read_bytes().splitlines(keepends=True)
        path.write_bytes(b"".join(lines[:-1]))
        Ledger(path).verify()  # the shortened chain is internally consistent: the known limit
        with pytest.raises(LedgerError, match="head is .* but .* was expected"):
            Ledger(path).verify(head=head)

    def test_deleting_everything_is_caught_with_the_head(self, tmp_path: Path) -> None:
        path = tmp_path / "l.jsonl"
        head = build(path).head()
        path.write_bytes(b"")
        with pytest.raises(LedgerError, match="expected"):
            Ledger(path).verify(head=head)

    def test_the_right_head_passes(self, tmp_path: Path) -> None:
        ledger = build(tmp_path / "l.jsonl")
        ledger.verify(head=ledger.head())

    def test_deleting_a_middle_line_swapping_and_duplicating_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "l.jsonl"
        build(path, 4)
        lines = path.read_bytes().splitlines(keepends=True)
        for variant in (
            [lines[0], lines[2], lines[3]],
            [lines[0], lines[2], lines[1], lines[3]],
            [lines[0], lines[1], lines[1], lines[2]],
            [lines[1], lines[2], lines[3]],
        ):
            path.write_bytes(b"".join(variant))
            with pytest.raises(LedgerError):
                Ledger(path).verify()

    def test_a_blank_line_a_crlf_conversion_and_a_missing_final_newline(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "l.jsonl"
        build(path)
        good = path.read_bytes()
        for bad in (good + b"\n", good.replace(b"\n", b"\r\n"), good[:-1], b"\n" + good):
            path.write_bytes(bad)
            with pytest.raises(LedgerError):
                Ledger(path).verify()

    def test_a_consistently_rewritten_ledger_passes_verify(self, tmp_path: Path) -> None:
        """The stated limit: someone who rebuilds the whole chain cannot be seen from inside."""
        original = build(tmp_path / "a.jsonl")
        forged = Ledger(tmp_path / "b.jsonl", clock=fixed_clock())
        forged.append("freeze", prereg_sha256="c" * 64, code_sha256="b" * 64)
        forged.verify()
        assert forged.head() != original.head()  # which is why the head is published elsewhere

    def test_a_non_utf8_or_non_json_line(self, tmp_path: Path) -> None:
        path = tmp_path / "l.jsonl"
        for junk in (b"\xff\xfe\n", b"not json\n", b"[1,2]\n", b'{"a":1}\n'):
            path.write_bytes(junk)
            with pytest.raises(LedgerError):
                Ledger(path).verify()

    def test_an_unknown_kind_written_by_hand_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "l.jsonl"
        ledger = build(path, 1)
        entry = ledger.entries()[0]
        entry["kind"] = "frozen"
        # recompute the hash so only the kind check can object
        from closed_loop.audit.ledger import _canonical, _hash_of

        entry["hash"] = _hash_of(entry)
        path.write_bytes(_canonical(entry).encode() + b"\n")
        with pytest.raises(LedgerError, match="unknown kind"):
            Ledger(path).verify()


def _forge(path: Path, entries: list[dict]) -> None:
    """Write entries whose own hashes are valid, whatever their links say (a careful forger)."""
    from closed_loop.audit.ledger import _canonical, _hash_of

    lines = []
    for entry in entries:
        entry = dict(entry)
        entry["hash"] = _hash_of(entry)
        lines.append(_canonical(entry))
    path.write_bytes(("\n".join(lines) + "\n").encode("ascii"))


class TestChainLinks:
    """Each link check on its own, against entries whose individual hashes are valid.

    The byte-flip tests cannot tell these checks apart, because any one of them catches a
    spliced file; the pass over the code found that deleting any single one of them survived.
    """

    def _entries(self, tmp_path: Path) -> list[dict]:
        return build(tmp_path / "src.jsonl", 3).entries()

    def test_a_wrong_prev_hash_with_a_valid_own_hash_and_the_right_seq(
        self, tmp_path: Path
    ) -> None:
        a, b, c = self._entries(tmp_path)
        b = {**b, "prev_hash": GENESIS_HASH}  # points nowhere
        path = tmp_path / "forged.jsonl"
        _forge(path, [a, b])
        with pytest.raises(LedgerError, match="prev_hash"):
            Ledger(path).verify()

    def test_a_middle_line_removed_and_the_survivors_renumbered_honestly(
        self, tmp_path: Path
    ) -> None:
        a, b, c = self._entries(tmp_path)
        c = {**c, "seq": 1}  # renumbered, valid hash, but prev_hash still names the removed b
        path = tmp_path / "forged.jsonl"
        _forge(path, [a, c])
        with pytest.raises(LedgerError, match="prev_hash"):
            Ledger(path).verify()

    def test_wrong_seq_with_perfectly_linked_hashes(self, tmp_path: Path) -> None:
        from closed_loop.audit.ledger import _hash_of

        a, b, c = self._entries(tmp_path)
        b = {**b, "seq": 7}
        b["hash"] = _hash_of(b)
        c = {**c, "prev_hash": b["hash"]}
        path = tmp_path / "forged.jsonl"
        _forge(path, [a, b, c])  # every link is consistent; only the numbering is wrong
        with pytest.raises(LedgerError, match="seq is 7, expected 1"):
            Ledger(path).verify()

    def test_the_hash_is_the_sha256_of_the_canonical_entry_including_prev_hash(
        self, tmp_path: Path
    ) -> None:
        import hashlib

        path = tmp_path / "l.jsonl"
        build(path, 3)
        previous = GENESIS_HASH
        for line in path.read_text(encoding="ascii").splitlines():
            entry = json.loads(line)
            body = {k: v for k, v in entry.items() if k != "hash"}
            text = json.dumps(body, sort_keys=True, separators=(",", ":"))
            assert entry["hash"] == hashlib.sha256(text.encode("ascii")).hexdigest()
            assert body["prev_hash"] == previous
            previous = entry["hash"]

    def test_changing_only_prev_hash_changes_the_entry_hash(self) -> None:
        from closed_loop.audit.ledger import _hash_of

        entry = {"seq": 1, "kind": "full", "prev_hash": "a" * 64}
        assert _hash_of(entry) != _hash_of({**entry, "prev_hash": "b" * 64})


class TestHeldoutCounter:
    def test_counter_adds_n_for_every_heldout_use_and_nothing_else(self, tmp_path: Path) -> None:
        ledger = build(tmp_path / "l.jsonl", 2)
        assert ledger.heldout_evaluations() == 0
        ledger.append("heldout-use", n=10)
        ledger.append("full", x=1)
        ledger.append("heldout-use", n=5)
        assert ledger.heldout_evaluations() == 15
        counts = [e["heldout_evaluations"] for e in ledger.entries()]
        assert counts == [0, 0, 10, 10, 15]

    def test_the_counter_is_inside_the_hash(self, tmp_path: Path) -> None:
        path = tmp_path / "l.jsonl"
        ledger = build(path, 1)
        ledger.append("heldout-use", n=10)
        text = path.read_text(encoding="ascii").replace(
            '"heldout_evaluations":10', '"heldout_evaluations":0'
        )
        path.write_text(text, encoding="ascii", newline="\n")
        with pytest.raises(LedgerError):
            Ledger(path).verify()


class TestLocking:
    def test_two_threads_appending_do_not_lose_or_duplicate_entries(self, tmp_path: Path) -> None:
        path = tmp_path / "l.jsonl"
        errors: list[BaseException] = []

        def worker(tag: str) -> None:
            try:
                ledger = Ledger(path)
                for i in range(25):
                    ledger.append("full", tag=tag, i=i)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in ("a", "b")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        entries = Ledger(path).entries()
        assert [e["seq"] for e in entries] == list(range(50))
        assert sorted(e["i"] for e in entries if e["tag"] == "a") == list(range(25))

    def test_two_processes_appending(self, tmp_path: Path) -> None:
        path = tmp_path / "l.jsonl"
        code = (
            "import sys\n"
            "from pathlib import Path\n"
            "from closed_loop.audit.ledger import Ledger\n"
            "ledger = Ledger(Path(sys.argv[1]))\n"
            "for i in range(15):\n"
            "    ledger.append('full', tag=sys.argv[2], i=i)\n"
        )
        procs = [
            subprocess.Popen(
                [sys.executable, "-c", code, str(path), tag],
                cwd=REPO,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )  # fmt: skip
            for tag in ("p", "q")
        ]
        for p in procs:
            out, err = p.communicate(timeout=180)
            assert p.returncode == 0, err.decode()
        assert [e["seq"] for e in Ledger(path).entries()] == list(range(30))

    def test_a_fresh_lock_blocks_and_times_out(self, tmp_path: Path) -> None:
        path = tmp_path / "l.jsonl"
        lock = tmp_path / "l.jsonl.lock"
        lock.write_text("pid=1\n")
        ledger = Ledger(path, lock_timeout_s=0.2, stale_after_s=3600)
        with pytest.raises(LedgerError, match="locked"):
            ledger.append("full")
        assert lock.exists() and not path.exists()

    def test_a_stale_lock_is_cleared(self, tmp_path: Path) -> None:
        path = tmp_path / "l.jsonl"
        lock = tmp_path / "l.jsonl.lock"
        lock.write_text("pid=1\n")
        old = time.time() - 7200
        os.utime(lock, (old, old))
        Ledger(path, stale_after_s=60).append("full")
        assert not lock.exists()
        Ledger(path).verify()

    def test_the_lock_is_released_after_a_refused_append(self, tmp_path: Path) -> None:
        ledger = Ledger(tmp_path / "l.jsonl")
        with pytest.raises(LedgerError):
            ledger.append("full", x=float("nan"))
        assert not (tmp_path / "l.jsonl.lock").exists()
        ledger.append("full", x=1)
