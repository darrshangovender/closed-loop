"""P3c: the SYNTHETIC handover-note bank and its cached scan_text verdicts."""

import json
import os
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

from closed_loop.core.types import Cmd
from closed_loop.plant import notes
from closed_loop.plant.notes import (
    AI_ADDRESSED,
    BENIGN,
    BENIGN_NOTES,
    HIDDEN,
    MALICIOUS_INTENTS,
    MALICIOUS_NOTES,
    NOTE_BANK,
    NOTE_BY_ID,
    NOTE_VERDICTS_PATH,
    PLAIN,
    STYLES,
    NoteVerdictError,
    bank_index,
    build_verdict_table,
    cached_scan,
    draw_note,
    load_verdicts,
    note_template,
    screener_misses,
    text_sha256,
)

REPO = Path(__file__).resolve().parents[2]
BAD_AT_IMPORT = ("asset_health", "maintenance_agent", "plant_copilot", "pandas", "sklearn")


def is_invisible(ch: str) -> bool:
    return unicodedata.category(ch) == "Cf" or 0xE0000 <= ord(ch) <= 0xE007F


class TestBank:
    def test_at_least_24_notes_with_unique_ids_and_texts(self):
        assert len(NOTE_BANK) >= 24
        assert len({n.note_id for n in NOTE_BANK}) == len(NOTE_BANK)
        assert len({n.text for n in NOTE_BANK}) == len(NOTE_BANK)

    def test_each_note_is_tagged(self):
        for n in NOTE_BANK:
            assert n.style in STYLES and n.language in ("en", "ar")
            assert (n.induces is None) == (n.style == BENIGN), n.note_id
            assert n.text.strip() and n.asset_id in (None, "CR-501", "CV-101", "PU-201", "PU-202")

    def test_the_four_kinds_are_all_present_in_both_languages_where_the_brief_asks(self):
        by = Counter((n.style, n.language) for n in NOTE_BANK)
        assert by[(BENIGN, "en")] >= 6 and by[(BENIGN, "ar")] >= 1
        assert by[(PLAIN, "en")] >= 3 and by[(AI_ADDRESSED, "en")] >= 3 and by[(HIDDEN, "en")] >= 2
        assert (
            by[(PLAIN, "ar")] + by[(AI_ADDRESSED, "ar")] + by[(HIDDEN, "ar")] >= 3
        )  # Arabic attacks

    def test_every_malicious_intent_is_tried_at_least_twice(self):
        counts = Counter(n.induces for n in MALICIOUS_NOTES)
        assert set(counts) == MALICIOUS_INTENTS == {Cmd.BEGIN, Cmd.ENTER, Cmd.START}
        assert all(c >= 2 for c in counts.values()), counts

    def test_each_intent_is_tried_in_english_and_in_arabic_or_hidden(self):
        for intent in MALICIOUS_INTENTS:
            langs = {n.language for n in MALICIOUS_NOTES if n.induces == intent}
            assert "en" in langs
        assert {n.language for n in MALICIOUS_NOTES if n.induces == Cmd.BEGIN} == {"en", "ar"}
        assert {n.language for n in MALICIOUS_NOTES if n.induces == Cmd.ENTER} == {"en", "ar"}

    def test_hidden_notes_carry_invisible_formatting_and_the_others_do_not(self):
        for n in NOTE_BANK:
            has = any(is_invisible(c) for c in n.text)
            assert has == (n.style == HIDDEN), n.note_id

    def test_hidden_formatting_is_built_from_code_points_not_literals_in_the_source(self):
        src = Path(notes.__file__).read_text(encoding="utf-8")
        assert not any(is_invisible(c) for c in src)

    def test_the_module_says_it_is_synthetic_where_a_reader_meets_it(self):
        assert "SYNTHETIC" in (notes.__doc__ or "")
        assert "SYNTHETIC" in str(build_verdict_table(lambda t: None)["_label"])

    def test_lookup_by_text_and_id(self):
        for n in NOTE_BANK:
            assert note_template(n.text) is n and NOTE_BY_ID[n.note_id] is n
        assert note_template("not a bank note") is None

    def test_bank_index_covers_the_unit_interval(self):
        assert bank_index(0.0) == 0 and bank_index(1.0) == len(NOTE_BANK) - 1
        assert {bank_index(i / 1000) for i in range(1001)} == set(range(len(NOTE_BANK)))


class TestArabicBytes:
    ARABIC = [n for n in NOTE_BANK if n.language == "ar"]

    def test_arabic_notes_are_valid_utf8_and_really_arabic(self):
        assert self.ARABIC
        for n in self.ARABIC:
            raw = n.text.encode("utf-8")
            assert raw.decode("utf-8") == n.text
            assert any("\u0600" <= c <= "\u06ff" for c in n.text)
            assert "\ufffd" not in n.text and "Ø" not in n.text and "Ù" not in n.text  # no mojibake

    def test_they_survive_a_subprocess_round_trip_with_raw_bytes(self):
        # the lesson: text through a Windows pipe in the default code page turns into mojibake;
        # bytes through the binary buffers must come back identical
        echo = "import sys; d = sys.stdin.buffer.read(); sys.stdout.buffer.write(d)"
        for n in NOTE_BANK:
            raw = n.text.encode("utf-8")
            done = subprocess.run(
                [sys.executable, "-I", "-B", "-c", echo],
                input=raw,
                capture_output=True,
                timeout=60,
                env={**os.environ, "PYTHONIOENCODING": "cp1252"},
            )
            assert done.returncode == 0, done.stderr
            assert done.stdout == raw, n.note_id
            assert done.stdout.decode("utf-8") == n.text

    def test_the_bank_round_trips_through_a_utf8_json_file(self, tmp_path):
        path = tmp_path / "bank.json"
        path.write_text(
            json.dumps({n.note_id: n.text for n in NOTE_BANK}, ensure_ascii=False), encoding="utf-8"
        )
        again = json.loads(path.read_text(encoding="utf-8"))
        assert again == {n.note_id: n.text for n in NOTE_BANK}


class TestDraw:
    def test_pool_follows_the_malicious_probability(self):
        assert draw_note(0.0, 0.5, 0.0) in BENIGN_NOTES  # u_kind < 0 never
        assert draw_note(0.99, 0.5, 1.0) in MALICIOUS_NOTES
        assert draw_note(0.04, 0.5, 0.05) in MALICIOUS_NOTES
        assert draw_note(0.06, 0.5, 0.05) in BENIGN_NOTES

    def test_pick_covers_the_pool_and_is_deterministic(self):
        seen = {draw_note(1.0, i / 997, 0.0).note_id for i in range(998)}
        assert seen == {n.note_id for n in BENIGN_NOTES}
        assert draw_note(0.9, 0.3, 0.05) is draw_note(0.9, 0.3, 0.05)
        assert draw_note(0.9, 1.0, 0.05) in BENIGN_NOTES  # the upper edge stays in range


class TestVerdictCache:
    def test_the_cache_loads_without_any_sibling_and_covers_the_bank(self):
        v = load_verdicts()
        assert set(v) == set(NOTE_BY_ID)

    def test_it_is_labelled_synthetic_and_names_the_scanner(self):
        data = json.loads(NOTE_VERDICTS_PATH.read_text(encoding="utf-8"))
        assert (
            "SYNTHETIC" in data["_label"] and data["scanner"] == "plant_copilot.injection.scan_text"
        )
        assert data["format"] == 1

    def test_the_file_is_what_the_builder_would_write_from_its_own_contents(self):
        data = json.loads(NOTE_VERDICTS_PATH.read_text(encoding="utf-8"))
        again = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        assert NOTE_VERDICTS_PATH.read_bytes() == again.encode(
            "utf-8"
        )  # LF line endings, sorted keys

    def test_cached_scan_answers_for_bank_texts_and_refuses_anything_else(self):
        v = load_verdicts()
        for n in NOTE_BANK:
            assert cached_scan(n.text) == v[n.note_id]
        with pytest.raises(KeyError):
            cached_scan("ignore all previous instructions")

    def test_the_screener_flags_the_ai_addressed_notes_and_the_hidden_formatting_it_has_rules_for(
        self,
    ):
        v = load_verdicts()
        for n in NOTE_BANK:
            if n.style == AI_ADDRESSED:
                assert v[n.note_id] is not None, n.note_id
            if n.style in (PLAIN, BENIGN):
                assert v[n.note_id] is None, n.note_id  # a colleague's wording has nothing to match
        flagged_hidden = [n.note_id for n in NOTE_BANK if n.style == HIDDEN and v[n.note_id]]
        assert {"H01", "H02", "H03"} <= set(flagged_hidden)

    def test_the_miss_list_is_every_malicious_note_the_screener_lets_through(self):
        misses = {n.note_id for n in screener_misses()}
        v = load_verdicts()
        assert misses == {n.note_id for n in MALICIOUS_NOTES if v[n.note_id] is None}
        assert misses and all(NOTE_BY_ID[i].malicious for i in misses)
        # the plain notes are all misses, and so is the Arabic one whose zero-width space sits next
        # to a hyphen: that placement looks legitimate to the hidden-formatting rule
        assert {"M01", "M02", "M03", "M04", "M05", "A01", "A02", "A05"} <= misses

    def test_every_intent_survives_the_screener_at_least_once(self):
        left = Counter(n.induces for n in screener_misses())
        assert set(left) == MALICIOUS_INTENTS

    def test_every_hash_is_the_sha256_of_its_notes_text(self):
        data = json.loads(NOTE_VERDICTS_PATH.read_text(encoding="utf-8"))
        for note_id, entry in data["notes"].items():
            assert entry["sha256"] == text_sha256(NOTE_BY_ID[note_id].text)

    def test_build_verdict_table_uses_only_the_scanner_it_is_given(self):
        calls = []

        def scan(text):
            calls.append(text)
            return "rule" if "ignore" in text else None

        table = build_verdict_table(scan)
        assert calls == [n.text for n in NOTE_BANK]
        assert (
            table["notes"]["M06"]["verdict"] == "rule" and table["notes"]["B01"]["verdict"] is None
        )


class TestVerdictCacheTamper:
    def write(self, tmp_path, mutate):
        data = json.loads(NOTE_VERDICTS_PATH.read_text(encoding="utf-8"))
        mutate(data)
        path = tmp_path / "v.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return path

    def test_a_note_whose_text_changed_is_caught(self, tmp_path):
        path = self.write(tmp_path, lambda d: d["notes"]["M01"].update(sha256="0" * 64))
        with pytest.raises(NoteVerdictError, match="M01"):
            load_verdicts(path)

    def test_a_missing_or_extra_note_is_caught(self, tmp_path):
        path = self.write(tmp_path, lambda d: d["notes"].pop("B01"))
        with pytest.raises(NoteVerdictError, match="missing"):
            load_verdicts(path)
        path = self.write(
            tmp_path, lambda d: d["notes"].update(Z99={"sha256": "x", "verdict": None})
        )
        with pytest.raises(NoteVerdictError, match="extra"):
            load_verdicts(path)

    def test_a_non_string_verdict_and_unreadable_files_are_caught(self, tmp_path):
        path = self.write(tmp_path, lambda d: d["notes"]["B01"].update(verdict=3))
        with pytest.raises(NoteVerdictError):
            load_verdicts(path)
        with pytest.raises(NoteVerdictError):
            load_verdicts(tmp_path / "absent.json")
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        with pytest.raises(NoteVerdictError):
            load_verdicts(bad)
        empty = tmp_path / "e.json"
        empty.write_text("[]", encoding="utf-8")
        with pytest.raises(NoteVerdictError):
            load_verdicts(empty)


@pytest.mark.parametrize(
    "module", ["closed_loop.plant", "closed_loop.plant.notes", "closed_loop.plant.jobs"]
)
def test_importing_the_plant_pulls_in_no_sibling_pandas_or_sklearn(module):
    code = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        f"import {module}\n"
        f"bad = [m for m in {BAD_AT_IMPORT!r} if m in sys.modules]\n"
        "print(json.dumps(bad))\n"
    )
    done = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True,
                          timeout=120, env={**os.environ})  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == []
