"""P3c / 0003 A2: the cached scan_text verdicts equal a recompute through the real plant_copilot.

Marked ``f2``: skipped, with the missing module names, when a sibling is not installed. Everything
in the F1 runs reads the cache, so this is the test that keeps the cache honest.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from closed_loop.plant.notes import (
    NOTE_BANK,
    NOTE_VERDICTS_PATH,
    build_verdict_table,
    load_verdicts,
    screener_misses,
)

pytestmark = pytest.mark.f2

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def real_scan():
    from plant_copilot.injection import scan_text

    return scan_text


def test_the_cache_file_equals_a_recompute_through_the_real_scan_text(real_scan):
    on_disk = json.loads(NOTE_VERDICTS_PATH.read_text(encoding="utf-8"))
    assert on_disk == build_verdict_table(real_scan)


def test_every_cached_verdict_is_the_real_one_note_by_note(real_scan):
    cached = load_verdicts()
    for n in NOTE_BANK:
        assert cached[n.note_id] == real_scan(n.text), n.note_id


def test_the_builder_script_agrees_with_the_file_on_disk():
    done = subprocess.run(
        [sys.executable, "-I", "-B", str(REPO / "scripts" / "build_note_verdicts.py"), "--check"],
        capture_output=True, text=True, timeout=300, env={**os.environ},
    )  # fmt: skip
    assert done.returncode == 0, done.stdout + done.stderr


def test_a_single_altered_verdict_would_fail_the_comparison(real_scan):
    # proof that the comparison has teeth: flip one verdict in a copy and it no longer matches
    table = build_verdict_table(real_scan)
    notes = table["notes"]
    assert isinstance(notes, dict)
    victim = next(k for k, v in notes.items() if v["verdict"] is None)
    mutated = json.loads(json.dumps(table))
    mutated["notes"][victim]["verdict"] = "override"
    assert mutated != table
    victim2 = next(k for k, v in notes.items() if v["verdict"] is not None)
    mutated2 = json.loads(json.dumps(table))
    mutated2["notes"][victim2]["verdict"] = None
    assert mutated2 != table


def test_the_miss_list_of_the_real_screener_is_what_the_cache_says(real_scan):
    real_misses = {n.note_id for n in NOTE_BANK if n.malicious and real_scan(n.text) is None}
    assert {n.note_id for n in screener_misses()} == real_misses
