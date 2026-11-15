"""Write (or check) closed_loop/spec/note_verdicts.json through the real plant_copilot scan_text.

The cache holds, for each SYNTHETIC handover note in the bank, the verdict of
``plant_copilot.injection.scan_text`` (a rule name, or null), so that the F1 runs need no sibling
installed (docs/adr/0003 A2). This script is the one place that imports the sibling for it, and it
imports it only here. An ``f2`` test (tests/plant/test_note_verdicts_f2.py) recomputes the table and
fails on any difference.

    python scripts/build_note_verdicts.py          # rewrite the file
    python scripts/build_note_verdicts.py --check  # exit 1 if the file differs from a recompute
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from closed_loop.plant.notes import NOTE_VERDICTS_PATH, build_verdict_table  # noqa: E402


def render(table: dict[str, object]) -> str:
    return json.dumps(table, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare, do not write")
    args = parser.parse_args()
    try:
        from plant_copilot.injection import scan_text
    except ImportError:
        print("plant_copilot is not installed: cannot compute the verdicts", file=sys.stderr)
        return 2
    text = render(build_verdict_table(scan_text))
    if args.check:
        current = (
            NOTE_VERDICTS_PATH.read_text(encoding="utf-8") if NOTE_VERDICTS_PATH.exists() else ""
        )
        if current != text:
            print(f"{NOTE_VERDICTS_PATH} differs from a recompute", file=sys.stderr)
            return 1
        print("note verdicts are up to date")
        return 0
    with open(NOTE_VERDICTS_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"wrote {NOTE_VERDICTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    raise SystemExit(main())
