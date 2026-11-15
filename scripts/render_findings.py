"""Print the markdown of ``docs/FINDINGS.md`` from the checked rows of ``closed_loop.f2.findings``.

    python scripts/render_findings.py                    the document, with the sections that need
                                                         a measurement marked "not generated"
    python scripts/render_findings.py --run-checks       also runs the 33 checks and lists what each
                                                         saw (a few minutes; needs the siblings)
    python scripts/render_findings.py --episodes FILE --mismatch FILE
                                                         the JSON of ``scripts/f2_measure.py``
    python scripts/render_findings.py --out docs/FINDINGS.md

Without ``--out`` the text goes to standard output and nothing is written. The table is data in
``closed_loop/f2/findings.py``, which ``tests/f2/test_findings.py`` checks against the installed
siblings, so the document cannot say something the checks do not hold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROGRESS = ROOT / "docs" / "adr" / "progress"


def load_json(path: Path | None, default: Path) -> dict[str, object] | None:
    target = path if path is not None else default
    if not target.exists():
        return None
    data: dict[str, object] = json.loads(target.read_text(encoding="utf-8"))
    return data


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-checks", action="store_true")
    parser.add_argument("--episodes", type=Path, default=None)
    parser.add_argument("--mismatch", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    from closed_loop.f2.findings import render_document, render_measured

    evidence = None
    if args.run_checks:
        from closed_loop.f2.mismatch import run_checks

        evidence = [(c.id, c.holds, c.evidence) for c in run_checks()]
    measured = render_measured(
        load_json(args.episodes, PROGRESS / "P11c-f2-measure-episodes.json"),
        load_json(args.mismatch, PROGRESS / "P11c-f2-measure-mismatch.json"),
        evidence,
    )
    text = render_document(measured) + "\n"
    if args.out is None:
        sys.stdout.write(text)
    else:
        args.out.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
