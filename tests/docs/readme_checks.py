"""The checks on ``README.md``, as plain functions of text, so that a test can run them on the real
README and, with a small fixture, prove that each one fails on what it is meant to catch.

Four checks live here:

* ``block_problems``: the Results block is exactly what ``scripts/render_readme_results.py`` makes
  from the records.
* ``unknown_numbers``: every number in the prose (outside code, links and the generated block) is
  among the records or on a short allow-list with a reason.
* ``standards_problems``: the README meets ``_assets/STANDARDS.md``.
* ``absolute_hits`` and ``deny_hits``: no absolute claims, and none of the names on a deny-list
  that is kept outside the repository.
"""

from __future__ import annotations

import importlib.util
import math
import re
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Final

import numpy as np

from closed_loop.core.results import ResultRecord

ROOT: Final[Path] = Path(__file__).resolve().parents[2]
SCRIPTS: Final[Path] = ROOT / "scripts"

FOOTER: Final[str] = (
    "Built by [Darrshan Govender](https://github.com/darrshangovender) · "
    "[Agulhas Code](https://agulhascode.co.za)"
)
REQUIRED_SECTIONS: Final[tuple[str, ...]] = (
    "Why this exists",
    "Quick start",
    "How it works",
    "Results",
    "Design decisions",
    "Limitations",
    "Project layout",
    "Tests",
)
STUB_MARKERS: Final[tuple[str, ...]] = ("Scaffold only",)
SKIP_DIRS: Final[frozenset[str]] = frozenset(
    {".git", ".cache", "__pycache__", ".venv", "venv", ".mypy_cache", ".ruff_cache",
     ".pytest_cache", "node_modules", ".hypothesis"}
)  # fmt: skip
ABSOLUTE_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(never|always|guarantee[sd]?|guaranteeing|every)\b|100\s?%", re.IGNORECASE
)


def load_script(name: str) -> ModuleType:
    """Import ``scripts/<name>.py`` by path (``scripts`` is not a package)."""
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------- reading the README


def is_stub(readme: str) -> bool:
    """True while the README is the scaffold: it says so, or it has no section at all."""
    return any(m in readme for m in STUB_MARKERS) or not re.search(r"^## ", readme, re.MULTILINE)


def split_fences(text: str) -> Iterator[tuple[bool, str]]:
    """``(is_code, text)`` pieces: fenced blocks (``` or ~~~) are code, the rest is prose."""
    prose: list[str] = []
    code: list[str] = []
    fence = ""
    for line in text.splitlines():
        stripped = line.lstrip()
        marker = stripped[:3]
        if not fence and marker in ("```", "~~~"):
            if prose:
                yield False, "\n".join(prose)
                prose = []
            fence = marker
            code = [line]
        elif fence:
            code.append(line)
            if marker == fence:
                yield True, "\n".join(code)
                fence = ""
        else:
            prose.append(line)
    if fence and code:
        yield True, "\n".join(code)
    if prose:
        yield False, "\n".join(prose)


def prose_of(readme: str, block_span: tuple[int, int] | None = None) -> str:
    """The README prose: no fenced code, no inline code, no link targets, no badges, no HTML
    comments, and, if ``block_span`` is given, nothing inside the generated Results block."""
    if block_span is not None:
        readme = readme[: block_span[0]] + "\n" + readme[block_span[1] :]
    text = "\n".join(piece for is_code, piece in split_fences(readme) if not is_code)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)  # images and badges
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # a link keeps its text, loses its target
    text = re.sub(r"`[^`\n]*`", " ", text)  # inline code
    text = re.sub(r"https?://\S+", " ", text)
    return text


# ----------------------------------------------------------------------------- the Results block


def block_problems(readme: str, records: Sequence[ResultRecord]) -> list[str]:
    """Why the Results block is not what the script makes from ``records`` (empty if it is)."""
    script = load_script("render_readme_results")
    try:
        have = script.current_block(readme.replace("\r\n", "\n"))
    except script.MarkerError as exc:
        return [str(exc)]
    want = script.render_block(records)
    if have == want:
        return []
    have_lines, want_lines = have.splitlines(), want.splitlines()
    for i, (a, b) in enumerate(zip(have_lines, want_lines, strict=False)):
        if a != b:
            return [f"line {i + 1} of the block differs:\n  README:    {a}\n  generated: {b}"]
    return [f"the block has {len(have_lines)} lines and the generated one {len(want_lines)}"]


# ----------------------------------------------------------------------------- numbers in the prose

ROUNDED_FROM: Final[int] = 3  # decimals from which a number may be a rounded record value
NUMBER_RE: Final[re.Pattern[str]] = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(%?)(?![\w])")


def number_tokens(text: str) -> list[tuple[str, bool]]:
    """``(digits, is_percent)`` of every number in ``text``; thousands commas are dropped, and a
    comma that ends a sentence is not part of the number."""
    out = []
    for m in NUMBER_RE.finditer(text):
        raw = m.group(1).rstrip(",")
        if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", raw):
            raw = raw.replace(",", "")
        elif "," in raw:
            for part in raw.split(","):  # "1,2,3": three numbers
                if part:
                    out.append((part, bool(m.group(2))))
            continue
        out.append((raw, bool(m.group(2))))
    return out


class NumberIndex:
    """What counts as 'among the records': every numeric field of a record (value, n, bounds, each
    numeric ``extra``), and every number written in a record's text (``measured_on``, a verdict's
    headline and notes)."""

    def __init__(self, records: Iterable[ResultRecord]) -> None:
        values: list[float] = []
        text_tokens: set[str] = set()
        for rec in records:
            values += [rec.value, float(rec.n)]
            for bound in (rec.ci_lo, rec.ci_hi):
                if bound is not None:
                    values.append(float(bound))
            texts = [rec.measured_on, rec.name]
            for v in rec.extra.values():
                if isinstance(v, bool):
                    continue
                if isinstance(v, int | float):
                    values.append(float(v))
                else:
                    texts.append(str(v))
            for t in texts:
                text_tokens.update(tok for tok, _pct in number_tokens(t))
        finite = [v for v in values if math.isfinite(v)]
        # the prose writes a magnitude; the sign is read from the table beside it
        self.values = np.array(sorted({abs(v) for v in finite}), dtype=np.float64)
        self.text_tokens = text_tokens

    def matches(self, digits: str, percent: bool = False) -> bool:
        """True if ``digits`` is a record's number or the text of one. A number written with three
        or more decimals may be a record's value rounded to that many places; a shorter one (an
        integer, ``0.05``, ``93.4``) must be exactly a record's number, or it could be any record
        that happens to round to it."""
        if digits in self.text_tokens:
            return True
        x = float(digits)
        places = len(digits.split(".")[1]) if "." in digits else 0
        scaled = self.values * (100.0 if percent else 1.0)
        if places >= ROUNDED_FROM:
            scaled = np.round(scaled, places)
        return bool(np.any(np.abs(scaled - x) < 1e-9 * max(1.0, abs(x))))


def unknown_numbers(
    prose: str, records: Sequence[ResultRecord], allow: Mapping[str, str]
) -> list[str]:
    """The numbers of ``prose`` that are neither among the records nor on the allow-list.

    An ``allow`` key is the digits as written (``"0.05"``) or a regular expression wrapped in
    slashes (``"/^000[1-3]$/"``); every entry carries its reason.
    """
    index = NumberIndex(records)
    patterns = [(re.compile(k[1:-1]), k) for k in allow if k.startswith("/") and k.endswith("/")]
    bad: list[str] = []
    for digits, percent in number_tokens(prose):
        shown = digits + ("%" if percent else "")
        if digits in allow or shown in allow:
            continue
        if any(p.search(digits) for p, _k in patterns):
            continue
        if not index.matches(digits, percent):
            bad.append(shown)
    return sorted(set(bad), key=lambda t: (len(t), t))


def empty_reasons(allow: Mapping[str, str]) -> list[str]:
    """Allow-list entries that give no reason."""
    return [k for k, v in allow.items() if not v.strip()]


# ------------------------------------------------------------------ the owner's standards

BADGE_RE: Final[re.Pattern[str]] = re.compile(r"\[!\[([^\]]*)\]\(([^)]*)\)\]\(([^)]*)\)")


def standards_problems(readme: str) -> list[str]:
    """How the README departs from ``_assets/STANDARDS.md`` (empty if it does not)."""
    problems: list[str] = []
    lines = [
        ln for is_code, piece in split_fences(readme) if not is_code for ln in piece.splitlines()
    ]
    h1 = [ln for ln in lines if re.match(r"# \S", ln)]
    if len(h1) != 1:
        problems.append(f"expected one H1, found {len(h1)}")
    badges = BADGE_RE.findall(readme)
    if len(badges) != 4:
        problems.append(f"expected exactly 4 badges, found {len(badges)}")
    elif not re.search(r"\b(ci|tests?|workflow)\b|tests\.yml", " ".join(badges[0]), re.IGNORECASE):
        problems.append(f"the first badge should be CI, it is {badges[0][0]!r}")
    first_h2 = next((i for i, ln in enumerate(lines) if ln.startswith("## ")), len(lines))
    head = [
        ln
        for ln in lines[:first_h2]
        if ln.strip() and not BADGE_RE.search(ln) and not re.match(r"# \S", ln)
    ]
    if not head or not head[0].startswith(">") or sum(ln.startswith(">") for ln in head) != 1:
        problems.append("the line after the badges should be a one-line blockquote, once")
    titles = [ln[3:].strip() for ln in lines if ln.startswith("## ")]
    at = -1
    for section in REQUIRED_SECTIONS:
        hit = next((i for i, t in enumerate(titles) if t.lower().startswith(section.lower())), -1)
        if hit < 0:
            problems.append(f"missing section: {section}")
        elif hit < at:
            problems.append(f"section out of order: {section}")
        else:
            at = hit
    if "```mermaid" not in readme:
        problems.append("How it works needs a mermaid diagram")
    last = next((ln for ln in reversed(readme.splitlines()) if ln.strip()), "")
    if last.strip() != FOOTER:
        problems.append("the last line is not the author footer")
    return problems


# ----------------------------------------------------------------------------- absolutes, names


def sentences(prose: str) -> list[str]:
    flat = re.sub(r"\s+", " ", prose)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])", flat) if s.strip()]


def absolute_hits(prose: str, allow: Mapping[str, str]) -> list[tuple[str, str]]:
    """``(word, sentence)`` for each absolute in ``prose`` whose sentence is not on the allow-list
    (an allow key is a piece of the sentence)."""
    hits = []
    for s in sentences(prose):
        for m in ABSOLUTE_RE.finditer(s):
            if not any(key in s for key in allow):
                hits.append((m.group(0).lower(), s))
    return hits


def read_deny_list(path: Path) -> list[str]:
    names = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            names.append(line)
    return names


def tree_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.is_file():
            yield path


def deny_hits(root: Path, names: Sequence[str]) -> list[tuple[str, int]]:
    """``(relative path, line number)`` of every line of a text file under ``root`` that holds a
    name from the list (case-insensitive). The names are never put in the result."""
    lowered = [n.lower() for n in names if n]
    hits: list[tuple[str, int]] = []
    for path in tree_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            low = line.lower()
            if any(n in low for n in lowered):
                hits.append((path.relative_to(root).as_posix(), i))
    return hits
