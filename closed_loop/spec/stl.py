"""Bounded-past signal temporal logic: syntax and vectorised (offline) robustness.

Interface text: docs/adr/0002-interfaces.md, section 3.4; fragment: docs/adr/0001-architecture.md,
section 5. ``closed_loop.spec.monitor.StreamingMonitor`` computes the same numbers one quarter at
a time.

Syntax: ``Atom(name)``, ``Not``, ``And``, ``Or``, ``Implies``, ``Once(f, a, b)`` (``O[a,b]``),
``Hist(f, a, b)`` (``H[a,b]``) and ``Always(f)`` (``G``, from the start of the trace to now).
Windows ``a`` and ``b`` count quarters, inclusive, into the past: at quarter ``t``, ``Once`` looks
at quarters ``t-b .. t-a``.

Robustness at quarter ``t`` is a real number whose sign says whether the formula holds there
(``> 0`` holds, ``< 0`` fails, ``0`` is the boundary and counts as holding) and whose size says by
how much, in the units of the atoms:

* ``Atom``: the signal value; ``Not``: negation; ``And``: minimum; ``Or``: maximum;
  ``Implies(p, q)``: ``max(-p, q)``.
* ``Once`` is the maximum and ``Hist`` the minimum over the window. Only quarters that exist are
  used, so near the start the window is cut at quarter 0. A window with no quarter at all (for
  instance ``Once(f, 2, 2)`` at quarter 1) is ``-inf`` for ``Once`` (nothing happened) and
  ``+inf`` for ``Hist`` (nothing contradicts). The infinities only meet ``min``, ``max`` and
  negation, so they never turn into NaN.
* ``Always(f)`` at ``t`` is the minimum of ``f`` over quarters ``0 .. t``.

All values are float32.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

__all__ = [
    "And",
    "Always",
    "Atom",
    "Formula",
    "Hist",
    "Implies",
    "Not",
    "Once",
    "Or",
    "all_of",
    "any_of",
    "min_robustness",
    "robustness",
]

_INF = np.float32(np.inf)


class Formula:
    """Base class of the syntax tree. Nodes are immutable and compare by structure."""

    def atoms(self) -> frozenset[str]:
        """Names of every signal the formula reads."""
        raise NotImplementedError

    def lookback(self) -> int:
        """How many quarters into the past the value at quarter ``t`` can depend on."""
        raise NotImplementedError

    def __and__(self, other: Formula) -> Formula:
        return And(self, other)

    def __or__(self, other: Formula) -> Formula:
        return Or(self, other)

    def __invert__(self) -> Formula:
        return Not(self)


def _check_window(a: int, b: int) -> None:
    for name, value in (("a", a), ("b", b)):
        if isinstance(value, bool) or not isinstance(value, int | np.integer):
            raise TypeError(f"window bound {name} must be an int number of quarters, got {value!r}")
    if not 0 <= a <= b:
        raise ValueError(f"window needs 0 <= a <= b, got a={a}, b={b}")


@dataclass(frozen=True)
class Atom(Formula):
    """A margin signal looked up by name in the signal dictionary."""

    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("an Atom needs a non-empty name")

    def atoms(self) -> frozenset[str]:
        return frozenset({self.name})

    def lookback(self) -> int:
        return 0

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Not(Formula):
    arg: Formula

    def atoms(self) -> frozenset[str]:
        return self.arg.atoms()

    def lookback(self) -> int:
        return self.arg.lookback()

    def __str__(self) -> str:
        return f"!({self.arg})"


@dataclass(frozen=True)
class And(Formula):
    left: Formula
    right: Formula

    def atoms(self) -> frozenset[str]:
        return self.left.atoms() | self.right.atoms()

    def lookback(self) -> int:
        return max(self.left.lookback(), self.right.lookback())

    def __str__(self) -> str:
        return f"({self.left} & {self.right})"


@dataclass(frozen=True)
class Or(Formula):
    left: Formula
    right: Formula

    def atoms(self) -> frozenset[str]:
        return self.left.atoms() | self.right.atoms()

    def lookback(self) -> int:
        return max(self.left.lookback(), self.right.lookback())

    def __str__(self) -> str:
        return f"({self.left} | {self.right})"


@dataclass(frozen=True)
class Implies(Formula):
    left: Formula
    right: Formula

    def atoms(self) -> frozenset[str]:
        return self.left.atoms() | self.right.atoms()

    def lookback(self) -> int:
        return max(self.left.lookback(), self.right.lookback())

    def __str__(self) -> str:
        return f"({self.left} -> {self.right})"


@dataclass(frozen=True)
class Once(Formula):
    """``O[a,b] f``: ``f`` held at some quarter in ``t-b .. t-a``."""

    arg: Formula
    a: int
    b: int

    def __post_init__(self) -> None:
        _check_window(self.a, self.b)

    def atoms(self) -> frozenset[str]:
        return self.arg.atoms()

    def lookback(self) -> int:
        return self.b + self.arg.lookback()

    def __str__(self) -> str:
        return f"O[{self.a},{self.b}]({self.arg})"


@dataclass(frozen=True)
class Hist(Formula):
    """``H[a,b] f``: ``f`` held at every quarter in ``t-b .. t-a``."""

    arg: Formula
    a: int
    b: int

    def __post_init__(self) -> None:
        _check_window(self.a, self.b)

    def atoms(self) -> frozenset[str]:
        return self.arg.atoms()

    def lookback(self) -> int:
        return self.b + self.arg.lookback()

    def __str__(self) -> str:
        return f"H[{self.a},{self.b}]({self.arg})"


@dataclass(frozen=True)
class Always(Formula):
    """``G f``: ``f`` held at every quarter from the start of the trace to now."""

    arg: Formula

    def atoms(self) -> frozenset[str]:
        return self.arg.atoms()

    def lookback(self) -> int:
        # unbounded in principle; the streaming monitor keeps one running minimum instead
        return self.arg.lookback()

    def __str__(self) -> str:
        return f"G({self.arg})"


def all_of(*formulas: Formula) -> Formula:
    """Conjunction of one or more formulas, left to right."""
    if not formulas:
        raise ValueError("all_of needs at least one formula")
    out = formulas[0]
    for f in formulas[1:]:
        out = And(out, f)
    return out


def any_of(*formulas: Formula) -> Formula:
    """Disjunction of one or more formulas, left to right."""
    if not formulas:
        raise ValueError("any_of needs at least one formula")
    out = formulas[0]
    for f in formulas[1:]:
        out = Or(out, f)
    return out


def _signal_length(f: Formula, signals: Mapping[str, NDArray]) -> int:
    lengths = {}
    for name in sorted(f.atoms()):
        if name not in signals:
            raise KeyError(f"no signal named {name!r}; have {sorted(signals)}")
        arr = np.asarray(signals[name])
        if arr.ndim != 1:
            raise ValueError(f"signal {name!r} must be one-dimensional, got shape {arr.shape}")
        lengths[name] = arr.shape[0]
    if len(set(lengths.values())) > 1:
        raise ValueError(f"signals differ in length: {lengths}")
    return next(iter(lengths.values()))


def _window(values: NDArray[np.float32], a: int, b: int, take_max: bool) -> NDArray[np.float32]:
    """Max (or min) of ``values[t-b .. t-a]`` for every ``t``, cut at the start of the trace."""
    n = values.shape[0]
    if n == 0:  # a window cannot be wider than the padded input, and there is nothing to compute
        return values
    fill = -_INF if take_max else _INF
    padded = np.concatenate([np.full(b, fill, dtype=np.float32), values])
    windows = sliding_window_view(padded, b - a + 1)[:n]
    out = windows.max(axis=-1) if take_max else windows.min(axis=-1)
    return np.asarray(out, dtype=np.float32)


def _eval(f: Formula, signals: Mapping[str, NDArray], n: int) -> NDArray[np.float32]:
    if isinstance(f, Atom):
        arr = np.asarray(signals[f.name], dtype=np.float32)
        if np.isnan(arr).any():
            raise ValueError(f"signal {f.name!r} contains NaN")
        return arr
    if isinstance(f, Not):
        return np.negative(_eval(f.arg, signals, n))
    if isinstance(f, And):
        return np.minimum(_eval(f.left, signals, n), _eval(f.right, signals, n))
    if isinstance(f, Or):
        return np.maximum(_eval(f.left, signals, n), _eval(f.right, signals, n))
    if isinstance(f, Implies):
        return np.maximum(np.negative(_eval(f.left, signals, n)), _eval(f.right, signals, n))
    if isinstance(f, Once):
        return _window(_eval(f.arg, signals, n), f.a, f.b, take_max=True)
    if isinstance(f, Hist):
        return _window(_eval(f.arg, signals, n), f.a, f.b, take_max=False)
    if isinstance(f, Always):
        inner = _eval(f.arg, signals, n)
        return np.asarray(np.minimum.accumulate(inner), dtype=np.float32)
    raise TypeError(f"not a formula: {f!r}")


def robustness(f: Formula, signals: Mapping[str, NDArray[np.float32]]) -> NDArray[np.float32]:
    """Vectorised robustness at every quarter. Margin > 0 means satisfied.

    Raises ``KeyError`` for a missing signal and ``ValueError`` for signals of different lengths,
    not one-dimensional, or containing NaN.
    """
    if not isinstance(f, Formula):
        raise TypeError(f"not a formula: {f!r}")
    n = _signal_length(f, signals)
    return _eval(f, signals, n)


def min_robustness(f: Formula, signals: Mapping[str, NDArray]) -> float:
    """Smallest robustness over all quarters (``+inf`` for an empty trace)."""
    values = robustness(f, signals)
    return float(np.min(values)) if values.size else float("inf")
