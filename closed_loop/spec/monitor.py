"""Streaming (online) monitor for the bounded-past fragment.

Interface text: docs/adr/0002-interfaces.md, section 3.4. Contract: fed the signals one quarter at
a time, ``StreamingMonitor.push`` returns exactly what ``closed_loop.spec.stl.robustness`` returns
at that quarter, for every formula.

Memory is proportional to the windows in the formula, not to the length of the run. ``Once`` and
``Hist`` keep a delay line of ``a`` values (the window starts ``a`` quarters back) and a monotonic
deque over the window, so each push is amortised constant time per operator. ``Always`` keeps one
running minimum.

Atom values are rounded to float32 on the way in, as the offline evaluator does, so the two agree
bit for bit even when the caller passes float64.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping

import numpy as np

from closed_loop.spec.stl import (
    Always,
    And,
    Atom,
    Formula,
    Hist,
    Implies,
    Not,
    Once,
    Or,
)

__all__ = ["StreamingMonitor"]

_Env = Mapping[str, float]


class _Node:
    def push(self, env: _Env) -> float:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class _AtomNode(_Node):
    """Reads an already-validated value (``StreamingMonitor.push`` checks before any node runs)."""

    def __init__(self, name: str) -> None:
        self.name = name

    def push(self, env: _Env) -> float:
        return env[self.name]


class _NotNode(_Node):
    def __init__(self, arg: _Node) -> None:
        self.arg = arg

    def push(self, env: _Env) -> float:
        return -self.arg.push(env)

    def reset(self) -> None:
        self.arg.reset()


class _BinaryNode(_Node):
    def __init__(self, left: _Node, right: _Node) -> None:
        self.left, self.right = left, right

    def reset(self) -> None:
        self.left.reset()
        self.right.reset()


class _AndNode(_BinaryNode):
    def push(self, env: _Env) -> float:
        return min(self.left.push(env), self.right.push(env))


class _OrNode(_BinaryNode):
    def push(self, env: _Env) -> float:
        return max(self.left.push(env), self.right.push(env))


class _ImpliesNode(_BinaryNode):
    def push(self, env: _Env) -> float:
        # both children are always pushed, in order, so their windows never miss a quarter
        left = self.left.push(env)
        right = self.right.push(env)
        return max(-left, right)


class _WindowNode(_Node):
    """Sliding max (``Once``) or min (``Hist``) over quarters ``t-b .. t-a``."""

    def __init__(self, arg: _Node, a: int, b: int, take_max: bool) -> None:
        self.arg, self.a, self.b, self.take_max = arg, int(a), int(b), take_max
        self.empty = -math.inf if take_max else math.inf
        self.reset()

    def reset(self) -> None:
        self.arg.reset()
        self.t = 0
        self.delay: deque[float] = deque()
        self.mono: deque[tuple[int, float]] = deque()  # (quarter, value), values monotone

    def push(self, env: _Env) -> float:
        value = self.arg.push(env)
        t = self.t
        self.t += 1
        self.delay.append(value)
        if len(self.delay) > self.a:  # the quarter t-a now exists
            entering = self.delay.popleft()
            if self.take_max:
                while self.mono and self.mono[-1][1] <= entering:
                    self.mono.pop()
            else:
                while self.mono and self.mono[-1][1] >= entering:
                    self.mono.pop()
            self.mono.append((t - self.a, entering))
        while self.mono and self.mono[0][0] < t - self.b:
            self.mono.popleft()
        return self.mono[0][1] if self.mono else self.empty


class _AlwaysNode(_Node):
    def __init__(self, arg: _Node) -> None:
        self.arg = arg
        self.low = math.inf

    def reset(self) -> None:
        self.arg.reset()
        self.low = math.inf

    def push(self, env: _Env) -> float:
        self.low = min(self.low, self.arg.push(env))
        return self.low


def _compile(f: Formula) -> _Node:
    if isinstance(f, Atom):
        return _AtomNode(f.name)
    if isinstance(f, Not):
        return _NotNode(_compile(f.arg))
    if isinstance(f, And):
        return _AndNode(_compile(f.left), _compile(f.right))
    if isinstance(f, Or):
        return _OrNode(_compile(f.left), _compile(f.right))
    if isinstance(f, Implies):
        return _ImpliesNode(_compile(f.left), _compile(f.right))
    if isinstance(f, Once):
        return _WindowNode(_compile(f.arg), f.a, f.b, take_max=True)
    if isinstance(f, Hist):
        return _WindowNode(_compile(f.arg), f.a, f.b, take_max=False)
    if isinstance(f, Always):
        return _AlwaysNode(_compile(f.arg))
    raise TypeError(f"not a formula: {f!r}")


class StreamingMonitor:
    """Online robustness of one formula, one quarter per ``push``."""

    def __init__(self, f: Formula) -> None:
        if not isinstance(f, Formula):
            raise TypeError(f"not a formula: {f!r}")
        self.formula = f
        self._atoms = tuple(sorted(f.atoms()))
        self._root = _compile(f)
        self.n = 0
        self.last: float | None = None

    def push(self, signals_now: Mapping[str, float]) -> float:
        """Robustness at this quarter, from this quarter's atom values.

        Raises ``KeyError`` if an atom is missing and ``ValueError`` if one is NaN. Every atom is
        checked before any state changes, so a refused push leaves the monitor as it was.
        """
        env: dict[str, float] = {}
        for name in self._atoms:
            if name not in signals_now:
                raise KeyError(f"no signal named {name!r}; have {sorted(signals_now)}")
            value = float(np.float32(signals_now[name]))
            if math.isnan(value):
                raise ValueError(f"signal {name!r} is NaN")
            env[name] = value
        value = self._root.push(env)
        self.n += 1
        self.last = value
        return value

    def reset(self) -> None:
        """Forget everything, as if no quarter had been pushed."""
        self._root.reset()
        self.n = 0
        self.last = None
