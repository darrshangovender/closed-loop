"""P2: the streaming monitor returns exactly the offline robustness, quarter by quarter."""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from closed_loop.spec.monitor import StreamingMonitor
from closed_loop.spec.properties import SpecParams, property_formulas, signals_from_trace
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
    robustness,
)
from tests.spec._helpers import (
    ASSETS,
    CONVEYOR,
    CRUSHER,
    PUMPS,
    clean_trace,
    random_trace,
    violating_trace,
)

ATOMS = ("a", "b", "c")


def stream(f: Formula, signals: dict[str, np.ndarray]) -> np.ndarray:
    monitor = StreamingMonitor(f)
    n = len(next(iter(signals.values())))
    out = np.empty(n, dtype=np.float32)
    for t in range(n):
        out[t] = monitor.push({k: float(v[t]) for k, v in signals.items()})
    return out


@st.composite
def formulas(draw: st.DrawFn, depth: int = 4) -> Formula:
    if depth == 0 or draw(st.integers(0, 4)) == 0:
        return Atom(draw(st.sampled_from(ATOMS)))
    kind = draw(st.sampled_from(["not", "and", "or", "implies", "once", "hist", "always"]))
    sub = formulas(depth=depth - 1)
    if kind == "not":
        return Not(draw(sub))
    if kind == "always":
        return Always(draw(sub))
    if kind in ("once", "hist"):
        a = draw(st.integers(0, 4))
        b = a + draw(st.integers(0, 5))
        return (Once if kind == "once" else Hist)(draw(sub), a, b)
    left, right = draw(sub), draw(sub)
    return {"and": And, "or": Or, "implies": Implies}[kind](left, right)


class TestStreamingEqualsOffline:
    @settings(max_examples=250)
    @given(
        formulas(),
        st.integers(0, 40),
        st.randoms(use_true_random=False),
    )
    def test_random_formulas_on_random_signals(self, f: Formula, n: int, rnd) -> None:
        # a small value grid makes ties common, which is where monotonic deques go wrong
        grid = [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 7.0]
        signals = {
            name: np.array([rnd.choice(grid) for _ in range(n)], dtype=np.float32) for name in ATOMS
        }
        got = stream(f, signals) if n else np.empty(0, np.float32)
        want = robustness(f, signals)
        assert np.array_equal(got, want), f"{f}\n{got}\n{want}"

    @pytest.mark.parametrize("params", [SpecParams(), SpecParams(retest_q=8, break_q=8)])
    def test_every_property_formula_on_200_random_traces(self, params: SpecParams) -> None:
        rng = np.random.default_rng(2026)
        formulas_ = property_formulas(params)
        for i in range(200):
            tr = random_trace(rng, int(rng.integers(0, 70)))
            signals = signals_from_trace(tr, params, CRUSHER, PUMPS)
            for pid, f in formulas_.items():
                got = stream(f, signals) if tr.n_q else np.empty(0, np.float32)
                assert np.array_equal(got, robustness(f, signals)), (i, pid)

    def test_every_property_formula_on_the_hand_traces(self) -> None:
        params = SpecParams()
        for tr in [clean_trace()] + [violating_trace(p) for p in property_formulas(params)]:
            signals = signals_from_trace(tr, params, CRUSHER, PUMPS)
            for pid, f in property_formulas(params).items():
                assert np.array_equal(stream(f, signals), robustness(f, signals)), pid

    def test_long_windows(self) -> None:
        rng = np.random.default_rng(9)
        signals = {"a": rng.normal(size=500).astype(np.float32)}
        f = Hist(Once(Atom("a"), 5, 60), 2, 40)
        assert np.array_equal(stream(f, signals), robustness(f, signals))

    def test_float64_input_is_rounded_like_the_offline_float32_path(self) -> None:
        value = 0.1  # not representable in float32
        monitor = StreamingMonitor(Atom("a"))
        assert monitor.push({"a": value}) == float(np.float32(value))


class TestBoundaryQuarters:
    def test_window_longer_than_history_uses_what_exists(self) -> None:
        m = StreamingMonitor(Hist(Atom("a"), 0, 5))
        assert m.push({"a": 4.0}) == 4.0  # one quarter of history
        assert m.push({"a": 2.0}) == 2.0
        assert m.push({"a": 9.0}) == 2.0

    def test_empty_window_at_the_start(self) -> None:
        once = StreamingMonitor(Once(Atom("a"), 2, 3))
        hist = StreamingMonitor(Hist(Atom("a"), 2, 3))
        assert [once.push({"a": 1.0}) for _ in range(3)] == [-math.inf, -math.inf, 1.0]
        assert [hist.push({"a": 1.0}) for _ in range(3)] == [math.inf, math.inf, 1.0]

    def test_a_value_leaves_the_window_exactly_after_b_quarters(self) -> None:
        m = StreamingMonitor(Once(Atom("a"), 0, 2))
        seen = [m.push({"a": v}) for v in (9.0, 0.0, 0.0, 0.0, 0.0)]
        assert seen == [9.0, 9.0, 9.0, 0.0, 0.0]  # the 9 is visible at t = 0, 1, 2

    def test_a_value_enters_the_window_exactly_a_quarters_late(self) -> None:
        m = StreamingMonitor(Once(Atom("a"), 2, 4))
        seen = [m.push({"a": v}) for v in (9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
        assert seen == [-math.inf, -math.inf, 9.0, 9.0, 9.0, 0.0, 0.0]


class TestMemoryAndState:
    def test_memory_is_bounded_by_the_window_not_the_run(self) -> None:
        f = Always(Once(Atom("a"), 3, 10))
        m = StreamingMonitor(f)
        for t in range(5000):
            m.push({"a": float(t % 7)})
        node = m._root.arg  # type: ignore[attr-defined]
        assert len(node.delay) <= 3 and len(node.mono) <= 8

    def test_reset_starts_over(self) -> None:
        m = StreamingMonitor(Always(Atom("a")))
        m.push({"a": -5.0})
        assert m.push({"a": 3.0}) == -5.0
        m.reset()
        assert m.n == 0 and m.last is None
        assert m.push({"a": 3.0}) == 3.0

    def test_counters(self) -> None:
        m = StreamingMonitor(Atom("a"))
        m.push({"a": 1.0})
        m.push({"a": 2.0})
        assert m.n == 2 and m.last == 2.0

    def test_a_refused_push_changes_nothing(self) -> None:
        f = Always(Once(And(Atom("a"), Atom("b")), 0, 3))
        reference = StreamingMonitor(f)
        probed = StreamingMonitor(f)
        script = [({"a": 1.0, "b": 2.0}), ({"a": -1.0, "b": 0.5}), ({"a": 2.0, "b": 2.0})]
        out_ref = [reference.push(s) for s in script]
        out = [probed.push(script[0])]
        with pytest.raises(KeyError):
            probed.push({"a": 1.0})  # b missing
        with pytest.raises(ValueError, match="NaN"):
            probed.push({"a": float("nan"), "b": 1.0})
        out += [probed.push(s) for s in script[1:]]
        assert out == out_ref

    def test_extra_signals_are_ignored(self) -> None:
        m = StreamingMonitor(Atom("a"))
        assert m.push({"a": 1.0, "unused": float("nan")}) == 1.0

    def test_not_a_formula(self) -> None:
        with pytest.raises(TypeError):
            StreamingMonitor("a")  # type: ignore[arg-type]


def test_assets_constant_is_what_the_helpers_use() -> None:
    assert set(ASSETS) == {CRUSHER, *PUMPS, CONVEYOR}  # P1 reads the conveyor too (A12.3)
