"""P2: robustness of the bounded-past fragment, checked against a naive definition."""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from closed_loop.spec.stl import (
    Always,
    And,
    Atom,
    Hist,
    Implies,
    Not,
    Once,
    Or,
    all_of,
    any_of,
    min_robustness,
    robustness,
)

INF = math.inf


def sig(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float32)


def naive_once(x: list[float], a: int, b: int) -> list[float]:
    out = []
    for t in range(len(x)):
        window = [x[tau] for tau in range(max(0, t - b), t - a + 1)]
        out.append(max(window) if window else -INF)
    return out


def naive_hist(x: list[float], a: int, b: int) -> list[float]:
    out = []
    for t in range(len(x)):
        window = [x[tau] for tau in range(max(0, t - b), t - a + 1)]
        out.append(min(window) if window else INF)
    return out


class TestBooleanLayer:
    def test_atoms_and_connectives(self) -> None:
        s = {"a": sig(1, -2, 3), "b": sig(-1, 5, 2)}
        assert robustness(Atom("a"), s).tolist() == [1, -2, 3]
        assert robustness(Not(Atom("a")), s).tolist() == [-1, 2, -3]
        assert robustness(And(Atom("a"), Atom("b")), s).tolist() == [-1, -2, 2]
        assert robustness(Or(Atom("a"), Atom("b")), s).tolist() == [1, 5, 3]
        assert robustness(Implies(Atom("a"), Atom("b")), s).tolist() == [-1, 5, 2]

    def test_result_is_float32(self) -> None:
        out = robustness(Always(Implies(Atom("a"), Atom("b"))), {"a": sig(1, 2), "b": sig(3, 4)})
        assert out.dtype == np.float32

    def test_operators_and_folds(self) -> None:
        a, b, c = Atom("a"), Atom("b"), Atom("c")
        assert (a & b) == And(a, b) and (a | b) == Or(a, b) and ~a == Not(a)
        assert all_of(a, b, c) == And(And(a, b), c) and any_of(a, b, c) == Or(Or(a, b), c)
        with pytest.raises(ValueError):
            all_of()
        with pytest.raises(ValueError):
            any_of()

    @given(
        hnp.arrays(np.float32, st.integers(0, 30), elements=st.floats(-1e3, 1e3, width=32)),
        st.data(),
    )
    def test_implies_is_max_of_negated_antecedent_and_consequent(
        self, a: np.ndarray, data: st.DataObject
    ) -> None:
        b = data.draw(hnp.arrays(np.float32, a.shape, elements=st.floats(-1e3, 1e3, width=32)))
        got = robustness(Implies(Atom("a"), Atom("b")), {"a": a, "b": b})
        assert np.array_equal(got, np.maximum(-a, b))
        assert np.array_equal(got, robustness(Or(Not(Atom("a")), Atom("b")), {"a": a, "b": b}))


class TestTemporal:
    def test_once_matches_the_definition_with_a_cut_window(self) -> None:
        x = [3.0, -1.0, 4.0, 1.0, -5.0, 9.0, 2.0, -6.0]
        for a, b in [(0, 0), (0, 2), (1, 3), (2, 2), (3, 5), (0, 20)]:
            got = robustness(Once(Atom("x"), a, b), {"x": sig(*x)})
            assert got.tolist() == naive_once(x, a, b), (a, b)

    def test_hist_matches_the_definition_with_a_cut_window(self) -> None:
        x = [3.0, -1.0, 4.0, 1.0, -5.0, 9.0, 2.0, -6.0]
        for a, b in [(0, 0), (0, 2), (1, 3), (2, 2), (3, 5), (0, 20)]:
            got = robustness(Hist(Atom("x"), a, b), {"x": sig(*x)})
            assert got.tolist() == naive_hist(x, a, b), (a, b)

    def test_window_with_no_quarter_is_minus_inf_for_once_and_plus_inf_for_hist(self) -> None:
        s = {"x": sig(5, 5, 5, 5)}
        assert robustness(Once(Atom("x"), 2, 3), s).tolist() == [-INF, -INF, 5, 5]
        assert robustness(Hist(Atom("x"), 2, 3), s).tolist() == [INF, INF, 5, 5]

    def test_window_bounds_are_inclusive_at_both_ends(self) -> None:
        # a single spike at quarter 3 is seen by Once[1,2] at quarters 4 and 5 only
        s = {"x": sig(-1, -1, -1, 7, -1, -1, -1, -1)}
        assert robustness(Once(Atom("x"), 1, 2), s).tolist() == [
            -INF, -1, -1, -1, 7, 7, -1, -1,
        ]  # fmt: skip

    def test_always_is_the_running_minimum_from_the_start(self) -> None:
        s = {"x": sig(4, 2, 5, -1, 3, 8)}
        assert robustness(Always(Atom("x")), s).tolist() == [4, 2, 2, -1, -1, -1]
        assert min_robustness(Always(Atom("x")), s) == -1.0

    def test_nested_operators(self) -> None:
        x = [1.0, -2.0, 3.0, -4.0, 5.0, 6.0]
        s = {"x": sig(*x)}
        inner = naive_hist(x, 0, 1)
        expected = naive_once(inner, 1, 2)
        got = robustness(Once(Hist(Atom("x"), 0, 1), 1, 2), s)
        assert got.tolist() == expected

    def test_infinities_never_turn_into_nan(self) -> None:
        s = {"a": sig(1, 1, 1), "b": sig(-1, -1, -1)}
        f = Always(Implies(Once(Atom("a"), 2, 2), Not(Hist(Atom("b"), 1, 2))))
        out = robustness(f, s)
        assert not np.isnan(out).any()

    def test_empty_trace(self) -> None:
        s = {"x": sig()}
        assert robustness(Always(Atom("x")), s).shape == (0,)
        assert min_robustness(Always(Atom("x")), s) == INF

    @pytest.mark.parametrize("make", [Once, Hist])
    @pytest.mark.parametrize("a, b", [(0, 0), (0, 3), (2, 5)])
    def test_windows_over_an_empty_trace(self, make, a: int, b: int) -> None:
        """Regression: sliding_window_view raised when the window was wider than the padding."""
        out = robustness(make(Atom("x"), a, b), {"x": sig()})
        assert out.shape == (0,) and out.dtype == np.float32

    def test_long_window_equals_a_naive_scan(self) -> None:
        rng = np.random.default_rng(5)
        x = rng.normal(size=400).astype(np.float32)
        got = robustness(Once(Atom("x"), 3, 150), {"x": x})
        assert got.tolist() == naive_once(x.tolist(), 3, 150)


class TestFormulaObjects:
    def test_atoms_and_lookback(self) -> None:
        f = Always(Implies(Once(Atom("a"), 1, 4), Hist(And(Atom("b"), Atom("c")), 0, 2)))
        assert f.atoms() == {"a", "b", "c"}
        assert f.lookback() == 4

    def test_structural_equality_and_hash(self) -> None:
        assert Once(Atom("a"), 1, 2) == Once(Atom("a"), 1, 2)
        assert Once(Atom("a"), 1, 2) != Once(Atom("a"), 1, 3)
        assert len({Once(Atom("a"), 1, 2), Once(Atom("a"), 1, 2)}) == 1

    def test_string_form(self) -> None:
        assert str(Always(Implies(Atom("a"), Once(Atom("b"), 0, 2)))) == "G((a -> O[0,2](b)))"

    @pytest.mark.parametrize("a, b", [(-1, 2), (3, 2), (0, -1)])
    def test_bad_windows_are_refused(self, a: int, b: int) -> None:
        with pytest.raises(ValueError, match="0 <= a <= b"):
            Once(Atom("x"), a, b)
        with pytest.raises(ValueError, match="0 <= a <= b"):
            Hist(Atom("x"), a, b)

    def test_non_integer_windows_are_refused(self) -> None:
        with pytest.raises(TypeError):
            Once(Atom("x"), 0.5, 2)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            Hist(Atom("x"), 0, True)

    def test_empty_atom_name_is_refused(self) -> None:
        with pytest.raises(ValueError):
            Atom("")


class TestSignalChecks:
    def test_missing_signal_is_named(self) -> None:
        with pytest.raises(KeyError, match="'b'"):
            robustness(And(Atom("a"), Atom("b")), {"a": sig(1)})

    def test_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="differ in length"):
            robustness(And(Atom("a"), Atom("b")), {"a": sig(1, 2), "b": sig(1)})

    def test_nan_is_refused(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            robustness(Atom("a"), {"a": sig(1, float("nan"))})

    def test_two_dimensional_signal_is_refused(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            robustness(Atom("a"), {"a": np.zeros((2, 2), np.float32)})

    def test_not_a_formula(self) -> None:
        with pytest.raises(TypeError):
            robustness("a", {"a": sig(1)})  # type: ignore[arg-type]
