import dataclasses
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from closed_loop.core.types import (
    BIN_EDGES_DAYS,
    FORCED_CMDS,
    N_BINS,
    Cmd,
    Command,
    Fidelity,
    Op,
    Permit,
    PlanCmd,
    bin_of_rul,
    bins_of_interval,
)


class TestFrozenVocabulary:
    """The enums are frozen interface: a rename or reorder must turn a test red."""

    def test_fidelity(self):
        assert [(f.name, f.value) for f in Fidelity] == [
            ("F0", "F0"),
            ("F1", "F1"),
            ("F1D", "F1+D"),
            ("F2", "F2"),
        ]

    def test_op_order_is_the_trace_encoding(self):
        assert [o.value for o in Op] == ["RUN", "STOP", "LOCKED", "ISO", "WORK", "FAIL"]

    def test_permit_order_is_the_trace_encoding(self):
        assert [p.value for p in Permit] == ["NONE", "VALID", "REVAL"]

    def test_cmd(self):
        assert [c.value for c in Cmd] == [
            "NOOP", "START", "STOP", "LOCK", "PROVE", "ISSUE", "REVAL", "BEGIN", "TEST",
            "ENTER", "EXIT", "COMPLETE", "CLOSE", "UNISO", "DERATE",
        ]  # fmt: skip

    def test_plan_cmd(self):
        assert [c.value for c in PlanCmd] == ["INSPECT", "ORDER", "RAISE_WO"]

    def test_cmd_and_plan_cmd_values_never_collide(self):
        # Command.cmd is Cmd | PlanCmd and is told apart by value in logs and tables.
        assert {c.value for c in Cmd}.isdisjoint({c.value for c in PlanCmd})

    def test_names_equal_values(self):
        for enum in (Fidelity, Op, Permit, Cmd, PlanCmd):
            for member in enum:
                if enum is not Fidelity:
                    assert member.name == member.value

    def test_forced_commands_are_stop_and_exit(self):
        assert isinstance(FORCED_CMDS, frozenset)
        assert FORCED_CMDS == {Cmd.STOP, Cmd.EXIT}

    def test_bins(self):
        assert N_BINS == 4
        assert BIN_EDGES_DAYS == (30.0, 10.0, 3.0)


class TestCommand:
    def test_defaults(self):
        c = Command("pump-a", Cmd.STOP)
        assert (c.asset_id, c.cmd, c.arg, c.source) == ("pump-a", Cmd.STOP, None, "")

    @pytest.mark.parametrize("bad", ["BEGIN", "STOP", "INSPECT", "", None, 3, b"STOP", ("STOP",)])
    def test_a_command_that_is_neither_a_cmd_nor_a_plancmd_is_refused(self, bad):
        # P12g, G6: both are StrEnums, so a plain string equals a member and the plant would run
        # it, while the shield's isinstance test would not gate it
        with pytest.raises(TypeError, match="Cmd or a PlanCmd"):
            Command("pump-a", bad)

    @pytest.mark.parametrize("good", [*Cmd, *PlanCmd])
    def test_every_member_of_either_enum_is_accepted(self, good):
        assert Command("pump-a", good).cmd is good

    def test_replace_goes_through_the_same_check(self):
        c = Command("pump-a", Cmd.STOP)
        with pytest.raises(TypeError):
            dataclasses.replace(c, cmd="BEGIN")

    def test_frozen(self):
        c = Command("pump-a", Cmd.STOP)
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.arg = "x"  # type: ignore[misc]

    def test_value_equality_and_hash(self):
        a = Command("p", PlanCmd.RAISE_WO, arg="bearing", source="B3")
        b = Command("p", PlanCmd.RAISE_WO, arg="bearing", source="B3")
        assert a == b and hash(a) == hash(b)
        assert a != Command("p", PlanCmd.RAISE_WO, arg="liner", source="B3")
        assert len({a, b}) == 1

    def test_fields_in_interface_order(self):
        assert [f.name for f in dataclasses.fields(Command)] == [
            "asset_id", "cmd", "arg", "source",
        ]  # fmt: skip


class TestBinOfRul:
    @pytest.mark.parametrize(
        ("rul", "expected"),
        [
            (1e9, 0),
            (math.inf, 0),
            (30.0001, 0),
            (30.0, 1),  # "> 30" is strict
            (29.9999, 1),
            (10.0001, 1),
            (10.0, 2),  # on two bins in the text; the more critical one
            (9.9999, 2),
            (3.0, 2),  # "< 3" is strict
            (2.9999, 3),
            (0.0001, 3),
            (0.0, 3),
            (-0.0, 3),
            (-1.0, 3),
            (-1e9, 3),
            (-math.inf, 3),
        ],
    )
    def test_edges(self, rul, expected):
        assert bin_of_rul(rul) == expected

    def test_int_input(self):
        assert [bin_of_rul(x) for x in (31, 30, 11, 10, 4, 3, 2, 0, -3)] == [
            0, 1, 1, 2, 2, 2, 3, 3, 3,
        ]  # fmt: skip

    def test_nan_is_refused(self):
        with pytest.raises(ValueError, match="NaN"):
            bin_of_rul(math.nan)

    @given(st.floats(allow_nan=False), st.floats(allow_nan=False))
    def test_monotone_non_increasing_in_rul(self, a, b):
        lo, hi = sorted((a, b))
        assert bin_of_rul(lo) >= bin_of_rul(hi)

    @given(st.floats(allow_nan=False))
    def test_range(self, rul):
        assert 0 <= bin_of_rul(rul) < N_BINS


class TestBinsOfInterval:
    @pytest.mark.parametrize(
        ("lo", "hi", "expected"),
        [
            (31.0, 40.0, (0, 0)),
            (30.0, 30.0, (1, 1)),
            (10.0, 10.0, (2, 2)),
            (3.0, 3.0, (2, 2)),
            (5.0, 8.0, (2, 2)),
            (2.0, 12.0, (1, 3)),
            (-5.0, 100.0, (0, 3)),  # wider than the whole range
            (-5.0, -1.0, (3, 3)),  # entirely below failure
            (-5.0, 0.0, (3, 3)),
            (-1.0, 2.9, (3, 3)),
            (-1.0, 3.0, (2, 3)),
            (10.0, 30.0, (1, 2)),  # 10 is bin 2, 30 is bin 1
            (0.0, math.inf, (0, 3)),
            (30.5, math.inf, (0, 0)),
        ],
    )
    def test_cases(self, lo, hi, expected):
        assert bins_of_interval(lo, hi) == expected

    def test_empty_interval_is_refused(self):
        with pytest.raises(ValueError, match="empty interval"):
            bins_of_interval(5.0, 4.0)

    @pytest.mark.parametrize(("lo", "hi"), [(math.nan, 1.0), (1.0, math.nan)])
    def test_nan_is_refused(self, lo, hi):
        with pytest.raises(ValueError, match="NaN"):
            bins_of_interval(lo, hi)

    @given(
        st.floats(min_value=-100, max_value=200),
        st.floats(min_value=0, max_value=100),
        st.floats(min_value=0, max_value=1),
    )
    def test_touches_exactly_the_bins_of_its_points(self, lo, width, frac):
        hi = lo + width
        low_bin, high_bin = bins_of_interval(lo, hi)
        assert low_bin <= high_bin
        point = min(max(lo + frac * width, lo), hi)
        assert low_bin <= bin_of_rul(point) <= high_bin
        # the end points are touched, so the range is tight
        assert bin_of_rul(hi) == low_bin
        assert bin_of_rul(lo) == high_bin
