import copy
import dataclasses

import numpy as np
import pytest

from closed_loop.core.trace import (
    GAS_AGE_NEVER_MIN,
    PERMIT_MARGIN_NONE_MIN,
    Trace,
    op_index,
    permit_index,
)
from closed_loop.core.types import N_BINS, Cmd, Command, Op, Permit

PER_ASSET_DTYPES = {
    "op": np.int8,
    "permit": np.int8,
    "occupied": np.bool_,
    "gas_age_min": np.float32,
    "alarm": np.bool_,
    "iso_proven": np.bool_,
    "permit_margin_min": np.float32,
    "true_rul_days": np.float32,
    "health_bin": np.int8,
    "set_lo": np.int8,
    "set_hi": np.int8,
    "uncontrolled_failure": np.bool_,
}
IDS = ("crusher", "pump-a", "pump-b")


def arrays_for(n_q, ids=IDS):
    """A hand-built set of arrays, independent of Trace.empty."""
    fields = {
        name: {a: np.zeros(n_q, dtype=dt) for a in ids} for name, dt in PER_ASSET_DTYPES.items()
    }
    return {**fields, "throughput": np.ones(n_q, dtype=np.float32)}


class TestConstruction:
    def test_empty_trace(self):
        t = Trace(n_q=0, asset_ids=())
        assert t.n_q == 0 and t.asset_ids == () and t.commands == []
        assert t.throughput.shape == (0,) and t.throughput.dtype == np.float32

    def test_empty_trace_with_assets_but_no_quarters(self):
        t = Trace.empty(0, IDS)
        assert all(t.op[a].shape == (0,) for a in IDS)

    def test_from_arrays(self):
        t = Trace(n_q=5, asset_ids=IDS, **arrays_for(5))
        assert t.n_q == 5
        assert set(t.op) == set(IDS)
        assert t.throughput.tolist() == [1.0] * 5
        assert t.commands == []

    def test_from_arrays_with_commands(self):
        cmd = Command("crusher", Cmd.STOP, source="B3")
        t = Trace(n_q=5, asset_ids=IDS, commands=[(2, cmd, True, False)], **arrays_for(5))
        assert t.commands == [(2, cmd, True, False)]

    def test_fields_in_interface_order(self):
        assert [f.name for f in dataclasses.fields(Trace)] == [
            "n_q", "asset_ids", "op", "permit", "occupied", "gas_age_min", "alarm", "iso_proven",
            "permit_margin_min", "true_rul_days", "health_bin", "set_lo", "set_hi", "throughput",
            "uncontrolled_failure", "commands",
        ]  # fmt: skip

    def test_a_trace_with_assets_needs_arrays(self):
        with pytest.raises(ValueError, match="missing assets"):
            Trace(n_q=3, asset_ids=("crusher",))


class TestEmptyFactory:
    def test_shapes_and_dtypes(self):
        t = Trace.empty(96, IDS)
        for name, dtype in PER_ASSET_DTYPES.items():
            for a in IDS:
                arr = getattr(t, name)[a]
                assert arr.shape == (96,), name
                assert arr.dtype == np.dtype(dtype), name
        assert t.throughput.shape == (96,) and t.throughput.dtype == np.float32

    def test_neutral_values(self):
        t = Trace.empty(4, ("crusher",))
        a = "crusher"
        assert (t.op[a] == op_index(Op.RUN)).all()
        assert (t.permit[a] == permit_index(Permit.NONE)).all()
        assert not t.occupied[a].any() and not t.alarm[a].any() and not t.iso_proven[a].any()
        assert (t.gas_age_min[a] == GAS_AGE_NEVER_MIN).all()
        assert (t.permit_margin_min[a] == PERMIT_MARGIN_NONE_MIN).all()
        assert np.isnan(t.true_rul_days[a]).all()
        assert (t.health_bin[a] == 0).all()
        assert (t.set_lo[a] == 0).all() and (t.set_hi[a] == N_BINS - 1).all()
        assert not t.uncontrolled_failure[a].any()
        assert (t.throughput == 0).all()

    def test_gas_age_never_is_large_and_finite(self):
        assert GAS_AGE_NEVER_MIN > 24 * 60 * 90  # more than the whole horizon, in minutes
        assert np.isfinite(np.float32(GAS_AGE_NEVER_MIN))

    def test_arrays_are_independent(self):
        t = Trace.empty(4, IDS)
        t.op["pump-a"][1] = op_index(Op.LOCKED)
        t.alarm["pump-a"][2] = True
        assert (t.op["pump-b"] == 0).all() and (t.op["crusher"] == 0).all()
        assert not t.alarm["pump-b"].any()

    def test_commands_list_is_not_shared(self):
        a, b = Trace.empty(2, ("x",)), Trace.empty(2, ("x",))
        a.commands.append((0, Command("x", Cmd.NOOP), True, False))
        assert b.commands == []

    def test_asset_ids_may_be_passed_as_a_list(self):
        assert Trace.empty(2, ["x", "y"]).asset_ids == ("x", "y")  # type: ignore[arg-type]

    def test_mutation_after_construction_is_visible(self):
        t = Trace.empty(3, ("x",))
        t.throughput[1] = 0.5
        t.op["x"][2] = op_index(Op.FAIL)
        assert t.throughput.tolist() == [0.0, 0.5, 0.0]
        assert t.op["x"].tolist() == [0, 0, 5]
        t.validate()


class TestEncoding:
    def test_indices_are_positions_in_the_enum_list(self):
        assert [op_index(o) for o in Op] == list(range(len(list(Op))))
        assert [permit_index(p) for p in Permit] == list(range(len(list(Permit))))
        assert op_index(Op.FAIL) == 5 and permit_index(Permit.REVAL) == 2

    def test_indices_fit_int8(self):
        assert max(op_index(o) for o in Op) < 127


class TestValidation:
    def test_wrong_length(self):
        arrays = arrays_for(5)
        arrays["alarm"]["pump-a"] = np.zeros(4, dtype=np.bool_)
        with pytest.raises(ValueError, match=r"alarm\['pump-a'\]"):
            Trace(n_q=5, asset_ids=IDS, **arrays)

    def test_wrong_throughput_length(self):
        arrays = arrays_for(5)
        arrays["throughput"] = np.ones(6, dtype=np.float32)
        with pytest.raises(ValueError, match="throughput"):
            Trace(n_q=5, asset_ids=IDS, **arrays)

    def test_two_dimensional_array(self):
        arrays = arrays_for(5)
        arrays["op"]["crusher"] = np.zeros((5, 1), dtype=np.int8)
        with pytest.raises(ValueError, match="shape"):
            Trace(n_q=5, asset_ids=IDS, **arrays)

    @pytest.mark.parametrize(
        ("field", "wrong"),
        [
            ("gas_age_min", np.float64),
            ("op", np.int64),
            ("alarm", np.int8),
            ("health_bin", np.uint8),
        ],
    )
    def test_wrong_dtype(self, field, wrong):
        arrays = arrays_for(5)
        arrays[field]["crusher"] = np.zeros(5, dtype=wrong)
        with pytest.raises(TypeError, match=field):
            Trace(n_q=5, asset_ids=IDS, **arrays)

    def test_wrong_throughput_dtype(self):
        arrays = arrays_for(5)
        arrays["throughput"] = np.ones(5, dtype=np.float64)
        with pytest.raises(TypeError, match="throughput"):
            Trace(n_q=5, asset_ids=IDS, **arrays)

    def test_not_an_array(self):
        arrays = arrays_for(5)
        arrays["occupied"]["crusher"] = [False] * 5
        with pytest.raises(TypeError, match="numpy array"):
            Trace(n_q=5, asset_ids=IDS, **arrays)

    def test_missing_asset_in_one_field(self):
        arrays = arrays_for(5)
        del arrays["set_hi"]["pump-b"]
        with pytest.raises(ValueError, match=r"set_hi.*missing assets \['pump-b'\]"):
            Trace(n_q=5, asset_ids=IDS, **arrays)

    def test_unexpected_asset_in_one_field(self):
        arrays = arrays_for(5)
        arrays["set_lo"]["ghost"] = np.zeros(5, dtype=np.int8)
        with pytest.raises(ValueError, match=r"unexpected \['ghost'\]"):
            Trace(n_q=5, asset_ids=IDS, **arrays)

    def test_duplicate_asset_ids(self):
        with pytest.raises(ValueError, match="duplicates"):
            Trace.empty(3, ("a", "a"))

    @pytest.mark.parametrize("n_q", [-1, 2.5, True])
    def test_bad_n_q(self, n_q):
        with pytest.raises(ValueError, match="n_q"):
            Trace(n_q=n_q, asset_ids=())

    def test_bad_command_entry(self):
        with pytest.raises(ValueError, match="commands"):
            Trace(n_q=0, asset_ids=(), commands=[(0, "STOP", True, False)])  # type: ignore[list-item]

    def test_validate_catches_an_in_place_replacement(self):
        t = Trace.empty(4, ("x",))
        t.op["x"] = np.zeros(3, dtype=np.int8)
        with pytest.raises(ValueError, match="shape"):
            t.validate()


class TestIdenticalTo:
    def make(self):
        t = Trace.empty(6, IDS)
        t.op["pump-a"][3] = op_index(Op.LOCKED)
        t.throughput[:] = [1, 1, 0.6, 0.6, 1, 1]
        t.true_rul_days["crusher"][:] = np.linspace(40, 0, 6, dtype=np.float32)
        t.commands.append((3, Command("pump-a", Cmd.LOCK, source="x"), True, False))
        return t

    def test_equal_copies(self):
        a, b = self.make(), self.make()
        assert a.identical_to(b) and b.identical_to(a)
        assert a.identical_to(copy.deepcopy(a))

    def test_nan_equals_nan(self):
        # true_rul_days is NaN in the neutral trace
        assert Trace.empty(3, ("x",)).identical_to(Trace.empty(3, ("x",)))

    @pytest.mark.parametrize(
        "edit",
        [
            lambda t: t.op["pump-b"].__setitem__(0, 1),
            lambda t: t.alarm["crusher"].__setitem__(5, True),
            lambda t: t.throughput.__setitem__(0, 0.999),
            lambda t: t.true_rul_days["crusher"].__setitem__(2, 7.0),
            lambda t: t.commands.append((4, Command("pump-a", Cmd.NOOP), True, False)),
            lambda t: t.commands.__setitem__(0, (3, t.commands[0][1], False, False)),
            lambda t: t.commands.__setitem__(0, (3, t.commands[0][1], True, True)),
        ],
    )
    def test_any_difference_is_seen(self, edit):
        a, b = self.make(), self.make()
        edit(b)
        assert not a.identical_to(b) and not b.identical_to(a)

    def test_negative_zero_is_different(self):
        a, b = Trace.empty(2, ("x",)), Trace.empty(2, ("x",))
        a.true_rul_days["x"][0] = 0.0
        b.true_rul_days["x"][0] = -0.0
        assert not a.identical_to(b)

    def test_different_assets_or_length(self):
        assert not Trace.empty(3, ("x",)).identical_to(Trace.empty(3, ("y",)))
        assert not Trace.empty(3, ("x",)).identical_to(Trace.empty(4, ("x",)))

    def test_equality_operator_is_identity(self):
        a = self.make()
        assert a == a
        assert a != self.make()  # no ambiguous-truth error from comparing arrays
