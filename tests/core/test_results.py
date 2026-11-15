import dataclasses
import json
import math

import numpy as np
import pytest

from closed_loop.core.results import ResultRecord
from closed_loop.core.types import Fidelity


def make(**overrides):
    fields = {
        "name": "M-VIOL",
        "value": 0.0,
        "n": 500,
        "fidelity": Fidelity.F1,
        "measured_on": "held-out seeds, nominal condition, 500 episodes",
        "script": "closed_loop.eval.report",
    }
    fields.update(overrides)
    return ResultRecord(**fields)


class TestValidation:
    @pytest.mark.parametrize("bad", ["", "   ", "\t\n"])
    def test_empty_measured_on_is_refused(self, bad):
        with pytest.raises(ValueError, match="measured_on"):
            make(measured_on=bad)

    @pytest.mark.parametrize("bad", ["", "  "])
    def test_empty_script_is_refused(self, bad):
        with pytest.raises(ValueError, match="script"):
            make(script=bad)

    @pytest.mark.parametrize("bad", [math.nan, np.nan, np.float32("nan"), float("NaN")])
    def test_nan_value_is_refused(self, bad):
        with pytest.raises(ValueError, match="NaN"):
            make(value=bad)

    def test_empty_name_is_refused(self):
        with pytest.raises(ValueError, match="name"):
            make(name="")

    def test_negative_n_is_refused(self):
        with pytest.raises(ValueError, match="negative"):
            make(n=-1)

    def test_fractional_n_is_refused(self):
        with pytest.raises(TypeError):
            make(n=2.5)

    @pytest.mark.parametrize("field", ["ci_lo", "ci_hi"])
    def test_nan_interval_bound_is_refused(self, field):
        with pytest.raises(ValueError, match="NaN"):
            make(**{field: math.nan})

    def test_inverted_interval_is_refused(self):
        with pytest.raises(ValueError, match="ci_lo"):
            make(ci_lo=0.5, ci_hi=0.1)

    def test_unknown_fidelity_is_refused(self):
        with pytest.raises(ValueError):
            make(fidelity="F9")

    def test_missing_required_fields(self):
        with pytest.raises(TypeError):
            ResultRecord(name="x", value=1.0, n=1)  # type: ignore[call-arg]


class TestAcceptedValues:
    def test_minimal_record(self):
        r = make()
        assert r.ci_lo is None and r.ci_hi is None and r.ci_kind == "" and r.extra == {}

    def test_zero_and_infinite_values_are_legitimate(self):
        # a detection delay that never happened is +inf; zero violations is 0
        assert make(value=0.0).value == 0.0
        assert make(value=math.inf).value == math.inf

    def test_zero_n(self):
        assert make(n=0).n == 0

    def test_interval_fields(self):
        r = make(value=0.01, ci_lo=0.0, ci_hi=0.02, ci_kind="confseq-0.05")
        assert (r.ci_lo, r.ci_hi, r.ci_kind) == (0.0, 0.02, "confseq-0.05")

    def test_half_open_interval_is_allowed(self):
        r = make(ci_hi=0.0015, ci_kind="zero-event-0.05")
        assert r.ci_lo is None and r.ci_hi == 0.0015

    def test_string_fidelity_is_coerced(self):
        assert make(fidelity="F1+D").fidelity is Fidelity.F1D

    def test_numpy_scalars_become_python_numbers(self):
        r = make(
            value=np.float32(0.25), n=np.int64(40), ci_lo=np.float64(0.1), ci_hi=np.float32(0.5)
        )
        assert type(r.value) is float and type(r.n) is int
        assert type(r.ci_lo) is float and type(r.ci_hi) is float
        json.dumps(r.to_dict())  # must not raise


class TestImmutability:
    def test_frozen(self):
        r = make()
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.value = 1.0  # type: ignore[misc]

    def test_extra_is_copied(self):
        source = {"k": 3, "label": "a"}
        r = make(extra=source)
        source["k"] = 99
        assert r.extra == {"k": 3, "label": "a"}

    def test_default_extra_is_not_shared(self):
        a, b = make(), make()
        assert a.extra is not b.extra

    def test_value_equality(self):
        assert make() == make()
        assert make() != make(value=0.1)


class TestSerialisation:
    def test_round_trip_through_json(self):
        r = make(
            value=0.0125,
            ci_lo=0.001,
            ci_hi=0.03,
            ci_kind="confseq-0.05",
            fidelity=Fidelity.F1D,
            extra={"k": 3, "kind": "wilson", "rate": 0.5},
        )
        back = ResultRecord.from_dict(json.loads(json.dumps(r.to_dict())))
        assert back == r
        assert back.fidelity is Fidelity.F1D

    def test_dict_keys_are_the_field_names(self):
        assert set(make().to_dict()) == {f.name for f in dataclasses.fields(ResultRecord)}

    def test_from_dict_applies_validation(self):
        d = make().to_dict()
        d["measured_on"] = ""
        with pytest.raises(ValueError, match="measured_on"):
            ResultRecord.from_dict(d)


def test_field_order_is_the_interface_order():
    assert [f.name for f in dataclasses.fields(ResultRecord)] == [
        "name", "value", "n", "fidelity", "measured_on", "script",
        "ci_lo", "ci_hi", "ci_kind", "extra",
    ]  # fmt: skip
