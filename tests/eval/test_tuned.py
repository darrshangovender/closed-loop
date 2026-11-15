"""``eval.tuned``: the tuned values, their provisional fallback and the file format."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from closed_loop.audit.prereg import file_sha256
from closed_loop.eval import tuned
from closed_loop.eval.tuned import (
    PROVISIONAL,
    TUNABLE,
    TunedError,
    load_tuned,
    use_tuned,
)

GOOD = {
    "k_b3": 9.0,
    "b2_limit": 6.0,
    "b1_fraction": 0.4,
    "delta_plan": 0.03,
    "tuned_on": "dev seeds 1000-1099, 100 episodes per cell",
    "script": "scripts/tune.py",
}


def write(tmp_path: Path, doc: dict[str, object]) -> Path:
    path = tmp_path / "tuned.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


class TestAbsentFile:
    def test_a_missing_file_gives_the_provisional_defaults_and_says_so(
        self, tmp_path: Path
    ) -> None:
        values = load_tuned(tmp_path / "nothing.json")
        assert values is PROVISIONAL
        assert values.provisional is True
        assert values.sha256 is None
        assert "provisional" in values.source

    def test_the_defaults_are_the_numbers_the_packets_used_on_dev(self) -> None:
        assert (PROVISIONAL.k_b3, PROVISIONAL.b2_limit) == (7.0, 5.5)
        assert (PROVISIONAL.b1_fraction, PROVISIONAL.delta_plan) == (0.5, 0.02)

    def test_the_repository_tuned_file_is_the_one_the_arms_read(self) -> None:
        # This test said "the repository has no tuned file yet" and asked to be retired when the
        # file appeared. It has: benchmarks/tuned.json is final. What stays true is that the arms
        # read that path, and that it is either absent (provisional) or a valid, hashed file.
        values = load_tuned(tuned.TUNED_PATH)
        if tuned.TUNED_PATH.exists():
            assert values.provisional is False and values.sha256 == file_sha256(tuned.TUNED_PATH)
            assert values.source == "benchmarks/tuned.json"  # relative: no machine path (P12g)
        else:
            assert values is PROVISIONAL


class TestPresentFile:
    def test_a_valid_file_is_read_and_hashed(self, tmp_path: Path) -> None:
        path = write(tmp_path, GOOD)
        values = load_tuned(path)
        assert values.provisional is False
        assert (values.k_b3, values.b2_limit, values.b1_fraction, values.delta_plan) == (
            9.0,
            6.0,
            0.4,
            0.03,
        )
        assert values.sha256 == file_sha256(path)
        assert values.tuned_on.startswith("dev seeds") and values.script == "scripts/tune.py"

    @pytest.mark.parametrize("missing", ["k_b3", "b2_limit", "b1_fraction", "delta_plan"])
    def test_a_missing_number_is_refused_not_defaulted(self, tmp_path: Path, missing: str) -> None:
        doc = {k: v for k, v in GOOD.items() if k != missing}
        with pytest.raises(TunedError):
            load_tuned(write(tmp_path, doc))

    def test_an_unknown_key_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(TunedError):
            load_tuned(write(tmp_path, {**GOOD, "k_b4": 3.0}))

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("k_b3", 0.0),
            ("k_b3", 120.0),
            ("delta_plan", 1.0),
            ("delta_plan", 0.0),
            ("b1_fraction", -1),
        ],
    )
    def test_an_absurd_number_is_refused(self, tmp_path: Path, key: str, value: float) -> None:
        with pytest.raises(TunedError):
            load_tuned(write(tmp_path, {**GOOD, key: value}))

    @pytest.mark.parametrize("key", ["tuned_on", "script"])
    def test_a_number_without_provenance_is_refused(self, tmp_path: Path, key: str) -> None:
        with pytest.raises(TunedError):
            load_tuned(write(tmp_path, {**GOOD, key: ""}))

    def test_a_file_that_is_not_json_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "tuned.json"
        path.write_text("{ not json", encoding="utf-8")
        with pytest.raises(TunedError):
            load_tuned(path)


class TestProvisionalNames:
    def test_provisional_values_are_listed_for_the_arm_that_uses_them(self) -> None:
        assert PROVISIONAL.provisional_names(("k_b3",)) == ("k_b3",)
        assert PROVISIONAL.provisional_names(()) == ()

    def test_tuned_values_are_never_listed(self, tmp_path: Path) -> None:
        assert load_tuned(write(tmp_path, GOOD)).provisional_names(("k_b3", "delta_plan")) == ()

    def test_a_name_that_is_not_tunable_is_an_error(self) -> None:
        with pytest.raises(TunedError):
            PROVISIONAL.provisional_names(("k_b9",))
        with pytest.raises(TunedError):
            PROVISIONAL.value("nope")

    def test_value_reads_each_tunable(self) -> None:
        assert [PROVISIONAL.value(n) for n in TUNABLE] == [7.0, 5.5, 0.5, 0.02]


class TestOverride:
    def test_use_tuned_changes_what_current_returns_and_restores_it(self) -> None:
        before = tuned.current()
        with use_tuned(k_b3=11.0) as inside:
            assert tuned.current().k_b3 == 11.0 == inside.k_b3
            assert tuned.current().b2_limit == before.b2_limit
        assert tuned.current() == before

    def test_use_tuned_restores_after_an_exception(self) -> None:
        before = tuned.current()
        with pytest.raises(RuntimeError), use_tuned(k_b3=3.0):
            raise RuntimeError("boom")
        assert tuned.current() == before


class TestPrimaryVariant:
    """P12b: the primary perception variant travels in tuned.json (optional, validated)."""

    def test_a_file_without_a_variant_leaves_the_provisional_one_and_says_so(
        self, tmp_path: Path
    ) -> None:
        values = load_tuned(write(tmp_path, GOOD))
        assert values.primary_variant == PROVISIONAL.primary_variant == "V3"
        assert values.variant_tuned is False
        assert values.provisional_names(("primary_variant", "k_b3")) == ("primary_variant",)

    def test_a_file_that_names_one_is_used_and_is_not_provisional(self, tmp_path: Path) -> None:
        values = load_tuned(write(tmp_path, {**GOOD, "primary_variant": "V5"}))
        assert values.primary_variant == "V5" and values.variant_tuned is True
        assert values.provisional_names(("primary_variant", "k_b3")) == ()

    def test_the_absent_file_makes_it_provisional_too(self) -> None:
        assert PROVISIONAL.provisional_names(("primary_variant",)) == ("primary_variant",)

    @pytest.mark.parametrize("bad", ["V9", "v3", "", "V1 "])
    def test_an_unknown_variant_is_refused(self, tmp_path: Path, bad: str) -> None:
        with pytest.raises(TunedError):
            load_tuned(write(tmp_path, {**GOOD, "primary_variant": bad}))

    def test_the_accepted_names_are_the_variants_of_the_perception_layer(self) -> None:
        from closed_loop.eval.tuned import _VARIANTS
        from closed_loop.perception.conformal import ALL_VARIANTS

        assert tuple(_VARIANTS) == tuple(ALL_VARIANTS)

    def test_use_tuned_can_change_the_variant(self) -> None:
        with use_tuned(primary_variant="V2L") as inside:
            assert inside.primary_variant == "V2L" and tuned.current().primary_variant == "V2L"


class TestMonitorMode:
    """0003 A15: the monitor mode of P2 and of C-MON travels in tuned.json, chosen on dev."""

    def test_a_file_without_a_mode_leaves_the_provisional_one_and_says_so(
        self, tmp_path: Path
    ) -> None:
        values = load_tuned(write(tmp_path, GOOD))
        assert values.monitor_mode == PROVISIONAL.monitor_mode == "proxy"
        assert values.mode_tuned is False
        assert values.provisional_names(("monitor_mode", "k_b3")) == ("monitor_mode",)

    def test_a_file_that_names_one_is_used_and_is_not_provisional(self, tmp_path: Path) -> None:
        values = load_tuned(write(tmp_path, {**GOOD, "monitor_mode": "inf"}))
        assert values.monitor_mode == "inf" and values.mode_tuned is True
        assert values.provisional_names(("monitor_mode", "k_b3")) == ()

    def test_the_absent_file_makes_it_provisional_too(self) -> None:
        assert PROVISIONAL.provisional_names(("monitor_mode",)) == ("monitor_mode",)

    @pytest.mark.parametrize("bad", ["audit", "Proxy", "", "loud"])
    def test_an_unknown_mode_is_refused(self, tmp_path: Path, bad: str) -> None:
        with pytest.raises(TunedError):
            load_tuned(write(tmp_path, {**GOOD, "monitor_mode": bad}))

    def test_the_accepted_modes_are_the_four_registered_ones(self) -> None:
        from closed_loop.eval.tuned import _MODES
        from closed_loop.perception.monitor import MODES

        assert tuple(_MODES) == tuple(MODES)

    def test_the_arm_that_acts_on_the_monitor_lists_the_mode_among_its_tuned_names(self) -> None:
        from closed_loop.eval.arms import ARMS

        assert "monitor_mode" in ARMS["P2"].tuned
        assert "monitor_mode" not in ARMS["P1"].tuned  # monitors are passive there

    def test_the_arm_registry_reads_the_mode_from_the_current_values(self) -> None:
        from closed_loop.eval import arms

        assert arms.MONITOR_MODE == tuned.current().monitor_mode
