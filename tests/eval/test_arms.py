"""``eval.arms``: the registry is the pre-registration's arm table, as data, and it pickles."""

from __future__ import annotations

import dataclasses
import functools
import inspect
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from closed_loop.eval import arms as arms_module
from closed_loop.eval import tuned
from closed_loop.eval.arms import (
    ARMS,
    CARELESS_P_SKIP,
    CONDITIONS,
    MONITOR_MODE,
    PRIMARY_VARIANT,
    Arm,
    NoInspectPlanner,
    arm,
    arms_for,
    make_bound_rule,
    make_calendar,
    make_careless,
    make_fixed_limit,
    make_gullible,
    make_planner,
    make_run_to_failure,
)
from closed_loop.eval.context import PreparedContext, use_context
from closed_loop.eval.tuned import use_tuned
from closed_loop.plant import make_plant
from closed_loop.proposers import BoundRule, Careless, Gullible
from closed_loop.proposers.baselines import Calendar, FixedLimit, RunToFailure

BOTH = ("nominal", "shifted")
NOMINAL = ("nominal",)

# Prereg section 4 (and 8), one row per arm: shield, monitor, interlock, perception, conditions,
# plus the fields that differ from the default.
TABLE: dict[str, tuple[bool, bool, bool, str, tuple[str, ...], dict[str, object]]] = {
    "B0": (False, False, False, "none", BOTH, {}),
    "B1": (False, False, False, "none", BOTH, {}),
    "B2": (False, False, False, "none", BOTH, {}),
    **{
        f"B3-V{v}": (False, False, False, f"V{v}", BOTH, {})
        for v in ("1", "2", "3", "4", "5", "6", "7", "2L", "3L", "7L")
    },
    "B3-V4-audit": (False, True, False, "V4", BOTH, {"monitor_mode": "audit"}),
    # P12b: the "B3 with shield" row that C-PLAN names (prereg section 7) and section 4 omits
    "B3-S": (True, False, False, PRIMARY_VARIANT, BOTH, {}),
    "P0": (False, False, False, PRIMARY_VARIANT, BOTH, {}),
    "P1": (True, False, False, PRIMARY_VARIANT, BOTH, {}),
    "P2": (True, True, False, PRIMARY_VARIANT, BOTH, {"monitor_mode": MONITOR_MODE}),
    "P1-corr": (True, False, False, PRIMARY_VARIANT, BOTH, {"diagnoser": "corr"}),
    "P1-oracle": (True, False, False, PRIMARY_VARIANT, BOTH, {"diagnoser": "oracle"}),
    "P1-I": (False, False, True, PRIMARY_VARIANT, BOTH, {}),
    "P1-step": (True, False, False, PRIMARY_VARIANT, BOTH, {"shield_set": "step99"}),
    "P1-nopair": (True, False, False, PRIMARY_VARIANT, BOTH, {"operator_pair": False}),
    "H-C0": (False, False, False, "V3", NOMINAL, {}),
    "H-C1": (True, False, False, "V3", NOMINAL, {}),
    "H-C-I": (False, False, True, "V3", NOMINAL, {}),
    "H-G0": (False, False, False, "V3", NOMINAL, {"screen": False}),
    "H-G1": (True, False, False, "V3", NOMINAL, {"screen": False}),
    "H-G-I": (False, False, True, "V3", NOMINAL, {"screen": False}),
    "H-G2": (False, False, False, "V3", NOMINAL, {"screen": True}),
    "S-stale": (True, False, False, "V3", NOMINAL, {"spec": "stale"}),
    "S-clause": (True, False, False, "V3", NOMINAL, {"spec": "clause"}),
    "P0-noinspect": (False, False, False, PRIMARY_VARIANT, BOTH, {}),
    "P0-mis10": (False, False, False, PRIMARY_VARIANT, BOTH, {"planner_mismatch": 0.10}),
    "P0-mis25": (False, False, False, PRIMARY_VARIANT, BOTH, {"planner_mismatch": 0.25}),
}


class TestRegistry:
    def test_it_holds_exactly_the_arms_of_the_preregistration(self) -> None:
        assert set(ARMS) == set(TABLE)
        assert len(ARMS) == 35

    def test_the_names_the_brief_lists_are_all_there(self) -> None:
        listed = (
            "B0 B1 B2 B3-V1 B3-V2 B3-V3 B3-V4 B3-V5 B3-V6 B3-V7 B3-V2L B3-V3L B3-V7L "
            "P0 P1 P2 P1-corr P1-oracle P1-I P1-step "
            "H-C0 H-C1 H-C-I H-G0 H-G1 H-G-I H-G2 S-stale S-clause"
        ).split()
        assert [n for n in listed if n not in ARMS] == []

    @pytest.mark.parametrize("name", sorted(TABLE))
    def test_each_arm_is_the_row_of_the_table(self, name: str) -> None:
        shield, monitor, interlock, perception, conditions, extra = TABLE[name]
        a = ARMS[name]
        assert a.name == name
        assert (a.shield, a.monitor, a.interlock) == (shield, monitor, interlock)
        assert a.perception == perception
        assert a.conditions == conditions
        defaults = Arm.__dataclass_fields__
        for field in (
            "screen",
            "spec",
            "planner_mismatch",
            "diagnoser",
            "shield_set",
            "operator_pair",
        ):
            expected = extra.get(field, defaults[field].default)
            assert getattr(a, field) == expected, (name, field)
        if "monitor_mode" in extra:
            assert a.monitor_mode == extra["monitor_mode"]

    def test_arm_looks_one_up_and_names_the_choices_when_it_cannot(self) -> None:
        assert arm("P1") is ARMS["P1"]
        with pytest.raises(KeyError, match="unknown arm"):
            arm("P9")

    def test_conditions_are_data(self) -> None:
        nominal = {a.name for a in arms_for("nominal")}
        shifted = {a.name for a in arms_for("shifted")}
        assert shifted < nominal
        assert nominal - shifted == {n for n, row in TABLE.items() if row[4] == NOMINAL}
        assert {"H-C1", "H-G2", "S-stale", "H-G-I", "H-C-I"} <= nominal - shifted
        assert {"P1-I", "B3-V1", "B0"} <= shifted
        with pytest.raises(ValueError):
            arms_for("windy")

    def test_the_hostile_and_spec_arms_use_v3(self) -> None:
        for name in ("H-C0", "H-C1", "H-C-I", "H-G0", "H-G1", "H-G-I", "H-G2", "S-stale"):
            assert ARMS[name].perception == "V3"

    def test_the_planner_arms_read_the_one_module_level_variant(self) -> None:
        planners = [a for n, a in ARMS.items() if n.startswith("P")]
        assert planners
        assert {a.perception for a in planners} == {arms_module.PRIMARY_VARIANT}

    def test_the_provisional_choices_are_marked_as_such_in_the_source(self) -> None:
        source = Path(arms_module.__file__).read_text(encoding="utf-8")
        for name in ("PRIMARY_VARIANT", "MONITOR_MODE"):
            before = source[: source.index(f"{name}: Final")]
            assert "PROVISIONAL" in before.rsplit("\n\n", 1)[-1], name

    def test_only_the_stale_arm_has_a_stale_spec(self) -> None:
        assert [n for n, a in ARMS.items() if a.spec == "stale"] == ["S-stale"]

    def test_the_registered_ablations_are_planner_arms_without_a_shield(self) -> None:
        assert ARMS["P0-mis10"].planner_mismatch == 0.10
        assert ARMS["P0-mis25"].planner_mismatch == 0.25
        assert ARMS["P0-noinspect"].planner_mismatch == 0.0
        assert not any(ARMS[n].shield for n in ("P0-mis10", "P0-mis25", "P0-noinspect"))

    def test_the_tuned_names_each_arm_reads(self) -> None:
        assert ARMS["B0"].tuned == ()
        assert ARMS["B1"].tuned == ("b1_fraction",)
        assert ARMS["B2"].tuned == ("b2_limit",)
        assert ARMS["B3-V3"].tuned == ("k_b3",) and ARMS["H-C1"].tuned == ("k_b3",)
        # P12b: the planner arms name the primary variant they read as well
        assert ARMS["P1"].tuned == ("delta_plan", "primary_variant")


class TestValidation:
    def make(self, **kw: object) -> Arm:
        base: dict[str, Any] = {
            "name": "X",
            "proposer": make_run_to_failure,
            "shield": False,
            "monitor": False,
        }
        return Arm(**{**base, **kw})

    @pytest.mark.parametrize(
        "kw",
        [
            {"name": " "},
            {"perception": "V9"},
            {"spec": "rev1"},
            {"shield_set": "step95"},
            {"diagnoser": "magic"},
            {"shield": True, "interlock": True, "perception": "V3"},
            {"monitor_mode": "loud"},
            {"conditions": ()},
            {"conditions": ("nominal", "windy")},
            {"planner_mismatch": -0.1},
            {"shield": True, "perception": "none"},  # a shield needs a perception layer
            {"monitor": True, "perception": "none"},
        ],
    )
    def test_nonsense_is_refused(self, kw: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            self.make(**kw)

    def test_the_defaults_are_the_frozen_ones_of_0002_and_0003(self) -> None:
        a = self.make()
        assert (a.perception, a.screen, a.spec, a.planner_mismatch) == ("V3", False, "clause", 0.0)
        assert (a.diagnoser, a.interlock, a.shield_set) == ("scm", False, "life95")

    def test_an_arm_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            ARMS["P1"].shield = False  # type: ignore[misc]


class TestPickling:
    @pytest.mark.parametrize("name", sorted(TABLE))
    def test_every_arm_round_trips_through_pickle(self, name: str) -> None:
        a = ARMS[name]
        b = pickle.loads(pickle.dumps(a))
        assert b.name == a.name and b.conditions == a.conditions
        assert (b.shield, b.monitor, b.perception, b.spec) == (
            a.shield,
            a.monitor,
            a.perception,
            a.spec,
        )
        pa, pb = a.proposer, b.proposer
        if isinstance(pa, functools.partial):
            assert isinstance(pb, functools.partial)
            assert (pb.func, pb.args, pb.keywords) == (pa.func, pa.args, pa.keywords)
        else:
            assert pb is pa

    @pytest.mark.parametrize("name", sorted(TABLE))
    def test_no_proposer_is_a_lambda_or_a_closure(self, name: str) -> None:
        p = ARMS[name].proposer
        func = p.func if isinstance(p, functools.partial) else p
        assert inspect.isfunction(func) or inspect.isclass(func)
        assert "<lambda>" not in func.__qualname__ and "<locals>" not in func.__qualname__
        assert func.__module__ == "closed_loop.eval.arms"

    def test_the_whole_registry_pickles_in_one_go(self) -> None:
        assert set(pickle.loads(pickle.dumps(dict(ARMS)))) == set(ARMS)


class TestFactories:
    def test_the_rule_arms_build_their_baseline(self, small_context: PreparedContext) -> None:
        with use_context(small_context):
            assert isinstance(make_run_to_failure(), RunToFailure)
            assert isinstance(make_calendar(), Calendar)
            assert isinstance(make_fixed_limit(), FixedLimit)

    def test_the_tuned_numbers_reach_the_baselines(self, small_context: PreparedContext) -> None:
        with use_context(small_context), use_tuned(k_b3=4.5, b2_limit=6.25, b1_fraction=0.3):
            b3 = make_bound_rule("scm")
            assert isinstance(b3, BoundRule) and b3.k == 4.5
            fixed = make_fixed_limit()
            assert isinstance(fixed, FixedLimit) and fixed.limit == 6.25
            cal = make_calendar()
            assert isinstance(cal, Calendar) and cal.fraction == 0.3

    def test_b3_gets_the_fitted_diagnoser_of_the_context(
        self, small_context: PreparedContext
    ) -> None:
        with use_context(small_context):
            b3 = make_bound_rule("scm")
            assert isinstance(b3, BoundRule)
            assert b3.diagnoser is small_context.diagnosers["scm"]
            none = make_bound_rule(None)
            assert isinstance(none, BoundRule) and none.diagnoser is None

    def test_the_hostile_arms_wrap_b3(self, small_context: PreparedContext) -> None:
        with use_context(small_context):
            c = make_careless()
            assert isinstance(c, Careless) and c.p_skip == CARELESS_P_SKIP
            assert isinstance(c.inner, BoundRule)
            g_on, g_off = make_gullible(True), make_gullible(False)
            assert isinstance(g_on, Gullible) and g_on.screen is True
            assert isinstance(g_off, Gullible) and g_off.screen is False

    def test_the_planner_gets_the_tuned_delta_and_the_mismatch(
        self, small_context: PreparedContext
    ) -> None:
        with use_context(small_context), use_tuned(delta_plan=0.04):
            p: Any = make_planner("scm", 0.25)
            assert p.delta_plan == 0.04 and p.mismatch == 0.25

    def test_the_oracle_planner_gets_an_unbound_oracle(
        self, small_context: PreparedContext
    ) -> None:
        with use_context(small_context):
            a: Any = make_planner("oracle")
            b: Any = make_planner("oracle")
            assert a.diagnoser is not b.diagnoser
            assert a.diagnoser.name == "oracle"

    def test_the_corr_planner_shares_the_fitted_correlational_model(
        self, small_context: PreparedContext
    ) -> None:
        with use_context(small_context):
            p: Any = make_planner("corr")
            assert p.diagnoser is small_context.diagnosers["corr"]

    def test_the_no_inspect_planner_has_no_inspection_slot(
        self, small_context: PreparedContext
    ) -> None:
        plant = make_plant(small_context.params)
        obs = plant.reset(1100)
        with use_context(small_context):
            normal: Any = make_planner("scm", 0.0, True)
            blind: Any = make_planner("scm", 0.0, False)
            assert isinstance(blind, NoInspectPlanner) and not isinstance(normal, NoInspectPlanner)
            assert normal._constraints(obs).inspect_slots > 0
            assert blind._constraints(obs).inspect_slots == 0

    def test_a_registered_arm_builds_its_proposer(self, small_context: PreparedContext) -> None:
        with use_context(small_context):
            for name in ("B0", "B3-V3", "H-G2", "P1-oracle", "P0-mis25", "P0-noinspect"):
                assert ARMS[name].proposer().name

    def test_the_conditions_are_the_two_of_the_preregistration(self) -> None:
        assert CONDITIONS == ("nominal", "shifted")


class TestHygiene:
    def test_importing_the_eval_modules_pulls_in_no_sibling_and_no_pandas(self) -> None:
        """0002 rule 4: the siblings, pandas and scikit-learn stay out of F1."""
        code = (
            "import sys, closed_loop.eval.arms, closed_loop.eval.episode, closed_loop.eval.metrics;"
            "bad = [m for m in ('asset_health', 'maintenance_agent', 'plant_copilot', 'pandas',"
            " 'sklearn') if m in sys.modules]; print(bad)"
        )
        root = Path(arms_module.__file__).resolve().parents[2]
        out = subprocess.run(
            [sys.executable, "-c", code], cwd=root, capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == "[]", out.stdout


class TestAddedByP12b:
    def test_the_planner_has_the_preregistered_number_of_particles(
        self, small_context: PreparedContext
    ) -> None:
        from closed_loop.eval.arms import PLANNER_PARTICLES

        assert PLANNER_PARTICLES == 256  # prereg section 2: N = 256, not the class default of 128
        with use_context(small_context):
            p: Any = make_planner("scm")
            assert p.n_particles == PLANNER_PARTICLES

    def test_b3_behind_the_shield_is_the_comparison_row_of_c_plan(self) -> None:
        a = ARMS["B3-S"]
        assert (a.shield, a.interlock, a.monitor) == (True, False, False)
        assert a.perception == PRIMARY_VARIANT and a.shield_set == "life95"
        assert a.conditions == BOTH and a.tuned == ("k_b3", "primary_variant")

    def test_the_primary_variant_is_read_from_the_tuned_values(self) -> None:
        assert PRIMARY_VARIANT == tuned.current().primary_variant

    def test_every_arm_that_uses_the_variant_says_so(self) -> None:
        for name in ("P0", "P1", "P2", "P1-corr", "P1-oracle", "P1-I", "P1-step", "B3-S"):
            assert "primary_variant" in ARMS[name].tuned, name
