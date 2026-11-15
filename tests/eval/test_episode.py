"""``eval.episode``: the loop wires the layers, fills the result, keeps the arms paired.

Every episode here runs on a development seed (1000 to 1999) with the small prepared context of
``conftest.py``; nothing touches a held-out or F2 seed.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from closed_loop.core.timegrid import DAY_Q
from closed_loop.core.trace import op_index
from closed_loop.core.types import Command, Fidelity, Op
from closed_loop.eval import episode as episode_module
from closed_loop.eval.arms import ARMS, PRIMARY_VARIANT, Arm
from closed_loop.eval.context import PreparedContext
from closed_loop.eval.episode import (
    EpisodeResult,
    run_episode,
    scenario_for,
    simulate_episode,
    summary,
)
from closed_loop.eval.tuned import PROVISIONAL, TunedValues, use_tuned
from closed_loop.perception.contract import health_set
from closed_loop.perception.labels import RulInterval
from closed_loop.plant.observe import simulate_lives
from closed_loop.plant.params import ASSET_IDS
from closed_loop.plant.plant import Observation
from closed_loop.plant.scenario import Scenario
from closed_loop.proposers.base import Proposer

# pinned before benchmarks/tuned.json existed: they run on the provisional defaults (conftest)
pytestmark = pytest.mark.usefixtures("provisional_defaults_module")

GOLDEN_DIR = Path(__file__).parent / "golden"
REGENERATE = "CLOSED_LOOP_REGENERATE_GOLDEN"
SEED = 1100  # a development seed outside the planner's world-model fit (1000-1019 here)


class QuietProposer:
    """A proposer that never issues a command: it cannot violate anything."""

    name = "quiet"

    def reset(self, seed: int) -> None:
        pass

    def propose(self, obs: Observation) -> list[Command]:
        return []


def make_quiet() -> Proposer:
    return QuietProposer()


def quiet_arm(*, shield: bool, perception: str = "V3") -> Arm:
    return Arm("quiet", make_quiet, shield=shield, monitor=False, perception=perception)


# ----------------------------------------------------------------------------- fields


@pytest.fixture(scope="module")
def result(small_context: PreparedContext) -> EpisodeResult:
    return run_episode(ARMS["B3-V3"], SEED, scenario_for("nominal"), context=small_context)


class TestTheGasClockRowsOfAnEpisode:
    """0003 A18, F5: the episode carries the two secondary counts, read from its own trace."""

    @pytest.mark.parametrize("seed", [1100, 1101, 1103])
    def test_the_fields_are_the_trace_functions_and_the_plants_job_count(
        self, seed: int, small_context: PreparedContext
    ) -> None:
        from closed_loop.core.types import Cmd
        from closed_loop.eval.metrics import crusher_work_rows, late_gas_tests

        with use_tuned(PROVISIONAL):
            rec = simulate_episode(
                ARMS["B3-V3"], seed, scenario_for("nominal"), context=small_context
            )
        r, tr = rec.result, rec.trace
        assert r.crusher_work_q == crusher_work_rows(tr)
        assert r.late_gas_tests == late_gas_tests(tr, small_context.spec_clause.retest_q)
        closes = sum(
            1
            for _q, c, ex, _ov in tr.commands
            if c.asset_id == "CR-501" and c.cmd is Cmd.CLOSE and ex
        )
        # a job is counted when the plant completes it; the runner's CLOSE follows a few quarters
        # later, so a job that completes at the very end of the episode has no CLOSE yet
        assert closes <= r.crusher_jobs_done <= closes + 1
        if r.crusher_jobs_done:
            assert r.crusher_work_q > 0  # a completed crusher job stood in WORK

    def test_some_seed_completes_a_crusher_job_so_the_check_above_is_not_empty(
        self, small_context: PreparedContext
    ) -> None:
        with use_tuned(PROVISIONAL):
            done = [
                simulate_episode(
                    ARMS["B3-V3"], seed, scenario_for("nominal"), context=small_context
                ).result.crusher_jobs_done
                for seed in (1100, 1101, 1103)
            ]
        assert sum(done) > 0, done


class TestResultFields:
    def test_the_fields_of_0002_are_filled(self, result: EpisodeResult) -> None:
        r = result
        assert (r.arm, r.seed, r.condition, r.fidelity) == ("B3-V3", SEED, "nominal", Fidelity.F1)
        assert set(r.violations) == set(r.min_robustness) == {"P1", "P2", "P3", "P4", "P5", "P6"}
        assert all(isinstance(n, int) and n >= 0 for n in r.violations.values())
        assert all(math.isfinite(v) for v in r.min_robustness.values())
        assert 0.0 <= r.throughput <= 1.0
        assert r.cost >= 0 and r.planned_jobs >= 0 and r.unplanned_failures >= 0
        assert r.shield_overrides == r.shield_admitted == 0  # no shield in this arm
        assert r.wall_s > 0
        assert len(r.labels) == len(r.checkpoints) == 360  # 4 assets x 90 days, one stream

    def test_the_cost_is_the_formula_of_the_preregistration(self, result: EpisodeResult) -> None:
        # jobs, failures, inspections (0.05 each), rejected work orders (0.02 each, counted by the
        # plant and not carried in the result) and lost quarters (0.02 each): 0003 A18, ruling R7
        want = (
            result.planned_jobs
            + 8.0 * result.unplanned_failures
            + 0.05 * result.inspections
            + 0.02 * result.lost_quarters
        )
        assert result.cost >= want - 1e-9
        extra = result.cost - want
        assert extra / 0.02 == pytest.approx(round(extra / 0.02), abs=1e-6)  # whole rejected orders
        assert result.lost_quarters == pytest.approx((1.0 - result.throughput) * 8640, rel=1e-4)

    def test_the_provisional_tuned_values_are_flagged_in_the_result(
        self, result: EpisodeResult, small_context: PreparedContext
    ) -> None:
        assert result.provisional == ("k_b3",)
        assert "provisional" in result.tuning
        b0 = run_episode(ARMS["B0"], SEED, scenario_for("nominal"), context=small_context)
        assert b0.provisional == ()  # B0 reads no tuned number
        with use_tuned(TunedValues(7.0, 5.5, 0.5, 0.02, False, "tuned.json", "ab" * 32)):
            tuned = run_episode(ARMS["B3-V3"], SEED, scenario_for("nominal"), context=small_context)
        assert tuned.provisional == () and tuned.tuning.endswith("sha256:" + "ab" * 6)

    def test_the_checkpoints_and_labels_carry_the_sim_only_truth_for_scoring(
        self, result: EpisodeResult
    ) -> None:
        assert all(
            cp.true_rul is not None and cp.health_bin is not None for cp in result.checkpoints
        )
        assert all(lab.true_rul is not None for lab in result.labels)
        assert all(cp.level == pytest.approx(0.9) for cp in result.checkpoints)

    def test_the_contract_counts_cover_the_shield_set(self, result: EpisodeResult) -> None:
        assert result.contract_source == "life95"
        assert result.contract.checkpoints == 360 and result.contract.lives > 0

    def test_all_four_monitors_run_passively(self, result: EpisodeResult) -> None:
        assert set(result.monitors) == {"opt", "pess", "inf", "proxy"}
        assert result.monitor_fired_day is None and result.monitor_mode == ""

    def test_an_arm_without_perception_has_no_perception_records(
        self, small_context: PreparedContext
    ) -> None:
        r = run_episode(ARMS["B0"], SEED, scenario_for("nominal"), context=small_context)
        assert r.labels == () and r.checkpoints == () and r.monitors == {}
        assert r.coverage.n_total == 0 and r.contract.checkpoints == 0 and r.contract_source == ""

    def test_the_condition_and_the_shift_day_come_from_the_scenario(
        self, small_context: PreparedContext
    ) -> None:
        nominal = run_episode(ARMS["B0"], SEED, scenario_for("nominal"), context=small_context)
        shifted = run_episode(ARMS["B0"], SEED, scenario_for("shifted"), context=small_context)
        assert nominal.condition == "nominal" and nominal.shift_day is None
        assert shifted.condition == "shifted" and 30.0 <= (shifted.shift_day or 0.0) <= 60.0

    def test_the_fidelity_label_is_carried_and_f2_is_refused(
        self, small_context: PreparedContext
    ) -> None:
        f1d = run_episode(
            ARMS["B0"], SEED, scenario_for("nominal"), Fidelity.F1D, context=small_context
        )
        assert f1d.fidelity is Fidelity.F1D
        for bad in (Fidelity.F0, Fidelity.F2):
            with pytest.raises(ValueError, match="F1"):
                run_episode(ARMS["B0"], SEED, scenario_for("nominal"), bad, context=small_context)

    def test_scenario_for_is_nominal_or_shifted(self) -> None:
        sc = scenario_for("shifted")
        assert isinstance(sc, Scenario)
        assert (sc.theta, sc.condition, sc.regime) == (None, "shifted", "in")
        assert scenario_for("nominal", "out").regime == "out"
        with pytest.raises(ValueError):
            scenario_for("stormy")


# ----------------------------------------------------------------------------- determinism, pairing


class TestDeterminism:
    def test_the_same_arm_seed_and_scenario_give_the_same_episode(
        self, small_context: PreparedContext
    ) -> None:
        a = run_episode(ARMS["H-G0"], SEED, scenario_for("nominal"), context=small_context)
        b = run_episode(ARMS["H-G0"], SEED, scenario_for("nominal"), context=small_context)
        assert dataclasses.replace(a, wall_s=0.0) == dataclasses.replace(b, wall_s=0.0)

    def test_the_result_does_not_depend_on_what_ran_before(
        self, small_context: PreparedContext
    ) -> None:
        first = run_episode(ARMS["B3-V3"], SEED, scenario_for("nominal"), context=small_context)
        run_episode(ARMS["H-C1"], SEED + 1, scenario_for("nominal"), context=small_context)
        again = run_episode(ARMS["B3-V3"], SEED, scenario_for("nominal"), context=small_context)
        assert dataclasses.replace(first, wall_s=0.0) == dataclasses.replace(again, wall_s=0.0)


class TestPairedDesign:
    """The same seed gives the same exogenous streams in every arm (0001 section 11)."""

    ARM_NAMES = ("B0", "B3-V3", "H-G0", "H-C1", "P1-I")

    @pytest.fixture(scope="class")
    @classmethod
    def plants(cls, small_context: PreparedContext) -> dict[str, Any]:
        return {
            name: simulate_episode(
                ARMS[name], SEED, scenario_for("shifted"), context=small_context
            ).plant
            for name in cls.ARM_NAMES
        }

    def test_the_physical_streams_are_identical(self, plants: dict[str, Any]) -> None:
        ref = plants["B0"]._phys
        for name, plant in plants.items():
            phys = plant._phys
            for aid in ref.load_day:
                for field in ("load_day", "rho_day", "cum", "cum_design"):
                    assert np.array_equal(getattr(phys, field)[aid], getattr(ref, field)[aid]), (
                        name,
                        aid,
                        field,
                    )
            assert phys.shift_day == ref.shift_day

    def test_the_observation_noise_and_data_quality_events_are_identical(
        self, plants: dict[str, Any]
    ) -> None:
        ref = plants["B0"]._obs_streams
        for name, plant in plants.items():
            for aid, streams in plant._obs_streams.items():
                for field in ("hi_noise", "sig_noise", "load_noise", "masked", "drift_frac"):
                    assert np.array_equal(getattr(streams, field), getattr(ref[aid], field)), (
                        name,
                        aid,
                        field,
                    )

    def test_the_des_streams_are_identical(self, plants: dict[str, Any]) -> None:
        ref = plants["B0"].des.st
        for name, plant in plants.items():
            st = plant.des.st
            assert np.array_equal(st.notes, ref.notes), name
            for field in ("u_refuse", "u_holder", "u_gas", "appr_z", "appr_c", "insp_z"):
                for aid in getattr(ref, field):
                    assert np.array_equal(getattr(st, field)[aid], getattr(ref, field)[aid]), (
                        name,
                        aid,
                        field,
                    )

    def test_every_arm_lives_the_life_table_until_it_first_stops_an_asset(
        self, plants: dict[str, Any], small_context: PreparedContext
    ) -> None:
        """The no-intervention life table is the same world each arm starts in: the realised RUL
        the trace holds before an asset's first stop equals the table's, to float32 precision."""
        table = simulate_lives(small_context.params, SEED, scenario_for("shifted"))
        compared = 0
        for name, plant in plants.items():
            tr = plant.trace()
            for aid in tr.asset_ids:
                stopped = np.flatnonzero(tr.op[aid] != op_index(Op.RUN))
                first_stop = int(stopped[0]) if stopped.size else tr.n_q
                days = [d for d in range(1, 90) if DAY_Q * d - 1 < first_stop]
                if not days:
                    continue
                got = np.array([tr.true_rul_days[aid][DAY_Q * d - 1] for d in days])
                assert got == pytest.approx(table.true_rul_days[aid][days], abs=1e-4), (name, aid)
                compared += len(days)
        assert compared > 500

    def test_a_checkpoints_truth_is_the_life_tables_realised_rul_of_that_day(
        self, small_context: PreparedContext
    ) -> None:
        """Alignment of the sim-only truth with the checkpoint: the realised RUL put on the
        checkpoint of day ``d`` is the table's value at the start of day ``d``."""
        table = simulate_lives(small_context.params, SEED, scenario_for("nominal"))
        rec = simulate_episode(ARMS["B3-V3"], SEED, scenario_for("nominal"), context=small_context)
        first_stop = {}
        for aid in ASSET_IDS:
            stopped = np.flatnonzero(rec.trace.op[aid] != op_index(Op.RUN))
            first_stop[aid] = int(stopped[0]) if stopped.size else rec.trace.n_q
        checked = 0
        for cp in rec.result.checkpoints:
            if cp.life_id == 0 and DAY_Q * cp.day < first_stop[cp.asset_id]:
                want = table.true_rul_days[cp.asset_id][cp.day]
                assert cp.true_rul == pytest.approx(want, abs=1e-4), (cp.asset_id, cp.day)
                checked += 1
        assert checked > 150

    def test_a_different_seed_gives_different_streams(self, small_context: PreparedContext) -> None:
        other = simulate_episode(
            ARMS["B0"], SEED + 1, scenario_for("shifted"), context=small_context
        ).plant
        base = simulate_episode(
            ARMS["B0"], SEED, scenario_for("shifted"), context=small_context
        ).plant
        assert not np.array_equal(other._phys.load_day["CR-501"], base._phys.load_day["CR-501"])


# ----------------------------------------------------------------------------- shield on and off


SHIELD_FIELDS = {
    "arm",
    "shield_overrides",
    "shield_admitted",
    "allowed_crusher",
    "allowed_pumps",
    "uncovered",
    "crusher_stopped",
    "crusher_restart_allowed",
    "wall_s",
}


def differing(a: EpisodeResult, b: EpisodeResult) -> set[str]:
    return {
        f.name
        for f in dataclasses.fields(EpisodeResult)
        if getattr(a, f.name) != getattr(b, f.name)
    }


class TestShieldOffAndOn:
    def test_a_proposer_that_does_not_violate_differs_only_in_the_shield_fields(
        self, small_context: PreparedContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Sets that exclude every dangerous bin, so the table has nothing to refuse: the shield
        # is on the path and must change nothing the plant sees.
        monkeypatch.setattr(episode_module, "health_set", lambda interval: (0, 0))
        off = run_episode(
            quiet_arm(shield=False), SEED, scenario_for("nominal"), context=small_context
        )
        on = run_episode(
            quiet_arm(shield=True), SEED, scenario_for("nominal"), context=small_context
        )
        assert differing(off, on) <= SHIELD_FIELDS
        assert on.shield_overrides == 0 and on.allowed_crusher[1] > 0  # it was consulted
        assert off.shield_overrides == off.allowed_crusher[1] == 0

    def test_the_same_holds_for_b0_when_the_shield_has_nothing_to_refuse(
        self, small_context: PreparedContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(episode_module, "health_set", lambda interval: (0, 0))
        base = dataclasses.replace(ARMS["B0"], perception="V3")
        on = dataclasses.replace(base, name="B0-shield", shield=True)
        a = run_episode(base, SEED, scenario_for("nominal"), context=small_context)
        b = run_episode(on, SEED, scenario_for("nominal"), context=small_context)
        assert a.planned_jobs + a.unplanned_failures > 0  # something happened to refuse or admit
        if b.shield_overrides == 0:
            assert differing(a, b) <= SHIELD_FIELDS
        else:  # the table is stricter than the properties; then only the trace may differ
            assert b.shield_admitted + b.shield_overrides > 0

    def test_with_the_real_sets_the_shield_stops_the_crusher_and_every_restart_offer_is_refused(
        self, small_context: PreparedContext
    ) -> None:
        """The finding of the P12a note, as it stood until 0003 A18: the table allows only STOP for
        a running crusher under the default per-life set and never a START, and no proposer issues
        a START outside a job. It was pinned here as ``throughput < 0.1`` and
        ``crusher_restart_allowed == 0``. Since ruling R4 an operator offers a restart of every
        stopped asset with no job once a day, and the throughput is 0.607 on this seed: the plant
        runs for 57 days (5,473 quarters) and the shield then stops the crusher on its band (an
        expected result of the band, not a fault). From there each daily offer reaches the shield
        and is refused, because the table allows no START under these sets, so the stop stands to
        the end of the episode and ``crusher_restart_allowed`` is still zero."""
        off = run_episode(
            quiet_arm(shield=False), SEED, scenario_for("nominal"), context=small_context
        )
        on = run_episode(
            quiet_arm(shield=True), SEED, scenario_for("nominal"), context=small_context
        )
        assert off.throughput > 0.9  # nobody repairs anything, so some assets fail
        assert 0.3 < on.throughput < off.throughput  # the plant runs behind the shield
        assert on.crusher_stopped > 0 and on.crusher_restart_allowed == 0
        assert on.shield_overrides > 0
        # the quiet proposer says nothing; every command the counter saw is an operator's offer
        assert on.commands_proposed > 0 and off.commands_proposed == 0

    def test_the_commands_the_shield_replaced_are_marked_in_the_trace(
        self, small_context: PreparedContext
    ) -> None:
        rec = simulate_episode(ARMS["H-C1"], SEED, scenario_for("nominal"), context=small_context)
        log = rec.trace.commands
        replaced = [c for _q, c, _executed, overridden in log if overridden]
        assert replaced and all(c.source == "shield" for c in replaced)
        assert all(overridden for _q, c, _e, overridden in log if c.source == "shield")
        forced = simulate_episode(
            ARMS["H-C-I"], SEED, scenario_for("nominal"), context=small_context
        )
        marked = [c for _q, c, _e, overridden in forced.trace.commands if overridden]
        assert marked and all(c.source.startswith("interlock") for c in marked)

    def test_restart_opportunities_are_counted_when_the_table_would_allow_a_start(
        self, small_context: PreparedContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A set that contains the top bin for days 10 to 19 stops the crusher; from day 20 the set
        excludes it and the table allows a START at every stopped decision. The quiet proposer
        never issues one (before 0003 A18 nobody did and the crusher stood stopped for 70 days:
        more than 5000 decisions); now the operator offers one at the next day's first quarter, the
        shield admits it, and the stop ends after at most a day."""
        calls = {"n": 0}

        def widening(interval: object) -> tuple[int, int]:
            day = calls["n"] // 4  # four assets, one call each per day
            calls["n"] += 1
            return (0, 3) if 10 <= day < 20 else (0, 0)

        monkeypatch.setattr(episode_module, "health_set", widening)
        r = run_episode(
            quiet_arm(shield=True), SEED, scenario_for("nominal"), context=small_context
        )
        assert 0 < r.crusher_restart_allowed < r.crusher_stopped
        assert r.crusher_restart_allowed <= 96  # at most a day of them, not the 70 days of before
        assert r.throughput > 0.5

    def test_the_interlock_and_the_shield_are_alternatives_in_the_loop(
        self, small_context: PreparedContext
    ) -> None:
        r = run_episode(ARMS["H-C-I"], SEED, scenario_for("nominal"), context=small_context)
        assert r.shield_overrides == 0 and r.interlock_admitted > 0
        assert r.commands_proposed > 0 and set(r.interlock_by_rule)


# ----------------------------------------------------------------------------- the loop itself


class TestLoop:
    def test_quarter_zero_is_a_warm_up_nobody_commands(
        self, small_context: PreparedContext
    ) -> None:
        rec = simulate_episode(ARMS["B3-V3"], SEED, scenario_for("nominal"), context=small_context)
        assert all(q > 0 for q, _cmd, _ex, _ov in rec.trace.commands)
        first = rec.trace.set_hi["CR-501"][0]
        assert 0 <= first <= 3  # the day-0 set was handed over before the first decision

    def test_the_set_in_the_trace_is_the_health_set_of_the_days_shield_interval(
        self, small_context: PreparedContext
    ) -> None:
        rec = simulate_episode(ARMS["B3-V3"], SEED, scenario_for("nominal"), context=small_context)
        tr = rec.trace
        checked = 0
        for cp in rec.shield_checkpoints:
            want = health_set(RulInterval(cp.lo, cp.m, cp.hi, cp.level, cp.prior))
            row = cp.day * DAY_Q
            assert (int(tr.set_lo[cp.asset_id][row]), int(tr.set_hi[cp.asset_id][row])) == want
            checked += 1
        assert checked == 360

    def test_the_step99_set_is_a_per_step_variant_never_the_life_band(
        self, small_context: PreparedContext
    ) -> None:
        bad = dataclasses.replace(ARMS["P1-step"], perception="V6")
        with pytest.raises(ValueError, match="per-step variant"):
            run_episode(bad, SEED, scenario_for("nominal"), context=small_context)

    def test_an_audit_monitor_needs_perception_v4(self, small_context: PreparedContext) -> None:
        bad = dataclasses.replace(
            ARMS["B3-V4-audit"], perception="none", shield=False, monitor=False
        )
        with pytest.raises(ValueError, match="audit monitor"):
            run_episode(bad, SEED, scenario_for("nominal"), context=small_context)

    def test_an_oracle_arm_binds_its_diagnoser_and_a_missing_one_is_an_error(
        self, small_context: PreparedContext
    ) -> None:
        r = run_episode(ARMS["P1-oracle"], SEED, scenario_for("nominal"), context=small_context)
        assert r.arm == "P1-oracle"
        bad = dataclasses.replace(ARMS["B0"], name="B0-oracle", diagnoser="oracle")
        with pytest.raises(ValueError, match="oracle"):
            run_episode(bad, SEED, scenario_for("nominal"), context=small_context)

    def test_the_acting_monitor_widens_the_sets_and_is_recorded(
        self, small_context: PreparedContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A monitor that fires on day 1 is the acting one: from then on the plant reports it."""
        from closed_loop.perception.monitor import CoverageMonitor

        real = CoverageMonitor.observe_day

        def fire_early(self: CoverageMonitor, *a: Any, **k: Any) -> None:
            real(self, *a, **k)
            if self.mode == "proxy" and a[1] >= 5 and not self.fired:
                self.fired = True
                self.fired_day = float(a[1])

        monkeypatch.setattr(CoverageMonitor, "observe_day", fire_early)
        arm = dataclasses.replace(
            ARMS["B3-V3"], name="B3-monitored", monitor=True, monitor_mode="proxy"
        )
        rec = simulate_episode(arm, SEED, scenario_for("nominal"), context=small_context)
        assert rec.result.monitor_fired_day == 5.0 and rec.result.monitor_mode == "proxy"
        assert rec.result.fallback_row is not None and rec.result.fallback_row >= 5 * 96
        assert rec.plant._monitor_fired is True
        # an unmonitored arm records the passive firing but never acts on it
        passive = simulate_episode(
            ARMS["B3-V3"], SEED, scenario_for("nominal"), context=small_context
        )
        assert passive.result.monitors["proxy"] == 5.0
        assert passive.result.monitor_fired_day is None and passive.result.fallback_row is None


class TestLabelsAgreeWithTheTruth:
    """Resolved on the operating clock, a label never contradicts the twin world's truth."""

    @pytest.mark.parametrize("name", ["B0", "B1", "B3-V1"])
    def test_censored_labels_never_outlive_the_truth_and_failures_match_it(
        self, name: str, small_context: PreparedContext
    ) -> None:
        tol = 0.25  # days: a wrong-mode job stops a part mid-life, and the load path then differs
        arm = dataclasses.replace(ARMS[name], perception="V1")  # B0 and B1 get a layer to label
        uncensored = censored = off = 0
        for seed in range(1100, 1106):
            for cond in ("nominal", "shifted"):
                r = run_episode(arm, seed, scenario_for(cond), context=small_context)
                for lab in r.labels:
                    assert lab.true_rul is not None
                    seen = lab.end_day - lab.checkpoint_day
                    if lab.censored:
                        censored += 1
                        assert seen <= lab.true_rul + tol
                    else:
                        uncensored += 1
                        off += abs(seen - lab.true_rul) > tol
        assert censored > 0
        assert uncensored > 0 or name == "B3-V1"  # B3 prevents nearly every failure
        assert off == 0


# ----------------------------------------------------------------------------- golden episodes

GOLDEN_CASES = [
    ("B3-V3", "nominal"),
    ("H-G0", "nominal"),
    ("P1-step", "shifted"),
]


def _approx_equal(got: Any, want: Any, path: str = "") -> None:
    if isinstance(want, dict):
        assert isinstance(got, dict) and set(got) == set(want), path
        for k in want:
            _approx_equal(got[k], want[k], f"{path}/{k}")
    elif isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), path
        for i, (g, w) in enumerate(zip(got, want, strict=True)):
            _approx_equal(g, w, f"{path}[{i}]")
    elif isinstance(want, float):
        assert got == pytest.approx(want, rel=1e-9, abs=1e-12), path
    else:
        assert got == want, path


class TestGolden:
    """One episode's summary per case, pinned. Regenerate only on purpose:

        CLOSED_LOOP_REGENERATE_GOLDEN=1 python -m pytest tests/eval/test_episode.py -k Golden

    and read the diff of the JSON file before keeping it.
    """

    @pytest.mark.parametrize(("name", "cond"), GOLDEN_CASES)
    def test_the_summary_is_the_pinned_one(
        self, name: str, cond: str, small_context: PreparedContext
    ) -> None:
        arm = ARMS[name]
        if arm.perception == PRIMARY_VARIANT != PROVISIONAL.primary_variant:
            # the planner arms read the primary variant when arms.py is imported, which is the
            # tuned one (V2L) now; the pin was taken with the provisional one (V3)
            arm = dataclasses.replace(arm, perception=PROVISIONAL.primary_variant)
        r = run_episode(arm, SEED, scenario_for(cond), context=small_context)
        got = json.loads(json.dumps(summary(r)))
        path = GOLDEN_DIR / f"episode_{name}_{SEED}_{cond}.json"
        if os.environ.get(REGENERATE) == "1":
            path.write_text(json.dumps(got, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        assert path.exists(), f"no golden file {path.name}; regenerate with {REGENERATE}=1"
        want = json.loads(path.read_text(encoding="utf-8"))
        _approx_equal(got, want)

    def test_there_is_a_golden_file_for_every_case(self) -> None:
        for name, cond in GOLDEN_CASES:
            assert (GOLDEN_DIR / f"episode_{name}_{SEED}_{cond}.json").exists()
