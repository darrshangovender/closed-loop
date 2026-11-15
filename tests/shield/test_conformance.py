"""The F1 abstraction map and the conformance test (packet P5c).

Synthetic traces carry one injected defect each, so that every count is known by construction. The
real plant is used for four things: a correct scripted job conforms completely (a positive control
that also pins the gas convention), a wrong-mode completion is a matched transition (ADR 0003 A11),
the reconstruction of ``soon`` from a trace equals what the plant's own view said (A12.1), and an
episode of a careless proposer runs through the whole pipeline. Since A12 there are three games: the
conveyor CV-101 is the third, and a locked pump is a pump state of its own (A12.2).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from closed_loop.core.results import ResultRecord
from closed_loop.core.trace import Trace, op_index, permit_index
from closed_loop.core.types import Cmd, Command, Op, Permit, PlanCmd
from closed_loop.plant.params import ASSET_IDS
from closed_loop.shield import runtime
from closed_loop.shield.abstraction import (
    CRUSHER_COMMANDS,
    PUMP_COMMANDS,
    ConveyorModel,
    CrusherJobModel,
    PumpPairModel,
)
from closed_loop.shield.conformance import (
    A_PHYS_KINDS,
    F1AbstractionMap,
    _soon,
    conformance,
    result_records,
)
from closed_loop.shield.table import ShieldTable
from closed_loop.spec.properties import SpecParams, evaluate
from tests.plant._scripts import (
    advance,
    confined_job,
    drive,
    fresh,
    isolate,
    merge,
    run_with_runner,
    seed_with_mode,
)
from tests.plant.test_wrong_mode import wrong_worlds

CR, CV, PA, PB = ASSET_IDS
TWO = {"crusher": CrusherJobModel(), "pumps": PumpPairModel()}
MODELS = {**TWO, "conveyor": ConveyorModel()}
LEGACY = {
    "crusher": CrusherJobModel(spike_semantics=True),
    "pumps": PumpPairModel(spike_semantics=True),
}
MAP = F1AbstractionMap()


def view(**kw: object) -> SimpleNamespace:
    base: dict[str, object] = dict(
        op=Op.RUN, permit=Permit.NONE, occupied=False, gas_age_q=99, alarm=False, soon=False
    )
    base.update(kw)
    return SimpleNamespace(**base)


def command(asset: str, cmd: Cmd | PlanCmd) -> Command:
    return Command(asset, cmd, None, "test")


def quiet(n: int = 200) -> Trace:
    """Nothing happens: every asset runs at bin 0, nobody works, no test, no permit."""
    return Trace.empty(n, ASSET_IDS)


def issue(tr: Trace, q: int, asset: str, cmd: Cmd) -> None:
    tr.commands.append((q, command(asset, cmd), True, False))


class TestTheMap:
    def test_the_crusher_state_is_the_view_with_the_given_bin(self) -> None:
        s = MAP.crusher_state(
            view(
                op=Op.WORK, permit=Permit.VALID, occupied=True, gas_age_q=1, alarm=True, soon=True
            ),
            2,
        )
        assert s == ("WORK", 2, "VALID", 1, 1, 1, 1)
        assert s in set(MODELS["crusher"].states())

    @pytest.mark.parametrize(
        ("gas_age_q", "gas"), [(0, 1), (1, 1), (2, 2), (3, 3), (4, 3), (99, 3)]
    )
    def test_gas_age_one_is_a_test_from_the_previous_quarter(
        self, gas_age_q: int, gas: int
    ) -> None:
        assert MAP.crusher_state(view(gas_age_q=gas_age_q), 0)[4] == gas  # type: ignore[index]

    def test_a_longer_period_gives_a_longer_range(self) -> None:
        slow = F1AbstractionMap(retest_q=8)
        assert [slow.crusher_state(view(gas_age_q=a), 0)[4] for a in (1, 5, 8, 9, 50)] == [  # type: ignore[index]
            1, 5, 8, 9, 9,
        ]  # fmt: skip
        for bad in (0, -1, 2.0, True):
            with pytest.raises(ValueError, match="retest_q"):
                F1AbstractionMap(retest_q=bad)  # type: ignore[arg-type]

    def test_the_pump_pair_state(self) -> None:
        s = MAP.pump_pair_state(view(op=Op.RUN), view(op=Op.WORK), (1, 3))
        assert s == ("RUN", 1, "WORK", 3)
        assert s in set(MODELS["pumps"].states())

    def test_a_locked_pump_is_a_locked_one(self) -> None:
        """0003 A12.2: the map used to send LOCKED to STOP (a locked pump counted as available)."""
        s = MAP.pump_pair_state(view(op=Op.LOCKED), view(op=Op.ISO), (0, 0))
        assert s == ("LOCKED", 0, "ISO", 0) and s in set(MODELS["pumps"].states())

    def test_the_conveyor_state(self) -> None:
        for op in Op:
            s = MAP.conveyor_state(view(op=op), 2)
            assert s == (op.value, 2) and s in set(MODELS["conveyor"].states())

    def test_pump_and_conveyor_lock_and_prove_map_to_themselves(self) -> None:
        for asset in (PA, PB, CV):
            assert MAP.command(command(asset, Cmd.LOCK)) == "LOCK"
            assert MAP.command(command(asset, Cmd.PROVE)) == "PROVE"
        assert MAP.command(command(CR, Cmd.PROVE)) == "PROVE"
        assert MAP.command(command(CR, Cmd.LOCK)) == "LOCK"

    @pytest.mark.parametrize(
        "cmd", [Cmd.ISSUE, Cmd.REVAL, Cmd.TEST, Cmd.ENTER, Cmd.EXIT, Cmd.CLOSE, Cmd.DERATE]
    )
    def test_commands_a_pump_has_no_name_for_are_no_ops(self, cmd: Cmd) -> None:
        assert MAP.command(command(PA, cmd)) == "NOOP"
        assert MAP.command(command(CV, cmd)) == "NOOP"

    def test_planning_commands_and_derate_are_outside_the_proof(self) -> None:
        for c in (PlanCmd.INSPECT, PlanCmd.ORDER, PlanCmd.RAISE_WO):
            assert MAP.command(command(CR, c)) == "NOOP"
        assert MAP.command(command(CR, Cmd.DERATE)) == "NOOP"

    def test_every_plant_command_maps_into_the_model_alphabet(self) -> None:
        for c in Cmd:
            assert MAP.command(command(CR, c)) in CRUSHER_COMMANDS
            assert MAP.command(command(PA, c)) in PUMP_COMMANDS
            assert MAP.command(command(CV, c)) in PUMP_COMMANDS

    def test_the_runtimes_concrete_command_is_the_inverse_of_the_map(self) -> None:
        for abstract in PUMP_COMMANDS:
            for asset in (PA, CV):
                assert MAP.command(runtime._concrete(asset, abstract)) == abstract, (
                    asset,
                    abstract,
                )
        for abstract in CRUSHER_COMMANDS:
            assert MAP.command(runtime._concrete(CR, abstract)) == abstract

    def test_the_runtime_accepts_it(self) -> None:
        table = ShieldTable("m", {}, ("STOP",), {})
        shielded = runtime.ShieldedProposer(
            SimpleNamespace(name="x", reset=lambda s: None, propose=lambda o: []),
            table,
            table,
            MAP,
            conveyor=table,
            crusher_model=MODELS["crusher"],
            pumps_model=MODELS["pumps"],
            conveyor_model=MODELS["conveyor"],
        )
        assert shielded.mapper is MAP


class TestSoonIsRebuiltFromTheRows:
    """0003 A12.1: ``view.soon`` is a handover next, or the permit reaching its 12 hours next. The
    old reconstruction knew the handover only; the trace holds no limit, so it is the permit's first
    row plus the limit."""

    @staticmethod
    def held(tr: Trace, first: int, last: int, permit: Permit = Permit.VALID) -> None:
        tr.permit[CR][first : last + 1] = permit_index(permit)

    def test_without_a_permit_only_the_handover_announces(self) -> None:
        tr = quiet(100)
        got = _soon(tr, CR, 48)
        assert [r for r in range(100) if got[r]] == [30, 62, 94]  # row r: step r + 1 ends a shift

    def test_a_permit_announces_its_limit_on_the_last_row_it_is_held(self) -> None:
        tr = quiet(100)
        self.held(tr, 5, 51)  # issued at 5: limit 53, spent at the end of step 52
        got = _soon(tr, CR, 48)
        assert [r for r in range(100) if got[r]] == [30, 51, 62, 94]
        # the limit counts from the first row of the run, not from the last revalidation
        tr2 = quiet(100)
        self.held(tr2, 5, 31)
        self.held(tr2, 32, 51, Permit.REVAL)
        assert list(_soon(tr2, CR, 48)) == list(got)

    def test_a_new_run_has_a_new_limit(self) -> None:
        tr = quiet(150)
        self.held(tr, 5, 51)
        self.held(tr, 60, 106)  # issued again at 60: spent at the end of step 107
        got = [r for r, v in enumerate(_soon(tr, CR, 48)) if v]
        assert got == [30, 51, 62, 94, 106, 126]

    def test_a_shorter_limit_is_the_parameter(self) -> None:
        tr = quiet(60)
        self.held(tr, 3, 13)
        assert [r for r, v in enumerate(_soon(tr, CR, 12)) if v] == [13, 30]

    def test_it_equals_the_plants_own_view_in_a_run_with_revalidation_and_a_spent_permit(
        self,
    ) -> None:
        """The reconstruction against the truth: every decision's ``view.soon`` of every asset."""
        plant = fresh(5, permit_max_h=6.0)  # 24 quarters, so it ends inside the shift too
        script = merge(
            isolate(0, CR),
            {40: [command(CR, Cmd.ISSUE)]},
            isolate(10, PA),
            {33: [command(PA, Cmd.REVAL)]},
        )
        soon = {a: [] for a in ASSET_IDS}  # the view at decision q is built from the step q - 1
        for q in range(200):
            r = plant.step(script.get(q, []))
            for a in ASSET_IDS:
                soon[a].append(r.obs.assets[a].soon)
        tr = plant.trace()
        for a in (CR, PA):
            rebuilt = _soon(tr, a, 24)
            assert list(rebuilt[:200]) == [int(v) for v in soon[a]], a
        assert any(_soon(tr, CR, 24)) and sum(soon[CR]) > sum(soon[CV])  # the limit announced

    def test_with_a_trace_from_a_12_hour_plant_the_default_is_right(self) -> None:
        plant = fresh(5)
        script = merge(isolate(0, CR), {32: [command(CR, Cmd.REVAL)]})
        soon = []
        for q in range(120):
            soon.append(plant.step(script.get(q, [])).obs.assets[CR].soon)
        assert list(_soon(plant.trace(), CR, 48)) == [int(v) for v in soon]
        assert [r for r, v in enumerate(soon) if v] == [30, 49, 62, 94]  # handovers, one limit


class TestASecondAnnouncementIsMatched:
    """A permit issued at quarter 15 of a shift has its limit on the last quarter but one of a later
    shift (15 + 48 = 63), so ``view.soon`` is true on two rows in a row (61 and 62: the limit, then
    the handover). Before the owner's ruling on 0003 A12.1 the games turned ``soon`` into 0 after an
    announced step and that tick was unmatched in ``crusher.soon`` (36 of them in 200 episodes);
    now the game contains it."""

    def test_the_plant_shows_it_and_the_report_has_nothing_unmatched(self) -> None:
        plant = fresh(5)
        drive(plant, isolate(12), until=80)  # STOP, LOCK, PROVE at 12-14 and ISSUE at 15
        tr = plant.trace()
        assert [r for r in range(81) if _soon(tr, CR, 48)[r]] == [30, 61, 62]
        rep = conformance([tr], MAP, MODELS)
        assert rep.unmatched == 0 and dict(rep.by_variable) == {}, rep.unmatched_examples

    def test_the_old_game_could_not_follow_it(self) -> None:
        """The same trace against the game of the older reading: one unmatched tick."""
        from tests.shield.test_abstraction import TestASecondAnnouncementInARow as Old

        plant = fresh(5)
        drive(plant, isolate(12), until=80)
        models = {**MODELS, "crusher": Old.Once()}
        rep = conformance([plant.trace()], MAP, models)
        assert rep.unmatched == 1 and dict(rep.by_variable) == {"crusher.soon": 1}
        assert rep.unmatched_when_safe == 1 and rep.missed_violations == 0

    @pytest.mark.parametrize("issue_q", [14, 16, 20])
    def test_a_permit_issued_a_little_earlier_or_later_conforms_as_before(
        self, issue_q: int
    ) -> None:
        plant = fresh(5)
        drive(plant, isolate(issue_q - 3), until=80)
        assert conformance([plant.trace()], MAP, MODELS).unmatched == 0


class TestTheGasConventionAgainstThePlant:
    """The plant's own view, not a stub: a test that passed last quarter is ``gas == 1``."""

    def test_after_a_passing_test_the_next_view_is_gas_one_and_enter_is_allowed(self) -> None:
        seed = seed_with_mode(CR, "LINER_WEAR")
        plant = fresh(seed)
        script = {
            0: [command(CR, Cmd.STOP)], 1: [command(CR, Cmd.LOCK)], 2: [command(CR, Cmd.PROVE)],
            3: [command(CR, Cmd.ISSUE)], 4: [command(CR, Cmd.BEGIN)], 5: [command(CR, Cmd.TEST)],
        }  # fmt: skip
        views = {}
        for q in range(7):
            res = plant.step(script.get(q, []))
            views[q] = res.obs.assets[CR]
        assert views[4].gas_age_q > 3 and MAP.crusher_state(views[4], 0)[4] == 3  # type: ignore[index]
        assert views[5].gas_age_q == 1  # the test passed in step 5
        assert MAP.crusher_state(views[5], 0)[4] == 1  # type: ignore[index]
        assert views[6].gas_age_q == 2 and MAP.crusher_state(views[6], 0)[4] == 2  # type: ignore[index]
        # the abstraction admits the entry in the state the plant reports after the test
        state = MAP.crusher_state(views[5], 0)
        outcomes = MODELS["crusher"].step(state, "ENTER")
        assert "VIOL" not in outcomes


class TestQuietAndImpossible:
    def test_a_quiet_plant_conforms_everywhere(self) -> None:
        rep = conformance([quiet(200)], MAP, MODELS)
        assert rep.unmatched == 0 and rep.by_variable == {}
        assert rep.ticks == rep.ticks_total == 3 * 199
        assert rep.ticks_by_model == {"crusher": 199, "pumps": 199, "conveyor": 199}
        assert rep.a_phys_violations == 0 and rep.a_phys_ticks == 4 * 199
        assert rep.missed_violations == 0 and rep.violation_edges == 0
        assert rep.wilson[0] == 0.0 and 0.0 < rep.wilson[1] < 0.05

    def test_one_impossible_transition_is_counted_once_under_its_variable(self) -> None:
        tr = quiet(200)
        tr.op[CR][100:] = op_index(Op.ISO)  # RUN to ISO with no command at all
        rep = conformance([tr], MAP, MODELS)
        assert rep.unmatched == 1
        assert dict(rep.by_variable) == {"crusher.op": 1}
        assert rep.unmatched_by_model == {"crusher": 1, "pumps": 0, "conveyor": 0}
        assert rep.unmatched_when_safe == 1  # NOOP can never reach VIOL here
        assert dict(rep.by_variable_when_safe) == {"crusher.op": 1}
        game, prev, cmd, new, n, viol = rep.unmatched_examples[0]
        assert (game, cmd, n, viol) == ("crusher", "NOOP", 1, False)
        assert prev[0] == "RUN" and new[0] == "ISO"

    def test_an_alarm_with_nobody_inside_is_an_impossible_alarm(self) -> None:
        tr = quiet(200)
        tr.alarm[CR][100:] = True
        rep = conformance([tr], MAP, MODELS)
        assert rep.unmatched == 1 and dict(rep.by_variable) == {"crusher.alarm": 1}

    def test_an_impossible_pump_transition_is_counted_under_the_pump_variable(self) -> None:
        tr = quiet(200)
        tr.op[PB][100:] = op_index(Op.ISO)
        rep = conformance([tr], MAP, MODELS)
        assert rep.unmatched == 1 and dict(rep.by_variable) == {"pumps.op2": 1}

    def test_two_variables_wrong_in_one_tick_count_once_and_under_both(self) -> None:
        tr = quiet(120)  # short, so that no handover (which would void the permit) falls inside
        tr.op[CR][100:] = op_index(Op.ISO)
        tr.permit[CR][100:] = permit_index(Permit.VALID)
        rep = conformance([tr], MAP, MODELS)
        assert rep.unmatched == 1
        assert dict(rep.by_variable) == {"crusher.op": 1, "crusher.permit": 1}

    def test_a_health_bin_that_falls_without_a_renewal_is_unmatched(self) -> None:
        tr = quiet(200)
        tr.health_bin[CR][:100] = 2
        tr.health_bin[CR][100:] = 1
        rep = conformance([tr], MAP, MODELS)
        assert rep.unmatched == 1 and dict(rep.by_variable) == {"crusher.h": 1}
        assert rep.a_phys_violations == 0, "a fall is not one of the A_phys violations"

    def test_an_impossible_conveyor_transition_is_counted_under_the_conveyor_variable(self) -> None:
        tr = quiet(200)
        tr.op[CV][100:] = op_index(Op.ISO)  # RUN to ISO with no command at all
        rep = conformance([tr], MAP, MODELS)
        assert rep.unmatched == 1 and dict(rep.by_variable) == {"conveyor.op": 1}
        assert rep.unmatched_by_model == {"crusher": 0, "pumps": 0, "conveyor": 1}

    def test_without_a_conveyor_model_the_conveyor_is_not_tested(self) -> None:
        tr = quiet(200)
        tr.op[CV][100:] = op_index(Op.ISO)
        rep = conformance([tr], MAP, TWO)
        assert rep.unmatched == 0 and set(rep.ticks_by_model) == {"crusher", "pumps"}
        assert rep.a_phys_ticks == 3 * 199  # and its bin is not looked at either

    def test_a_command_the_game_refuses_to_follow_is_matched_by_its_refusal(self) -> None:
        tr = quiet(200)
        issue(tr, 50, CR, Cmd.PROVE)  # impossible on a running crusher: refused, nothing changes
        assert conformance([tr], MAP, MODELS).unmatched == 0


class TestAPhys:
    def test_a_bin_skip_is_one_violation(self) -> None:
        tr = quiet(200)
        tr.health_bin[CR][100:] = 2  # 0 to 2 in one tick
        rep = conformance([tr], MAP, MODELS)
        assert rep.a_phys_violations == 1
        assert dict(rep.a_phys_by_kind) == {"bin_skip": 1, "failure_below_top": 0}
        assert rep.unmatched == 1 and dict(rep.by_variable) == {"crusher.h": 1}

    def test_a_rise_of_one_bin_is_not_a_violation(self) -> None:
        tr = quiet(200)
        tr.health_bin[PA][100:] = 1
        assert conformance([tr], MAP, MODELS).a_phys_violations == 0

    def test_the_largest_jump_counts_once(self) -> None:
        tr = quiet(200)
        tr.health_bin[CR][100:] = 3
        rep = conformance([tr], MAP, MODELS)
        assert rep.a_phys_violations == 1 and rep.a_phys_by_kind["bin_skip"] == 1

    def test_a_skip_in_a_pump_counts_too(self) -> None:
        tr = quiet(200)
        tr.health_bin[PB][100:] = 3
        rep = conformance([tr], MAP, MODELS)
        assert rep.a_phys_violations == 1

    def test_a_skip_in_the_conveyor_counts_too(self) -> None:
        tr = quiet(200)
        tr.health_bin[CV][100:] = 3
        rep = conformance([tr], MAP, MODELS)
        assert rep.a_phys_violations == 1 and dict(rep.by_variable) == {"conveyor.h": 1}

    def test_a_failure_from_a_bin_below_the_top_is_a_violation_and_a_missed_one(self) -> None:
        tr = quiet(200)
        tr.health_bin[CR][:100] = 2
        tr.health_bin[CR][100:] = 3
        tr.op[CR][100:] = op_index(Op.FAIL)
        tr.uncontrolled_failure[CR][100] = True
        rep = conformance([tr], MAP, MODELS)
        assert rep.a_phys_by_kind["failure_below_top"] == 1
        assert rep.a_phys_violations == 1
        assert rep.violation_edges == 1 and rep.missed_violations == 1
        assert rep.ticks_in_violation == 1  # the tick after the failure starts inside it

    def test_a_failure_from_the_top_bin_is_the_game_sink_and_not_a_violation_of_a_phys(
        self,
    ) -> None:
        tr = quiet(200)
        tr.health_bin[CR][:] = 3
        tr.op[CR][100:] = op_index(Op.FAIL)
        tr.uncontrolled_failure[CR][100] = True
        rep = conformance([tr], MAP, MODELS)
        assert rep.a_phys_violations == 0
        assert rep.violation_edges == 1 and rep.missed_violations == 0

    def test_the_kinds_are_the_published_ones(self) -> None:
        assert A_PHYS_KINDS == ("bin_skip", "failure_below_top")
        assert set(conformance([quiet(10)], MAP, MODELS).a_phys_by_kind) == set(A_PHYS_KINDS)


class TestBinMoves:
    """Moves of the health bin the games cannot produce, counted apart from A_phys."""

    def test_a_quiet_trace_has_none_and_the_keys_are_published(self) -> None:
        rep = conformance([quiet(50)], MAP, MODELS)
        assert dict(rep.bin_moves) == {
            "fall": 0, "rise_while_not_running": 0,
            "completion_renewed": 0, "completion_unrenewed": 0,
        }  # fmt: skip

    def test_a_fall_while_running(self) -> None:
        tr = quiet(100)
        tr.health_bin[PA][:50] = 2
        tr.health_bin[PA][50:] = 1
        assert conformance([tr], MAP, MODELS).bin_moves["fall"] == 1

    def test_a_rise_while_stopped_or_isolated_but_not_while_running(self) -> None:
        stopped = quiet(100)
        stopped.op[PA][:] = op_index(Op.ISO)
        stopped.health_bin[PA][50:] = 1
        assert conformance([stopped], MAP, MODELS).bin_moves["rise_while_not_running"] == 1
        running = quiet(100)
        running.health_bin[PA][50:] = 1
        assert conformance([running], MAP, MODELS).bin_moves["rise_while_not_running"] == 0

    @pytest.mark.parametrize(
        ("renewed", "key", "other"),
        [(True, "completion_renewed", "completion_unrenewed"),
         (False, "completion_unrenewed", "completion_renewed")],
    )  # fmt: skip
    def test_completions_are_counted_by_what_the_bin_did(
        self, renewed: bool, key: str, other: str
    ) -> None:
        tr = quiet(20)
        tr.op[PA][:10] = op_index(Op.WORK)
        tr.iso_proven[PA][:] = True
        tr.op[PA][10:] = op_index(Op.ISO)
        tr.health_bin[PA][:10] = 2
        tr.health_bin[PA][10:] = 0 if renewed else 2
        issue(tr, 10, PA, Cmd.COMPLETE)
        rep = conformance([tr], MAP, MODELS)
        assert rep.bin_moves[key] == 1 and rep.bin_moves[other] == 0
        assert rep.bin_moves["fall"] == 0, "the renewal is not a fall"

    def test_a_completion_at_bin_zero_is_neither(self) -> None:
        tr = quiet(20)
        tr.op[PA][:10] = op_index(Op.WORK)
        tr.iso_proven[PA][:] = True
        tr.op[PA][10:] = op_index(Op.ISO)
        rep = conformance([tr], MAP, MODELS)
        assert rep.bin_moves["completion_renewed"] == rep.bin_moves["completion_unrenewed"] == 0

    def test_the_two_rates_are_published(self) -> None:
        tr = quiet(100)
        tr.health_bin[PA][:50] = 2
        tr.health_bin[PA][50:] = 1
        names = {
            r.name: r
            for r in result_records(conformance([tr], MAP, MODELS), measured_on="x", script="y")
        }
        assert names["conformance.bin_move.fall"].extra["k"] == 1
        assert names["conformance.bin_move.rise_while_not_running"].extra["k"] == 0


class TestWhatTheBreakItPassFound:
    """One regression per mutant of the break-it pass that survived the first tests."""

    @staticmethod
    def entered(n: int = 40) -> Trace:
        """A clean entry: tests at 4 and every quarter from 6, ENTER at 5 after the test at 4."""
        tr = quiet(n)
        tr.op[CR][:] = op_index(Op.WORK)
        tr.iso_proven[CR][:] = True
        tr.permit[CR][:] = permit_index(Permit.VALID)
        tr.gas_age_min[CR][:4] = 1.0e6
        return tr

    def test_an_alarm_that_comes_on_with_someone_inside_is_matched_and_without_is_not(self) -> None:
        tr = self.entered(24)
        # a passing test in every quarter from 4 to 19; ENTER at 5 (no test in that quarter)
        for r in range(4, 20):
            tr.gas_age_min[CR][r] = 0.0
            issue(tr, r, CR, Cmd.TEST)
        tr.commands = [c for c in tr.commands if c[0] != 5]
        issue(tr, 5, CR, Cmd.ENTER)
        tr.gas_age_min[CR][5] = 15.0
        tr.occupied[CR][5:22] = True
        tr.alarm[CR][20:22] = True  # onset in a quarter with no test
        tr.gas_age_min[CR][20] = 15.0
        tr.gas_age_min[CR][21] = 30.0
        issue(tr, 22, CR, Cmd.EXIT)
        rep = conformance([tr], MAP, MODELS)
        assert dict(rep.by_variable).get("crusher.alarm", 0) == 0, rep.unmatched_examples
        nobody = quiet(40)
        nobody.alarm[CR][20:] = True
        assert conformance([nobody], MAP, MODELS).by_variable["crusher.alarm"] == 1

    def test_a_test_two_quarters_old_is_not_a_fresh_one_at_the_moment_of_entry(self) -> None:
        tr = self.entered(20)
        tr.gas_age_min[CR][4:] = 30.0  # the last passing test is 30 minutes old when they go in
        tr.occupied[CR][8:] = True
        issue(tr, 8, CR, Cmd.ENTER)
        rep = conformance([tr], MAP, MODELS, break_q=1)
        assert rep.violation_edges >= 1
        loose = conformance([tr], MAP, MODELS, break_q=2)
        assert loose.violation_edges < rep.violation_edges

    def test_the_game_sink_does_not_excuse_a_state_the_plant_is_happy_with(self) -> None:
        """A test two quarters old with someone inside: fine in the plant, stale to the game."""
        tr = self.entered(12)
        tr.gas_age_min[CR][9] = 0.0  # a pass at 9
        issue(tr, 9, CR, Cmd.TEST)
        tr.occupied[CR][10:] = True
        tr.gas_age_min[CR][10] = 15.0
        issue(tr, 10, CR, Cmd.ENTER)
        tr.gas_age_min[CR][11] = 30.0  # not older than the period: no violation in the plant
        rep = conformance([tr], MAP, MODELS)
        assert rep.violation_edges == 0
        assert rep.unmatched == 1 and dict(rep.by_variable) == {"crusher.state": 1}
        assert (
            rep.unmatched_when_safe == 0
        )  # the game's outcome is the sink: a shield never admits it

    def test_the_closest_outcome_decides_which_variable_is_blamed(self) -> None:
        tr = quiet(40)
        tr.op[CR][30:] = op_index(Op.ISO)  # row 30 is the one with a handover announced
        rep = conformance([tr], MAP, MODELS)
        assert rep.unmatched == 1 and dict(rep.by_variable) == {"crusher.op": 1}

    def test_an_occupant_whose_permit_lapsed_at_the_handover_is_a_caught_p2_edge(self) -> None:
        tr = self.entered(40)
        for r in range(4, 40):
            tr.gas_age_min[CR][r] = 0.0
            if r != 5:
                issue(tr, r, CR, Cmd.TEST)
        issue(tr, 5, CR, Cmd.ENTER)
        tr.gas_age_min[CR][5] = 15.0
        tr.occupied[CR][5:] = True
        tr.permit[CR][31:] = permit_index(Permit.REVAL)  # the handover after row 30
        rep = conformance([tr], MAP, MODELS)
        assert rep.violation_edges == 1 and rep.missed_violations == 0
        assert rep.unmatched == 0, rep.unmatched_examples


class TestViolationsAndSkippedTicks:
    def test_work_on_an_unproven_asset_is_a_caught_edge_and_the_rest_is_inside(self) -> None:
        tr = quiet(200)
        tr.op[CR][100:111] = op_index(Op.WORK)  # iso_proven stays False
        issue(tr, 100, CR, Cmd.BEGIN)
        rep = conformance([tr], MAP, MODELS)
        assert rep.violation_edges == 1 and rep.missed_violations == 0
        assert rep.ticks_in_violation >= 10  # every tick that begins inside the violation
        assert rep.ticks_total == 3 * 199

    def test_energising_during_work_is_a_violation_edge(self) -> None:
        tr = quiet(200)
        tr.op[CR][50:100] = op_index(Op.WORK)
        tr.iso_proven[CR][50:100] = True
        issue(tr, 100, CR, Cmd.START)  # WORK to RUN
        rep = conformance([tr], MAP, MODELS)
        assert rep.violation_edges >= 1 and rep.missed_violations == 0

    def test_both_pumps_locked_is_a_caught_p4_edge(self) -> None:
        """0003 A12.2: the edge is the second lock; it used to be the second PROVE."""
        tr = quiet(200)
        for pump in (PA, PB):
            tr.op[pump][90:] = op_index(Op.STOP)
            tr.op[pump][100:] = op_index(Op.LOCKED)
            issue(tr, 90, pump, Cmd.STOP)
            issue(tr, 100, pump, Cmd.LOCK)
        rep = conformance([tr], MAP, MODELS)
        assert rep.violation_edges == 1 and rep.missed_violations == 0
        assert rep.unmatched == 0, rep.unmatched_examples
        assert rep.ticks_in_violation == 99  # the rest of the trace begins inside it

    def test_a_locked_pump_and_a_failure_of_the_other_is_a_caught_p4_edge(self) -> None:
        """Seed 1167, tick 7399 in miniature: PU-201 locked, PU-202 fails from the top bin."""
        tr = quiet(200)
        tr.op[PA][80:] = op_index(Op.STOP)
        tr.op[PA][90:] = op_index(Op.LOCKED)
        issue(tr, 80, PA, Cmd.STOP)
        issue(tr, 90, PA, Cmd.LOCK)
        tr.health_bin[PB][:] = 3
        tr.op[PB][100:] = op_index(Op.FAIL)
        rep = conformance([tr], MAP, MODELS)
        assert rep.violation_edges == 1 and rep.missed_violations == 0
        assert rep.unmatched == 0, rep.unmatched_examples

    def test_a_locked_pump_with_the_other_running_is_not_a_violation(self) -> None:
        tr = quiet(200)
        tr.op[PA][80:] = op_index(Op.STOP)
        tr.op[PA][90:] = op_index(Op.LOCKED)
        issue(tr, 80, PA, Cmd.STOP)
        issue(tr, 90, PA, Cmd.LOCK)
        rep = conformance([tr], MAP, MODELS)
        assert rep.violation_edges == 0 and rep.ticks_in_violation == 0 and rep.unmatched == 0

    def test_work_on_an_unproven_conveyor_is_a_caught_p1_edge(self) -> None:
        """0003 A12.3 and P12a F3: the conveyor's P1, which no game covered before."""
        tr = quiet(200)
        tr.op[CV][100:111] = op_index(Op.WORK)  # iso_proven stays False
        issue(tr, 100, CV, Cmd.BEGIN)
        rep = conformance([tr], MAP, MODELS)
        assert rep.violation_edges == 1 and rep.missed_violations == 0
        assert rep.unmatched == 0, rep.unmatched_examples
        assert rep.ticks_in_violation >= 10

    def test_energising_a_conveyor_while_work_is_open_is_a_caught_edge(self) -> None:
        tr = quiet(200)
        tr.op[CV][50:100] = op_index(Op.WORK)
        tr.iso_proven[CV][50:100] = True
        issue(tr, 100, CV, Cmd.START)  # WORK to RUN
        rep = conformance([tr], MAP, MODELS)
        assert rep.violation_edges >= 1 and rep.missed_violations == 0

    def test_entering_on_a_stale_test_is_a_caught_edge(self) -> None:
        tr = quiet(200)
        tr.op[CR][:] = op_index(Op.WORK)
        tr.iso_proven[CR][:] = True
        tr.permit[CR][:] = permit_index(Permit.VALID)
        tr.occupied[CR][100:] = True  # entering with no test ever
        issue(tr, 100, CR, Cmd.ENTER)
        rep = conformance([tr], MAP, MODELS)
        assert rep.violation_edges >= 1 and rep.missed_violations == 0

    def test_two_commands_for_one_component_in_a_quarter_are_not_compared(self) -> None:
        tr = quiet(200)
        issue(tr, 100, CR, Cmd.STOP)
        issue(tr, 100, CR, Cmd.LOCK)
        rep = conformance([tr], MAP, MODELS)
        assert rep.multi_command_ticks == 1
        assert rep.ticks == rep.ticks_total - 1
        assert rep.unmatched == 0

    def test_noop_and_planning_commands_are_not_commands(self) -> None:
        tr = quiet(200)
        tr.commands.append((100, command(CR, Cmd.NOOP), True, False))
        tr.commands.append((100, command(CR, PlanCmd.RAISE_WO), True, False))
        tr.commands.append((100, command(CR, Cmd.STOP), True, False))
        rep = conformance([tr], MAP, MODELS)
        assert rep.multi_command_ticks == 0

    def test_ticks_add_up(self) -> None:
        tr = quiet(300)
        tr.op[CR][100:111] = op_index(Op.WORK)
        issue(tr, 100, CR, Cmd.BEGIN)
        issue(tr, 200, PA, Cmd.STOP)
        issue(tr, 200, PA, Cmd.START)
        rep = conformance([tr], MAP, MODELS)
        assert (
            rep.ticks + rep.ticks_in_violation + rep.violation_edges + rep.multi_command_ticks
            == rep.ticks_total
        )


class TestWrongModeCompletionIsMatched:
    """ADR 0003 A11: completing a job on the wrong mode does not renew the component."""

    def completion(self, asset: str, renewed: bool) -> Trace:
        tr = quiet(20)  # no handover inside the window
        if asset == CR:
            tr.op[CR][:10] = op_index(Op.WORK)
            tr.iso_proven[CR][:] = True
            tr.op[CR][10:] = op_index(Op.ISO)
            tr.permit[CR][:] = permit_index(Permit.VALID)
        else:
            tr.op[asset][:10] = op_index(Op.WORK)
            tr.iso_proven[asset][:] = True
            tr.op[asset][10:] = op_index(Op.ISO)
        tr.health_bin[asset][:10] = 2
        tr.health_bin[asset][10:] = 0 if renewed else 2
        issue(tr, 10, asset, Cmd.COMPLETE)
        return tr

    @pytest.mark.parametrize("asset", [CR, PA, CV])
    def test_the_unrenewed_completion_conforms(self, asset: str) -> None:
        rep = conformance([self.completion(asset, renewed=False)], MAP, MODELS)
        assert rep.unmatched == 0, rep.unmatched_examples

    @pytest.mark.parametrize("asset", [CR, PA, CV])
    def test_the_renewed_completion_conforms(self, asset: str) -> None:
        rep = conformance([self.completion(asset, renewed=True)], MAP, MODELS)
        assert rep.unmatched == 0

    @pytest.mark.parametrize(("asset", "variable"), [(CR, "crusher.h"), (PA, "pumps.h1")])
    def test_it_was_unmatched_under_the_old_relation(self, asset: str, variable: str) -> None:
        rep = conformance([self.completion(asset, renewed=False)], MAP, LEGACY)
        assert rep.unmatched == 1 and dict(rep.by_variable) == {variable: 1}
        assert conformance([self.completion(asset, renewed=True)], MAP, LEGACY).unmatched == 0

    def test_in_the_real_plant_a_wrong_mode_job_completes_and_the_tick_is_matched(self) -> None:
        plant, _base = wrong_worlds()
        tr = plant.trace()
        assert plant.stats().wrong_mode_jobs == 1
        rep = conformance([tr], MAP, MODELS)
        assert not [e for e in rep.unmatched_examples if e[2] == "COMPLETE"], rep.unmatched_examples
        completes = [q for q, c, ok, _ in tr.commands if c.cmd is Cmd.COMPLETE and ok]
        assert completes, "the job was completed"
        old = conformance([tr], MAP, LEGACY)
        assert [e for e in old.unmatched_examples if e[2] == "COMPLETE"], "and it was not before"


class TestRealPlant:
    def test_a_correct_scripted_job_conforms_completely(self) -> None:
        plant = fresh(seed_with_mode(CR, "LINER_WEAR"))
        advance(plant, confined_job("LINER_WEAR"), until=300)
        rep = conformance([plant.trace()], MAP, MODELS)
        assert rep.unmatched == 0 and rep.missed_violations == 0
        assert rep.violation_edges == 0 and rep.a_phys_violations == 0
        assert rep.ticks + rep.multi_command_ticks == rep.ticks_total > 700

    @pytest.mark.parametrize("hours", [12.0, 6.0, 4.0])
    def test_a_job_whose_permit_is_spent_and_replaced_conforms_too(self, hours: float) -> None:
        """A12.1 against the games: the runner leaves on `soon`, issues in WORK, goes back in."""
        plant = fresh(seed_with_mode(CR, "LINER_WEAR"), approver_catch_p=0.0, permit_max_h=hours)
        from closed_loop.plant.jobs import work_order_command

        mode = plant.truth.life_mode[CR]
        violations, _ = run_with_runner(plant, {2: [work_order_command(CR, mode)]}, until=400)
        assert violations == []
        rep = conformance([plant.trace()], MAP, MODELS, permit_max_h=hours)
        assert rep.unmatched == 0, rep.unmatched_examples
        assert rep.missed_violations == 0 and rep.violation_edges == 0

    def test_the_wrong_mode_world_has_no_a_phys_violation(self) -> None:
        plant, _ = wrong_worlds()
        assert conformance([plant.trace()], MAP, MODELS).a_phys_violations == 0


class TestDeterminism:
    def traces(self) -> list[Trace]:
        a, b = quiet(200), quiet(150)
        a.op[CR][100:] = op_index(Op.ISO)
        b.health_bin[PA][70:] = 2
        return [a, b]

    def test_same_traces_same_report(self) -> None:
        assert conformance(self.traces(), MAP, MODELS) == conformance(self.traces(), MAP, MODELS)

    def test_the_order_of_the_traces_does_not_matter(self) -> None:
        a, b = self.traces()
        x, y = conformance([a, b], MAP, MODELS), conformance([b, a], MAP, MODELS)
        assert (x.ticks, x.unmatched, x.by_variable, x.a_phys_violations) == (
            y.ticks, y.unmatched, y.by_variable, y.a_phys_violations,
        )  # fmt: skip
        assert x.unmatched_examples == y.unmatched_examples

    def test_reports_add_across_traces(self) -> None:
        a, b = self.traces()
        both = conformance([a, b], MAP, MODELS)
        parts = [conformance([a], MAP, MODELS), conformance([b], MAP, MODELS)]
        assert both.ticks == sum(p.ticks for p in parts)
        assert both.unmatched == sum(p.unmatched for p in parts)
        assert both.episodes == 2

    def test_an_empty_list_is_an_empty_report(self) -> None:
        rep = conformance([], MAP, MODELS)
        assert (rep.ticks, rep.unmatched, rep.a_phys_violations) == (0, 0, 0)
        assert rep.wilson == (0.0, 1.0)

    def test_a_trace_of_one_quarter_has_no_ticks(self) -> None:
        assert conformance([quiet(1)], MAP, MODELS).ticks == 0


class TestRecords:
    def report(self):  # type: ignore[no-untyped-def]
        tr = quiet(400)
        tr.op[CR][100:] = op_index(Op.ISO)
        tr.health_bin[CR][300:] = 2
        return conformance([tr], MAP, MODELS)

    def records(self) -> list[ResultRecord]:
        return result_records(
            self.report(), measured_on="synthetic trace, 1 episode", script="tests.shield"
        )

    def test_every_record_names_what_it_was_measured_on_and_has_a_denominator(self) -> None:
        recs = self.records()
        assert recs
        assert len({r.name for r in recs}) == len(recs)
        for r in recs:
            assert r.measured_on == "synthetic trace, 1 episode" and r.script == "tests.shield"
            assert r.fidelity.value == "F1"
            assert r.ci_lo is not None and r.ci_hi is not None and r.ci_kind == "wilson-0.95"
            assert r.ci_lo <= r.value <= r.ci_hi
            assert "k" in r.extra

    def test_the_headline_record_is_the_report(self) -> None:
        rep = self.report()
        head = next(r for r in self.records() if r.name == "conformance.unmatched")
        assert head.n == rep.ticks and head.extra["k"] == rep.unmatched
        assert head.value == pytest.approx(rep.unmatched / rep.ticks)
        assert (head.ci_lo, head.ci_hi) == pytest.approx(rep.wilson)

    def test_variables_with_no_event_do_not_appear_but_the_totals_with_zero_do(self) -> None:
        names = {r.name for r in self.records()}
        assert "conformance.unmatched.crusher.op" in names
        assert "conformance.unmatched.pumps.op1" not in names
        phys = next(r for r in self.records() if r.name == "conformance.a_phys")
        assert phys.extra["k"] == 1 and phys.n == 4 * 399
        none = next(r for r in self.records() if r.name == "conformance.a_phys.failure_below_top")
        assert none.extra["k"] == 0 and none.ci_lo == 0.0 and none.ci_hi > 0.0

    def test_an_empty_measured_on_is_refused(self) -> None:
        with pytest.raises(ValueError, match="measured_on"):
            result_records(self.report(), measured_on="", script="s")


class TestTheMapDrivesTheRuntime:
    """The map is what the runtime shield reads. With sets that contain the true bin, a careless
    proposer behind the shield does nothing the crusher job, the pump pair or the conveyor games
    forbid."""

    @staticmethod
    def sets_from_truth(plant) -> None:  # type: ignore[no-untyped-def]
        # the simulation side plays a perception layer that is told the truth, one bin of margin
        # each way because the view reaches the shield a quarter later
        for aid, b in plant.truth.health_bin.items():
            plant.set_perception(aid, set_lo=max(b - 1, 0), set_hi=min(b + 1, 3))

    def run(self, seed: int, *, shielded: bool, tables):  # type: ignore[no-untyped-def]
        from closed_loop.proposers import Careless, make_baseline
        from tests.proposers._loop import run_episode

        crusher_tables, pump_tables, conveyor_tables = tables

        def wrap(plant, proposer):  # type: ignore[no-untyped-def]
            if not shielded:
                return proposer
            return runtime.ShieldedProposer(
                proposer, crusher_tables[3], pump_tables[3], MAP, conveyor=conveyor_tables[3]
            )

        return run_episode(
            Careless(make_baseline("B2"), p_skip=0.2),
            seed,
            wrap=wrap,
            each_quarter=self.sets_from_truth,
        )

    def test_no_violation_on_any_covered_asset_behind_the_shield(
        self, crusher_tables, pump_tables, conveyor_tables
    ) -> None:  # type: ignore[no-untyped-def]
        covered = {CR, PA, PB, f"{PA}|{PB}", CV}
        tables = (crusher_tables, pump_tables, conveyor_tables)
        shielded_total = bare_total = bare_belt = 0
        for seed in (1, 2, 3, 4, 5, 6):
            behind = self.run(seed, shielded=True, tables=tables)
            bare = self.run(seed, shielded=False, tables=tables)
            shielded_total += sum(1 for _p, a, _q in behind.violations if a in covered)
            bare_total += sum(1 for _p, a, _q in bare.violations if a in covered)
            bare_belt += sum(1 for _p, a, _q in bare.violations if a == CV)
        assert shielded_total == 0
        assert bare_total > 0, "the careless fitter does violate without the shield"
        assert bare_belt > 0, "and on the belt too, which is what the conveyor game is for"

    def test_the_conveyor_is_no_longer_what_gets_through(
        self, crusher_tables, pump_tables, conveyor_tables
    ) -> None:  # type: ignore[no-untyped-def]
        """P5c's recorded gap (nine P1 violations on CV-101 behind the shield) is closed."""
        tables = (crusher_tables, pump_tables, conveyor_tables)
        seen = set()
        for seed in (1, 2, 3, 4, 5, 6):
            ep = self.run(seed, shielded=True, tables=tables)
            seen |= {(p, a) for p, a, _q in ep.violations}
        assert seen == set()


class TestACarelessConveyorCommand:
    """The regression of P12a F3 (41 plant P1 events on CV-101 in 200 episodes behind the shield,
    none of them in M-VIOL): a careless BEGIN or START on the belt is filtered by the shield, and
    with the shield off the plant records it (and, for BEGIN, the specification does too)."""

    @staticmethod
    def play(  # type: ignore[no-untyped-def]
        script, conveyor_tables, crusher_tables, pump_tables, *, shielded: bool
    ):
        """Quarter 0 is a warm-up (narrow sets reach the views from quarter 1); the script's keys
        are decision quarters from 1 on."""
        plant = fresh(7)
        for a in ASSET_IDS:
            plant.set_perception(a, set_lo=0, set_hi=0)
        inner = SimpleNamespace(
            name="script", reset=lambda s: None, propose=lambda o: list(script.get(o.quarter, []))
        )
        proposer = (
            runtime.ShieldedProposer(
                inner, crusher_tables[4], pump_tables[4], MAP, conveyor=conveyor_tables[4]
            )
            if shielded
            else inner
        )
        obs, violations = plant.step([]).obs, []
        for _ in range(16):
            res = plant.step(proposer.propose(obs))  # type: ignore[arg-type]
            violations += res.violations
            obs = res.obs
        return plant, violations, proposer

    def test_a_begin_on_the_running_belt(
        self, crusher_tables, pump_tables, conveyor_tables
    ) -> None:  # type: ignore[no-untyped-def]
        script = {4: [command(CV, Cmd.BEGIN)]}
        tables = dict(
            crusher_tables=crusher_tables, pump_tables=pump_tables, conveyor_tables=conveyor_tables
        )
        plant, violations, _ = self.play(script, shielded=False, **tables)
        assert violations == [("P1", CV, 4)]  # the plant's record, at the quarter of the BEGIN
        rob = evaluate(plant.trace(), SpecParams(), CR, (PA, PB))
        assert rob["P1"] < 0  # and the specification's, since A12.3
        plant, violations, shield = self.play(script, shielded=True, **tables)
        assert violations == [] and evaluate(plant.trace(), SpecParams(), CR, (PA, PB))["P1"] >= 0
        assert shield.overrides >= 1
        assert not any(
            c.cmd is Cmd.BEGIN for q, c, _ok, _o in plant.trace().commands if c.asset_id == CV
        )

    def test_a_start_while_the_work_is_open(
        self, crusher_tables, pump_tables, conveyor_tables
    ) -> None:  # type: ignore[no-untyped-def]
        script = {
            1: [command(CV, Cmd.STOP)], 2: [command(CV, Cmd.LOCK)], 3: [command(CV, Cmd.PROVE)],
            4: [command(CV, Cmd.ISSUE)], 5: [command(CV, Cmd.BEGIN)], 9: [command(CV, Cmd.START)],
        }  # fmt: skip
        tables = dict(
            crusher_tables=crusher_tables, pump_tables=pump_tables, conveyor_tables=conveyor_tables
        )
        plant, violations, _ = self.play(script, shielded=False, **tables)
        assert [v for v in violations if v[1] == CV] == [("P1", CV, 9)]  # an event of the plant
        assert plant.trace().op[CV][9] == list(Op).index(Op.RUN)
        # the state formula cannot see it (the A4 note): the specification stays silent
        assert evaluate(plant.trace(), SpecParams(), CR, (PA, PB))["P1"] >= 0
        plant, violations, shield = self.play(script, shielded=True, **tables)
        assert violations == [] and plant.trace().op[CV][9] == list(Op).index(Op.WORK)
        assert shield.overrides == 1  # the START, replaced by a STOP that changes nothing in WORK

    def test_the_proper_job_goes_through_the_shield_untouched(
        self, crusher_tables, pump_tables, conveyor_tables
    ) -> None:  # type: ignore[no-untyped-def]
        script = {
            1: [command(CV, Cmd.STOP)], 2: [command(CV, Cmd.LOCK)], 3: [command(CV, Cmd.PROVE)],
            4: [command(CV, Cmd.ISSUE)], 5: [command(CV, Cmd.BEGIN)],
            9: [command(CV, Cmd.COMPLETE)], 10: [command(CV, Cmd.CLOSE)],
            11: [command(CV, Cmd.UNISO)], 12: [command(CV, Cmd.START)],
        }  # fmt: skip
        plant, violations, shield = self.play(
            script,
            crusher_tables=crusher_tables,
            pump_tables=pump_tables,
            conveyor_tables=conveyor_tables,
            shielded=True,
        )
        assert violations == [] and shield.overrides == 0
        assert plant.trace().op[CV][13] == list(Op).index(Op.RUN)
