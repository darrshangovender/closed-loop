"""P3b: twin worlds. Same exogenous streams, replayed with do(): abduction, action, prediction.

A policy change must not perturb the noise of an unrelated asset, of the same asset's other lives,
or of any later event. All randomness is keyed through rng_for(root_seed, stream, *ids), so these
tests compare worlds that differ in one action and require everything else to match bit for bit.
"""

import numpy as np
import pytest

from closed_loop.core.types import Cmd, Command, Fidelity, PlanCmd
from closed_loop.plant import DO_REPLACE, Scenario, make_plant
from closed_loop.plant.observe import observe_day, simulate_lives

from ._helpers import asset_arrays_equal
from ._scripts import (
    CR,
    CV,
    P1,
    P2,
    C,
    advance,
    confined_job,
    fresh,
    scripted_params,
    seed_with_mode,
    shifted,
)

OTHERS = (CR, CV, P2)


def replace_cmd(asset: str) -> Command:
    return Command(asset, PlanCmd.RAISE_WO, "any", source=DO_REPLACE)


def run_to_end(plant, commands_by_q=None):
    commands_by_q = commands_by_q or {}
    while True:
        r = plant.step(list(commands_by_q.get(plant.des.q, ())))
        if r.done:
            return


class TestTwinOfNothing:
    def test_twin_with_no_intervention_reproduces_the_factual_trace_bit_for_bit(self):
        factual = fresh(3, p_fail_in_episode=0.6)
        run_to_end(factual)
        twin = factual.twin([])
        run_to_end(twin)
        assert factual.trace().identical_to(twin.trace())

    def test_the_twin_replays_a_factual_run_that_had_commands(self):
        sched = confined_job("LINER_WEAR")
        seed = seed_with_mode(CR, "LINER_WEAR")
        factual = fresh(seed, p_fail_in_episode=0.6)
        advance(factual, sched, until=300)
        twin = factual.twin([])
        advance(twin, sched, until=300)
        assert factual.trace().identical_to(twin.trace())

    def test_the_same_commands_given_as_interventions_give_the_same_trace(self):
        sched = confined_job("LINER_WEAR")
        seed = seed_with_mode(CR, "LINER_WEAR")
        factual = fresh(seed, p_fail_in_episode=0.6)
        advance(factual, sched, until=300)
        replay = factual.twin([(q, c) for q, cmds in sched.items() for c in cmds])
        advance(replay, {}, until=300)
        assert factual.trace().identical_to(replay.trace())

    def test_a_twin_starts_from_a_clean_reset_with_the_same_seed_and_scenario(self):
        factual = fresh(9)
        run_to_end(factual)
        twin = factual.twin([])
        assert twin.des.q == 0 and twin._seed == 9
        assert twin.initial_observation.assets == factual.initial_observation.assets

    def test_a_twin_of_a_twin_is_still_the_same_world(self):
        factual = fresh(3, p_fail_in_episode=0.6)
        run_to_end(factual)
        t2 = factual.twin([(500, C(Cmd.NOOP, CV))]).twin([])
        run_to_end(t2)
        assert factual.trace().identical_to(t2.trace())  # interventions are not inherited

    def test_resetting_a_twin_drops_its_interventions(self):
        factual = fresh(3, p_fail_in_episode=0.6)
        run_to_end(factual)
        twin = factual.twin([(200, replace_cmd(P1))])
        twin.reset(3, Scenario())  # a reset starts a fresh world
        run_to_end(twin)
        assert factual.trace().identical_to(twin.trace())

    def test_negative_intervention_quarters_are_refused(self):
        with pytest.raises(ValueError):
            fresh(1).twin([(-1, C(Cmd.NOOP))])


class TestDoReplace:
    def pump_failure_seed(self):
        params = scripted_params(p_fail_in_episode=0.6)
        for s in range(300):
            t = simulate_lives(params, s, Scenario())
            if t.fails_in_horizon(P1) and t.failure_day[P1] > 40:
                return s
        raise AssertionError

    def test_replacing_one_asset_leaves_every_other_asset_identical(self):
        seed = self.pump_failure_seed()
        factual = fresh(seed, p_fail_in_episode=0.6)
        run_to_end(factual)
        twin = factual.twin([(1500, replace_cmd(P1))])
        run_to_end(twin)
        tf, tt = factual.trace(), twin.trace()
        for aid in OTHERS:
            assert asset_arrays_equal(tf, tt, aid), aid
        assert not asset_arrays_equal(tf, tt, P1)

    def test_replacing_before_the_failure_prevents_it_and_starts_a_new_life(self):
        seed = self.pump_failure_seed()
        factual = fresh(seed, p_fail_in_episode=0.6)
        run_to_end(factual)
        fq = int(np.argmax(factual.trace().uncontrolled_failure[P1]))
        assert factual.trace().uncontrolled_failure[P1][fq]
        twin = factual.twin([(1500, replace_cmd(P1))])
        run_to_end(twin)
        fail = twin.trace().uncontrolled_failure[P1]
        assert not fail[fq]  # the old component's failure never happens
        # the new component (onset at least 24 days after the swap) may fail late in the episode,
        # but far after the old one would have
        assert not fail.any() or int(np.argmax(fail)) > fq + 96
        assert twin.truth.life_id[P1] == 1 and factual.truth.life_id[P1] == 0
        assert twin.stats().planned_jobs == 0  # a do() is an intervention, not a paid job

    def test_before_the_intervention_the_two_worlds_are_one(self):
        seed = self.pump_failure_seed()
        factual = fresh(seed, p_fail_in_episode=0.6)
        advance(factual, {}, until=3000)
        twin = factual.twin([(1500, replace_cmd(P1))])
        advance(twin, {}, until=1499)
        prefix = factual.trace()
        assert asset_arrays_equal(prefix, twin.trace(), P1)  # compared on the shorter length

    def test_a_replace_of_a_failed_asset_returns_it_to_service(self):
        seed = self.pump_failure_seed()
        factual = fresh(seed, p_fail_in_episode=0.6)
        run_to_end(factual)
        fq = int(np.argmax(factual.trace().uncontrolled_failure[P1]))
        twin = factual.twin([(fq + 10, replace_cmd(P1))])
        advance(twin, {}, until=fq + 20)
        assert twin.des.by_id[P1].op.value == "RUN" and twin.truth.life_id[P1] == 1


class TestAbductionActionPrediction:
    """The exogenous noise is the same under a different action; only the response differs."""

    def residual(self, plant, aid, day):
        """Reading minus its noise-free model: the exogenous term, from the observation."""
        view = plant.des.by_id[aid]
        obs = plant._last_obs[aid]
        q = day * 96
        sev = view.clock.severity(q)
        load = float(plant._phys.load_day[aid][day])
        quiet = scripted_params(
            hi_noise_sd=0.0, sig_noise_sd=0.0, load_obs_sd=0.0, p_fail_in_episode=0.6
        )
        clean = observe_day(quiet, plant._obs_streams[aid], day, sev, view.life.signature, load)
        assert obs is not None and obs.hi is not None and clean.hi is not None
        # observe_day with sd 0 still adds the drift; subtract both from the noisy reading
        return (obs.hi - clean.hi) / plant.params.hi_noise_sd

    def test_the_noise_of_the_replaced_asset_is_the_same_before_and_after_the_action(self):
        factual = fresh(5, p_fail_in_episode=0.6)
        twin = factual.twin([(2000, replace_cmd(P1))])
        z = factual._obs_streams[P1].hi_noise
        for plant in (factual, twin):
            for day in range(40, 60):
                advance(plant, {}, until=day * 96 - 1)
                plant.step(())
                if plant._last_obs[P1].hi is not None:
                    assert self.residual(plant, P1, day) == pytest.approx(z[day], abs=1e-9), (day,)

    def test_the_streams_are_the_very_same_arrays_in_both_worlds(self):
        factual = fresh(5)
        twin = factual.twin([(2000, replace_cmd(P1))])
        for aid in (CR, CV, P1, P2):
            a, b = factual._obs_streams[aid], twin._obs_streams[aid]
            assert a.hi_noise.tobytes() == b.hi_noise.tobytes()
            assert a.sig_noise.tobytes() == b.sig_noise.tobytes()
            assert a.masked.tobytes() == b.masked.tobytes()
            for stream in ("u_refuse", "u_holder", "u_gas", "appr_z", "appr_c", "insp_z"):
                assert (
                    getattr(factual.des.st, stream)[aid].tobytes()
                    == getattr(twin.des.st, stream)[aid].tobytes()
                )
        assert factual.des.st.notes.tobytes() == twin.des.st.notes.tobytes()
        assert factual._phys.cum[P1].tobytes() == twin._phys.cum[P1].tobytes()

    def test_a_job_on_one_asset_does_not_perturb_the_others_or_any_note(self):
        seed = seed_with_mode(CR, "LINER_WEAR")
        idle = fresh(seed, p_fail_in_episode=0.6)
        busy = fresh(seed, p_fail_in_episode=0.6)
        notes_idle, notes_busy = [], []
        sched = shifted(confined_job("LINER_WEAR"), 96 * 10)
        while True:
            a = idle.step(())
            b = busy.step(list(sched.get(busy.des.q, ())))
            notes_idle.append(a.obs.note)
            notes_busy.append(b.obs.note)
            if a.done:
                break
        ti, tb = idle.trace(), busy.trace()
        for aid in (CV, P1, P2):
            assert asset_arrays_equal(ti, tb, aid), aid
        assert notes_idle == notes_busy
        assert not asset_arrays_equal(ti, tb, CR)
        # and the readings of the untouched assets, day by day
        for aid in (CV, P1, P2):
            assert idle._last_obs[aid] == busy._last_obs[aid]

    def test_a_later_work_orders_latency_does_not_depend_on_when_the_earlier_one_was_raised(self):
        def latency_of_second(first_at: int) -> int:
            plant = make_plant(scripted_params(approver_catch_p=1.0, approver_sigma=0.6,
                                               approver_median_h=2.0))  # fmt: skip
            plant.reset(4, Scenario())
            true_mode = plant.truth.life_mode[P1]
            wrong = "BEARING_DEFECT" if true_mode == "IMPELLER_WEAR" else "IMPELLER_WEAR"
            while plant.des.q < first_at:
                plant.step(())
            plant.step([C(PlanCmd.RAISE_WO, P1, wrong)])  # caught: the work order disappears
            while plant.des.q < 100:
                plant.step(())
            plant.step([C(PlanCmd.RAISE_WO, P1, true_mode)])
            raised = 100
            r = plant.step(())
            while (
                r.obs.assets[P1].job is not None
                and r.obs.assets[P1].job.status == "AWAITING_APPROVAL"
            ):
                r = plant.step(())
            return r.obs.quarter - raised

        assert latency_of_second(0) == latency_of_second(40) == latency_of_second(70)

    def test_a_work_orders_latency_belongs_to_its_asset_not_to_a_global_count(self):
        def latency(with_crusher_order_first: bool) -> int:
            plant = make_plant(
                scripted_params(approver_catch_p=0.0, approver_sigma=0.6, approver_median_h=2.0)
            )
            plant.reset(4, Scenario())
            if with_crusher_order_first:
                plant.step(
                    [C(PlanCmd.RAISE_WO, CR, "BEARING_DEFECT")]
                )  # an unrelated asset's order
            while plant.des.q < 50:
                plant.step(())
            plant.step([C(PlanCmd.RAISE_WO, P1, "BEARING_DEFECT")])
            r = plant.step(())
            while r.obs.assets[P1].job.status == "AWAITING_APPROVAL":
                r = plant.step(())
            return r.obs.quarter - 50

        assert latency(False) == latency(True)

    def test_the_refusal_draw_belongs_to_the_quarter_not_to_the_call_order(self):
        a, b = fresh(2, p_refuse=0.5), fresh(2, p_refuse=0.5)
        outcomes_a = [bool(a.step([C(Cmd.START, CV)]).refused) for _ in range(200)]
        # b makes other calls in between; its START at the same quarter meets the same draw
        outcomes_b = []
        for _ in range(200):
            r = b.step([C(Cmd.START, P1), C(Cmd.START, CV), C(Cmd.START, P2)])
            outcomes_b.append(any(c.asset_id == CV for c in r.refused))
        assert outcomes_a == outcomes_b


class TestDerateTwin:
    def test_do_derate_extends_the_remaining_life_and_leaves_other_assets_alone(self):
        params = scripted_params(p_fail_in_episode=0.6)
        seed = next(
            s for s in range(300)
            if simulate_lives(params, s, Scenario()).fails_in_horizon(P1)
            and simulate_lives(params, s, Scenario()).failure_day[P1] > 40
        )  # fmt: skip
        factual = make_plant(params, fidelity=Fidelity.F1D)
        factual.reset(seed, Scenario())
        advance(factual, {}, until=2000)
        twin = factual.twin([(1000, Command(P1, Cmd.DERATE, source="do"))])
        advance(twin, {}, until=2000)
        assert twin.truth.failure_time_q[P1] > factual.truth.failure_time_q[P1] + 96
        tf, tt = factual.trace(), twin.trace()
        for aid in OTHERS:
            assert asset_arrays_equal(tf, tt, aid)
