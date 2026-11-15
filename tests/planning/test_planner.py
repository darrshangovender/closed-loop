"""P8b: the belief planner on hand-built cases whose answers are worked out on paper.

Each class states a situation small enough to solve by hand (a deterministic clock, a two-state
trap, a doomed-or-safe pair) and checks the planner against the arithmetic, then against the exact
value of the same instance (``exactdp``) where the two should agree.
"""

from __future__ import annotations

import numpy as np
import pytest

from closed_loop.core.types import Op, PlanCmd
from closed_loop.planning.exactdp import exact_value, validation_beliefs, validation_model
from closed_loop.planning.planner import (
    AssetAssessment,
    BeliefPlanner,
    PlanConstraints,
    PlanCosts,
    mean_life_days,
    particles_from_belief,
)
from closed_loop.planning.worldmodel import (
    ACTION_CONTINUE,
    ACTION_INSPECT,
    ACTION_REPLACE,
    Particles,
    TabularWorldModel,
)
from closed_loop.plant.jobs import work_order_command
from tests.planning._helpers import plant_model, quiet_model
from tests.planning._obs import FakeTracker, observation, plant_particles, view

PLANT_COSTS = PlanCosts.for_plant(
    planned_job_cost=1.0, unplanned_cost_ratio=8.0, inspect_cost=0.05, mean_life_days=90.0
)
G = 1.0 / 90.0


# ------------------------------------------------------------------------------ hand-built models
def clock_model() -> TabularWorldModel:
    """Six levels that rise by one every step, with certainty; level 5 is the failure."""
    n = 6
    up = np.zeros((n, n))
    for s in range(n - 1):
        up[s, s + 1] = 1.0
    up[n - 1, n - 1] = 1.0
    replace = np.zeros((n, n))
    replace[:, 0] = 1.0
    blind = np.full((n, 2), 0.5)
    cost = {"continue": 0.0, "inspect": 0.1, "replace": 1.0, "failure": 2.0}
    return TabularWorldModel(np.stack([up, up, replace]), blind, cost)


def trap_model(failure: float = 1.0) -> TabularWorldModel:
    """Healthy (0), degraded (1), failed (2): a degraded asset fails with chance 0.03 a step."""
    t = np.array([[0.9, 0.1, 0.0], [0.0, 0.97, 0.03], [0.0, 0.0, 1.0]])
    replace = np.zeros((3, 3))
    replace[:, 0] = 1.0
    blind = np.full((3, 2), 0.5)
    cost = {"continue": 0.0, "inspect": 0.1, "replace": 1.0, "failure": failure}
    return TabularWorldModel(np.stack([t, t, replace]), blind, cost)


def doomed_model(price: float) -> TabularWorldModel:
    """Safe (0) or doomed (1) assets look alike to a daily reading; an inspection tells them apart.

    A doomed asset becomes critical (2) at the next step and fails (3) at the one after. The
    daily reading says nothing; the inspection reads the safe level as symbol 0 and the others as
    symbol 1.
    """
    t = np.zeros((4, 4))
    t[0, 0] = 1.0
    t[1, 2] = 1.0
    t[2, 3] = 1.0
    t[3, 3] = 1.0
    replace = np.zeros((4, 4))
    replace[:, 0] = 1.0
    blind = np.full((4, 2), 0.5)
    sharp = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    cost = {"continue": 0.0, "inspect": price, "replace": 1.0, "failure": 4.0}
    return TabularWorldModel(np.stack([t, t, replace]), blind, cost, O_inspect=sharp)


def point(n_states: int, level: int) -> Particles:
    b = np.zeros(n_states)
    b[level] = 1.0
    return particles_from_belief(b)


# ------------------------------------------------------------------------------------------ costs
class TestCosts:
    def test_negative_or_nonfinite_terms_are_refused(self) -> None:
        for kw in ({"replace": -1.0}, {"fail_event": float("nan")}, {"lead_days": 1.0}):
            with pytest.raises(ValueError):
                PlanCosts(**kw)

    def test_an_inspection_may_not_cost_less_than_continuing(self) -> None:
        with pytest.raises(ValueError):
            PlanCosts(cont=0.5, inspect=0.1)

    def test_the_tabular_costs_are_read_from_the_model(self) -> None:
        c = PlanCosts.from_tabular(validation_model())
        assert (c.replace, c.cont, c.inspect, c.fail_step, c.fail_event) == (1.0, 0.0, 0.1, 2.0, 0)
        assert c.inspect_price == pytest.approx(0.1)

    def test_the_plant_costs_charge_the_failure_and_the_wasted_day_not_the_job(self) -> None:
        c = PlanCosts.for_plant(
            planned_job_cost=1.0, unplanned_cost_ratio=8.0, inspect_cost=0.05, mean_life_days=100.0
        )
        assert c.replace == 0.0 and c.fail_event == 8.0
        assert c.waste_per_day == pytest.approx(0.01) and c.inspect_price == pytest.approx(0.05)

    def test_a_model_other_than_the_tabular_one_needs_costs(self) -> None:
        with pytest.raises(ValueError):
            BeliefPlanner(quiet_model()).assess("PU-201", plant_particles(0.5))

    def test_the_mean_life_of_a_fresh_pump_is_a_few_months(self) -> None:
        # onset 24-60 days after installation and 20-75 days of wear: about 85 to 95 days
        life = mean_life_days(plant_model(), seed=1)
        assert 75.0 < life < 105.0


# ------------------------------------------------------------------- the cost of each plan by hand
class TestPlanCostsOnADeterministicClock:
    """Level 2 of a six-level clock fails at step 3: failed states at steps 3, 4 and 5."""

    def planner(self, n: int = 16) -> BeliefPlanner:
        return BeliefPlanner(
            clock_model(), n_particles=n, horizon_days=5, delta_plan=1.0, full_evaluation=True
        )

    def test_the_cost_of_replacing_at_each_step_and_of_never_replacing(self) -> None:
        a = self.planner().assess("A", point(6, 2))
        # replace at 0, 1, 2: 1; at 3: one failed step then the job (1 + 2); at 4: 1 + 4; never: 6
        assert a.open_loop_cost == pytest.approx([1.0, 1.0, 1.0, 3.0, 5.0, 6.0])
        assert a.open_loop_p == pytest.approx([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    def test_the_first_action_values_and_the_choice(self) -> None:
        a = self.planner().assess("A", point(6, 2))
        # replace now costs 1; continuing and replacing at step 1 costs 1; an inspection adds 0.1
        assert a.q == pytest.approx([1.0, 1.1, 1.0])
        assert a.action in (ACTION_CONTINUE, ACTION_REPLACE)
        assert a.voi == pytest.approx(0.0)

    def test_the_value_equals_the_exact_value_when_nothing_is_random(self) -> None:
        planner = self.planner()
        for level in range(5):
            belief = np.zeros(6)
            belief[level] = 1.0
            exact, _ = exact_value(clock_model(), belief, 5)
            assert planner.assess("A", particles_from_belief(belief)).value == pytest.approx(exact)

    def test_an_asset_that_has_already_failed_is_charged_every_step_until_replaced(self) -> None:
        a = self.planner().assess("A", point(6, 5))
        assert a.open_loop_cost == pytest.approx([1.0, 3.0, 5.0, 7.0, 9.0, 10.0])
        assert a.action == ACTION_REPLACE and a.open_loop_p[0] == pytest.approx(1.0)


class TestPlanCostsAgainstTheChain:
    """The cost of a plan is the failure charge summed over the chain's own distribution."""

    @pytest.mark.parametrize("seed", range(3))
    def test_matches_the_expectation_computed_from_the_transition_matrices(self, seed: int) -> None:
        m = validation_model()
        b = validation_beliefs(3, seed=7)[seed]
        h = 5
        a = BeliefPlanner(
            m, n_particles=20000, horizon_days=h, delta_plan=1.0, seed=seed, full_evaluation=True
        ).assess("A", particles_from_belief(b))
        # P(failed at step t) for t = 1..h from the belief, and for a fresh component
        fail_at = []
        cur = b.copy()
        for _ in range(h):
            cur = cur @ m.T[0]
            fail_at.append(cur[m.failure_state])
        fresh_fail = []
        cur = np.zeros(5)
        cur[0] = 1.0
        for _ in range(h):
            cur = cur @ m.T[0]
            fresh_fail.append(cur[m.failure_state])
        fail = m.cost["failure"]
        expected = []
        for j in range(h):
            before = sum(fail_at[:j])
            after = sum(fresh_fail[: h - 1 - j])
            # a failed state is charged at its own step and the job at step j clears it
            expected.append(m.cost["replace"] + fail * (before + after))
        expected.append(fail * sum(fail_at))
        assert a.open_loop_cost == pytest.approx(expected, abs=0.03)


class TestWasteLeadAndEventCosts:
    """The plant's costs on a deterministic pump: 40 days of wear at rate one."""

    def planner(self, lead: float = 0.0, p: float = 1.0) -> BeliefPlanner:
        costs = PlanCosts.for_plant(
            planned_job_cost=1.0,
            unplanned_cost_ratio=8.0,
            inspect_cost=0.05,
            mean_life_days=90.0,
            lead_days=lead,
        )
        return BeliefPlanner(
            quiet_model(), n_particles=32, delta_plan=p, costs=costs, full_evaluation=True
        )

    def test_replacing_early_wastes_the_days_that_were_left(self) -> None:
        # x = 0.5 of 40 days: 20 days left, no failure inside the 14-day window
        a = self.planner().assess("PU-201", plant_particles(0.5, 32), load_mean=1.0)
        j = np.arange(14)
        assert a.open_loop_cost[:14] == pytest.approx(G * (20.0 - j))
        assert a.open_loop_cost[14] == 0.0 and a.open_loop_p == pytest.approx(np.zeros(15))

    def test_a_failure_before_the_replacement_costs_the_unplanned_ratio(self) -> None:
        # x = 0.9: four days left, the asset fails during step 4
        a = self.planner().assess("PU-201", plant_particles(0.9, 32), load_mean=1.0)
        assert a.open_loop_cost[:4] == pytest.approx(G * (3.5 - np.arange(4)))
        assert a.open_loop_cost[4:] == pytest.approx(np.full(11, 8.0))
        assert a.open_loop_p == pytest.approx([0, 0, 0, 0] + [1] * 11)

    def test_the_lead_time_makes_the_step_before_a_failure_risky(self) -> None:
        a = self.planner(lead=0.1).assess("PU-201", plant_particles(0.9, 32), load_mean=1.0)
        # replacing at step 3 still leaves a tenth of the failing step before the asset stops
        assert a.open_loop_cost[3] == pytest.approx(G * 0.5 + 8.0 * 0.1)
        assert a.open_loop_p[3] == pytest.approx(0.1)
        assert a.open_loop_cost[2] == pytest.approx(G * 1.5)  # two steps away: not affected

    def test_a_model_without_a_remaining_life_cannot_price_the_waste(self) -> None:
        class Bare:
            name = "bare"

            def prior(self, rng, n):  # type: ignore[no-untyped-def]
                raise NotImplementedError

            def step(self, p, action, rng):  # type: ignore[no-untyped-def]
                return p

            def loglik(self, p, obs, noise_scale=1.0):  # type: ignore[no-untyped-def]
                return np.zeros(p.n)

            def failed(self, p):  # type: ignore[no-untyped-def]
                return np.zeros(p.n, dtype=bool)

        planner = BeliefPlanner(Bare(), n_particles=4, costs=PLANT_COSTS)
        p = Particles({"x": np.zeros(4)}, np.full(4, 0.25))
        with pytest.raises(ValueError, match="remaining_life_days"):
            planner.assess("A", p)


# --------------------------------------------------------------------------------- the Lagrangian
class TestLagrangian:
    """A degraded asset whose cheapest plan is to wait, at a failure risk above the budget."""

    def assess(self, delta: float, *, level: int = 1, n: int = 4096) -> AssetAssessment:
        planner = BeliefPlanner(
            trap_model(), n_particles=n, horizon_days=5, delta_plan=delta, seed=3,
            full_evaluation=True,
        )  # fmt: skip
        return planner.assess("A", point(3, level))

    def test_without_a_binding_constraint_the_multiplier_stays_at_zero(self) -> None:
        a = self.assess(0.5)
        # about 14% chance of failing in five steps, cost about 0.43 of a job: waiting is cheaper
        assert a.lam == 0.0 and a.action in (ACTION_CONTINUE, ACTION_INSPECT) and a.feasible
        assert 0.10 < a.p_chosen < 0.18

    def test_the_multiplier_is_raised_until_the_failure_probability_is_within_budget(self) -> None:
        a = self.assess(0.02)
        assert a.lam > 0.0 and a.feasible
        assert a.p_chosen <= 0.02 and a.action == ACTION_REPLACE

    def test_a_tighter_budget_never_lowers_the_multiplier(self) -> None:
        lams = [self.assess(d).lam for d in (0.5, 0.2, 0.1, 0.05, 0.02)]
        assert lams == sorted(lams) and lams[0] == 0.0 and lams[-1] > 0.0

    def test_the_search_is_a_bracket_plus_at_most_six_halvings(self) -> None:
        a = self.assess(0.02)
        assert 1 < a.n_lambda <= 1 + 12 + 6

    def test_a_budget_that_cannot_be_met_is_reported_not_hidden(self) -> None:
        # a tenth of the belief is already failed: no action can bring the probability under 2%
        belief = np.array([0.0, 0.9, 0.1])
        planner = BeliefPlanner(
            trap_model(), n_particles=4096, horizon_days=5, delta_plan=0.02, seed=3,
            full_evaluation=True,
        )  # fmt: skip
        a = planner.assess("A", particles_from_belief(belief))
        assert a.feasible is False
        assert a.action == ACTION_REPLACE  # the most cautious action still on the table
        assert a.p_chosen > 0.02

    def test_the_constraint_changes_the_plan_it_does_not_just_report(self) -> None:
        """Mutation target: with the multiplier dropped the planner waits, which it must not."""
        loose = self.assess(1.0)
        tight = self.assess(0.02)
        assert loose.action != ACTION_REPLACE
        assert tight.action == ACTION_REPLACE
        assert tight.q_lagrangian[ACTION_REPLACE] < tight.q_lagrangian[ACTION_CONTINUE]
        assert loose.p_chosen > 0.02  # the loose plan really did break the budget

    def test_the_plant_planner_replaces_early_enough_to_meet_the_budget(self) -> None:
        # x = 0.8: 8 days left. Waiting is cheap (a day of waste is 0.011) but the chance of
        # failing in the window under a noisy model is not zero; with budget 0.02 it must hold.
        planner = BeliefPlanner(
            plant_model(), n_particles=256, delta_plan=0.02, costs=PLANT_COSTS, seed=2
        )
        for x in (0.2, 0.8, 0.95, 0.99):
            a = planner.assess("PU-201", plant_particles(x, 128), load_mean=1.0, day=int(x * 100))
            assert a.p_chosen <= 0.02 or not a.feasible


# ------------------------------------------------------------------------- value of information
class TestInspection:
    def assess(self, price: float, *, belief: tuple[float, ...] = (0.5, 0.5, 0.0, 0.0)):
        planner = BeliefPlanner(
            doomed_model(price), n_particles=64, horizon_days=3, delta_plan=1.0, seed=1,
            full_evaluation=True,
        )  # fmt: skip
        return planner.assess("A", particles_from_belief(np.array(belief)))

    def test_the_value_of_information_is_worked_out_by_hand(self) -> None:
        a = self.assess(0.1)
        # no information: replace now, or wait and replace at step 1: 1.0 either way
        # perfect information after step 1: safe -> never replace (0), doomed -> replace (1): 0.5
        assert a.q[ACTION_REPLACE] == pytest.approx(1.0)
        assert a.q[ACTION_CONTINUE] == pytest.approx(1.0)
        assert a.q[ACTION_INSPECT] == pytest.approx(0.1 + 0.5)
        assert a.voi == pytest.approx(0.5)

    def test_it_inspects_when_the_information_is_worth_more_than_its_price(self) -> None:
        assert self.assess(0.1).action == ACTION_INSPECT
        assert self.assess(0.49).action == ACTION_INSPECT

    def test_it_does_not_inspect_when_the_price_exceeds_the_value(self) -> None:
        for price in (0.51, 0.6, 2.0):
            a = self.assess(price)
            assert a.action != ACTION_INSPECT
            assert a.voi == pytest.approx(0.5)  # the value is the same, the price is what moved

    def test_it_does_not_inspect_when_there_is_nothing_to_learn(self) -> None:
        # a belief that is already certain about which kind of asset this is
        for belief in ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)):
            a = self.assess(0.1, belief=belief)
            assert a.voi == pytest.approx(0.0) and a.action != ACTION_INSPECT

    def test_on_a_plant_belief_an_ambiguous_asset_has_a_value_and_a_certain_one_has_none(
        self,
    ) -> None:
        planner = BeliefPlanner(
            quiet_model(), n_particles=64, costs=PLANT_COSTS, seed=4, full_evaluation=True
        )
        sure = planner.assess("PU-201", plant_particles(0.80, 64), load_mean=1.0)
        assert sure.voi == pytest.approx(0.0, abs=1e-9)
        # two clusters whose remaining lives differ by a few days but whose readings are close
        a = plant_particles(0.80, 32)
        b = plant_particles(0.86, 32)
        mix = Particles(
            {k: np.concatenate([a.state[k], b.state[k]]) for k in a.state}, np.full(64, 1.0 / 64)
        )
        unsure = planner.assess("PU-201", mix, load_mean=1.0)
        assert unsure.voi > 0.0 and unsure.full


# -------------------------------------------------------------------- the shortcut and the draws
class TestShortcutAndDeterminism:
    def test_the_cheap_path_gives_the_same_action_as_the_full_evaluation(self) -> None:
        m = validation_model()
        cheap = BeliefPlanner(m, n_particles=128, horizon_days=5, delta_plan=1.0, seed=5)
        full = BeliefPlanner(
            m, n_particles=128, horizon_days=5, delta_plan=1.0, seed=5, full_evaluation=True
        )
        used = 0
        healthy = [np.array([1.0, 0, 0, 0, 0]), np.array([0.9, 0.1, 0, 0, 0])]
        for b in [*healthy, *validation_beliefs(40, seed=3)]:
            a = cheap.assess("A", particles_from_belief(b), day=2)
            if not a.full:
                used += 1
                assert full.assess("A", particles_from_belief(b), day=2).action == a.action
        assert used > 0

    def test_the_cheap_path_is_taken_for_an_asset_that_is_plainly_healthy(self) -> None:
        planner = BeliefPlanner(quiet_model(), n_particles=32, costs=PLANT_COSTS)
        a = planner.assess("PU-201", plant_particles(0.2, 32), load_mean=1.0)
        assert a.full is False and a.action == ACTION_CONTINUE and a.n_lambda == 0

    def test_the_same_inputs_give_the_same_assessment_bit_for_bit(self) -> None:
        def run() -> AssetAssessment:
            planner = BeliefPlanner(
                plant_model(), n_particles=128, costs=PLANT_COSTS, seed=9, full_evaluation=True
            )
            return planner.assess("PU-201", plant_particles(0.9, 128), load_mean=1.0, day=41)

        a, b = run(), run()
        assert np.array_equal(a.q, b.q) and np.array_equal(a.p_fail, b.p_fail)
        assert (a.lam, a.voi, a.action) == (b.lam, b.voi, b.action)

    def test_the_draws_depend_on_the_seed_and_the_day(self) -> None:
        def q(seed: int, day: int) -> np.ndarray:
            planner = BeliefPlanner(
                plant_model(), n_particles=64, costs=PLANT_COSTS, seed=seed, full_evaluation=True
            )
            return planner.assess("PU-201", plant_particles(0.9, 64), load_mean=1.0, day=day).q

        assert not np.array_equal(q(1, 5), q(2, 5)) and not np.array_equal(q(1, 5), q(1, 6))

    def test_the_result_does_not_depend_on_which_other_asset_was_assessed_first(self) -> None:
        planner = BeliefPlanner(plant_model(), n_particles=64, costs=PLANT_COSTS, seed=2)
        alone = planner.assess("PU-201", plant_particles(0.9, 64), load_mean=1.0, day=3)
        other = BeliefPlanner(plant_model(), n_particles=64, costs=PLANT_COSTS, seed=2)
        other.assess("PU-202", plant_particles(0.5, 64), load_mean=1.0, day=3)
        again = other.assess("PU-201", plant_particles(0.9, 64), load_mean=1.0, day=3)
        assert np.array_equal(alone.q, again.q)

    def test_the_belief_is_rolled_out_with_n_particles_whatever_the_tracker_holds(self) -> None:
        """Regression: a 5-particle belief was rolled out as 5 particles, not as n_particles."""
        seen: list[int] = []

        class Spy(TabularWorldModel):
            def step(self, p, action, rng):  # type: ignore[no-untyped-def]
                seen.append(p.n)
                return super().step(p, action, rng)

        base = validation_model()
        spy = Spy(base.T, base.O, base.cost)
        BeliefPlanner(spy, n_particles=200, horizon_days=3, delta_plan=1.0).assess(
            "A", particles_from_belief(np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
        )
        # the first three calls are the belief's own three-step rollout; the later ones are the
        # separate batch that profiles a freshly installed component
        assert seen[:3] == [200, 200, 200]

    def test_bad_arguments_are_refused(self) -> None:
        m = validation_model()
        for kw in (
            {"n_particles": 1},
            {"horizon_days": 0},
            {"delta_plan": 1.5},
            {"k_readings": 1},
            {"bisection_steps": -1},
        ):
            with pytest.raises(ValueError):
                BeliefPlanner(m, **kw)


# --------------------------------------------------------------------------------------- the limits
def an_assessment(aid: str, action: int, benefit: float = 1.0) -> AssetAssessment:
    """A planner verdict with a stated cost of waiting, to test the joint choice on its own."""
    q = np.array([benefit, benefit, 0.0])
    zero = np.zeros(15)
    return AssetAssessment(aid, action, q, q, np.zeros(3), 0.0, True, 0.0, True, 0, zero, zero)


PART = "SYN-IMP-2001"


def setup(wanting, *, spares=None, ops=None, jobs=(), constraints=None, benefit=None):
    """A planner, a tracker, an observation and constraints in which ``wanting`` want a job."""
    benefit = benefit or {}
    assets = ("CR-501", "CV-101", "PU-201", "PU-202")
    ops = ops or {}
    views = {
        aid: view(
            aid,
            op=ops.get(aid, Op.RUN),
            job_open=aid in jobs,
            spares={PART: 2} if spares is None else spares,
        )
        for aid in assets
    }
    obs = observation(views)
    planner = BeliefPlanner(
        {aid: plant_model() for aid in assets}, n_particles=16, costs=PLANT_COSTS, seed=1
    )
    verdict = {
        aid: an_assessment(aid, ACTION_REPLACE if aid in wanting else ACTION_CONTINUE,
                           benefit.get(aid, 1.0))
        for aid in assets
    }  # fmt: skip
    planner._assess_cached = lambda aid, tracker: verdict[aid]  # type: ignore[method-assign]
    tracker = FakeTracker({aid: plant_particles(0.5, 16) for aid in assets}).as_tracker()
    c = constraints or PlanConstraints(
        part_for={(aid, "IMPELLER_WEAR"): PART for aid in assets}, pump_pair=("PU-201", "PU-202")
    )
    return planner, obs, tracker, c


MODE = {"mode_of": lambda aid: "IMPELLER_WEAR"}


class TestJointChoice:
    def test_assets_that_do_not_want_a_job_get_none(self) -> None:
        planner, obs, tracker, c = setup(wanting=())
        d = planner.decide(obs, tracker, c, **MODE)
        assert d.replace == () and not [x for x in d.commands if x.cmd is PlanCmd.RAISE_WO]

    def test_a_work_order_carries_the_mode_and_the_current_shift(self) -> None:
        planner, obs, tracker, c = setup(wanting=("CV-101",))
        (cmd,) = [x for x in planner.plan(obs, tracker, c, **MODE) if x.cmd is PlanCmd.RAISE_WO]
        shift = obs.quarter // 32
        assert cmd == work_order_command("CV-101", "IMPELLER_WEAR", shift, source="planner")

    def test_crew_limits_the_number_of_open_work_orders(self) -> None:
        planner, obs, tracker, c = setup(wanting=("CR-501", "CV-101", "PU-201"))
        d = planner.decide(obs, tracker, c, **MODE)
        assert len(d.replace) == 2 and set(d.deferred.values()) == {"crew"}

    def test_a_work_order_already_open_uses_one_of_the_crew(self) -> None:
        planner, obs, tracker, c = setup(wanting=("CR-501", "CV-101"), jobs=("PU-201",))
        d = planner.decide(obs, tracker, c, **MODE)
        assert len(d.replace) == 1 and "crew" in d.deferred.values()

    def test_the_pump_pair_is_never_given_two_work_orders_at_once(self) -> None:
        planner, obs, tracker, c = setup(wanting=("PU-201", "PU-202"))
        d = planner.decide(obs, tracker, c, **MODE)
        assert len(d.replace) == 1 and list(d.deferred.values()) == ["pump pair"]

    def test_the_pump_that_costs_more_to_leave_goes_first(self) -> None:
        planner, obs, tracker, c = setup(
            wanting=("PU-201", "PU-202"), benefit={"PU-201": 1.0, "PU-202": 5.0}
        )
        assert planner.decide(obs, tracker, c, **MODE).replace == ("PU-202",)
        planner, obs, tracker, c = setup(
            wanting=("PU-201", "PU-202"), benefit={"PU-201": 5.0, "PU-202": 1.0}
        )
        assert planner.decide(obs, tracker, c, **MODE).replace == ("PU-201",)

    def test_a_pump_is_not_started_while_the_other_is_isolated_or_in_work(self) -> None:
        for op in (Op.ISO, Op.WORK):
            planner, obs, tracker, c = setup(wanting=("PU-201",), ops={"PU-202": op})
            d = planner.decide(obs, tracker, c, **MODE)
            assert d.replace == () and d.deferred == {"PU-201": "pump pair"}

    def test_a_pump_waits_for_a_partner_whose_work_order_is_open(self) -> None:
        planner, obs, tracker, c = setup(wanting=("PU-201",), jobs=("PU-202",))
        assert planner.decide(obs, tracker, c, **MODE).replace == ()

    def test_a_failed_pump_is_repaired_first_and_the_other_waits(self) -> None:
        planner, obs, tracker, c = setup(wanting=("PU-202",), ops={"PU-201": Op.FAIL})
        d = planner.decide(obs, tracker, c, **MODE)
        assert d.replace == ("PU-201",) and d.deferred == {"PU-202": "pump pair"}

    def test_a_failed_asset_takes_a_place_in_the_crew_before_the_others(self) -> None:
        planner, obs, tracker, c = setup(
            wanting=("CR-501", "CV-101"),
            ops={"PU-201": Op.FAIL},
            benefit={"CR-501": 5.0, "CV-101": 4.0},
        )
        d = planner.decide(obs, tracker, c, **MODE)
        assert set(d.replace) == {"PU-201", "CR-501"} and d.deferred == {"CV-101": "crew"}

    def test_a_failed_pump_may_be_repaired_while_the_partners_job_has_not_isolated(self) -> None:
        planner, obs, tracker, c = setup(wanting=(), ops={"PU-201": Op.FAIL}, jobs=("PU-202",))
        assert planner.decide(obs, tracker, c, **MODE).replace == ("PU-201",)

    def test_a_failed_pump_is_not_repaired_while_the_partner_is_in_work(self) -> None:
        planner, obs, tracker, c = setup(
            wanting=(), ops={"PU-201": Op.FAIL, "PU-202": Op.WORK}, jobs=("PU-202",)
        )
        assert planner.decide(obs, tracker, c, **MODE).replace == ()

    def test_one_confined_space_job_at_a_time(self) -> None:
        c = PlanConstraints(
            part_for={(a, "IMPELLER_WEAR"): PART for a in ("CR-501", "CV-101", "PU-201", "PU-202")},
            confined_jobs=frozenset({("CR-501", "IMPELLER_WEAR"), ("CV-101", "IMPELLER_WEAR")}),
            max_open_jobs=4,
        )
        planner, obs, tracker, c = setup(wanting=("CR-501", "CV-101", "PU-201"), constraints=c)
        d = planner.decide(obs, tracker, c, **MODE)
        assert len(d.replace) == 2 and "PU-201" in d.replace
        assert d.deferred and set(d.deferred.values()) == {"confined space"}

    def test_a_confined_job_already_open_blocks_a_second(self) -> None:
        c = PlanConstraints(
            part_for={("CR-501", "IMPELLER_WEAR"): PART},
            confined_jobs=frozenset({("CR-501", "IMPELLER_WEAR")}),
            confined_open=1,
        )
        planner, obs, tracker, c = setup(wanting=("CR-501",), constraints=c)
        d = planner.decide(obs, tracker, c, **MODE)
        assert d.replace == () and d.deferred == {"CR-501": "confined space"}

    def test_a_spare_that_is_not_there_blocks_the_work_order_and_is_ordered(self) -> None:
        planner, obs, tracker, c = setup(wanting=("CV-101",), spares={PART: 0})
        d = planner.decide(obs, tracker, c, **MODE)
        assert d.replace == () and d.deferred == {"CV-101": f"no spare {PART}"}
        orders = [x for x in d.commands if x.cmd is PlanCmd.ORDER]
        assert [(o.arg) for o in orders] == [PART] and d.orders == (PART,)

    def test_one_spare_serves_one_work_order(self) -> None:
        planner, obs, tracker, c = setup(wanting=("CR-501", "CV-101"), spares={PART: 1})
        d = planner.decide(obs, tracker, c, **MODE)
        assert len(d.replace) == 1 and list(d.deferred.values()) == [f"no spare {PART}"]

    def test_a_spare_already_ordered_is_not_ordered_again(self) -> None:
        c = PlanConstraints(on_order=frozenset({PART}))
        planner, obs, tracker, c = setup(wanting=(), spares={PART: 0}, constraints=c)
        assert planner.decide(obs, tracker, c, **MODE).orders == ()

    def test_an_unknown_mode_is_refused_not_guessed(self) -> None:
        planner, obs, tracker, c = setup(wanting=("CV-101",))
        d = planner.decide(obs, tracker, c, mode_of=lambda aid: "NO_SUCH_MODE")
        assert d.replace == () and d.deferred == {"CV-101": "unknown mode"}

    def test_a_diagnosis_that_is_not_degradation_raises_no_work_order(self) -> None:
        planner, obs, tracker, c = setup(wanting=("CV-101", "PU-201"))
        d = planner.decide(
            obs, tracker, c, mode_of=lambda aid: None if aid == "CV-101" else "IMPELLER_WEAR"
        )
        assert d.replace == ("PU-201",) and "CV-101" in d.deferred

    def test_assets_with_a_work_order_or_in_cooldown_or_stopped_are_left_alone(self) -> None:
        planner, obs, tracker, c = setup(
            wanting=("CR-501", "CV-101", "PU-201"),
            jobs=("CR-501",),
            ops={"PU-201": Op.STOP},
        )
        c = PlanConstraints(
            part_for=c.part_for, busy=frozenset({"CV-101"}), pump_pair=("PU-201", "PU-202")
        )
        assert planner.decide(obs, tracker, c, **MODE).replace == ()

    def test_inspections_are_limited_by_the_free_fitters(self) -> None:
        planner, obs, tracker, c = setup(wanting=())
        verdict = {a: an_assessment(a, ACTION_INSPECT) for a in obs.assets}
        planner._assess_cached = lambda aid, tr: verdict[aid]  # type: ignore[method-assign]
        for slots in (0, 1, 3):
            c2 = PlanConstraints(inspect_slots=slots)
            d = planner.decide(obs, tracker, c2, **MODE)
            assert len(d.inspect) == slots

    def test_an_asset_replaced_this_call_is_not_also_inspected(self) -> None:
        planner, obs, tracker, c = setup(wanting=("CV-101",))
        d = planner.decide(obs, tracker, c, **MODE)
        assert "CV-101" not in d.inspect

    def test_no_inspect_keeps_a_recently_inspected_asset_out_of_the_list(self) -> None:
        planner, obs, tracker, c = setup(wanting=())
        verdict = {a: an_assessment(a, ACTION_INSPECT) for a in obs.assets}
        planner._assess_cached = lambda aid, tr: verdict[aid]  # type: ignore[method-assign]
        c2 = PlanConstraints(inspect_slots=4, no_inspect=frozenset({"CV-101", "PU-201"}))
        assert set(planner.decide(obs, tracker, c2, **MODE).inspect) == {"CR-501", "PU-202"}

    def test_the_default_mode_is_the_most_probable_one_under_the_belief(self) -> None:
        planner, obs, tracker, c = setup(wanting=("PU-201",))
        (cmd,) = [x for x in planner.plan(obs, tracker, c) if x.cmd is PlanCmd.RAISE_WO]
        assert cmd.arg is not None and cmd.arg.startswith("IMPELLER_WEAR:")  # mode index 1


class TestJointChoiceOnRealBeliefs:
    """The same limits through the whole assessment, with a belief for every asset."""

    def test_two_pumps_about_to_fail_are_serialised_and_the_urgent_one_goes_first(self) -> None:
        assets = ("CR-501", "CV-101", "PU-201", "PU-202")
        views = {a: view(a, spares={PART: 4}) for a in assets}
        obs = observation(views)
        beliefs = {
            "CR-501": plant_particles(0.3, 64),
            "CV-101": plant_particles(0.3, 64),
            "PU-201": plant_particles(0.995, 64),
            "PU-202": plant_particles(0.99, 64),
        }
        planner = BeliefPlanner(
            {a: quiet_model() for a in assets}, n_particles=64, costs=PLANT_COSTS, seed=1
        )
        c = PlanConstraints(
            part_for={(a, "IMPELLER_WEAR"): PART for a in assets}, pump_pair=("PU-201", "PU-202")
        )
        d = planner.decide(obs, FakeTracker(beliefs).as_tracker(), c, **MODE)
        assert len(d.replace) == 1 and d.replace[0] in ("PU-201", "PU-202")
        assert d.deferred == {({"PU-201", "PU-202"} - set(d.replace)).pop(): "pump pair"}
        assert d.assessments["CR-501"].action == ACTION_CONTINUE


# ------------------------------------------------------------------------- the readings tree
class TestLookahead:
    """The preposterior looks ahead ``lookahead_readings`` readings; one is the ADR's version."""

    def test_the_hand_worked_value_of_information_does_not_depend_on_the_depth_here(self) -> None:
        # the daily reading says nothing in this model, so a second one adds no information
        for depth in (1, 2):
            planner = BeliefPlanner(
                doomed_model(0.1), n_particles=64, horizon_days=3, delta_plan=1.0, seed=1,
                lookahead_readings=depth, full_evaluation=True,
            )  # fmt: skip
            a = planner.assess("A", particles_from_belief(np.array([0.5, 0.5, 0.0, 0.0])))
            assert a.voi == pytest.approx(0.5) and a.action == ACTION_INSPECT

    @pytest.mark.parametrize("seed", range(3))
    def test_with_enough_readings_and_no_use_for_inspection_the_planner_is_the_exact_dp(
        self, seed: int
    ) -> None:
        """Horizon 3, two readings: every reading the exact tree uses is in the planner's tree."""
        base = validation_model()
        cost = {**base.cost, "inspect": 50.0}  # an inspection is never worth it, as in the DP
        m = TabularWorldModel(base.T, base.O, cost)
        belief = validation_beliefs(3, seed=11)[seed]
        planner = BeliefPlanner(
            m, n_particles=40000, horizon_days=3, delta_plan=1.0, seed=seed,
            lookahead_readings=2, full_evaluation=True,
        )  # fmt: skip
        a = planner.assess("A", particles_from_belief(belief))
        from closed_loop.planning.exactdp import exact_q_values

        exact = exact_q_values(m, belief, 3)
        assert a.q[ACTION_CONTINUE] == pytest.approx(exact[ACTION_CONTINUE], abs=0.03)
        assert a.q[ACTION_REPLACE] == pytest.approx(exact[ACTION_REPLACE], abs=0.03)

    def test_a_deeper_tree_never_values_waiting_below_the_best_fixed_plan(self) -> None:
        # the tree contains every fixed plan as a policy, so it cannot cost more than the best one
        m = validation_model()
        for belief in validation_beliefs(8, seed=5):
            a = BeliefPlanner(
                m, n_particles=256, horizon_days=5, delta_plan=1.0, seed=2,
                lookahead_readings=3, full_evaluation=True,
            ).assess("A", particles_from_belief(belief))  # fmt: skip
            assert a.q[ACTION_CONTINUE] <= float(np.min(a.open_loop_cost[1:])) + 1e-9

    def test_the_depth_is_capped_by_the_horizon(self) -> None:
        planner = BeliefPlanner(
            validation_model(), n_particles=64, horizon_days=2, delta_plan=1.0,
            lookahead_readings=5, full_evaluation=True,
        )  # fmt: skip
        a = planner.assess("A", particles_from_belief(np.array([0.2, 0.2, 0.2, 0.2, 0.2])))
        assert a.full and np.isfinite(a.q).all()

    def test_a_plant_belief_is_read_through_sampled_readings(self) -> None:
        planner = BeliefPlanner(
            quiet_model(), n_particles=64, costs=PLANT_COSTS, seed=4, lookahead_readings=3,
            full_evaluation=True,
        )  # fmt: skip
        a = planner.assess("PU-201", plant_particles(0.85, 64), load_mean=1.0)
        assert a.full and np.isfinite(a.q).all() and a.voi == pytest.approx(0.0, abs=1e-9)

    def test_a_model_with_neither_finite_readings_nor_a_gaussian_mean_is_refused(self) -> None:
        class Opaque:
            name = "opaque"

            def prior(self, rng, n):  # type: ignore[no-untyped-def]
                return Particles({"x": np.zeros(n)}, np.full(n, 1.0 / n))

            def step(self, p, action, rng):  # type: ignore[no-untyped-def]
                return p

            def loglik(self, p, obs, noise_scale=1.0):  # type: ignore[no-untyped-def]
                return np.zeros(p.n)

            def failed(self, p):  # type: ignore[no-untyped-def]
                fail = np.zeros(p.n, dtype=bool)
                fail[0] = True
                return fail

        costs = PlanCosts(inspect=0.1, fail_step=1.0)
        p = Particles({"x": np.zeros(8)}, np.full(8, 0.125))
        with pytest.raises(TypeError, match="observation_support"):
            BeliefPlanner(Opaque(), n_particles=8, horizon_days=3, costs=costs).assess("A", p)

    def test_bad_depth_is_refused(self) -> None:
        with pytest.raises(ValueError):
            BeliefPlanner(validation_model(), lookahead_readings=0)
        with pytest.raises(ValueError):
            BeliefPlanner(validation_model(), k_later=1)


# ---------------------------------------------------------------------------------- the screen
class TestScreen:
    """A belief that cannot fail inside the window is answered without a rollout."""

    def mixed(self, near_weight: float):
        far = plant_particles(0.3, 100)
        close = plant_particles(0.97, 100)
        w = np.concatenate(
            [np.full(100, (1.0 - near_weight) / 100), np.full(100, near_weight / 100)]
        )
        return Particles({k: np.concatenate([far.state[k], close.state[k]]) for k in far.state}, w)

    def planner(self, **kw):
        return BeliefPlanner(quiet_model(), n_particles=64, costs=PLANT_COSTS, seed=1, **kw)

    def test_a_clear_belief_is_answered_without_a_rollout(self) -> None:
        steps: list[int] = []
        model = quiet_model()
        real = model.step

        def counting(p, action, rng, **kw):  # type: ignore[no-untyped-def]
            steps.append(p.n)
            return real(p, action, rng, **kw)

        model.step = counting  # type: ignore[method-assign]
        planner = BeliefPlanner(model, n_particles=64, costs=PLANT_COSTS, seed=1)
        a = planner.assess("PU-201", plant_particles(0.3, 64), load_mean=1.0)
        assert a.action == ACTION_CONTINUE and a.full is False and a.n_lambda == 0
        assert steps == []  # not one step of the model was taken

    def test_it_gives_the_verdict_and_the_costs_the_full_path_gives(self) -> None:
        belief = plant_particles(0.3, 64)
        screened = self.planner().assess("PU-201", belief, load_mean=1.0)
        full = self.planner(full_evaluation=True).assess("PU-201", belief, load_mean=1.0)
        assert screened.action == full.action == ACTION_CONTINUE
        assert screened.open_loop_cost == pytest.approx(full.open_loop_cost)
        assert screened.open_loop_p == pytest.approx(full.open_loop_p)
        assert screened.q[ACTION_REPLACE] == pytest.approx(full.q[ACTION_REPLACE])
        assert screened.asset_id == "PU-201"

    def test_a_belief_with_weight_near_failure_is_not_screened(self) -> None:
        a = self.planner().assess("PU-201", self.mixed(0.05), load_mean=1.0)
        assert a.open_loop_p[-1] > 0.0  # a rollout found the particles that fail

    def test_a_sliver_of_weight_below_the_limit_does_not_defeat_the_screen(self) -> None:
        a = self.planner().assess("PU-201", self.mixed(5e-4), load_mean=1.0)
        assert a.full is False and a.open_loop_p[-1] == 0.0

    def test_a_budget_below_the_limit_turns_the_screen_off(self) -> None:
        a = self.planner(delta_plan=0.0).assess("PU-201", plant_particles(0.3, 64), load_mean=1.0)
        assert a.n_lambda == 0 and a.full is False  # the rollout path's own cheap verdict
        steps: list[int] = []
        model = quiet_model()
        real = model.step
        model.step = lambda p, action, rng, **kw: (steps.append(1), real(p, action, rng, **kw))[1]  # type: ignore[method-assign]
        BeliefPlanner(model, n_particles=64, costs=PLANT_COSTS, delta_plan=0.0).assess(
            "PU-201", plant_particles(0.3, 64), load_mean=1.0
        )
        assert steps  # with a zero budget the planner does roll out

    def test_a_finite_model_has_no_screen(self) -> None:
        planner = BeliefPlanner(clock_model(), n_particles=8, horizon_days=3, delta_plan=0.5)
        a = planner.assess("A", point(6, 0))
        assert np.isfinite(a.q).all()


# -------------------------------------------------------- the pieces of the search, one by one
def risky_fresh_model() -> TabularWorldModel:
    """A fresh part (0) fails with chance one half a step; a replacement restores it."""
    t = np.array([[0.5, 0.5], [0.0, 1.0]])
    replace = np.array([[1.0, 0.0], [1.0, 0.0]])
    blind = np.full((2, 2), 0.5)
    cost = {"continue": 0.0, "inspect": 0.1, "replace": 1.0, "failure": 2.0}
    return TabularWorldModel(np.stack([t, t, replace]), blind, cost)


class TestFreshComponentAfterAReplacement:
    def test_a_replacement_is_charged_for_the_failures_of_the_part_it_fits(self) -> None:
        h = 3
        a = BeliefPlanner(
            risky_fresh_model(), n_particles=40000, horizon_days=h, delta_plan=1.0, seed=2,
            full_evaluation=True,
        ).assess("A", point(2, 0))  # fmt: skip
        f = [1.0 - 0.5**m for m in range(0, 4)]  # chance a fresh part is failed after m steps
        expected = []
        for j in range(h):
            before = sum(f[1 : j + 1])  # failed steps of the old part before the replacement
            after = sum(f[1 : h - j])  # failed steps of the fresh part after it
            expected.append(1.0 + 2.0 * (before + after))
        expected.append(2.0 * sum(f[1 : h + 1]))
        assert a.open_loop_cost == pytest.approx(expected, abs=0.05)


class TestTreeArithmetic:
    """The tree value and the action values on trees small enough to add up by hand."""

    def tree(self):
        from closed_loop.planning.planner import _Level, _Tree

        # one reading with two outcomes (probability 0.25 and 0.75); at each, replacing costs
        # (3, 1) with failure chance (0.0, 0.5); going on leaves a plan costing (2, 4) with
        # failure chance (0.0, 0.0)
        level = _Level(np.array([[0.25, 0.75]]), np.array([3.0, 1.0]), np.array([0.0, 0.5]))
        return _Tree([level], np.array([[2.0], [4.0]]), np.array([[0.0], [0.0]]))

    def test_without_a_multiplier_each_node_takes_the_cheaper_of_stopping_and_going_on(
        self,
    ) -> None:
        lag, cost, prob = BeliefPlanner._tree_value(self.tree(), 0.0)
        # outcome 1: min(3, 2) = 2 going on; outcome 2: min(1, 4) = 1 stopping
        assert lag == pytest.approx(0.25 * 2.0 + 0.75 * 1.0)
        assert cost == pytest.approx(lag) and prob == pytest.approx(0.75 * 0.5)

    def test_the_multiplier_is_charged_on_stopping_inside_the_tree(self) -> None:
        # with lambda = 10, stopping at the second outcome costs 1 + 10 * 0.5 = 6, so it goes on (4)
        lag, cost, prob = BeliefPlanner._tree_value(self.tree(), 10.0)
        assert lag == pytest.approx(0.25 * 2.0 + 0.75 * 4.0)
        assert cost == pytest.approx(0.25 * 2.0 + 0.75 * 4.0) and prob == pytest.approx(0.0)

    def test_the_multiplier_is_charged_for_replacing_now_as_well(self) -> None:
        from closed_loop.planning.planner import _Context

        ctx = _Context(1.5, 0.2, self.tree(), self.tree(), 0.1)
        _, lag, cost, prob = BeliefPlanner._values(ctx, 5.0)
        assert lag[ACTION_REPLACE] == pytest.approx(1.5 + 5.0 * 0.2)
        assert cost[ACTION_REPLACE] == pytest.approx(1.5) and prob[ACTION_REPLACE] == 0.2
        assert lag[ACTION_INSPECT] == pytest.approx(lag[ACTION_CONTINUE] + 0.1)


class TestReadingsAreProperDistributions:
    def reader(self, model, n):
        from closed_loop.planning.planner import _Reader

        p = model.prior(np.random.default_rng(0), n)
        return _Reader(model, [p, p], 1.0)

    def test_a_finite_model_gives_probabilities_and_posteriors_that_sum_to_one(self) -> None:
        m = validation_model()
        reader = self.reader(m, 32)
        planner = BeliefPlanner(m, n_particles=32, horizon_days=3)
        ramp = np.linspace(1, 2, 32)
        w = np.vstack([np.full(32, 1 / 32), ramp / ramp.sum()])
        cond, post = planner._expand(reader, 1, w, np.random.default_rng(1), 1.0, 4)
        assert cond.shape == (2, m.n_symbols) and post.shape == (2 * m.n_symbols, 32)
        assert cond.sum(axis=1) == pytest.approx([1.0, 1.0])
        assert post.sum(axis=1) == pytest.approx(np.ones(2 * m.n_symbols))

    def test_a_continuous_model_gives_equal_probabilities_and_normalised_posteriors(self) -> None:
        m = quiet_model()
        reader = self.reader(m, 32)
        planner = BeliefPlanner(m, n_particles=32, costs=PLANT_COSTS)
        w = np.full((3, 32), 1 / 32)
        cond, post = planner._expand(reader, 1, w, np.random.default_rng(1), 1.0, 8)
        assert cond == pytest.approx(np.full((3, 8), 1 / 8))
        assert post.shape == (24, 32) and post.sum(axis=1) == pytest.approx(np.ones(24))


class TestTheMultiplierSearch:
    def lam(self, delta: float, steps: int) -> float:
        planner = BeliefPlanner(
            trap_model(), n_particles=4096, horizon_days=5, delta_plan=delta, seed=3,
            full_evaluation=True, bisection_steps=steps,
        )  # fmt: skip
        return planner.assess("A", point(3, 1)).lam

    def test_six_halvings_land_within_a_few_per_cent_of_the_smallest_multiplier(self) -> None:
        ratios = [self.lam(d, 6) / self.lam(d, 30) for d in (0.02, 0.05, 0.1, 0.12)]
        assert all(1.0 <= r <= 1.03 for r in ratios), ratios

    def test_without_the_halvings_the_multiplier_would_be_too_high(self) -> None:
        # the check above has teeth: the bracket alone is not that precise
        ratios = [self.lam(d, 0) / self.lam(d, 30) for d in (0.02, 0.05, 0.1, 0.12)]
        assert max(ratios) > 1.03


class TestTheShortcutKeepsTheBudget:
    def test_a_small_chance_of_failure_above_the_budget_is_not_waved_through(self) -> None:
        far = plant_particles(0.3, 100)
        close = plant_particles(0.995, 100)
        w = np.concatenate([np.full(100, 0.996 / 100), np.full(100, 0.004 / 100)])
        state = {k: np.concatenate([far.state[k], close.state[k]]) for k in far.state}
        planner = BeliefPlanner(
            quiet_model(), n_particles=200, costs=PLANT_COSTS, delta_plan=0.001, seed=1
        )
        a = planner.assess("PU-201", Particles(state, w), load_mean=1.0)
        # waiting costs 0.004 * 8 = 0.03, below an inspection's price, so the shortcut would say
        # continue; but 0.4% is four times the budget
        assert a.action == ACTION_REPLACE and a.p_chosen <= 0.001


class TestSourcesOfTheReadings:
    """The sampled readings keep the particles that fail inside the window in the tree."""

    def draw(self, w, doomed, k=16, seed=0):
        return BeliefPlanner._sources(w, doomed, k, np.random.default_rng(seed))

    def test_the_failing_stratum_gets_its_own_readings_worth_its_own_weight(self) -> None:
        n = 100
        doomed = np.zeros(n, dtype=bool)
        doomed[:4] = True
        w = np.full((1, n), 0.996 / 96)
        w[0, :4] = 0.004 / 4
        idx, share = self.draw(w, doomed)
        kd = 4  # k // 4
        assert doomed[idx[0, :kd]].all() and not doomed[idx[0, kd:]].any()
        assert share[0, :kd].sum() == pytest.approx(0.004) and share.sum() == pytest.approx(1.0)

    def test_with_nothing_failing_every_reading_is_from_the_rest_and_equally_likely(self) -> None:
        w = np.full((2, 20), 0.05)
        idx, share = self.draw(w, np.zeros(20, dtype=bool), k=8)
        assert share == pytest.approx(np.full((2, 8), 1 / 8)) and idx.max() < 20

    def test_with_everything_failing_every_reading_is_from_the_failing_stratum(self) -> None:
        w = np.full((1, 20), 0.05)
        idx, share = self.draw(w, np.ones(20, dtype=bool), k=8)
        assert share == pytest.approx(np.full((1, 8), 1 / 8))

    def test_each_parent_is_split_by_its_own_weights(self) -> None:
        doomed = np.zeros(10, dtype=bool)
        doomed[:2] = True
        w = np.vstack([np.full(10, 0.1), np.r_[np.zeros(2), np.full(8, 0.125)]])
        idx, share = self.draw(w, doomed, k=8)
        assert share[0, :2].sum() == pytest.approx(0.2)  # first parent: a fifth is failing
        assert share[1].sum() == pytest.approx(1.0) and not doomed[idx[1]].any()  # second: none
