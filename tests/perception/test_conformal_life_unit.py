"""Follow-up to P4b: the life-unit forms V2L, V3L, V7L, and proof that V2, V3, V7 did not change.

SYNTHETIC data only. One update per resolved life, ``gamma`` unchanged at 0.02, using the life's
mean miss indicator under the variant's label rule.
"""

import json

import numpy as np
import pytest

from closed_loop.perception.conformal import (
    ALL_VARIANTS,
    GAMMA,
    LIFE_VARIANTS,
    VARIANTS,
    make_provider,
    replay_open_loop,
)
from closed_loop.perception.coverage import open_loop_lives
from closed_loop.perception.estimator import EstimatorOutput
from closed_loop.perception.labels import Label, LabelStatus, RulInterval
from closed_loop.plant.params import PlantParams
from tests.perception.golden_label_unit import GOLDEN, golden_run, run_stream

H, M, LH, U = LabelStatus.HIT, LabelStatus.MISS, LabelStatus.LOWER_HIT, LabelStatus.UNKNOWN
ALPHA = 0.10
LEVEL = 0.90


def provider(name, *, scores=None, seed=7, **kw):
    scores = np.random.default_rng(0).standard_normal(1000) if scores is None else scores
    return make_provider(name, alpha=ALPHA, development_scores=scores, seed=seed, **kw)


def est(m=0.0, sigma=1.0, status="estimated"):
    return EstimatorOutput("PU-201", 10, status, m, sigma, 5.0)


def lab(
    status,
    *,
    life=0,
    asset="PU-201",
    day=10,
    resolved=50,
    censored=False,
    prior=False,
    truth=None,
    lo=-1.0,
    hi=1.0,
    audit=False,
):
    return Label(
        life,
        asset,
        day,
        lo,
        hi,
        LEVEL,
        resolved,
        status,
        day + 3.0,
        censored,
        audit,
        prior,
        true_rul=truth,
        m=0.0,
        sigma=1.0,
    )


def a_est(p):
    return p.state()["alpha_t_est"]


class TestNames:
    def test_the_registered_set_is_unchanged_and_the_life_forms_are_additional(self):
        assert VARIANTS == ("V1", "V2", "V3", "V4", "V5", "V6", "V7")
        assert LIFE_VARIANTS == ("V2L", "V3L", "V7L")
        assert ALL_VARIANTS == (*VARIANTS, *LIFE_VARIANTS)

    def test_each_life_form_is_built_and_named(self):
        for name in LIFE_VARIANTS:
            p = provider(name)
            assert p.name == name and p.state()["unit_life"] == 1.0

    def test_the_per_label_forms_do_not_report_a_unit(self):
        for name in ("V2", "V3", "V7"):
            assert "unit_life" not in provider(name).state()

    def test_only_the_three_requested_life_forms_exist(self):
        for bad in ("V1L", "V4L", "V5L", "V6L", "V8L"):
            with pytest.raises(ValueError, match="unknown variant"):
                provider(bad)

    def test_a_life_form_starts_from_the_same_pool_as_its_label_form(self):
        scores = np.random.default_rng(3).standard_normal(3000)
        for base in ("V2", "V3", "V7"):
            a = provider(base, scores=scores).predict(est(m=20.0, sigma=3.0), LEVEL)
            b = provider(base + "L", scores=scores).predict(est(m=20.0, sigma=3.0), LEVEL)
            assert a == b


class TestOneUpdatePerLife:
    def test_v2l_exact_alpha_t_for_a_life(self):
        """Four observed failures, one a miss: mean miss indicator 1/4, ONE step of gamma."""
        life = [lab(M, hi=0.5, lo=-0.5), lab(H), lab(H), lab(H)]
        p = provider("V2L")
        p.observe(life)
        assert a_est(p) == pytest.approx(ALPHA + GAMMA * (ALPHA - 0.25))
        s = p.state()
        assert s["updates_est"] == 1.0 and s["n_labels"] == 4.0

    def test_the_per_label_form_steps_once_per_label_on_the_same_life(self):
        life = [lab(M, hi=0.5, lo=-0.5), lab(H), lab(H), lab(H)]
        p = provider("V2")
        p.observe(life)
        assert a_est(p) == pytest.approx(ALPHA + GAMMA * (ALPHA - 1.0) + 3 * GAMMA * ALPHA)
        assert p.state()["updates_est"] == 4.0

    def test_v3l_uses_the_mean_over_the_informative_labels_only(self):
        # UNKNOWN is left out; the censored MISS counts as a miss, LOWER_HIT as a hit
        life = [
            lab(U, censored=True),
            lab(M, censored=True),
            lab(LH, censored=True),
            lab(LH, censored=True),
        ]
        p = provider("V3L")
        p.observe(life)
        assert a_est(p) == pytest.approx(ALPHA + GAMMA * (ALPHA - 1.0 / 3.0))

    def test_v2l_counts_every_censored_label_as_a_hit(self):
        life = [lab(M, censored=True), lab(U, censored=True), lab(LH, censored=True)]
        p = provider("V2L")
        p.observe(life)
        assert a_est(p) == pytest.approx(ALPHA + GAMMA * ALPHA)  # mean miss 0

    def test_v7l_reads_the_truth_not_the_status(self):
        # interval [-1, 1]; truths 0.2 (inside), 5 (outside), 0 (inside), 3 (outside)
        life = [
            lab(LH, censored=True, truth=0.2),
            lab(H, truth=5.0),
            lab(U, censored=True, truth=0.0),
            lab(H, truth=3.0),
        ]
        p = provider("V7L")
        p.observe(life)
        assert a_est(p) == pytest.approx(ALPHA + GAMMA * (ALPHA - 0.5))

    def test_v7l_needs_the_truth(self):
        with pytest.raises(ValueError, match="true_rul"):
            provider("V7L").observe([lab(H, truth=None)])

    def test_the_step_does_not_depend_on_how_many_labels_a_life_has(self):
        short, long_ = provider("V3L"), provider("V3L")
        short.observe([lab(M, hi=0.5, lo=-0.5)])
        long_.observe([lab(M, hi=0.5, lo=-0.5, day=10 + i) for i in range(40)])
        assert a_est(short) == pytest.approx(a_est(long_))

    def test_a_life_with_trend_and_prior_checkpoints_updates_each_class_once(self):
        life = [
            lab(M, hi=0.5, lo=-0.5),
            lab(H),
            lab(M, hi=0.5, lo=-0.5, prior=True),
            lab(M, hi=0.5, lo=-0.5, prior=True),
            lab(H, prior=True),
            lab(H, prior=True),
        ]
        p = provider("V2L")
        p.observe(life)
        s = p.state()
        assert s["alpha_t_est"] == pytest.approx(ALPHA + GAMMA * (ALPHA - 0.5))
        assert s["alpha_t_prior"] == pytest.approx(ALPHA + GAMMA * (ALPHA - 0.5))
        assert s["updates_est"] == 1.0 and s["updates_prior"] == 1.0

    def test_lives_are_told_apart_by_asset_and_life_id(self):
        p = provider("V3L")
        p.observe(
            [lab(M, hi=0.5, lo=-0.5, life=0), lab(M, hi=0.5, lo=-0.5, life=1)]
            + [lab(H, asset="PU-202", life=0)]
        )
        assert p.state()["updates_est"] == 3.0
        assert a_est(p) == pytest.approx(
            ALPHA + 2 * GAMMA * (ALPHA - 1.0) + GAMMA * ALPHA, abs=1e-12
        )

    def test_scores_of_observed_failures_still_enter_the_pool_one_by_one(self):
        p = provider("V3L")
        p.observe([lab(H, day=10 + i) for i in range(5)])
        assert p.state()["scores_added_est"] == 5.0
        q = provider("V3L")
        q.observe([lab(H, censored=True, day=10 + i) for i in range(5)])
        assert q.state()["scores_added_est"] == 0.0


class TestSkips:
    def test_a_life_of_unknown_labels_does_not_move_v3l(self):
        p = provider("V3L")
        p.observe([lab(U, censored=True, day=10 + i) for i in range(12)])
        assert a_est(p) == ALPHA
        s = p.state()
        assert s["updates_est"] == 0.0 and s["skipped_est"] == 1.0

    def test_the_same_life_does_move_v2l_as_all_hits(self):
        p = provider("V2L")
        p.observe([lab(U, censored=True, day=10 + i) for i in range(12)])
        assert a_est(p) == pytest.approx(ALPHA + GAMMA * ALPHA)

    def test_a_skipped_life_leaves_a_later_one_to_count_normally(self):
        p = provider("V3L")
        p.observe([lab(U, censored=True, life=0)])
        p.observe([lab(M, hi=0.5, lo=-0.5, life=1, resolved=51)])
        assert a_est(p) == pytest.approx(ALPHA + GAMMA * (ALPHA - 1.0))


class TestOrderAndGuards:
    def test_a_life_arrives_in_one_call(self):
        p = provider("V3L")
        p.observe([lab(H)])
        with pytest.raises(ValueError, match="already observed"):
            p.observe([lab(H, day=11)])

    def test_labels_must_arrive_in_time_order(self):
        p = provider("V2L")
        p.observe([lab(H, resolved=60)])
        with pytest.raises(ValueError, match="time order"):
            p.observe([lab(H, life=1, resolved=59)])

    def test_predict_is_pure_and_the_level_is_checked(self):
        p = provider("V3L")
        assert p.predict(est(), LEVEL) == p.predict(est(), LEVEL)
        with pytest.raises(ValueError, match="one provider per level"):
            p.predict(est(), 0.99)

    def test_no_truth_is_read_by_the_non_oracle_life_forms(self):
        a, b = provider("V3L"), provider("V2L")
        c, d = provider("V3L"), provider("V2L")
        a.observe([lab(H, truth=99.0)])
        b.observe([lab(M, hi=0.5, lo=-0.5, truth=99.0)])
        c.observe([lab(H, truth=None)])
        d.observe([lab(M, hi=0.5, lo=-0.5, truth=None)])
        assert a.state() == c.state() and b.state() == d.state()


class TestWhyTheLifeUnitMatters:
    """The finding the follow-up answers: correlated labels in bulk swing a per-label step."""

    def trajectory(self, name, n_lives=200):
        rng = np.random.default_rng(5)
        p = provider(name)
        path = []
        for k in range(n_lives):
            bad = rng.random() < ALPHA  # a life is wholly missed or wholly covered
            status, lo, hi = (M, -0.5, 0.5) if bad else (H, -1.0, 1.0)
            p.observe(
                [lab(status, lo=lo, hi=hi, life=k, day=10 + i, resolved=50 + k) for i in range(40)]
            )
            path.append(a_est(p))
        return np.array(path)

    def test_a_per_label_step_swings_alpha_t_far_more_than_a_per_life_step(self):
        label_path, life_path = self.trajectory("V3"), self.trajectory("V3L")
        swing_label = float(np.max(np.abs(label_path - ALPHA)))
        swing_life = float(np.max(np.abs(life_path - ALPHA)))
        assert swing_label > 10 * swing_life
        assert swing_life < 0.1  # one step of gamma moves alpha_t by at most 0.02 x 0.9

    def test_the_life_unit_keeps_alpha_t_near_alpha_when_misses_come_at_the_nominal_rate(self):
        path = self.trajectory("V3L", n_lives=400)
        assert abs(float(path[-1]) - ALPHA) < 0.15
        assert float(np.min(path)) > -0.2


class TestReplay:
    def test_the_oracle_life_form_is_given_the_truth_and_the_others_are_not(self):
        lives = open_loop_lives(PlantParams(), 1001)

        class Spy:
            def __init__(self, name):
                self.name, self.truth = name, []

            def predict(self, e, level):
                return RulInterval(e.m_days - 5.0, e.m_days, e.m_days + 5.0, level, False)

            def observe(self, labels):
                self.truth.append(any(x.true_rul is not None for x in labels))

            def state(self):
                return {}

        for name, expected in (("V7L", True), ("V3L", False), ("V2L", False), ("V7", True)):
            spy = Spy(name)
            replay_open_loop(lives, spy, 0.9)
            assert spy.truth and all(t is expected for t in spy.truth), name


class TestPerLabelFormsAreUnchanged:
    """V2, V3 and V7 with ``unit="label"`` are bit for bit what they were before the follow-up."""

    @pytest.mark.parametrize("name", ["V2", "V3", "V7"])
    def test_stream_and_replay_match_the_golden_run(self, name):
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        now = json.loads(json.dumps(golden_run()))
        assert now[name]["stream"] == golden[name]["stream"]
        assert now[name]["replay"] == golden[name]["replay"]

    def test_the_golden_file_was_written_before_the_life_forms_existed(self):
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        assert set(golden) == {"V2", "V3", "V7"}
        for entry in golden.values():
            assert "unit_life" not in entry["stream"]["state"]
            assert len(entry["stream"]["intervals"]) == 36
            assert entry["replay"]["n_checkpoints"] == entry["replay"]["n_labels"] > 100

    def test_the_golden_comparison_has_teeth(self):
        """Change V3's step from 0.02 to 0.021 and the recorded state no longer matches."""
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        changed = json.loads(json.dumps(run_stream("V3", gamma=0.021)))
        assert changed["state"] != golden["V3"]["stream"]["state"]
        assert changed["intervals"] != golden["V3"]["stream"]["intervals"]
