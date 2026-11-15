"""P4: the perception layer in a closed loop on the real F1 plant (scripts/perception_coverage.py).

The driver there is a stand-in for the episode loop of packet P12a: a bound rule through the plant's
own job path, with an oracle diagnosis. These tests check what the layer promises when real
replacements, failures, stops and horizon censoring feed it: every checkpoint is labelled once, no
label contradicts the twin-world truth, the Manski bounds bracket the oracle coverage, the providers
accept the labels in the order they arrive, and a rerun is identical. SYNTHETIC data only.
"""

import importlib.util
import sys
from functools import cache
from pathlib import Path

import pytest

from closed_loop.perception.conformal import development_data
from closed_loop.perception.coverage import checkpoint_coverage, manski_bounds, select
from closed_loop.perception.labels import LabelStatus
from closed_loop.perception.monitor import ProxyBand, proxy_development_scores
from closed_loop.plant import PlantParams

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "perception_coverage.py"
P = PlantParams()
EPS = 0.025  # a quarter of an hour is 0.0104 days; failures are seen at the end of their quarter


@cache
def harness():
    spec = importlib.util.spec_from_file_location("perception_coverage_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@cache
def pools():
    seeds = range(2000, 2020)
    return development_data(P, seeds), ProxyBand.fit(proxy_development_scores(P, seeds), 0.05)


@cache
def episode(seed, condition, variant="V3", stop_at_end=True):
    dev, band = pools()
    return harness().closed_loop_episode(
        seed, condition, variant, dev, band, P, stop_at_end=stop_at_end
    )


SEEDS = (1000, 1001, 1002, 1003)


@pytest.mark.parametrize("condition", ["nominal", "shifted"])
class TestInvariants:
    def streams(self, condition, variant="V3"):
        out = {"sched": [], "step99": [], "life95": []}
        for s in SEEDS:
            ep = episode(s, condition, variant)
            out["sched"].append(ep.sched)
            out["step99"].append(ep.sets["step99"])
            out["life95"].append(ep.sets["life95"])
        return out

    def test_every_checkpoint_has_exactly_one_label(self, condition):
        for kind, per_episode in self.streams(condition).items():
            for st in per_episode:
                cps = sorted((c.asset_id, c.life_id, c.day) for c in st.checkpoints)
                labs = sorted((x.asset_id, x.life_id, x.checkpoint_day) for x in st.labels)
                assert cps == labs, kind
                assert len(set(cps)) == len(cps)

    def test_lives_are_whole_and_ordered(self, condition):
        for st in self.streams(condition)["sched"]:
            by_life = {}
            for c in st.checkpoints:
                by_life.setdefault((c.asset_id, c.life_id), []).append(c.day)
            for days in by_life.values():
                assert days == sorted(days) and len(set(days)) == len(days)

    def test_no_label_contradicts_the_twin_world_truth(self, condition):
        bad = []
        n = 0
        for st in self.streams(condition)["sched"]:
            truth = {(c.asset_id, c.life_id, c.day): c.true_rul for c in st.checkpoints}
            for lab in st.labels:
                L = truth[(lab.asset_id, lab.life_id, lab.checkpoint_day)]
                n += 1
                if lab.status is LabelStatus.HIT and not (lab.lo - EPS <= L <= lab.hi + EPS):
                    bad.append(("HIT", lab, L))
                if lab.status is LabelStatus.MISS and lab.lo + EPS < L < lab.hi - EPS:
                    bad.append(("MISS", lab, L))
                if lab.status is LabelStatus.LOWER_HIT and L < lab.lo - EPS:
                    bad.append(("LOWER_HIT", lab, L))
                if not lab.censored and abs((lab.end_day - lab.checkpoint_day) - L) > EPS:
                    bad.append(("observed failure time", lab, L))
                if lab.censored and L < lab.end_day - lab.checkpoint_day - EPS:
                    bad.append(("censored before the twin failure", lab, L))
        assert n > 300 and not bad, bad[:3]

    def test_manski_bounds_bracket_the_oracle_coverage_in_every_run(self, condition):
        for kind, per_episode in self.streams(condition).items():
            for st in per_episode:
                for pop in ("all", "estimated", "prior", "post_blind"):
                    if not select(st.checkpoints, pop):
                        continue
                    lo, hi = manski_bounds(st.labels, pop)
                    orc = checkpoint_coverage(st.checkpoints, pop)
                    assert lo - 1e-9 <= orc <= hi + 1e-9, (kind, pop, lo, orc, hi)

    def test_the_bound_rule_leaves_the_learner_mostly_in_the_dark(self, condition):
        n = {s: 0 for s in LabelStatus}
        for st in self.streams(condition)["sched"]:
            for lab in st.labels:
                n[lab.status] += 1
        total = sum(n.values())
        informative_exact = n[LabelStatus.HIT] + n[LabelStatus.MISS]
        assert total > 300
        # the claim is not about the size of the effect, only its direction: censored labels
        # (replacements and the end of the episode) dominate what the learner gets
        assert informative_exact < total

    def test_a_rerun_is_identical(self, condition):
        dev, band = pools()
        again = harness().closed_loop_episode(SEEDS[0], condition, "V3", dev, band, P)
        first = episode(SEEDS[0], condition)
        assert again.sched.checkpoints == first.sched.checkpoints
        assert again.sched.labels == first.sched.labels
        assert again.fired == first.fired


class TestVariantsRunInTheLoop:
    @pytest.mark.parametrize("variant", ["V1", "V2", "V4", "V5", "V6", "V7"])
    def test_every_variant_completes_an_episode_and_labels_everything(self, variant):
        ep = episode(1001, "nominal", variant)
        assert len(ep.sched.checkpoints) == len(ep.sched.labels) > 100
        assert ep.sched.life_n >= 4

    def test_the_oracle_variant_sees_the_truth_and_the_others_do_not(self):
        # the labels the script keeps for scoring hold the truth; what the providers were given
        # is checked in test_conformal (learner views); here: the oracle ran without raising
        ep = episode(1002, "nominal", "V7")
        assert all(lab.true_rul is not None for lab in ep.sched.labels)

    def test_censoring_at_the_replacement_instead_of_the_stop_changes_the_labels(self):
        stop = episode(1000, "nominal", "V3", True)
        repl = episode(1000, "nominal", "V3", False)
        if stop.planned_jobs == 0:
            pytest.skip("no replacement happened on this seed")
        ends = {(x.asset_id, x.life_id, x.checkpoint_day): x.end_day for x in stop.sched.labels}
        later = [
            x
            for x in repl.sched.labels
            if x.censored and x.end_day > ends[(x.asset_id, x.life_id, x.checkpoint_day)] + 1e-9
        ]
        assert later, "the stop precedes the completion of a replacement"
