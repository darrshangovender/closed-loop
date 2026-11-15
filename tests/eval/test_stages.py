"""``eval.stages``: the registry, the plug-in point, and the records that claims read.

The cell stages run here on cheap, invented episodes (``EpisodeTask.__call__`` is replaced by a
function of the arm name and the seed), so what is tested is the plumbing: that the names a stage
writes are the names a claim reads, that pairing is on the seed, and that a planted defect or a
stopped plant changes the verdict the way the registered rules say.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import inspect
import pickle
import sys
import types
from collections import Counter
from collections.abc import Callable
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from closed_loop.audit.guard import HeldoutGuard
from closed_loop.audit.ledger import Ledger
from closed_loop.audit.seeds import f2_seed, search_seed
from closed_loop.core.results import ResultRecord
from closed_loop.core.types import Fidelity
from closed_loop.eval import assemble, stages
from closed_loop.eval.arms import ARMS
from closed_loop.eval.assemble import claim_records
from closed_loop.eval.claims import SHIELD_PAIRS, SearchPairs, claim_name
from closed_loop.eval.counts import ALPHA_DESCRIPTIVE, ALPHA_PRIMARY, FULL, QUICK, Counts
from closed_loop.eval.episode import EpisodeResult
from closed_loop.eval.metrics import Attribution, EpisodeCoverage
from closed_loop.eval.report import check_outputs, write_outputs
from closed_loop.eval.runner import CellRun, EpisodeTask, Failed, Runner
from closed_loop.eval.stages import (
    HOSTILE_ARMS,
    PERCEPTION_ARMS,
    POLICY_ARMS,
    SEARCH_KINDS,
    STAGE_KEYS,
    STAGES,
    SearchOutcome,
    SearchTask,
    StageContext,
    cell_records,
    f1_seed_plan,
    failed_search,
    finish,
    missing_modules,
    paired_record,
    run_pipeline,
    value_record,
)
from closed_loop.eval.tuned import PROVISIONAL
from closed_loop.eval.tuned import current as current_tuned
from closed_loop.stats import clopper_pearson

PROPS = ("P1", "P2", "P3", "P4", "P5", "P6")


def invented(arm: str, seed: int, condition: str, profile: dict[str, Any]) -> EpisodeResult:
    """An episode of ``arm`` whose values come from ``profile`` and the seed, nothing else."""
    mins = dict.fromkeys(PROPS, 1.0)
    violates = profile.get("violates", 0.0)
    if violates and (seed % 100) < violates * 100:
        mins["P2"] = -1.0
    attributions = ()
    if profile.get("defect_every") and seed % profile["defect_every"] == 0:
        attributions = (Attribution("P2", 10, "assumption_held"),)
        mins["P2"] = -1.0
    shifted = condition == "shifted"
    monitors = {m: (60.0 if shifted else None) for m in ("opt", "pess", "inf", "proxy")}
    return EpisodeResult(
        arm=arm, seed=seed, condition=condition, fidelity=Fidelity.F1,
        violations={p: int(mins[p] < 0) for p in PROPS}, min_robustness=mins,
        throughput=profile.get("throughput", 0.9),
        cost=profile.get("cost", 20.0) + (seed % 5), unplanned_failures=profile.get("u", 0),
        planned_jobs=2, shield_overrides=3, shield_admitted=7, monitor_fired_day=None, labels=(),
        wall_s=0.0, attributions=attributions, commands_proposed=100,
        coverage=EpisodeCoverage(profile.get("cov", 0.9), profile.get("cov", 0.9), 50,
                                 profile.get("cov", 0.9), profile.get("cov", 0.9), 20, 5, 55),
        monitors=monitors if profile.get("monitors") else {},
        shift_day=45.0 if shifted else None,
        learner={"post_blind.n": 30, "post_blind.hit": 20, "post_blind.miss": 4,
                 "post_blind.lower_hit": 2, "post_blind.unknown": 4, "post_blind.lower_ok": 25,
                 "post_blind.uncensored": 24,
                 "estimated.n": 10, "estimated.hit": 6, "estimated.miss": 2,
                 "estimated.lower_hit": 1, "estimated.unknown": 1, "estimated.lower_ok": 8,
                 "estimated.uncensored": 8},
    )  # fmt: skip


PROFILES: dict[str, dict[str, Any]] = {
    "B3-V1": {"cost": 170.0, "cov": 0.80},
    "B3-V3": {"cost": 170.0, "cov": 0.85, "u": 6},
    "B3-V2": {"cost": 170.0, "cov": 0.95},
    "B3-S": {"cost": 180.0, "throughput": 0.2, "u": 6},
    "P0": {"cost": 10.0},
    "P1": {"cost": 14.0, "throughput": 0.09, "monitors": True},
    "P1-nopair": {"cost": 40.0, "throughput": 0.07, "u": 2},
    "P1-I": {"cost": 30.0, "throughput": 0.8, "violates": 0.0},
    "H-C0": {"violates": 0.5, "throughput": 0.99},
    "H-C1": {"throughput": 0.9},
    "H-C-I": {"throughput": 0.7},
    "H-G0": {"violates": 0.4, "throughput": 0.99},
    "H-G1": {"throughput": 0.9},
    "H-G-I": {"throughput": 0.7},
}


def install_cheap(mp: pytest.MonkeyPatch, profiles: dict[str, dict[str, Any]]) -> None:
    def call(self: EpisodeTask, seed: int) -> EpisodeResult:
        return invented(
            self.arm.name, seed, self.scenario.condition, profiles.get(self.arm.name, {})
        )

    mp.setattr(EpisodeTask, "__call__", call)
    mp.setattr(stages, "PERCEPTION_ARMS", ("B3-V1", "B3-V2", "B3-V3"))
    mp.setattr(stages, "POLICY_ARMS", ("B3-S", "P0", "P1", "P1-nopair", "P1-I"))
    mp.setattr(stages, "HOSTILE_ARMS", ("H-C0", "H-C1", "H-C-I", "H-G0", "H-G1", "H-G-I"))


@pytest.fixture
def cheap_cells(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, Any]]:
    """Replace the episode of every task by an invented one; edit the returned profiles to plant."""
    profiles = {k: dict(v) for k, v in PROFILES.items()}
    install_cheap(monkeypatch, profiles)
    # these tests read the claim inputs, not the descriptive tables (their own tests are above)
    monkeypatch.setattr(stages, "cell_records", lambda *a, **k: [])
    return profiles


N = 60
SEEDS = tuple(range(1100, 1100 + N))
COUNTS = Counts(batch=20, n_max_perception=N, n_max_policy=N, n_max_planner=N, n_max_hostile=N)


def context_for(runner: Runner, counts: Counts = COUNTS) -> StageContext:
    return StageContext(
        mode="quick", counts=counts, runner=runner, context=None,  # type: ignore[arg-type]
        seeds=SEEDS, seed_role="dev", tuned=PROVISIONAL, primary_variant="V3", monitor_mode="proxy",
        root=Path("."),
    )  # fmt: skip


def run_cells(only: tuple[str, ...] = ("perception_closed", "policy", "hostile")) -> Any:
    runner = Runner(workers=1, headroom=lambda: 40.0, timeout_s=None)
    sc = context_for(runner)
    result = run_pipeline(sc, only)
    return sc, result


@pytest.fixture(scope="module")
def default_run() -> tuple[StageContext, Any, Any]:
    """The three cell stages on the default invented episodes, run once for the whole module."""
    mp = pytest.MonkeyPatch()
    install_cheap(mp, {k: dict(v) for k, v in PROFILES.items()})
    try:
        sc, result = run_cells()
        return sc, result, finish(sc, result, prereg_sha="cd" * 32)
    finally:
        mp.undo()


def record_named(records: Any, name: str) -> ResultRecord:
    return next(r for r in records if r.name == name)


# ----------------------------------------------------------------------------- the registry


class TestRegistry:
    def test_there_is_one_stage_per_row_of_the_budget_in_the_order_they_run(self) -> None:
        # "reach" (0003 A16, made-up data about the statistics) has no row of the budget: it is
        # the one stage that is not a row, and it comes first
        assert STAGE_KEYS == (
            "reach", "f0", "perception_open", "perception_closed", "policy", "hostile", "causal",
            "search", "f2", "conformance_dp",
        )  # fmt: skip
        rows = [r for s in STAGES for r in s.rows]
        assert rows == list(range(1, 11))  # rows 8 and 9 are the one F2 stage
        assert next(s for s in STAGES if s.key == "reach").rows == ()

    def test_the_arm_groups_cover_the_registry_without_overlap(self) -> None:
        union = set(PERCEPTION_ARMS) | set(POLICY_ARMS) | set(HOSTILE_ARMS)
        assert union == set(ARMS)
        assert len(PERCEPTION_ARMS) + len(POLICY_ARMS) + len(HOSTILE_ARMS) == len(ARMS)

    def test_the_arms_named_by_the_claims_and_ablations_exist(self) -> None:
        for shielded, unshielded, interlock in SHIELD_PAIRS.values():
            assert {shielded, unshielded, interlock} <= set(ARMS)
        assert set(assemble._SHIELDED_ARMS) <= set(ARMS)
        for _label, a, b in assemble._ABLATIONS:
            assert a in ARMS and b.format(P="V3") in ARMS, (a, b)

    def test_the_pump_deadlock_is_an_ablation_row_and_a_shielded_arm_of_the_defect_scan(
        self,
    ) -> None:
        """0003 A18 R8: P1-nopair is P1 as it was before the operator's pair stop; the paired
        difference keeps the deadlock a generated number. F4: it is also a shielded arm, so an
        assumption_held violation in it is a defect like any other (P12e left it out of the scan;
        the third review put it in)."""
        rows = {a: b for _label, a, b in assemble._ABLATIONS}
        assert rows.get("P1-nopair") == "P1"
        assert "P1-nopair" in assemble._SHIELDED_ARMS

    def test_stale_specification_is_not_in_the_defect_scan(self) -> None:
        assert "S-stale" not in assemble._SHIELDED_ARMS and "S-clause" in assemble._SHIELDED_ARMS


def hide_modules(mp: pytest.MonkeyPatch, *names: str) -> None:
    """Make ``find_spec`` say the named modules do not exist, whatever the repo has."""
    real = importlib.util.find_spec

    def find_spec(name: str, package: str | None = None) -> Any:
        return None if name in names else real(name, package)

    mp.setattr(importlib.util, "find_spec", find_spec)


F2_MODULES = ("closed_loop.f2.episode", "closed_loop.f2.replay", "closed_loop.eval.stage_f2")


class TestNotBuilt:
    def test_a_stage_whose_module_is_missing_lists_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        hide_modules(monkeypatch, "closed_loop.search.falsify", *F2_MODULES[1:])
        by_key = {s.key: s for s in STAGES}
        assert missing_modules(by_key["search"]) == ["closed_loop.search.falsify"]
        assert missing_modules(by_key["f2"]) == list(F2_MODULES[1:])

    def test_a_missing_stage_is_loud_and_the_run_does_not_claim_completeness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        hide_modules(monkeypatch, "closed_loop.search.falsify", *F2_MODULES)
        sc = context_for(Runner(workers=1, headroom=lambda: 40.0))
        lines: list[str] = []
        sc.log = lines.append  # type: ignore[method-assign]
        result = run_pipeline(sc, ("search", "f2"))
        assert result.not_built == ("search", "f2") and not result.complete
        assert all(o.status == "not built" for o in result.outputs)
        assert any("NOT BUILT" in line and "closed_loop.search.falsify" in line for line in lines)
        assert "P9b" in result.outputs[0].notes[0] and "P11c" in result.outputs[1].notes[0]

    def test_one_missing_stage_is_enough_for_the_run_to_be_incomplete(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        hide_modules(monkeypatch, "closed_loop.search.falsify")
        sc = context_for(Runner(workers=1, headroom=lambda: 40.0))
        result = run_pipeline(sc, ("search",))
        assert not result.complete and result.not_built == ("search",)

    def test_a_skipped_stage_also_means_incomplete(self, cheap_cells: Any) -> None:
        _sc, result = run_cells(("policy",))
        assert not result.complete and result.not_built == ()

    def test_an_unknown_stage_is_an_error(self) -> None:
        sc = context_for(Runner(workers=1, headroom=lambda: 40.0))
        with pytest.raises(ValueError, match="unknown stage"):
            run_pipeline(sc, ("nonsense",))


# ----------------------------------------------------------------------------- the search stage


@dataclasses.dataclass
class FakeEval:
    label_counts: dict[str, int]
    objective: float = -0.5
    theta: tuple[float, ...] = (0.5, 0.5)
    violated: tuple[str, ...] = ("P2",)
    prop: str = "P2"  # the property of the events (a violation of P6 alone has its own)

    @property
    def events(self) -> tuple[tuple[str, str], ...]:
        """What ``search.falsify.Evaluation.events`` gives: ``(property, label)`` pairs."""
        return tuple((self.prop, label) for label, n in self.label_counts.items() if n > 0)


@dataclasses.dataclass
class FakeRun:
    first_violation_eval: int | None
    evals: int


@dataclasses.dataclass
class FakeResult:
    run: FakeRun
    violations: tuple[FakeEval, ...]
    shielded: bool
    regime: str


CLASSIFY_CALLS: list[Any] = []


def fake_classify(shielded: bool, regime: str, events: Any) -> str:
    """The rule of ``search.falsify.classify`` (0003 A5 and A15), restated so the tests stand
    alone: it takes ``(property, label)`` pairs, and only P1 to P5 can make a defect."""
    pairs = list(events)
    CLASSIFY_CALLS.append(pairs)
    assert all(isinstance(e, tuple) and len(e) == 2 for e in pairs), pairs  # never bare labels
    if not pairs:
        return "none"
    if not shielded:
        return "unshielded"
    if any(label == "assumption_held" and prop != "P6" for prop, label in pairs):
        return "defect" if regime == "in" else "unexplained_beyond_assumptions"
    if any(label != "assumption_held" for _prop, label in pairs):
        return "contract_failure"
    return "p6_unexplained"


def install_fake_search(
    mp: pytest.MonkeyPatch,
    behave: Callable[[str, str, int, int], FakeResult],
    seen: list[Any] | None = None,
) -> None:
    def falsify_detailed(
        factory: Any, regime: str, algo: str, budget: int, seed: int, **_kw: Any
    ) -> FakeResult:
        if seen is not None:
            seen.append((regime, algo, budget, seed, factory is fake.target_arm, current_tuned()))
        return behave(regime, algo, budget, seed)

    fake = types.ModuleType("closed_loop.search.falsify")
    fake.__spec__ = ModuleSpec("closed_loop.search.falsify", None)
    fake.falsify_detailed = falsify_detailed  # type: ignore[attr-defined]
    fake.classify = fake_classify  # type: ignore[attr-defined]
    fake.target_arm = lambda: ARMS["H-G1"]  # type: ignore[attr-defined]
    mp.setitem(sys.modules, "closed_loop.search.falsify", fake)


def nothing_in_r_in(regime: str, algo: str, budget: int, seed: int) -> FakeResult:
    if regime == "in":
        return FakeResult(FakeRun(None, budget), (), True, regime)
    first = 40 + seed % 7 if algo == "cma" else (None if seed % 2 else 250)
    found = (FakeEval({"A_exec": 1}),) if first is not None else ()
    return FakeResult(FakeRun(first, budget), found, True, regime)


def search_context(runs: int = 6, budget: int = 300) -> StageContext:
    return context_for(
        Runner(workers=1, headroom=lambda: 40.0), Counts(search_runs=runs, search_budget=budget)
    )


class TestSearchStage:
    def test_it_runs_each_regime_and_optimiser_over_the_same_seeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[Any] = []
        install_fake_search(monkeypatch, nothing_in_r_in, seen)
        result = run_pipeline(search_context(), ("search",))
        assert result.not_built == () and result.outputs[0].status == "done"
        assert len(seen) == 24 and {c[3] for c in seen} == set(range(3000, 3006))
        assert {(c[0], c[1]) for c in seen} == {
            ("in", "cma"),
            ("in", "random"),
            ("out", "cma"),
            ("out", "random"),
        }
        assert all(c[2] == 300 and c[4] for c in seen)  # the budget, and the module's target_arm
        names = {r.name for r in result.records}
        for needed in (
            claim_name("C-SEARCH", "find_rate", "R_out", "cma"),
            claim_name("C-SEARCH", "find_rate", "R_out", "random"),
            claim_name("C-SEARCH", "find_diff", "R_out"),
            claim_name("C-SEARCH", "evals_diff", "R_out"),
            claim_name("C-SEARCH", "r_in_violations"),
            claim_name("C-SHIELD", "search_r_in"),
        ):
            assert needed in names, needed
        rin = record_named(result.records, claim_name("C-SEARCH", "r_in_violations"))
        assert rin.value == 0.0 and rin.n == 12

    def test_the_find_rate_and_the_pairing_are_read_from_the_runs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fake_search(monkeypatch, nothing_in_r_in)
        records = run_pipeline(search_context(), ("search",)).records
        cma = record_named(records, claim_name("C-SEARCH", "find_rate", "R_out", "cma"))
        rnd = record_named(records, claim_name("C-SEARCH", "find_rate", "R_out", "random"))
        assert (cma.value, cma.extra["k"], cma.n) == (1.0, 6, 6)
        assert rnd.extra["k"] == 3 and rnd.value == 0.5  # seeds 3000, 3002, 3004 find at 250
        diff = record_named(records, claim_name("C-SEARCH", "find_diff", "R_out"))
        assert diff.value == pytest.approx(0.5) and diff.n == 6
        first = record_named(records, "search.R_out.cma.evals_to_first")
        # 40 + seed % 7 for seeds 3000..3005 (3000 % 7 == 4)
        assert first.value == pytest.approx(np.mean([40 + s % 7 for s in range(3000, 3006)]))

    def test_the_run_pairs_are_recorded_with_an_exact_interval_at_the_claim_level(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fake_search(monkeypatch, nothing_in_r_in)
        records = run_pipeline(search_context(), ("search",)).records
        pairs = record_named(records, claim_name("C-SEARCH", "pairs", "R_out"))
        # CMA-ES finds at 40 + seed % 7; random search at 250 (even seeds) or never: CMA-ES is
        # first, or alone, in every one of the six pairs
        assert (pairs.extra["cma"], pairs.extra["random"]) == (6, 0)
        assert (pairs.extra["neither"], pairs.extra["tied"], pairs.extra["pairs"]) == (0, 0, 6)
        assert pairs.n == 6 and pairs.value == 1.0 and pairs.extra["k"] == 6
        assert (pairs.ci_lo, pairs.ci_hi) == clopper_pearson(6, 6, ALPHA_PRIMARY)
        assert pairs.ci_kind == f"clopper-pearson-{ALPHA_PRIMARY:g}"

    def test_six_decided_pairs_cannot_win_and_ten_clean_ones_do(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fake_search(monkeypatch, nothing_in_r_in)
        for runs, label in ((6, "inconclusive"), (10, "win")):
            sc = search_context(runs=runs)
            result = run_pipeline(sc, ("search",))
            by = {v.claim: v for v in finish(sc, result, prereg_sha=None).verdicts}
            assert by["C-SEARCH"].label == label, runs

    def test_random_search_first_in_every_pair_is_a_loss(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def behave(regime: str, algo: str, budget: int, seed: int) -> FakeResult:
            first = 10 if algo == "random" else 60
            found = (FakeEval({"A_exec": 1}),) if regime == "out" else ()
            run = FakeRun(first if regime == "out" else None, budget)
            return FakeResult(run, found, True, regime)

        install_fake_search(monkeypatch, behave)
        sc = search_context(runs=10)
        done = finish(sc, run_pipeline(sc, ("search",)), prereg_sha=None)
        v = {x.claim: x for x in done.verdicts}["C-SEARCH"]
        assert v.label == "loss"
        assert (v.numbers["pairs.cma"], v.numbers["pairs.random"]) == (0, 10)

    def test_the_pairs_are_taken_by_run_with_ties_and_blanks_dropped_and_counted(self) -> None:
        def out(first: int | None) -> SearchOutcome:
            return SearchOutcome(first, 200, int(first is not None))

        cma = {1: out(30), 2: out(None), 3: out(50), 4: out(90), 5: out(None), 6: out(60)}
        rand = {1: out(80), 2: out(40), 3: out(50), 4: out(20), 5: out(None), 6: out(None)}
        got = stages.search_pairs(cma, rand, [1, 2, 3, 4, 5, 6], 200)
        # 1 CMA-ES (30 < 80), 2 random (only), 3 tied, 4 random (20 < 90), 5 neither, 6 CMA-ES
        # (only)
        assert got == SearchPairs(cma=2, rand=2, neither=1, tied=1)

    def test_a_failed_run_is_a_pair_in_which_it_found_nothing(self) -> None:
        ok = SearchOutcome(70, 200, 1)
        cma = {1: failed_search(200), 2: ok}
        rand = {1: ok, 2: failed_search(200)}
        got = stages.search_pairs(cma, rand, [1, 2], 200)
        assert got == SearchPairs(cma=1, rand=1)

    def test_a_find_beyond_the_budget_is_not_a_find_in_the_pairs(self) -> None:
        late = SearchOutcome(250, 300, 1)  # found at 250 of a longer run than the budget asked for
        early = SearchOutcome(150, 300, 1)
        got = stages.search_pairs({1: late, 2: early}, {1: early, 2: late}, [1, 2], 200)
        assert got == SearchPairs(cma=1, rand=1)
        # and a pair in which neither found one inside the budget is "neither", also when a run
        # went on past it (it is not a tie, which needs two finds)
        none = SearchOutcome(None, 300, 0)
        got = stages.search_pairs(
            {1: late, 2: none, 3: late}, {1: none, 2: late, 3: late}, [1, 2, 3], 200
        )
        assert got == SearchPairs(cma=0, rand=0, neither=3, tied=0)

    def test_only_a_violation_with_every_assumption_held_is_a_defect_in_r_in(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def behave(regime: str, algo: str, budget: int, seed: int) -> FakeResult:
            if regime != "in":
                return FakeResult(FakeRun(None, budget), (), True, regime)
            if seed == 3001:  # a defect: found by both optimisers
                return FakeResult(
                    FakeRun(7, budget), (FakeEval({"assumption_held": 2}),), True, "in"
                )
            if seed == 3002:  # a contract failure: a violation with a broken assumption
                return FakeResult(FakeRun(9, budget), (FakeEval({"A_exec": 1}),), True, "in")
            return FakeResult(FakeRun(None, budget), (), True, "in")

        install_fake_search(monkeypatch, behave)
        out = run_pipeline(search_context(), ("search",))
        rin = record_named(out.records, claim_name("C-SEARCH", "r_in_violations"))
        shield = record_named(out.records, claim_name("C-SHIELD", "search_r_in"))
        assert (rin.value, rin.n) == (2.0, 12) == (shield.value, shield.n)  # seed 3001, twice
        assert rin.extra["contract_failure_runs"] == 2
        assert "2 of 12 runs with a defect, 2 with a contract failure" in out.outputs[0].notes[0]
        per_algo = record_named(out.records, "search.R_in.cma.defect_runs")
        assert (per_algo.extra["k"], per_algo.n) == (1, 6)

    def test_a_violation_in_r_out_with_no_broken_assumption_is_said(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def behave(regime: str, algo: str, budget: int, seed: int) -> FakeResult:
            if regime == "out" and seed == 3000:
                return FakeResult(
                    FakeRun(3, budget), (FakeEval({"assumption_held": 1}),), True, "out"
                )
            return FakeResult(FakeRun(None, budget), (), True, regime)

        install_fake_search(monkeypatch, behave)
        out = run_pipeline(search_context(), ("search",))
        assert any(
            "2 of the runs had a violation that none of A_exec, A_phys" in n
            for n in out.outputs[0].notes
        )
        rin = record_named(out.records, claim_name("C-SEARCH", "r_in_violations"))
        assert rin.value == 0.0  # a gap in regime out is not a defect of the proof

    def test_a_failed_run_counts_as_nothing_found_and_is_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def behave(regime: str, algo: str, budget: int, seed: int) -> FakeResult:
            if regime == "out" and algo == "cma" and seed == 3003:
                raise RuntimeError("the plant fell over")
            return FakeResult(FakeRun(5, budget), (FakeEval({"A_phys": 1}),), True, regime)

        install_fake_search(monkeypatch, behave)
        out = run_pipeline(search_context(), ("search",))
        cma = record_named(out.records, claim_name("C-SEARCH", "find_rate", "R_out", "cma"))
        assert cma.extra["k"] == 5 and cma.n == 6 and cma.extra["failed"] == 1
        assert "1 of the runs failed" in cma.measured_on
        assert "search: 1 of the runs failed and count as nothing found" in out.outputs[0].notes
        evals = record_named(out.records, "search.R_out.cma.evals_to_first")
        assert evals.value == pytest.approx((5 * 5 + 300) / 6)  # the whole budget for the failure

    def test_a_search_module_without_classify_is_not_built(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = types.ModuleType("closed_loop.search.falsify")
        fake.__spec__ = ModuleSpec("closed_loop.search.falsify", None)
        fake.falsify_detailed = lambda *a, **k: None  # type: ignore[attr-defined]
        fake.target_arm = lambda: ARMS["H-G1"]  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "closed_loop.search.falsify", fake)
        result = run_pipeline(search_context(), ("search",))
        assert result.not_built == ("search",) and "classify" in result.outputs[0].notes[0]

    def test_the_tuned_values_of_the_run_reach_the_search_and_its_cache_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[Any] = []
        install_fake_search(monkeypatch, nothing_in_r_in, seen)
        other = dataclasses.replace(PROVISIONAL, k_b3=PROVISIONAL.k_b3 + 1.0)
        run_pipeline(dataclasses.replace(search_context(2), tuned=other), ("search",))
        assert seen and all(c[5] == other for c in seen)
        a = SearchTask("out", "cma", 30, PROVISIONAL).fingerprint()
        b = SearchTask("out", "cma", 30, other).fingerprint()
        assert a != b and a["target"] == b["target"]

    def test_budget_regime_and_optimiser_are_part_of_the_cache_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fake_search(monkeypatch, nothing_in_r_in)
        base = SearchTask("out", "cma", 30, PROVISIONAL).fingerprint()
        for changed in (
            SearchTask("in", "cma", 30, PROVISIONAL),
            SearchTask("out", "random", 30, PROVISIONAL),
            SearchTask("out", "cma", 31, PROVISIONAL),
        ):
            assert changed.fingerprint() != base

    def test_the_run_deadline_makes_room_for_a_whole_search_and_is_given_back(self) -> None:
        runner = Runner(workers=1, headroom=lambda: 40.0, timeout_s=900.0)
        with stages._run_deadline(runner, 9000.0):
            assert runner.timeout_s == 9000.0
        assert runner.timeout_s == 900.0
        with stages._run_deadline(runner, 10.0):
            assert runner.timeout_s == 900.0  # never shortened
        unlimited = Runner(workers=1, headroom=lambda: 40.0, timeout_s=None)
        with stages._run_deadline(unlimited, 9000.0):
            assert unlimited.timeout_s is None

    def test_the_deadline_is_given_back_when_the_search_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner = Runner(workers=1, headroom=lambda: 40.0, timeout_s=900.0)
        with pytest.raises(KeyError), stages._run_deadline(runner, 9000.0):
            raise KeyError("boom")
        assert runner.timeout_s == 900.0

    def test_the_outcome_of_a_run_is_picklable_and_says_what_kind_it_found(self) -> None:
        o = SearchOutcome(12, 300, 3, (("contract_failure", 2), ("defect", 1)))
        assert pickle.loads(pickle.dumps(o)) == o
        assert (o.of_kind("defect"), o.of_kind("contract_failure"), o.of_kind("x")) == (1, 2, 0)
        assert o.found(300) and not o.found(10) and not SearchOutcome(None, 300, 0).found(300)
        assert failed_search(300).failed and not failed_search(300).found(300)
        assert (o.evals_used(300), o.evals_used(10)) == (12, 10)  # censored at the budget
        assert SearchOutcome(300, 300, 1).found(300)  # a violation at the last evaluation counts
        assert SearchOutcome(None, 300, 0).evals_used(300) == 300

    def test_a_search_task_reduces_a_run_to_its_kinds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def behave(regime: str, algo: str, budget: int, seed: int) -> FakeResult:
            evs = (
                FakeEval({"assumption_held": 1}),
                FakeEval({"A_exec": 3, "C1": 0}),
                FakeEval({"A_exec": 1}),
                FakeEval({}),  # a violation of P6 alone: no attribution, no kind
            )
            return FakeResult(FakeRun(4, budget), evs, True, regime)

        install_fake_search(monkeypatch, behave)
        o = SearchTask("in", "random", 60, PROVISIONAL)(3000)
        assert (o.first_violation_eval, o.evals, o.violating) == (4, 60, 4)
        assert o.kinds == (("contract_failure", 2), ("defect", 1)) and not o.failed
        assert o.first_find == ((0.5, 0.5), ("P2",))  # the first violating evaluation

    def test_a_task_keeps_the_first_violating_evaluation_and_not_the_most_severe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def behave(regime: str, algo: str, budget: int, seed: int) -> FakeResult:
            evs = tuple(
                FakeEval(
                    {"A_exec": 1}, objective=-0.1 * k, theta=(k / 10, 0.0),
                    violated=("P2",) if k == 1 else ("P3",),
                )
                for k in range(1, 9)
            )  # fmt: skip
            return FakeResult(FakeRun(1, budget), evs, True, regime)

        install_fake_search(monkeypatch, behave)
        o = SearchTask("out", "cma", 60, PROVISIONAL)(3000)
        assert o.violating == 8
        assert o.first_find == ((0.1, 0.0), ("P2",))  # evaluation order, with its own properties
        assert (
            pickle.loads(pickle.dumps(o)) == o
        )  # it travels back from a worker and into the cache

    def test_the_task_gives_classify_the_property_with_each_label(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        CLASSIFY_CALLS.clear()

        def behave(regime: str, algo: str, budget: int, seed: int) -> FakeResult:
            evs = (
                FakeEval({"assumption_held": 1}, prop="P2"),
                FakeEval({"assumption_held": 1, "C1": 2}, prop="P5"),
            )
            return FakeResult(FakeRun(1, budget), evs, True, regime)

        install_fake_search(monkeypatch, behave)
        SearchTask("in", "cma", 20, PROVISIONAL)(3000)
        assert CLASSIFY_CALLS == [
            [("P2", "assumption_held")],
            [("P5", "assumption_held"), ("P5", "C1")],
        ]

    def test_an_unexplained_violation_of_p6_alone_is_not_a_defect(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def behave(regime: str, algo: str, budget: int, seed: int) -> FakeResult:
            ev = FakeEval({"assumption_held": 1}, prop="P6", violated=("P6",))
            return FakeResult(FakeRun(1, budget), (ev,), True, regime)

        install_fake_search(monkeypatch, behave)
        o = SearchTask("in", "cma", 20, PROVISIONAL)(3000)
        assert o.of_kind("defect") == 0 and o.of_kind("p6_unexplained") == 1
        sc = search_context(runs=2)
        out = run_pipeline(sc, ("search",))
        rin = record_named(out.records, claim_name("C-SEARCH", "r_in_violations"))
        assert rin.value == 0.0  # not a defect of the proof

    def test_the_real_classify_reads_the_pairs_the_task_gives_it(self) -> None:
        if importlib.util.find_spec("closed_loop.search.falsify") is None:
            pytest.skip("packet P9b is not in this tree")
        classify = importlib.import_module("closed_loop.search.falsify").classify
        assert classify(True, "in", [("P2", "assumption_held")]) == "defect"
        assert classify(True, "in", [("P6", "assumption_held")]) != "defect"
        assert classify(True, "out", [("P5", "assumption_held")]) == (
            "unexplained_beyond_assumptions"
        )
        assert classify(True, "in", [("P5", "C1")]) == "contract_failure"

    def test_a_run_that_found_nothing_has_no_find(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_fake_search(
            monkeypatch, lambda r, a, b, s: FakeResult(FakeRun(None, b), (), True, r)
        )
        o = SearchTask("out", "cma", 60, PROVISIONAL)(3000)
        assert o.first_find is None and failed_search(60).first_find is None

    def test_the_replay_set_is_the_first_find_of_each_run_in_run_order_across_the_cells(
        self,
    ) -> None:
        def out(seed: int, regime: str, algo: str, found: bool = True) -> SearchOutcome:
            first = ((seed / 1e4, float(len(algo))), ("P2",)) if found else None
            return SearchOutcome(1 if found else None, 200, 1 if found else 0, (), first)

        seeds = (3000, 3001, 3002)
        runs = {
            (r, a): {s: out(s, r, a, found=not (s == 3001 and (r, a) == ("in", "random")))
                     for s in seeds}
            for r, a in stages.SEARCH_CELLS
        }  # fmt: skip
        got = stages.search_hand_off(runs, seeds, 25)
        assert [(f.seed, f.regime, f.algo) for f in got] == [
            (3000, "in", "cma"), (3000, "in", "random"),
            (3000, "out", "cma"), (3000, "out", "random"),
            (3001, "in", "cma"), (3001, "out", "cma"), (3001, "out", "random"),  # in/random: none
            (3002, "in", "cma"), (3002, "in", "random"),
            (3002, "out", "cma"), (3002, "out", "random"),
        ]  # fmt: skip
        assert got[0].props == ("P2",) and got[0].theta == (0.3, 3.0)

    def test_the_cap_takes_finds_across_all_four_cells_and_not_the_first_of_one(self) -> None:
        seeds = tuple(range(3000, 3010))
        runs = {
            cell: {s: SearchOutcome(1, 200, 1, (), ((float(s),), ("P2",))) for s in seeds}
            for cell in stages.SEARCH_CELLS
        }
        got = stages.search_hand_off(runs, seeds, 25)
        assert len(got) == 25
        assert {(f.regime, f.algo) for f in got} == set(stages.SEARCH_CELLS)
        counts = Counter((f.regime, f.algo) for f in got)
        assert max(counts.values()) - min(counts.values()) <= 1  # six or seven from each cell
        assert [f.seed for f in got[:4]] == [3000] * 4  # run order: the first seed of every cell

    def test_a_failed_run_adds_no_find_and_the_set_can_be_shorter_than_the_cap(self) -> None:
        seeds = (3000, 3001)
        runs = {
            cell: {3000: failed_search(200), 3001: failed_search(200)}
            for cell in stages.SEARCH_CELLS
        }
        assert stages.search_hand_off(runs, seeds, 25) == []

    def test_an_empty_hand_over_is_still_left_behind_for_the_f2_stage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fake_search(
            monkeypatch, lambda r, a, b, s: FakeResult(FakeRun(None, b), (), True, r)
        )
        sc = search_context(runs=3)
        run_pipeline(sc, ("search",))
        assert (
            sc.shared["search_violations"] == []
        )  # [] says "searched, found nothing"; absence does not

    def test_the_stage_hands_over_the_first_finds_not_the_most_severe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def behave(regime: str, algo: str, budget: int, seed: int) -> FakeResult:
            first = FakeEval({"A_exec": 1}, objective=-0.1, theta=(0.1, 0.1))
            worse = FakeEval({"A_exec": 1}, objective=-0.9, theta=(0.9, 0.9))
            return FakeResult(FakeRun(2, budget), (first, worse), True, regime)

        install_fake_search(monkeypatch, behave)
        counts = Counts(search_runs=2, search_budget=200, f2_replays=3)
        sc = dataclasses.replace(search_context(runs=2), counts=counts)
        result = run_pipeline(sc, ("search",))
        found = sc.shared["search_violations"]
        assert len(found) == 3 and all(f.theta == (0.1, 0.1) for f in found)
        assert [(f.seed, f.regime, f.algo) for f in found] == [
            (3000, "in", "cma"), (3000, "in", "random"), (3000, "out", "cma"),
        ]  # fmt: skip
        assert any("3 finds" in n and "at most 3" in n for n in result.outputs[0].notes)

    def test_the_real_module_has_what_the_stage_calls_if_it_is_there(self) -> None:
        if importlib.util.find_spec("closed_loop.search.falsify") is None:
            pytest.skip("packet P9b is not in this tree")
        module = importlib.import_module("closed_loop.search.falsify")
        params = list(inspect.signature(module.falsify_detailed).parameters)
        assert params[:5] == ["arm_factory", "regime", "algo", "budget", "seed"]
        assert list(inspect.signature(module.classify).parameters)[:3] == [
            "shielded",
            "regime",
            "labels",
        ]
        assert callable(module.target_arm) and module.target_arm().shield
        for kind in SEARCH_KINDS:
            assert kind in module.KINDS


# ----------------------------------------------------------------------------- helpers


class TestResultsAreReproducible:
    """0003 A15: results.json holds results; what depends on the state of the cache or on the clock
    goes to the run log."""

    def run_of(self, computed: int, cached: int) -> CellRun:
        results = [invented("B0", 1100 + i, "nominal", {}) for i in range(10)]
        return CellRun(
            "B0/nominal", tuple(range(1100, 1110)), results, [], 500, "n_max", 0.2, computed,
            cached, seconds=12.3 + computed,
        )  # fmt: skip

    def test_a_cell_summary_does_not_depend_on_how_many_episodes_came_from_the_cache(self) -> None:
        cold = cell_records("B0", "nominal", self.run_of(10, 0))
        warm = cell_records("B0", "nominal", self.run_of(0, 10))
        assert [dataclasses.asdict(r) for r in cold] == [dataclasses.asdict(r) for r in warm]
        summary = next(r for r in cold if r.name.endswith(".cell"))
        assert "computed" not in summary.extra and "cached" not in summary.extra

    def test_the_summary_of_the_sabotage_suite_has_no_clock_in_it(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        for name in ("test_p5b_sabotage.py", "test_sabotage.py"):
            (tmp_path / "tests" / "shield").mkdir(parents=True, exist_ok=True)
            (tmp_path / "tests" / "shield" / name).write_text("", encoding="utf-8")
        for line in ("24 passed in 3.2s", "1 failed, 23 passed in 118.30s (0:01:58)"):
            monkeypatch.setattr(
                stages.subprocess,
                "run",
                lambda cmd, line=line, **kw: types.SimpleNamespace(
                    stdout=line + "\n", returncode=0
                ),
            )
            note = stages._run_sabotage(tmp_path)[2]
            assert " in " not in note and "s)" not in note and "(0:" not in note, note


class TestAFailedOpenLoopSeed:
    """The same stand-in as a failed closed-loop episode: worst coverage, counted."""

    def test_it_counts_at_zero_coverage_in_every_variant_and_is_said(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def call(self: stages.OpenLoopTask, seed: int) -> dict[str, EpisodeCoverage]:
            if seed == 1101:
                raise RuntimeError("the lives would not render")
            return dict.fromkeys(self.variants, EpisodeCoverage(0.9, 0.9, 50, 0.9, 0.9, 20, 5, 55))

        monkeypatch.setattr(stages.OpenLoopTask, "__call__", call)
        runner = Runner(workers=1, headroom=lambda: 40.0, timeout_s=None)
        sc = context_for(runner, Counts(batch=4, open_loop_seeds=4))
        out = stages.run_perception_open(sc)
        cov = sc.shared["open_loop"][("nominal", "V3")]
        assert set(cov) == {1100, 1101, 1102, 1103}  # no seed is dropped
        assert cov[1101] == stages.WORST_COVERAGE and cov[1101].c == 0.0
        assert cov[1100].c == 0.9
        assert any("count at the worst coverage" in n for n in out.notes)
        record = record_named(out.records, "open.V3.nominal.m_cov")
        assert record.n == 4 and record.value == pytest.approx((0.9 * 3 + 0.0) / 4)
        assert record.extra["failed"] == 1


class TestRecordHelpers:
    def test_a_value_record_has_the_mean_the_interval_and_the_clip_count(self) -> None:
        r = value_record(
            "x", [0.2, 0.4, 1.5], hi=1.0, fidelity=Fidelity.F1, measured_on="three numbers"
        )
        assert r.value == pytest.approx((0.2 + 0.4 + 1.0) / 3) and r.n == 3
        assert r.extra["clipped"] == 1 and r.ci_lo is not None and r.ci_lo <= r.value <= r.ci_hi

    def test_a_proportion_with_no_event_has_its_zero_event_bound(self) -> None:
        r = value_record(
            "x", [0.0] * 50, fidelity=Fidelity.F1, measured_on="m", proportion=True, alpha=0.01
        )
        assert r.extra["k"] == 0 and 0.0 < r.extra["zero_event_upper"] < 1.0

    def test_a_paired_record_uses_the_seeds_both_cells_have(self) -> None:
        a = {1: 0.5, 2: 0.7, 3: 0.9}
        b = {2: 0.4, 3: 0.8, 4: 0.1}
        r = paired_record(
            "d", a, b, lo=0.0, hi=1.0, alpha=0.05, fidelity=Fidelity.F1, measured_on="m"
        )
        assert r is not None and r.n == 2 and r.value == pytest.approx(0.2)
        assert r.extra["seeds_a"] == 3 and r.extra["seeds_b"] == 3
        assert r.ci_lo is not None and r.ci_hi is not None and r.ci_lo <= r.value <= r.ci_hi

    def test_a_paired_record_reads_the_pairs_in_the_order_of_the_run_not_sorted(self) -> None:
        # held-out seeds have no order of their own: the order is the run's (batches of episodes),
        # and a betting sequence is built in the order it is fed. The spread-scaled sequence of
        # paired_cs (0003 A16) does not depend on the order, so what pins the order is the betting
        # sequence: the fallback for few pairs, and the whole of the safety leg of C-PLAN
        from closed_loop.stats import paired_betting_cs, paired_cs

        rng = np.random.default_rng(3)
        seeds = [int(x) for x in rng.permutation(10_000)[:150]]  # the run's order, not sorted
        a = {s: float(x) for s, x in zip(seeds, rng.uniform(0.2, 0.9, len(seeds)), strict=True)}
        b = {s: float(x) for s, x in zip(seeds, rng.uniform(0.0, 0.6, len(seeds)), strict=True)}
        va = np.array([a[s] for s in seeds])
        vb = np.array([b[s] for s in seeds])
        s_va = np.array([a[s] for s in sorted(seeds)])
        s_vb = np.array([b[s] for s in sorted(seeds)])
        betting = paired_record(
            "d", a, b, lo=0.0, hi=1.0, alpha=0.05, fidelity=Fidelity.F1, measured_on="m",
            betting_only=True,
        )  # fmt: skip
        assert betting is not None
        want = paired_betting_cs(va, vb, 0.0, 1.0, 0.05)
        in_sorted_order = paired_betting_cs(s_va, s_vb, 0.0, 1.0, 0.05)
        assert (betting.ci_lo, betting.ci_hi) == (min(want), max(want))
        assert want != in_sorted_order  # the data are such that the order shows
        # the default is paired_cs, which here is the spread-scaled sequence (150 pairs, no heavy
        # tail), narrower than the betting one, and the same in any order
        spread = paired_record(
            "d", a, b, lo=0.0, hi=1.0, alpha=0.05, fidelity=Fidelity.F1, measured_on="m"
        )
        assert spread is not None and (spread.ci_lo, spread.ci_hi) == pytest.approx(
            paired_cs(va, vb, 0.0, 1.0, 0.05), abs=1e-12
        )
        assert paired_cs(va, vb, 0.0, 1.0, 0.05) == pytest.approx(
            paired_cs(s_va, s_vb, 0.0, 1.0, 0.05), abs=1e-12
        )
        assert spread.ci_hi - spread.ci_lo < betting.ci_hi - betting.ci_lo

    def test_no_shared_seed_is_no_record(self) -> None:
        assert paired_record("d", {1: 0.1}, {2: 0.1}, lo=0, hi=1, alpha=0.05, fidelity=Fidelity.F1,
                             measured_on="m") is None  # fmt: skip

    def test_cell_records_carry_the_failures_and_the_provisional_flags(self) -> None:
        good = invented("B0", 1, "nominal", {})
        bad = Failed(2, "timeout", "no result after 900 s")
        from closed_loop.eval.runner import worst_episode

        stand_in = worst_episode(ARMS["B0"], 2, "nominal", Fidelity.F1, "timeout: no result")
        run = CellRun("B0/nominal", (1, 2), [good, stand_in], [bad], 500, "n_max", 0.2, 2, 0)
        recs = cell_records("B0", "nominal", run)
        cell = record_named(recs, "B0.nominal.cell")
        assert cell.extra["failed"] == 1 and "seed 2: timeout" in str(cell.extra["failures"])
        cost = record_named(recs, "B0.nominal.m_cost")
        assert cost.extra["failed"] == 1 and cost.n == 2


class TestContractAndLatency:
    def episodes(self) -> list[EpisodeResult]:
        from closed_loop.eval.interlock import ViolationLag
        from closed_loop.perception.contract import ContractCounts

        base = invented("H-C-I", 1, "nominal", {})
        counts = ContractCounts(10, 2, 1, 100, 5, 3, 250)
        lags = (ViolationLag("P2", "CR-501", 10, 10), ViolationLag("P2", "CR-501", 90, None))
        one = dataclasses.replace(base, contract=counts, contract_source="life95", online_lags=lags)
        two = dataclasses.replace(one, seed=2, online_lags=(ViolationLag("P3", "CR-501", 5, 7),))
        return [one, two]

    def test_both_readings_of_the_contract_are_recorded_with_their_denominators(self) -> None:
        recs = stages.contract_records("H-C-I", "nominal", self.episodes())
        by = {r.name.rsplit(".", 1)[1]: r for r in recs}
        assert by["any_miss_rul"].value == pytest.approx(4 / 20) and by["any_miss_rul"].n == 20
        assert by["any_miss_bin"].extra["k"] == 2 and by["any_miss_bin"].n == 20
        assert by["step_miss_bin"].value == pytest.approx(6 / 200) and by["step_miss_bin"].n == 200
        assert by["mean_width_bins"].value == pytest.approx(500 / 200)
        assert (
            by["any_miss_rul"].ci_kind == "wilson-0.95"
            and by["any_miss_rul"].extra["source"] == "life95"
        )
        assert (
            by["any_miss_rul"].ci_lo is not None
            and by["any_miss_rul"].ci_lo < 0.2 < by["any_miss_rul"].ci_hi
        )

    def test_an_arm_with_no_perception_has_no_contract_records(self) -> None:
        assert stages.contract_records("B0", "nominal", [invented("B0", 1, "nominal", {})]) == []

    def test_the_online_lag_histogram_is_recorded_per_property(self) -> None:
        recs = {r.name: r for r in stages.latency_records("H-C-I", "nominal", self.episodes())}
        assert (
            recs["latency.H-C-I.nominal.P2.0"].value == 1
            and recs["latency.H-C-I.nominal.P2.never"].value == 1
        )
        assert recs["latency.H-C-I.nominal.P2.0"].n == 2
        assert (
            recs["latency.H-C-I.nominal.P3.2-4"].value == 1
            and recs["latency.H-C-I.nominal.P3.0"].value == 0
        )
        assert not any(".P1." in name for name in recs)  # a property with no violation has no rows

    def test_cell_records_carry_them_and_the_report_prints_them(self) -> None:
        from closed_loop.eval.report import render_results

        eps = self.episodes()
        run = CellRun("H-C-I/nominal", (1, 2), eps, [], 500, "n_max", 0.2, 2, 0)
        recs = cell_records("H-C-I", "nominal", run)
        names = {r.name for r in recs}
        assert (
            "contract.H-C-I.nominal.any_miss_rul" in names
            and "latency.H-C-I.nominal.P2.never" in names
        )
        text = render_results(recs)
        from closed_loop.eval.report import FORBIDDEN

        assert not FORBIDDEN.search(text)  # the bin that was called "never" is "not noticed"
        assert "not noticed" in text
        assert "## The perception contract: per-life any-miss, both readings" in text
        assert "| H-C-I | nominal | life95 |" in text
        assert (
            "## Online monitors in the loop" in text
            and "| H-C-I | nominal | P2 | 2 | 1 | 0 | 0 | 0 | 1 |" in text
        )


class TestF0Parts:
    def fake_pytest(self, monkeypatch: pytest.MonkeyPatch, stdout: str, tmp_path: Path) -> None:
        for name in ("test_p5b_sabotage.py", "test_sabotage.py"):
            (tmp_path / "tests" / "shield").mkdir(parents=True, exist_ok=True)
            (tmp_path / "tests" / "shield" / name).write_text("", encoding="utf-8")

        def run(cmd: list[str], **kw: Any) -> Any:
            assert "addopts=" in cmd  # the project's own -q would suppress the summary line
            return types.SimpleNamespace(stdout=stdout, returncode=0)

        monkeypatch.setattr(stages.subprocess, "run", run)

    def test_the_summary_line_of_pytest_gives_passed_and_total(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self.fake_pytest(monkeypatch, "........ [100%]\n24 passed in 3.2s\n", tmp_path)
        # the time the suite took is not a result: it is cut from the line that goes in the records
        assert stages._run_sabotage(tmp_path) == (24, 24, "24 passed")

    def test_failures_and_errors_count_against_the_total(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self.fake_pytest(monkeypatch, "FAILED x\n5 failed, 19 passed, 2 errors in 4s\n", tmp_path)
        passed, total, tail = stages._run_sabotage(tmp_path)
        assert (passed, total) == (19, 26) and "5 failed" in tail

    def test_a_progress_line_is_not_taken_for_a_summary(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # the line ending in [100%] was once taken for the summary (and printed as "100%")
        self.fake_pytest(monkeypatch, "........ [100%]\n", tmp_path)
        assert stages._run_sabotage(tmp_path) == (0, 0, "pytest printed no summary line")

    def test_missing_test_files_are_said_not_run(self, tmp_path: Path) -> None:
        passed, total, why = stages._run_sabotage(tmp_path)
        assert (passed, total) == (0, 0) and "not in this checkout" in why

    def test_a_game_whose_reference_does_not_exist_yet_is_left_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from closed_loop.shield import reference

        monkeypatch.delattr(reference, "ConveyorReference", raising=False)
        keys = [k for k, _m, _r in stages._games(2)]
        assert keys[:2] == ["crusher", "pumps"] and "conveyor" not in keys


# ----------------------------------------------------------------------------- claim inputs


class TestClaimInputsFromCells:
    def test_the_cells_of_each_stage_are_stored_by_arm_and_condition(
        self, default_run: Any
    ) -> None:
        sc, _result, _out = default_run
        assert sc.store.has("P0", "nominal") and sc.store.has("P0", "shifted")
        assert sc.store.has("H-C0", "nominal") and not sc.store.has("H-C0", "shifted")
        assert len(sc.store.results("P0", "nominal")) == N

    def test_c_plan_pairs_the_planner_with_b3_on_the_seed(self, default_run: Any) -> None:
        _sc, result, _out = default_run
        recs = result.records
        d = record_named(recs, claim_name("C-PLAN", "cost_diff", "P0", "nominal"))
        # P0 costs 10 + seed % 5 over 200 and B3-V3 170 + seed % 5: the seed term cancels in pairs
        assert d.value == pytest.approx(-160.0 / 200.0) and d.n == N
        assert d.ci_hi is not None and d.ci_hi < -0.02
        p1 = record_named(recs, claim_name("C-PLAN", "cost_diff", "P1", "nominal"))
        assert p1.value == pytest.approx((14.0 - 180.0) / 200.0)  # P1 against B3-S, not B3-V3
        record_named(recs, claim_name("C-PLAN", "failure_diff", "P0", "shifted"))
        assert not [r for r in recs if "unplanned_diff" in r.name]  # renamed with the metric

    def test_the_safety_leg_of_c_plan_is_f_e_on_the_betting_sequence_and_the_cost_leg_is_not(
        self, default_run: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 0003 A16: f_e (an episode with an unplanned failure) over [0, 1] on paired_betting_cs, at
        # the same levels as before; the cost leg takes the default sequence
        sc, _result, _out = default_run
        calls: dict[str, tuple[float, float, float, bool]] = {}
        real = assemble.paired_record

        def spy(name: str, a: Any, b: Any, **kw: Any) -> Any:
            if name.startswith("claim/C-PLAN/"):
                calls[name] = (kw["lo"], kw["hi"], kw["alpha"], kw.get("betting_only", False))
            return real(name, a, b, **kw)

        monkeypatch.setattr(assemble, "paired_record", spy)
        claim_records(sc)
        fail = calls[claim_name("C-PLAN", "failure_diff", "P0", "nominal")]
        assert fail == (0.0, 1.0, ALPHA_PRIMARY, True)
        assert calls[claim_name("C-PLAN", "failure_diff", "P1", "shifted")][2:] == (
            ALPHA_DESCRIPTIVE,
            True,
        )
        cost = calls[claim_name("C-PLAN", "cost_diff", "P0", "nominal")]
        assert cost == (0.0, 1.0, ALPHA_PRIMARY, False)

    def test_f_e_is_one_when_the_episode_had_a_failure(self, default_run: Any) -> None:
        # B3-V3 is planted with 6 unplanned failures an episode and P0 with none: f_e is 1 and 0,
        # whatever the count, and the paired difference is exactly -1
        _sc, result, _out = default_run
        d = record_named(result.records, claim_name("C-PLAN", "failure_diff", "P0", "nominal"))
        assert d.value == -1.0 and d.extra["range_lo"] == 0.0 and d.extra["range_hi"] == 1.0

    def test_c_plan_verdicts_follow_from_the_planted_costs(self, default_run: Any) -> None:
        _sc, _result, out = default_run
        by = {v.claim: v for v in out.verdicts}
        assert by["C-PLAN.P0"].label == "win" and by["C-PLAN.P1"].label == "win"

    def test_a_dearer_planner_loses(self, cheap_cells: Any) -> None:
        cheap_cells["P0"]["cost"] = 190.0  # dearer than the rule, which costs 10
        cheap_cells["B3-V3"]["cost"] = 10.0
        sc, result = run_cells(("perception_closed", "policy"))
        by = {v.claim: v for v in finish(sc, result, prereg_sha=None).verdicts}
        assert by["C-PLAN.P0"].label == "loss" and by["C-PLAN.P1"].label == "win"

    def test_c_adapt_pairs_each_variant_with_v1_under_shifted(self, default_run: Any) -> None:
        _sc, result, _out = default_run
        d = record_named(result.records, claim_name("C-ADAPT", "diff_c_lo", "V2"))
        assert d.value == pytest.approx(0.95 - 0.80) and d.n == N and d.fidelity is Fidelity.F1
        cov = record_named(result.records, claim_name("C-ADAPT", "cov_lo_shifted", "V2"))
        # V3 is the primary variant here: only its rows are at the claim level, V2's are secondary
        assert cov.value == pytest.approx(0.95) and cov.ci_kind == f"confseq-{ALPHA_DESCRIPTIVE:g}"
        main = record_named(result.records, claim_name("C-ADAPT", "cov_lo_shifted", "V3"))
        assert main.ci_kind == f"confseq-{ALPHA_PRIMARY:g}"
        diff_main = record_named(result.records, claim_name("C-ADAPT", "diff_c_lo", "V3"))
        assert diff_main.ci_kind == f"confseq-{ALPHA_PRIMARY:g}" and d.ci_kind == cov.ci_kind
        record_named(result.records, claim_name("C-ADAPT", "cov_lo_nominal", "V3"))

    def test_gap_b_is_open_loop_minus_closed_loop_paired_on_the_seed(
        self, default_run: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sc0, _result, _out = default_run
        cov = {
            (cond, v): {s: EpisodeCoverage(0.9, 0.9, 50, 0.92, 0.92, 20, 5, 55) for s in SEEDS}
            for cond in ("nominal", "shifted")
            for v in stages.OPEN_LOOP_VARIANTS
        }
        sc = context_for(Runner(workers=1, headroom=lambda: 40.0))
        sc.store, sc.shared["open_loop"] = sc0.store, cov  # the stored cells, plus open loop
        records = claim_records(sc)
        pooled = record_named(records, claim_name("C-PERC", "gap_b", "pooled", "V1"))
        trend = record_named(records, claim_name("C-PERC", "gap_b", "trend", "V3"))
        assert pooled.value == pytest.approx(0.9 - 0.80) and pooled.n == N
        assert trend.value == pytest.approx(0.92 - 0.85)
        assert pooled.ci_kind.endswith(f"{0.05 / 18:g}")  # the stricter level of the two verdicts
        shifted = record_named(records, "gapb.shifted.pooled.V1")  # descriptive, level 0.05
        assert shifted.ci_kind == "confseq-0.05"
        assert not any(r.name.startswith("claim/C-PERC") and "shifted" in r.name for r in records)
        gap_a = record_named(records, "gapa.nominal.pooled.V1")
        assert gap_a.extra["c_lrn"] == pytest.approx(20 / 26)  # hit / (hit + miss + lower_hit)
        assert gap_a.value == pytest.approx(20 / 26 - 0.80)
        assert gap_a.extra["manski_lo"] == pytest.approx(20 / 30)
        assert gap_a.extra["manski_hi"] == pytest.approx(26 / 30)

    def test_c_mon_reads_the_passive_monitors_of_p1(self, default_run: Any) -> None:
        _sc, result, _out = default_run
        fa = record_named(result.records, claim_name("C-MON", "false_alarm", "proxy"))
        det = record_named(result.records, claim_name("C-MON", "detection", "proxy"))
        assert fa.extra["k"] == 0 and fa.value == 0.0 and det.value == 1.0
        assert "fired within 30 days" in det.measured_on


def shield_verdict(sc: StageContext, result: Any) -> Any:
    out = finish(sc, result, prereg_sha=None)
    return {v.claim: v for v in out.verdicts}["C-SHIELD"]


class TestShieldInputsFromCells:
    def test_a_clean_shield_that_runs_the_plant_is_a_win(self, default_run: Any) -> None:
        _sc, result, out = default_run
        v = {x.claim: x for x in out.verdicts}["C-SHIELD"]
        assert v.label == "win", v.headline
        ratio = record_named(result.records, claim_name("C-SHIELD", "throughput_ratio", "H-C"))
        assert ratio.value == pytest.approx(0.9 / 0.99) and ratio.extra["shielded_arm"] == "H-C1"
        # P1 stops the plant in the default profile (0.09 of P0's 0.9): the A13 label applies
        assert v.qualifier == "vacuous: the plant is mostly stopped"
        assert any("interlock against shield, P" in n for n in v.notes)

    def test_every_shielded_arm_of_the_scan_has_a_defect_and_a_plant_only_record(
        self, default_run: Any
    ) -> None:
        """0003 A18, F4: the plant-only share is a record of its own for each shielded arm that
        ran, P1-nopair included."""
        _sc, result, _out = default_run
        names = {r.name for r in result.records}
        for arm in ("H-C1", "H-G1", "P1", "P1-nopair"):
            assert claim_name("C-SHIELD", "defects", arm) in names, arm
            assert claim_name("C-SHIELD", "plant_only", arm) in names, arm

    def test_the_plant_only_record_leaves_out_the_events_the_spec_saw(
        self, cheap_cells: Any
    ) -> None:
        """0003 A18, F4: half of H-C1's episodes break P2 on the spec (the spec sees them); none is
        a plant-only event, so the plant-only record is 0 while M-VIOL (P1-P3) is not."""
        cheap_cells["H-C1"]["violates"] = 0.5
        _sc, result = run_cells(("hostile", "policy"))
        viol = record_named(result.records, claim_name("C-SHIELD", "viol_p13", "H-C1"))
        only = record_named(result.records, claim_name("C-SHIELD", "plant_only", "H-C1"))
        assert viol.value > 0.0 and only.value == 0.0

    def test_a_shield_that_runs_the_plant_in_every_pair_is_not_vacuous(
        self, cheap_cells: Any
    ) -> None:
        cheap_cells["P1"]["throughput"] = 0.8
        sc, result = run_cells(("hostile", "policy"))
        v = shield_verdict(sc, result)
        assert v.label == "win" and v.text == "win"

    def test_a_shield_that_stops_a_hostile_arm_makes_the_win_vacuous(
        self, cheap_cells: Any
    ) -> None:
        cheap_cells["P1"]["throughput"] = 0.8
        cheap_cells["H-G1"]["throughput"] = 0.05
        sc, result = run_cells(("hostile", "policy"))
        v = shield_verdict(sc, result)
        assert v.label == "win" and v.text == "win, vacuous: the plant is mostly stopped"

    def test_an_unshielded_arm_that_never_ran_gives_no_infinite_ratio(
        self, cheap_cells: Any
    ) -> None:
        from closed_loop.eval.report import records_to_json

        cheap_cells["H-C0"]["throughput"] = 0.0  # a ratio over zero is undefined, not infinite
        sc, result = run_cells(("hostile", "policy"))
        ratio = record_named(result.records, claim_name("C-SHIELD", "throughput_ratio", "H-C"))
        assert ratio.value == 1.0 and ratio.extra["unshielded_stopped"] == 1
        records_to_json(result.records)  # and the file can be written

    def test_the_planner_pair_counts_for_the_vacuity_label_too(self, default_run: Any) -> None:
        _sc, result, _out = default_run
        ratio = record_named(result.records, claim_name("C-SHIELD", "throughput_ratio", "P"))
        assert ratio.value == pytest.approx(0.09 / 0.9)  # P1 against P0: A13's own example

    def test_one_unexplained_violation_in_a_shielded_arm_is_a_loss(self, cheap_cells: Any) -> None:
        cheap_cells["H-C1"]["defect_every"] = 17  # an assumption_held violation now and then
        sc, result = run_cells(("hostile", "policy"))
        v = shield_verdict(sc, result)
        assert v.label == "loss" and "H-C1" in v.headline and v.qualifier == ""
        d = record_named(result.records, claim_name("C-SHIELD", "defects", "H-C1"))
        assert d.extra["k"] == len([s for s in SEEDS if s % 17 == 0])

    def test_a_defect_in_the_stale_specification_arm_is_not_a_defect_of_the_proof(
        self, cheap_cells: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(stages, "HOSTILE_ARMS", (*stages.HOSTILE_ARMS, "S-stale", "S-clause"))
        cheap_cells["S-stale"] = {"defect_every": 3}
        sc, result = run_cells(("hostile", "policy"))
        names = {r.name for r in result.records}
        assert claim_name("C-SHIELD", "defects", "S-stale") not in names
        assert claim_name("C-SHIELD", "defects", "S-clause") in names
        assert shield_verdict(sc, result).label == "win"

    def test_the_interlock_comparison_reaches_the_verdict(self, default_run: Any) -> None:
        _sc, result, out = default_run
        v = {x.claim: x for x in out.verdicts}["C-SHIELD"]
        row = record_named(result.records, claim_name("C-SHIELD", "interlock", "H-C"))
        assert row.value == 0.0 and row.extra["throughput_interlock"] == pytest.approx(0.7)
        assert "interlock against shield, H-C" in " ".join(v.notes)

    def test_a_failed_episode_in_a_shielded_cell_blocks_the_win_but_is_not_a_defect(
        self, cheap_cells: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = EpisodeTask.__call__

        def call(self: EpisodeTask, seed: int) -> EpisodeResult:
            if self.arm.name == "H-C1" and seed == SEEDS[5]:
                raise RuntimeError("the plant crashed")
            return original(self, seed)

        monkeypatch.setattr(EpisodeTask, "__call__", call)
        sc, result = run_cells(("hostile", "policy"))
        v = shield_verdict(sc, result)
        assert v.label == "inconclusive"  # one violation (the failure at its worst value) in H-C1
        defects = record_named(result.records, claim_name("C-SHIELD", "defects", "H-C1"))
        assert defects.extra["k"] == 0  # a crash is not an unexplained violation
        assert [f.seed for f in sc.store.run("H-C1", "nominal").failures] == [SEEDS[5]]


class TestAblations:
    def test_paired_ablation_records_exist_for_each_pair_and_metric(self, default_run: Any) -> None:
        _sc, result, _out = default_run
        abl = [r for r in result.records if r.name.startswith("ablation.")]
        labels = {r.extra["label"] for r in abl}
        assert "planner against rule" in labels and "shield on against off (rule)" in labels
        assert any(r.name.endswith(".nominal.m_cost") for r in abl)
        rec = next(
            r
            for r in abl
            if r.extra["label"] == "planner against rule" and r.name.endswith("nominal.m_cost")
        )
        assert rec.extra["arm"] == "P0" and rec.extra["reference"] == "B3-V3"
        assert rec.value == pytest.approx(-160.0 / 200.0) and rec.ci_kind == "confseq-0.05"


# ----------------------------------------------------------------------------- the whole run


class TestFinish:
    def test_the_run_records_say_what_the_verdicts_need_and_whether_it_is_complete(
        self, default_run: Any
    ) -> None:
        _sc, _result, out = default_run
        params = record_named(out.records, "run/params")
        assert params.extra["primary_variant"] == "V3" and params.extra["monitor_mode"] == "proxy"
        assert params.extra["n_max_hostile"] == N and params.extra["prereg_sha256"] == "cd" * 32
        comp = record_named(out.records, "run/completeness")
        assert comp.value == 0.0 and not out.complete
        assert set(str(comp.extra["skipped"]).split(",")) >= {"f0", "search", "f2"}
        stage = record_named(out.records, "run/stage/policy")
        assert stage.extra["status"] == "done" and stage.extra["rows"] == "4"

    def test_the_previous_runs_stay_in_the_report_next_to_this_one(self, default_run: Any) -> None:
        sc, result, _out = default_run
        out = finish(
            sc, result, prereg_sha=None,
            previous_headlines=({"verdict/C-SHIELD": "loss", "complete": False},),
        )  # fmt: skip
        assert record_named(out.records, "run/history").value == 1.0
        assert record_named(out.records, "run/headline/0/verdict/C-SHIELD").extra["text"] == "loss"
        text = record_named(out.records, "run/headline/1/verdict/C-SHIELD").extra["text"]
        assert text.startswith("win")

    def test_an_earlier_runs_text_is_shown_without_a_machine_path(self, default_run: Any) -> None:
        """The ledger keeps a first run's error as written; the report must not reprint it."""
        sc, result, _out = default_run
        frame = str(Path(__file__).resolve())
        out = finish(
            sc, result, prereg_sha=None,
            previous_headlines=({"numbers/C-TOGETHER": f'crashed | File "{frame}", line 1'},),
        )  # fmt: skip
        text = record_named(out.records, "run/headline/0/numbers/C-TOGETHER").extra["text"]
        assert str(Path.home()) not in text
        assert 'File "tests/eval/test_stages.py", line 1' in text

    def test_what_a_run_writes_checks_clean_and_a_forgery_does_not(
        self, default_run: Any, tmp_path: Path
    ) -> None:
        _sc, _result, out = default_run
        files = (tmp_path / "results.json", tmp_path / "RESULTS.md", tmp_path / "PROOF-SCOPE.md")
        digest = write_outputs(out.records, *files)
        assert check_outputs(*files, expected_sha256=digest) == []
        text = files[1].read_text(encoding="utf-8")
        assert "| C-SHIELD | win" in text
        files[1].write_text(text.replace("| C-SHIELD | win", "| C-SHIELD | loss"), encoding="utf-8")
        assert check_outputs(*files, expected_sha256=digest) != []

    def test_claim_records_alone_do_not_need_the_other_stages(self, cheap_cells: Any) -> None:
        sc, _ = run_cells(("policy",))
        names = {r.name for r in claim_records(sc)}
        assert claim_name("C-PLAN", "cost_diff", "P1", "nominal") in names
        assert not any(n.startswith("claim/C-SHIELD/viol_p13") for n in names)


class TestPublishedSearchSeeds:
    """0003 A14: the search seeds of a full run are derived from the frozen hash and counted."""

    def test_a_quick_run_searches_on_the_development_role(self) -> None:
        sc = search_context(runs=4)
        assert sc.published_search_seeds is None
        assert sc.search_seeds == (3000, 3001, 3002, 3003)

    def test_a_full_run_has_no_default_for_them(self) -> None:
        sc = dataclasses.replace(search_context(runs=4), mode="full")
        with pytest.raises(ValueError, match="held-out guard"):
            _ = sc.search_seeds

    def test_the_stage_runs_on_the_published_seeds_and_says_so(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[Any] = []
        install_fake_search(monkeypatch, nothing_in_r_in, seen)
        sha = "ab" * 32
        published = tuple(search_seed(sha, i) for i in range(3))
        sc = dataclasses.replace(
            search_context(runs=3), mode="full", published_search_seeds=published, prereg_sha=sha
        )
        out = run_pipeline(sc, ("search",))
        assert {c[3] for c in seen} == set(published) and len(seen) == 12
        record = record_named(out.records, claim_name("C-SEARCH", "find_rate", "R_out", "cma"))
        assert "search_seed(pre-registration hash, i) for i = 0 to 2" in record.measured_on
        assert "3000" not in record.measured_on

    def test_the_development_range_is_named_as_such_in_a_quick_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fake_search(monkeypatch, nothing_in_r_in)
        out = run_pipeline(search_context(runs=3), ("search",))
        record = record_named(out.records, claim_name("C-SEARCH", "find_rate", "R_out", "cma"))
        assert "development seeds (search-dev) 3000 to 3002" in record.measured_on

    def test_a_full_benchmark_draws_them_from_the_guard_and_the_ledger_counts_them(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        def pipeline(sc: StageContext, only: Any = None) -> str:
            captured["sc"] = sc
            return "pipeline"

        monkeypatch.setattr(stages, "run_pipeline", pipeline)
        monkeypatch.setattr(stages, "finish", lambda sc, result, **kw: "finished")
        ledger = Ledger(tmp_path / "ledger.jsonl")
        sha = "ab" * 32
        ledger.append("freeze", prereg_sha256=sha)
        done = stages.run_benchmark(
            mode="full", root=tmp_path, workers=1, guard=HeldoutGuard(ledger, sha),
            prereg_sha=sha, context=object(), headroom=lambda: 40.0, log=lambda _m: None,
            cache_dir=tmp_path / "cache",
        )  # type: ignore[arg-type]  # fmt: skip
        sc = captured["sc"]
        assert done == "finished"
        assert sc.published_search_seeds == tuple(search_seed(sha, i) for i in range(10))
        assert sc.search_seeds == sc.published_search_seeds
        assert sc.published_f2_seeds == tuple(f2_seed(sha, i) for i in range(FULL.f2_episodes))
        # 700 held-out F1 seeds, 10 search seeds and 60 F2 seeds (0003 A16)
        assert (
            ledger.heldout_evaluations()
            == (FULL.largest_cell + FULL.search_runs + FULL.f2_episodes)
            == 770
        )
        uses = [e for e in ledger.entries() if e["kind"] == "heldout-use"]
        assert [e.get("role") for e in uses] == [None, "search", "f2"]
        assert [e["n"] for e in uses] == [700, 10, 60]

    def test_a_quick_benchmark_draws_nothing_from_a_guard(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        def pipeline(sc: StageContext, only: Any = None) -> str:
            captured["sc"] = sc
            return "pipeline"

        monkeypatch.setattr(stages, "run_pipeline", pipeline)
        monkeypatch.setattr(stages, "finish", lambda sc, result, **kw: "finished")
        stages.run_benchmark(
            mode="quick", root=tmp_path, workers=1, guard=None, prereg_sha=None,
            context=object(), headroom=lambda: 40.0, log=lambda _m: None,
            cache_dir=tmp_path / "cache",
        )  # type: ignore[arg-type]  # fmt: skip
        assert captured["sc"].published_search_seeds is None
        assert captured["sc"].published_f2_seeds is None  # a quick run keeps development seeds
        assert captured["sc"].search_seeds == tuple(range(3000, 3000 + QUICK.search_runs))


class TestSeedPlan:
    def test_quick_uses_development_seeds_and_never_asks_for_a_guard(self) -> None:
        seeds, role = f1_seed_plan("quick", QUICK, None)
        assert role == "dev" and seeds == tuple(range(1100, 1100 + QUICK.largest_cell))
        assert all(1000 <= s < 2000 for s in seeds)

    def test_a_full_run_takes_its_seeds_from_the_guard_and_the_use_is_recorded(
        self, tmp_path: Path
    ) -> None:
        ledger = Ledger(tmp_path / "ledger.jsonl")
        sha = "ab" * 32
        ledger.append("freeze", prereg_sha256=sha)
        seeds, role = f1_seed_plan("full", FULL, HeldoutGuard(ledger, sha))
        assert role == "heldout" and len(seeds) == FULL.largest_cell == len(set(seeds))
        assert ledger.heldout_evaluations() == FULL.largest_cell == 700
        assert not any(1000 <= s < 4000 for s in seeds)

    def test_a_full_run_without_a_guard_is_refused(self) -> None:
        with pytest.raises(ValueError, match="guard"):
            f1_seed_plan("full", FULL, None)

    def test_a_full_run_cannot_pick_its_stages(self) -> None:
        with pytest.raises(ValueError, match="every stage"):
            stages.run_benchmark(
                mode="full", root=Path("."), workers=1, guard=None, prereg_sha=None, only=("f0",)
            )

    def test_an_unknown_mode_is_refused(self) -> None:
        with pytest.raises(ValueError, match="quick or full"):
            stages.run_benchmark(
                mode="medium", root=Path("."), workers=1, guard=None, prereg_sha=None
            )


def test_the_stage_context_carries_the_hash_f2_seeds_are_derived_from() -> None:
    sc = context_for(Runner(workers=1, headroom=lambda: 40.0))
    assert sc.prereg_sha is None  # a quick run has no frozen document, so no F2 seeds
    assert dataclasses.replace(sc, prereg_sha="ab" * 32).prereg_sha == "ab" * 32


def test_a_cheap_runner_is_what_the_tests_use() -> None:
    fn: Callable[..., Any] = invented
    assert fn("B0", 1, "nominal", {}).arm == "B0"


@pytest.mark.slow
class TestSearchOnTheRealModule:
    """Two short searches of the real P9b module through the pool (4 evaluations each)."""

    def cell(self, ctx: Any, workers: int) -> Any:
        task = SearchTask("out", "random", 4, PROVISIONAL)
        with Runner(workers=workers, context=ctx, headroom=lambda: 40.0, timeout_s=600.0) as r:
            return r.run_seeds(task, [3000, 3001], batch=2, n_max=2, use_cache=False)

    def test_a_search_run_goes_through_the_pool_and_is_the_same_for_one_and_two_workers(
        self, small_context: Any
    ) -> None:
        one = self.cell(small_context, 1)
        two = self.cell(small_context, 2)
        assert not one.failures and not two.failures
        assert one.values == two.values  # frozen dataclasses of ints: exact equality
        for outcome in one.values:
            assert isinstance(outcome, SearchOutcome) and outcome.evals == 4
            assert outcome.violating >= sum(n for _k, n in outcome.kinds)
