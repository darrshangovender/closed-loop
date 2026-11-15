"""P8b: the planner reads observations, the belief and the constraints it is handed. No truth.

0001 section 8 and the P8b packet: the planner must not read ``TruthView`` or any truth field. Five
independent checks, as for the belief in ``test_belief_truth_free.py``: the imports of the three
modules, the names they use, the shape of the public calls, a replay of a real plant episode in
which every route to the plant's truth is poisoned (any read raises), and a check that the poisoned
episode is the same episode as the unpoisoned one. A last test shows the planner is doing something
in that episode, so that the poisoning does not pass because the planner is idle.
"""

import ast
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

from closed_loop.planning import exactdp, planner, proposer
from closed_loop.planning.planner import BeliefPlanner
from closed_loop.planning.proposer import PlannerProposer

ROOT = Path(__file__).resolve().parents[2]
MODULES = {
    "planner.py": Path(planner.__file__),
    "exactdp.py": Path(exactdp.__file__),
    "proposer.py": Path(proposer.__file__),
}

# Everything that holds, derives from, or names a truth field of the plant or its trace, and the
# plant accessors that are not part of an observation.
FORBIDDEN_NAMES = frozenset(
    {
        "TruthView",
        "true_rul_days",
        "health_bin",
        "failure_time_q",
        "life_mode",
        "rate_rul_days",
        "alert_truth",
        "AlertTruth",
        "truth",
        "Trace",
        "trace",
        "set_lo",
        "set_hi",
        "uncontrolled_failure",
        "life_start_day",
        "stats",
        "twin",
    }
)
STDLIB_NUMPY = {"__future__", "collections", "dataclasses", "functools", "math", "statistics"}
STDLIB_NUMPY |= {"types", "typing", "numpy"}
CORE = {"closed_loop.core.rng", "closed_loop.core.timegrid", "closed_loop.core.types"}
OWN = {"closed_loop.planning.belief", "closed_loop.planning.worldmodel"}
PLANT = {"closed_loop.plant.jobs", "closed_loop.plant.params", "closed_loop.plant.plant"}
ALLOWED_PACKAGE_IMPORTS = {
    "planner.py": CORE | OWN,
    "exactdp.py": {"closed_loop.core.rng", "closed_loop.planning.worldmodel"},
    "proposer.py": CORE
    | OWN
    | PLANT
    | {"closed_loop.planning.planner", "closed_loop.planning.fitting"}
    | {"closed_loop.causal.diagnoser", "closed_loop.causal.records"}
    # the stall watch reads views only (0003 A18, ruling R5; its own imports are checked in
    # tests/proposers/test_no_truth.py)
    | {"closed_loop.proposers.stall"},
}
# imports that exist for the type checker only (`if TYPE_CHECKING:`)
TYPE_ONLY = {"planner.py": {"closed_loop.plant.plant"}}


def tree_of(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def imports_of(tree: ast.Module) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            out.add(("." * node.level) + (node.module or ""))
    return out


class TestImports:
    @pytest.mark.parametrize("name", MODULES)
    def test_only_the_expected_modules_are_imported(self, name: str) -> None:
        allowed = ALLOWED_PACKAGE_IMPORTS[name] | TYPE_ONLY.get(name, set())
        for imported in imports_of(tree_of(MODULES[name])):
            assert not imported.startswith("."), f"relative import {imported}"
            if imported.startswith("closed_loop"):
                assert imported in allowed, f"{name} imports {imported}"
            else:
                assert imported.split(".")[0] in STDLIB_NUMPY | {"__future__"}, (
                    f"{name}: {imported}"
                )

    def test_the_planner_imports_the_plant_for_the_type_checker_only(self) -> None:
        tree = tree_of(MODULES["planner.py"])
        plant_imports = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("closed_loop.plant")
        ]
        assert len(plant_imports) == 1
        guarded = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.If) and "TYPE_CHECKING" in ast.unparse(n.test)
        ]
        assert guarded and plant_imports[0] in [c for g in guarded for c in g.body]

    def test_importing_the_planner_does_not_import_the_plant_or_the_trace(self) -> None:
        code = (
            "import sys; sys.path.insert(0, '.');"
            "import closed_loop.planning.planner, closed_loop.planning.exactdp;"
            "bad = [m for m in sys.modules if m.startswith('closed_loop.plant')"
            " or m == 'closed_loop.core.trace'];"
            "print(bad)"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT
        )
        assert out.stdout.strip() == "[]"

    def test_importing_the_package_does_not_load_the_proposer_or_the_fitting(self) -> None:
        code = (
            "import sys; sys.path.insert(0, '.'); import closed_loop.planning;"
            "print('closed_loop.planning.proposer' in sys.modules,"
            " 'closed_loop.planning.fitting' in sys.modules, 'scipy' in sys.modules)"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT
        )
        assert out.stdout.strip() == "False False False"

    def test_the_lazy_names_resolve(self) -> None:
        import closed_loop.planning as pkg

        assert pkg.PlannerProposer is PlannerProposer
        assert callable(pkg.default_models)
        with pytest.raises(AttributeError):
            pkg.no_such_name  # noqa: B018


class TestNames:
    @pytest.mark.parametrize("name", MODULES)
    def test_no_truth_name_is_used_as_a_name_attribute_argument_or_string(self, name: str) -> None:
        tree = tree_of(MODULES[name])
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef):
                first = node.body[0] if node.body else None
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                    docstrings.add(id(first.value))
        used: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.Attribute):
                used.add(node.attr)
            elif isinstance(node, ast.arg):
                used.add(node.arg)
            elif isinstance(node, ast.keyword) and node.arg:
                used.add(node.arg)
            elif isinstance(node, ast.alias):
                used.add(node.name)  # `from x import TruthView`
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
            ):
                used.add(node.value)  # a getattr(x, "truth") would show up whole
        assert not (used & FORBIDDEN_NAMES), used & FORBIDDEN_NAMES


class TestSignatures:
    def test_plan_takes_the_observation_the_tracker_and_the_constraints(self) -> None:
        params = set(inspect.signature(BeliefPlanner.plan).parameters)
        assert params == {"self", "obs", "tracker", "constraints", "mode_of"}

    def test_the_proposer_takes_an_observation_and_nothing_else(self) -> None:
        assert set(inspect.signature(PlannerProposer.propose).parameters) == {"self", "obs"}

    def test_assess_takes_a_belief_a_load_a_day_and_a_budget(self) -> None:
        params = set(inspect.signature(BeliefPlanner.assess).parameters)
        assert params == {"self", "asset_id", "particles", "load_mean", "day", "delta_plan"}


# ----------------------------------------------------------------------------- the poisoned replay
@pytest.fixture(scope="module")
def models():
    from closed_loop.planning.proposer import default_models

    return default_models(fit_seeds=range(1000, 1030))


# every route to the plant's truth or internals other than reset, step and the Observation
POISONED_METHODS = ("trace", "alert_truth", "stats", "life_start_day", "twin")


def episode(models, *, poison: bool, seed: int = 1503, days: int = 55):
    """Run the planner on a real episode; with ``poison`` every route to the truth raises."""
    from closed_loop.core.timegrid import DAY_Q
    from closed_loop.plant import PlantParams, Scenario, make_plant
    from closed_loop.plant.plant import F1Plant

    plant = make_plant(PlantParams())
    obs = plant.reset(seed, Scenario())
    pr = PlannerProposer(models, diagnoser=None, seed=seed)

    def boom(*_a, **_k):
        raise AssertionError("the planner read the plant's truth")

    # What is poisoned and what is restored come from one tuple, and the originals are taken from
    # the class dictionary (the raw property or function), so the two cannot drift apart.
    saved = {name: F1Plant.__dict__[name] for name in ("truth", *POISONED_METHODS)}
    issued = []
    try:
        if poison:
            F1Plant.truth = property(boom)  # type: ignore[method-assign,assignment]
            for name in POISONED_METHODS:
                setattr(F1Plant, name, boom)
        for _ in range(days * DAY_Q):
            cmds = pr.propose(obs)
            issued.extend((obs.quarter, c) for c in cmds)
            res = plant.step(cmds)
            obs = res.obs
            if res.done:
                break
    finally:
        for name, value in saved.items():
            setattr(F1Plant, name, value)
    return issued, pr


class TestPoisonedTruth:
    def test_an_episode_runs_with_every_route_to_the_truth_poisoned(self, models) -> None:
        issued, pr = episode(models, poison=True)
        assert issued and pr.decisions

    def test_the_commands_are_the_same_with_and_without_the_poison(self, models) -> None:
        with_poison, _ = episode(models, poison=True)
        without, _ = episode(models, poison=False)
        assert with_poison == without

    def test_the_poison_is_taken_off_again_whatever_happens(self, models, monkeypatch) -> None:
        """Regression: ``twin`` was poisoned but not restored, which broke tests/plant."""
        from closed_loop.plant import PlantParams, Scenario, make_plant
        from closed_loop.plant.plant import F1Plant

        names = ("truth", *POISONED_METHODS)
        before = {name: F1Plant.__dict__[name] for name in names}
        episode(models, poison=True, days=5)
        assert {name: F1Plant.__dict__[name] for name in names} == before
        assert F1Plant.twin is before["twin"]
        plant = make_plant(PlantParams())
        plant.reset(1503, Scenario())
        assert isinstance(plant.twin([]), F1Plant)  # the real method works again
        assert plant.truth.life_id  # and so does the real truth

        def explode(self, obs):
            raise RuntimeError("the episode failed half way")

        monkeypatch.setattr(PlannerProposer, "propose", explode)
        with pytest.raises(RuntimeError, match="half way"):
            episode(models, poison=True, days=5)
        assert {name: F1Plant.__dict__[name] for name in names} == before

    def test_the_planner_is_not_idle_in_that_episode(self, models) -> None:
        issued, pr = episode(models, poison=False)
        kinds = {c.cmd.value for _, c in issued}
        assert "RAISE_WO" in kinds and len(kinds) > 3  # plans, then the runner's micro-commands
        assert any(a.full for _, d in pr.decisions for a in d.assessments.values())
