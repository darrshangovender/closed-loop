"""P8a: the belief reads observations only.

0001 section 8 and the P8a packet: the belief tracker must not read ``TruthView`` or any truth
field. Four independent checks: the imports of the two modules, the names they use, the shape of the
public update call, and a replay of a real plant episode in which the plant's truth is poisoned (any
read of it raises). A fifth check shows the tracker is useful on that same stream, so the poisoning
test is not passing because the tracker does nothing.
"""

import ast
import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from closed_loop.planning import belief, worldmodel
from closed_loop.planning.belief import BeliefTracker
from closed_loop.planning.worldmodel import PlantWorldModel

ROOT = Path(__file__).resolve().parents[2]
MODULES = (Path(belief.__file__), Path(worldmodel.__file__))

# Everything that holds, derives from, or names a truth field of the plant or its trace.
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
        "set_lo",
        "set_hi",
        "uncontrolled_failure",
    }
)
ALLOWED_IMPORT_ROOTS = frozenset(
    {"__future__", "collections", "dataclasses", "math", "types", "typing", "numpy"}
)
ALLOWED_PACKAGE_IMPORTS = frozenset({"closed_loop.core.rng", "closed_loop.planning.worldmodel"})


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
    @pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
    def test_only_numpy_the_standard_library_and_two_owned_modules_are_imported(self, path):
        for name in imports_of(tree_of(path)):
            assert not name.startswith("."), f"relative import {name}"
            if name.startswith("closed_loop"):
                assert name in ALLOWED_PACKAGE_IMPORTS, f"{path.name} imports {name}"
            else:
                assert name.split(".")[0] in ALLOWED_IMPORT_ROOTS, f"{path.name} imports {name}"

    @pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
    def test_the_plant_and_the_trace_are_not_imported(self, path):
        names = imports_of(tree_of(path))
        assert not any(n.startswith("closed_loop.plant") for n in names)
        assert "closed_loop.core.trace" not in names

    def test_importing_the_belief_does_not_import_the_plant(self):
        code = (
            "import sys; sys.path.insert(0, '.');"
            "import closed_loop.planning.belief, closed_loop.planning.worldmodel;"
            "bad = [m for m in sys.modules if m.startswith('closed_loop.plant')"
            " or m == 'closed_loop.core.trace'];"
            "print(bad)"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT
        )
        assert out.stdout.strip() == "[]"

    def test_the_package_init_leaves_the_fitting_module_out(self):
        code = (
            "import sys; sys.path.insert(0, '.'); import closed_loop.planning;"
            "print('closed_loop.planning.fitting' in sys.modules)"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT
        )
        assert out.stdout.strip() == "False"


class TestNames:
    @pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
    def test_no_truth_name_is_used_as_a_name_attribute_argument_or_string(self, path):
        tree = tree_of(path)
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
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
            ):
                used.add(node.value)  # a getattr(x, "truth") would show up whole
        assert not (used & FORBIDDEN_NAMES), used & FORBIDDEN_NAMES


class TestSignatures:
    def test_the_update_call_takes_a_day_a_reading_a_flag_and_a_load_only(self):
        params = set(inspect.signature(BeliefTracker.update).parameters)
        assert params == {"self", "asset_id", "day", "hi", "inspected", "load"}

    def test_the_model_methods_take_particles_numbers_and_generators_only(self):
        allowed = {
            "self",
            "p",
            "n",
            "rng",
            "action",
            "obs",
            "noise_scale",
            "load",
            "load_mean",
            "idx",
            "shrink_delta",
            "mismatch",
        }
        for name in ("prior", "renewal", "step", "loglik", "failed", "resample", "hi_mean"):
            params = set(inspect.signature(getattr(PlantWorldModel, name)).parameters)
            assert params <= allowed | {"load"}, (name, params - allowed)


@pytest.fixture(scope="module")
def fitted():
    from closed_loop.planning.fitting import fit_world_model

    return fit_world_model(range(1000, 1030))


@pytest.fixture(scope="module")
def pump_episode():
    """A real plant episode with a failing pump: readings per day, and the failure day."""
    from closed_loop.plant import PlantParams, Scenario, make_plant

    params = PlantParams()
    for seed in range(1200, 1300):
        plant = make_plant(params)
        plant.reset(seed, Scenario())
        if plant.truth.failure_time_q["PU-201"] < 8000:
            break
    else:
        raise AssertionError("no seed with an early pump failure")
    fail_day = plant.truth.failure_time_q["PU-201"] / 96.0
    return seed, fail_day


def replay(seed: int, fitted: dict[str, float], *, poison: bool):
    """Feed one real episode to a tracker through the Observation only; return the particles."""
    from closed_loop.plant import PlantParams, Scenario, make_plant
    from closed_loop.plant.plant import F1Plant

    plant = make_plant(PlantParams())
    obs = plant.reset(seed, Scenario())
    model = PlantWorldModel(fitted, asset_class="SLURRY_PUMP")
    tracker = BeliefTracker(model, seed=3)
    original = F1Plant.truth
    if poison:

        def boom(self):  # any read of the plant's truth is a failure of this test
            raise AssertionError("the belief tracker read the plant's truth")

        F1Plant.truth = property(boom)  # type: ignore[method-assign,assignment]
    try:
        seen = -1
        path: list[tuple[int, np.ndarray]] = []
        while True:
            if obs.day != seen:
                seen = obs.day
                view = obs.assets["PU-201"]
                reading = view.last_obs
                assert reading is not None
                tracker.update("PU-201", obs.day, reading.hi, False, load=reading.load)
                path.append((obs.day, tracker.particles("PU-201").state["x"].copy()))
            result = plant.step(())
            obs = result.obs
            if result.done:
                break
    finally:
        F1Plant.truth = original  # type: ignore[method-assign]
    return tracker, path


class TestPoisonedTruth:
    def test_an_episode_replays_with_the_plants_truth_poisoned(self, fitted, pump_episode):
        seed, _ = pump_episode
        tracker, path = replay(seed, fitted, poison=True)
        assert len(path) == 90 and tracker.day("PU-201") == 89

    def test_the_result_is_the_same_with_and_without_the_poison(self, fitted, pump_episode):
        seed, _ = pump_episode
        a, path_a = replay(seed, fitted, poison=True)
        b, path_b = replay(seed, fitted, poison=False)
        for (da, xa), (db, xb) in zip(path_a, path_b, strict=True):
            assert da == db and np.array_equal(xa, xb)
        assert np.array_equal(a.particles("PU-201").weights, b.particles("PU-201").weights)

    def test_the_tracker_is_not_idle_it_follows_the_degradation(self, fitted, pump_episode):
        seed, fail_day = pump_episode
        tracker, path = replay(seed, fitted, poison=False)
        mean_x = [(d, float(np.mean(x))) for d, x in path]
        early = np.mean([x for d, x in mean_x if 24 <= d < 30])
        late = np.mean([x for d, x in mean_x if d >= fail_day - 3])
        assert late > early + 0.3  # the belief moved a long way toward failure
        assert tracker.p_failed("PU-201") > 0.5 or fail_day > 89
