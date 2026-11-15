"""Fixtures shared by the tests of ``benchmarks/run.py`` (a miniature repository and the module)."""

import pytest

from tests.audit.test_run import (  # noqa: F401
    TheRealBenchmarkMustNotRunInATest,
    repo,
    run_module,
)


@pytest.fixture(autouse=True)
def _the_real_benchmark_does_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing in these tests may start the benchmark: those that exercise ``run_experiments``
    put a stand-in for ``run_benchmark`` in place, and a test that does not meets this refusal."""
    import closed_loop.eval.stages as stages

    def refuse(**kwargs: object) -> object:
        raise TheRealBenchmarkMustNotRunInATest("a test reached run_benchmark")

    monkeypatch.setattr(stages, "run_benchmark", refuse)
