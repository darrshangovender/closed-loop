"""Fixtures of the search tests: one small prepared context per session.

The same small spec as ``tests/eval/conftest.py`` (a dozen seconds to fit, deterministic). The tests
that need a real episode use it; the tests of the loop itself use a stand-in evaluation and need no
plant at all.
"""

from __future__ import annotations

import pytest

from closed_loop.eval.context import ContextSpec, PreparedContext, prepare_context

SMALL_SPEC = ContextSpec(
    conformal_seeds=(2000, 2040),
    proxy_seeds=(2000, 2040),
    diagnoser_seeds=(2300, 2314),
    world_model_seeds=(1000, 1020),
)


@pytest.fixture(scope="session")
def small_context() -> PreparedContext:
    return prepare_context(SMALL_SPEC)
