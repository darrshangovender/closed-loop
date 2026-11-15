"""Fixtures of the eval tests: one small prepared context per session.

The full context (``eval.context.ContextSpec()``) takes about half a minute to fit; the tests use a
smaller one built from fewer calibration seeds, which has the same shape and is deterministic. A
test that calls a proposer factory directly must do so inside ``use_context(small_context)``, or the
factory would build the full default context.
"""

from __future__ import annotations

import pytest

from closed_loop.eval.context import ContextSpec, PreparedContext, default_context, prepare_context

SMALL_SPEC = ContextSpec(
    conformal_seeds=(2000, 2040),
    proxy_seeds=(2000, 2040),
    diagnoser_seeds=(2300, 2314),
    world_model_seeds=(1000, 1020),
)


@pytest.fixture(scope="session")
def small_context() -> PreparedContext:
    return prepare_context(SMALL_SPEC)


@pytest.fixture(scope="session")
def full_context() -> PreparedContext:
    """The default context, as a real run uses it (about half a minute to fit); slow tests only."""
    return default_context()
