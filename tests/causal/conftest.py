"""Shared, session-scoped fixtures for the causal tests. SYNTHETIC lives only.

The expensive objects (observed lives, fitted models) are built once per session. The "quiet"
configuration switches off every noise source and data-quality event and the direct effect of load
on the health index, so the structural equations are all that is left and exactness tests can be
exact; the "noisy" one is the study's plant.
"""

from __future__ import annotations

import pytest

from closed_loop.causal import mdiag
from closed_loop.causal.diagnoser import make_diagnoser
from closed_loop.causal.scm import SCMEffectModel
from closed_loop.causal.twin import causal_params, generate_observational
from closed_loop.plant import PlantParams

QUIET_SEED = 3000
NOISY_SEED = 1000


@pytest.fixture(scope="session")
def quiet_params() -> PlantParams:
    return causal_params(
        hi_noise_sd=0.0,
        sig_noise_sd=0.0,
        load_noise_sd=0.0,
        load_level_sd=0.0,
        load_obs_sd=0.0,
        upset_p_per_circuit=0.0,
        p_gain_fault=0.0,
        p_stuck_day=0.0,
        p_spike_day=0.0,
        hi_load_coef=0.0,
    )


@pytest.fixture(scope="session")
def quiet_lives(quiet_params):
    return generate_observational(24, QUIET_SEED, "pi_obs", params=quiet_params)


@pytest.fixture(scope="session")
def quiet_scm(quiet_lives):
    model = SCMEffectModel(n_beta_lives=10)
    model.fit(quiet_lives)
    return model


@pytest.fixture(scope="session")
def noisy_params() -> PlantParams:
    return causal_params()


@pytest.fixture(scope="session")
def noisy_lives(noisy_params):
    return generate_observational(48, NOISY_SEED, "pi_obs", params=noisy_params)


@pytest.fixture(scope="session")
def diag_train():
    return mdiag.collect_lives(range(1000, 1090), mdiag.diagnosis_params(mdiag.DEFAULT_TILT))


@pytest.fixture(scope="session")
def fitted_diagnosers(diag_train):
    out = {}
    for kind in ("scm", "corr"):
        d = make_diagnoser(kind)
        d.fit(diag_train)
        out[kind] = d
    return out


@pytest.fixture(scope="session")
def noisy_scm(noisy_lives):
    model = SCMEffectModel(n_beta_lives=6)
    model.fit(noisy_lives)
    return model


@pytest.fixture(scope="session")
def noisy_misspecified(noisy_lives):
    model = SCMEffectModel(misspecified=True, n_beta_lives=6)
    model.fit(noisy_lives)
    return model


@pytest.fixture(scope="session")
def noisy_corr(noisy_lives):
    from closed_loop.causal.correlational import CorrelationalEffectModel

    model = CorrelationalEffectModel()
    model.fit(noisy_lives)
    return model


@pytest.fixture(scope="session")
def study_cases(noisy_params):
    from closed_loop.causal.queries import build_cases

    cases, skipped = build_cases(range(1200, 1206), noisy_params)
    return cases, skipped
