"""Structural causal model, correlational baseline, twin queries and diagnosis (packet P7).

Importing this package pulls in numpy, scipy and the plant (for the twin worlds and the asset
catalogue); no sibling repository, no pandas, no scikit-learn.
"""

from closed_loop.causal.correlational import (
    CorrelationalEffectModel,
    SoftmaxRegression,
)
from closed_loop.causal.diagnoser import (
    AlertTruthTable,
    CorrDiagnoser,
    Diagnoser,
    Diagnosis,
    OracleDiagnoser,
    ScmDiagnoser,
    make_diagnoser,
)
from closed_loop.causal.queries import (
    EffectModel,
    QueryConfig,
    build_cases,
    make_effect_model,
    paired_difference,
    per_life_scores,
    score,
)
from closed_loop.causal.records import (
    AlertLabel,
    HistoryRecorder,
    LifeRecord,
    ObservationHistory,
)
from closed_loop.causal.scm import SCMEffectModel
from closed_loop.causal.twin import (
    OperatorConfig,
    Query,
    TruthAnswer,
    causal_params,
    generate_observational,
    passive_world,
    run_world,
    twin_truth,
    twin_truths,
)

__all__ = [
    "AlertLabel",
    "AlertTruthTable",
    "CorrDiagnoser",
    "CorrelationalEffectModel",
    "Diagnoser",
    "Diagnosis",
    "EffectModel",
    "HistoryRecorder",
    "LifeRecord",
    "ObservationHistory",
    "OperatorConfig",
    "OracleDiagnoser",
    "Query",
    "QueryConfig",
    "SCMEffectModel",
    "ScmDiagnoser",
    "SoftmaxRegression",
    "TruthAnswer",
    "build_cases",
    "causal_params",
    "generate_observational",
    "make_diagnoser",
    "make_effect_model",
    "paired_difference",
    "passive_world",
    "per_life_scores",
    "run_world",
    "score",
    "twin_truth",
    "twin_truths",
]
