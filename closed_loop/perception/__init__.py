"""Estimator, labels, conformal intervals and the coverage contract (packets P4a to P4c).

Everything here is SYNTHETIC. Importing this package pulls in numpy and the plant's parameters and
nothing else: no sibling repository, no pandas, no scikit-learn (docs/adr/0002, rule 4).
"""

from closed_loop.perception.conformal import (
    DevelopmentData,
    IntervalProvider,
    development_data,
    is_audit_life,
    make_provider,
    replay_open_loop,
)
from closed_loop.perception.contract import (
    SHIELD_SETS,
    contract_report,
    health_set,
    make_shield_provider,
)
from closed_loop.perception.coverage import (
    CoverageRun,
    gaps,
    learner_coverage,
    manski_bounds,
    open_loop_lives,
    oracle_coverage,
    per_life_any_miss,
)
from closed_loop.perception.estimator import (
    EstimatorOutput,
    StreamingEstimator,
    TheilSenEstimator,
)
from closed_loop.perception.labels import (
    Checkpoint,
    Label,
    LabelStatus,
    RulInterval,
    learner_view,
    resolve_labels,
)
from closed_loop.perception.monitor import CoverageMonitor, ProxyBand

__all__ = [
    "SHIELD_SETS",
    "Checkpoint",
    "CoverageMonitor",
    "CoverageRun",
    "DevelopmentData",
    "EstimatorOutput",
    "IntervalProvider",
    "Label",
    "LabelStatus",
    "ProxyBand",
    "RulInterval",
    "StreamingEstimator",
    "TheilSenEstimator",
    "contract_report",
    "development_data",
    "gaps",
    "health_set",
    "is_audit_life",
    "learner_coverage",
    "learner_view",
    "make_provider",
    "make_shield_provider",
    "manski_bounds",
    "open_loop_lives",
    "oracle_coverage",
    "per_life_any_miss",
    "replay_open_loop",
    "resolve_labels",
]
