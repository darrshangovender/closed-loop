"""asset-health as an alert source for maintenance-agent (packet P11b). Everything is SYNTHETIC.

``AssetHealthAlertSource`` implements ``maintenance_agent.ports.AlertSource``. It turns the work
order the real asset-health pipeline writes for an open alert (``analyse_run``) into the ``Alert``
and ``EvidenceItem`` list the real orchestrator reads. It is a thin adapter: every number comes from
asset-health, and every place where the two vocabularies do not match is handled in the open and
commented with its number in docs/adr/0001-architecture.md, section 10.3.

The alert is a snapshot. The orchestrator re-reads the alert from its source before it releases
anything and treats a change as a stale reading (``alert_fingerprint``), so ``get_alert`` returns
the reading taken when the alert was raised, not a live one. Its ``rul.as_of`` is the simulation
date (mismatch 9): verifier rule V10 blocks a reading more than ``max_alert_age_days`` (3) old.

``RunCache`` keeps what the pipeline has computed for one episode. A run of ``run_asset`` costs
about 2 s and is flat in the length of the history, so it is made once per history (a history
changes only at a join or a stop) and every later day is answered by slicing with ``as_of``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from closed_loop.f2._siblings import load
from closed_loop.f2.render import RenderedTelemetry
from closed_loop.f2.splice import maintenance_events, splice
from closed_loop.plant.params import ASSET_IDS, MODES_BY_CLASS

# The rule layer opens its primary gates at 3.5 robust sigma (asset_health/detect/rules.py, the
# first Cond of the race, impeller and liner rules). Mismatch 7: the agent's evidence needs an
# alarm level and asset-health has none, so the alarm is defined as baseline plus this threshold
# in robust-sigma units. Our definition, not an upstream one.
Z_ALARM = 3.5
SIGMA_UNIT = "robust sigma"

# asset-health channel -> the maintenance-agent signal whose playbook entry it can support. The
# agent's vocabulary is wider (oil, seal, thickness) and narrower (no vane-pass band) than the
# generator's; a channel with no counterpart is dropped and counted. Mismatches 3 and 7.
FEATURE_SIGNAL: dict[str, str] = {
    "vib_band_bpfo": "vib_hf_envelope_g",
    "bpfo_share": "vib_hf_envelope_g",
    "vib_band_bpfi": "vib_hf_envelope_g",
    "bpfi_share": "vib_hf_envelope_g",
    "vib_band_hf": "vib_hf_envelope_g",
    "hf_share": "vib_hf_envelope_g",
    "vib_rms": "vib_overall_mm_s",
    "vib_band_proc": "vib_overall_mm_s",
    "proc_share": "vib_overall_mm_s",
    "vib_band_low": "vib_1x_mm_s",
    "low_share": "vib_1x_mm_s",
    "temp_c": "bearing_temp_c",
    "pressure_kpa": "disch_pressure_drop_pct",
    "flow_m3h": "flow_dev_pct",
    "power_kw": "mill_power_dev_pct",
    "current_a": "mill_power_dev_pct",
}
# asset-health fault hypothesis -> the maintenance-agent failure mode it names (None: the agent has
# no mode for it, mismatch 3: an overheating or an unexplained anomaly can only become an
# inspection-first draft).
HYPOTHESIS_MODE: dict[str, str | None] = {
    "bearing_outer_race": "BEARING_DEFECT",
    "bearing_inner_race": "BEARING_DEFECT",
    "bearing_generic": "BEARING_DEFECT",
    "impeller_wear": "IMPELLER_WEAR",
    "liner_wear": "LINER_WEAR",
    "overheating": None,
    "unexplained_anomaly": None,
}
# asset-health severity -> maintenance-agent severity (0001 section 10.2; mismatch 6).
SEVERITY: dict[str, str] = {
    "normal": "WARN",
    "watch": "WARN",
    "warning": "WARN",
    "urgent": "ALARM",
    "critical": "ALARM",
}
# agent component key of the part each plant mode consumes (playbook.json, entries).
COMPONENT_OF_MODE: dict[str, str] = {
    "BEARING_DEFECT": "bearing_de",
    "IMPELLER_WEAR": "impeller",
    "LINER_WEAR": "liner_set",
}


class DuplicateAlert(ValueError):
    """An alert id was raised twice (mismatch 11: the CMMS refuses a second order for one id)."""


def aware(instant: datetime) -> datetime:
    """UTC-aware ``instant``. The agent compares aware datetimes (approval expiry), so the whole F2
    clock is aware; a naive instant is taken to be UTC."""
    return instant.replace(tzinfo=UTC) if instant.tzinfo is None else instant.astimezone(UTC)


def naive(instant: datetime) -> datetime:
    """The naive UTC instant asset-health uses throughout (its ``timeutil`` convention)."""
    return aware(instant).replace(tzinfo=None)


class RunCache:
    """The pipeline's answers for one episode, by plant asset.

    ``set_history`` runs the real ``run_asset`` when the history differs from the cached one and
    does nothing otherwise; ``analysis`` runs the real ``analyse_run`` for an instant and remembers
    it. The counters and timers are what P11c logs against the caching rule, and the per-step split
    of the F2 episode time.
    """

    def __init__(
        self,
        pipeline_cfg: Any | None = None,
        analysis_cfg: Any | None = None,
        rul_model: Any | None = None,
    ) -> None:
        self._analysis_mod = load("asset_health.analysis")
        self._pipeline_mod = load("asset_health.pipeline")
        self._pd = load("pandas")
        cfg = analysis_cfg or self._analysis_mod.AnalysisConfig()
        if pipeline_cfg is not None:
            cfg = cfg.model_copy(update={"pipeline": pipeline_cfg})
        self.analysis_cfg = cfg
        self.rul_model = (
            rul_model if rul_model is not None else self._analysis_mod.load_default_rul_model()
        )
        self._keys: dict[str, tuple[Any, ...]] = {}
        self._runs: dict[str, Any] = {}
        self._segments: dict[str, tuple[RenderedTelemetry, ...]] = {}
        self._analyses: dict[tuple[str, int], Any] = {}
        self.run_asset_calls = 0
        self.analyse_calls = 0
        self.pipeline_seconds = 0.0  # instrumentation only: wall-clock cost of the sibling code
        self.analyse_seconds = 0.0

    @staticmethod
    def _key(segments: Sequence[RenderedTelemetry]) -> tuple[Any, ...]:
        return tuple(
            (s.latent.life_id, s.start, s.stop, s.latent.repair_at_start, s.n_rows)
            for s in segments
        )

    def set_history(self, asset_id: str, segments: Sequence[RenderedTelemetry]) -> bool:
        """Make ``segments`` the asset's history. True if the pipeline had to run."""
        if asset_id not in ASSET_IDS:
            raise KeyError(f"unknown plant asset {asset_id!r}")
        key = self._key(segments)
        if self._keys.get(asset_id) == key:
            return False
        raw = splice(segments)
        events = maintenance_events(segments)
        started = time.perf_counter()
        run = self._pipeline_mod.run_asset(
            raw, segments[0].spec, events, self.analysis_cfg.pipeline, None
        )
        self.pipeline_seconds += time.perf_counter() - started
        self.run_asset_calls += 1
        self._keys[asset_id] = key
        self._runs[asset_id] = run
        self._segments[asset_id] = tuple(segments)
        self._analyses = {k: v for k, v in self._analyses.items() if k[0] != asset_id}
        return True

    def has(self, asset_id: str) -> bool:
        return asset_id in self._runs

    def run(self, asset_id: str) -> Any:
        return self._runs[asset_id]

    def segments(self, asset_id: str) -> tuple[RenderedTelemetry, ...]:
        return self._segments[asset_id]

    def analysis(self, asset_id: str, instant: datetime) -> Any:
        """``analyse_run`` as of ``instant`` (an aware or naive UTC datetime), memoised."""
        stamp = self._pd.Timestamp(naive(instant))
        memo = (asset_id, stamp.value)
        hit = self._analyses.get(memo)
        if hit is not None:
            return hit
        started = time.perf_counter()
        result = self._analysis_mod.analyse_run(
            self._runs[asset_id], self.analysis_cfg, self.rul_model, as_of=stamp
        )
        self.analyse_seconds += time.perf_counter() - started
        self.analyse_calls += 1
        self._analyses[memo] = result
        return result


@dataclass(frozen=True)
class AlertRecord:
    """What asset-health itself said when the alert was raised (sim-side bookkeeping, not agent
    input): its diagnosis and the interval it published."""

    alert_id: str
    asset_id: str
    work_order_id: str
    hypothesis: str
    mode: str | None
    severity: str
    raised_at: datetime
    rul_status: str
    interval: tuple[float, float, float, float] | None  # lower, median, upper, level
    evidence_features: tuple[str, ...]
    dropped_features: tuple[str, ...]


@dataclass(frozen=True)
class AlertOutcome:
    """The result of ``raise_alert``. ``kind`` is one of

    * ``raised``: a new alert, ``alert_id`` is set;
    * ``existing``: asset-health's work order already has an alert, ``alert_id`` is that one;
    * ``none``: asset-health has no open work order for the asset at this instant;
    * ``no_interval``: it has one but the RUL status is not-degrading or insufficient-trend, so
      there is no interval to hand over (mismatch 4); counted as a missed-alert opportunity.
    """

    kind: str
    alert_id: str | None = None
    reason: str = ""


@dataclass
class AlertStats:
    raised: int = 0
    no_interval: int = 0
    score_clamped: int = 0
    dropped_features: int = 0
    mapped_features: int = 0
    missed_status: dict[str, int] = field(default_factory=dict)


def agent_assets() -> dict[str, Any]:
    """The asset register of the maintenance-agent fixtures for the plant's four assets, with the
    bill of materials made consistent with the plant's spares (mismatch 15).

    The agent's PU-202 uses a different impeller (``SYN-IMP-2002``, stock 0) from PU-201; the plant
    has one shared impeller stock, so the plant's part number is written into the register. This
    changes the agent's data and is recorded as a finding of the integration.
    """
    synthetic = load("maintenance_agent.adapters.synthetic")
    models = load("maintenance_agent.models")
    data = synthetic.load_fixture("alerts.json")
    register: dict[str, Any] = {}
    for item in data["alerts"]:
        info = models.Alert.model_validate(item).asset
        if info.asset_id in ASSET_IDS:
            register.setdefault(info.asset_id, info)
    missing = [a for a in ASSET_IDS if a not in register]
    if missing:
        raise LookupError(f"the agent fixtures hold no asset register entry for {missing}")
    out: dict[str, Any] = {}
    for asset_id, info in register.items():
        bom = dict(info.bom)
        for spec in MODES_BY_CLASS[info.asset_class]:
            bom[COMPONENT_OF_MODE[spec.mode]] = spec.part_no
        out[asset_id] = info.model_copy(update={"bom": bom})
    return out


class AssetHealthAlertSource:
    """Implements ``maintenance_agent.ports.AlertSource`` over the real asset-health pipeline.

    ``runs`` holds the pipeline's runs; ``clock`` is the simulation clock (UTC-aware datetimes);
    ``asset_map`` maps each plant asset id to its asset-health id (mismatch 1) and fixes which
    assets this source serves. ``interval`` optionally replaces asset-health's own p10/p50/p90 with
    a conformal interval from the arm under test: it receives the plant asset and the
    ``RULEstimate`` and returns ``(lower, median, upper, level)``, or ``None`` to keep
    asset-health's own.
    """

    def __init__(
        self,
        runs: RunCache,
        clock: Callable[[], datetime],
        asset_map: Mapping[str, str],
        *,
        assets: Mapping[str, Any] | None = None,
        interval: Callable[[str, Any], tuple[float, float, float, float] | None] | None = None,
    ) -> None:
        self._runs = runs
        self._clock = clock
        self._asset_map = dict(asset_map)
        self._assets = dict(assets) if assets is not None else agent_assets()
        self._interval = interval
        self._alerts: dict[str, Any] = {}
        self._evidence: dict[str, list[Any]] = {}
        self._records: dict[str, AlertRecord] = {}
        self._by_order: dict[tuple[str, str], str] = {}
        self._counter: dict[str, int] = {}
        self._level = float(runs.analysis_cfg.rul.target_coverage)
        self.stats = AlertStats()
        for plant_id in self._asset_map:
            if plant_id not in self._assets:
                raise KeyError(f"no agent asset register entry for {plant_id!r}")

    # ------------------------------------------------------------------ AlertSource protocol

    def list_alert_ids(self) -> list[str]:
        return sorted(self._alerts)

    def get_alert(self, alert_id: str) -> Any:
        if alert_id not in self._alerts:
            raise load("maintenance_agent.ports").NotFoundError(f"alert {alert_id!r} not found")
        return self._alerts[alert_id].model_copy(deep=True)

    def get_evidence(self, alert_id: str) -> list[Any]:
        if alert_id not in self._alerts:
            raise load("maintenance_agent.ports").NotFoundError(f"alert {alert_id!r} not found")
        return [e.model_copy(deep=True) for e in self._evidence[alert_id]]

    # ------------------------------------------------------------------ ours

    def record(self, alert_id: str) -> AlertRecord:
        return self._records[alert_id]

    def records(self) -> list[AlertRecord]:
        return [self._records[k] for k in sorted(self._records)]

    def raise_alert(
        self, asset_id: str, alert_id: str | None = None, *, refresh: bool = False
    ) -> AlertOutcome:
        """Raise an alert for ``asset_id`` from asset-health's state at the clock's instant.

        At most one alert per asset-health work order (a work order is one alert episode): asking
        again while the same episode is open returns the existing alert, unless ``refresh`` asks for
        a new reading under a fresh id (an escalation). An explicit ``alert_id`` that is already
        taken raises ``DuplicateAlert``.
        """
        if asset_id not in self._asset_map:
            raise KeyError(f"{asset_id!r} is not served by this alert source")
        now = aware(self._clock())
        analysis = self._runs.analysis(asset_id, now)
        if not analysis.work_orders:
            return AlertOutcome("none", reason="no open work order")
        order = analysis.work_orders[0]
        existing = self._by_order.get((asset_id, order.work_order_id))
        if existing is not None and not refresh:
            return AlertOutcome("existing", existing)
        status = order.rul.status.value if order.rul is not None else "no_rul"
        # mismatch 4: only an ESTIMATED reading has an interval; the agent cannot take one without
        if order.rul is None or order.rul.status.value != "estimated":
            self.stats.no_interval += 1
            self.stats.missed_status[status] = self.stats.missed_status.get(status, 0) + 1
            return AlertOutcome("no_interval", reason=f"RUL status is {status}")
        new_id = alert_id or self._next_id(asset_id)
        if new_id in self._alerts:
            raise DuplicateAlert(f"alert id {new_id!r} already exists")
        self._register(asset_id, new_id, order, analysis, now)
        return AlertOutcome("raised", new_id)

    def _next_id(self, asset_id: str) -> str:
        n = self._counter.get(asset_id, 0) + 1
        life = self._runs.segments(asset_id)[-1].latent.life_id
        while f"{asset_id}-L{life}-A{n}" in self._alerts:
            n += 1
        self._counter[asset_id] = n
        return f"{asset_id}-L{life}-A{n}"

    def _register(
        self, asset_id: str, alert_id: str, order: Any, analysis: Any, now: datetime
    ) -> None:
        models = load("maintenance_agent.models")
        rul = order.rul
        lower, median, upper, level = rul.p10_days, rul.p50_days, rul.p90_days, self._level
        if self._interval is not None:
            override = self._interval(asset_id, rul)
            if override is not None:
                lower, median, upper, level = override
        # mismatch 5: asset-health's interval is p10-p90 (80%); the field is carried. The agent
        # does not read it (maintenance_agent/models.py:207 defines it; nothing in the package
        # reads it), so it changes nothing downstream except the alert fingerprint.
        interval = models.RULInterval(
            as_of=now.date(),  # mismatch 9: the simulation date, so V10 sees a fresh reading
            lower_days=float(lower),
            median_days=float(median),
            upper_days=float(upper),
            confidence=float(level),
        )
        # mismatch 6: severity map; the fused score is not bounded to [0, 1], so it is clamped
        peak = float(analysis.traces[-1].episode.peak_score) if analysis.traces else 0.0
        score = min(1.0, max(0.0, peak))
        if score != peak:
            self.stats.score_clamped += 1
        hypothesis = order.failure_mode_hypothesis.value
        # mismatch 8: asset-health's latest_safe_date is ignored on purpose; the agent plans and
        # the verifier enforces its own policy_latest_date from the lower bound.
        evidence, kept, dropped = self._evidence_items(alert_id, order, now)
        alert = models.Alert(
            alert_id=alert_id,
            asset=self._assets[asset_id].model_copy(deep=True),
            raised_at=now,
            severity=SEVERITY[order.severity.value],
            summary=(
                f"{order.asset_name}: asset-health rule layer suggests "
                f"{hypothesis.replace('_', ' ')} ({order.severity.value})"
            ),
            anomaly_score=score,
            rul=interval,
        )
        self._alerts[alert_id] = alert
        self._evidence[alert_id] = evidence
        self._by_order[(asset_id, order.work_order_id)] = alert_id
        self._records[alert_id] = AlertRecord(
            alert_id=alert_id,
            asset_id=asset_id,
            work_order_id=order.work_order_id,
            hypothesis=hypothesis,
            mode=HYPOTHESIS_MODE[hypothesis],
            severity=order.severity.value,
            raised_at=now,
            rul_status=rul.status.value,
            interval=(float(lower), float(median), float(upper), float(level)),
            evidence_features=tuple(kept),
            dropped_features=tuple(dropped),
        )
        self.stats.raised += 1

    def _evidence_items(
        self, alert_id: str, order: Any, now: datetime
    ) -> tuple[list[Any], list[str], list[str]]:
        """Mismatch 7: asset-health evidence (observed, expected, z) -> agent evidence.

        Each kept channel becomes one item in robust-sigma units: ``value`` is |z|, ``baseline`` 0
        (the regime-expected level) and ``alarm`` the detection threshold, so the agent's
        exceedance (value over alarm) is |z| / 3.5. The physical readings go in the comment, which
        the agent scans and never obeys. Channels with no counterpart in the agent's vocabulary are
        dropped and listed.
        """
        models = load("maintenance_agent.models")
        items: list[Any] = []
        kept: list[str] = []
        dropped: list[str] = []
        for ev in order.evidence:
            signal = FEATURE_SIGNAL.get(ev.feature)
            if signal is None:
                dropped.append(ev.feature)
                self.stats.dropped_features += 1
                continue
            kept.append(ev.feature)
            self.stats.mapped_features += 1
            items.append(
                models.EvidenceItem(
                    id=f"{alert_id}-E{len(items) + 1}",
                    signal=signal,
                    value=abs(float(ev.z_score)),
                    unit=SIGMA_UNIT,
                    baseline=0.0,
                    alarm=Z_ALARM,
                    direction="high",
                    trend_per_day=0.0,
                    observed_at=now,
                    source="asset-health rule layer",
                    comment=(
                        f"SYNTHETIC. asset-health channel {ev.feature}: observed "
                        f"{ev.observed:.4g} {ev.unit}, regime-expected {ev.expected:.4g}"
                    ),
                )
            )
        return items, kept, dropped
