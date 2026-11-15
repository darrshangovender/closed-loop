"""Construct the real maintenance-agent orchestrator for an F2 episode (packet P11b).

``maintenance_agent.wiring.build_system`` constructs its own ``MockCMMS`` and takes no ``cmms``
argument (mismatch 12), so the ``Orchestrator`` is built directly here, with the same parts
``build_system`` uses (the rules planner, the signed procedure store, in-memory checkpoints and
audit logs) and the simulation clock everywhere a clock is accepted. Everything is SYNTHETIC.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from functools import lru_cache
from typing import Any

from closed_loop.f2._siblings import load
from closed_loop.f2.alert_source import aware


@lru_cache(maxsize=1)
def procedure_store() -> Any:
    """The signed SYNTHETIC procedure library, built once per process (about 0.7 s; read-only)."""
    return load("maintenance_agent.adapters.synthetic").SyntheticProcedureStore.from_fixtures()


def new_registry() -> Any:
    """An empty in-memory approver registry (trusted by construction, as in the sibling's eval)."""
    return load("maintenance_agent.approval").ApproverRegistry()


def register_approver(registry: Any, approver_id: str, name: str, roles: list[str]) -> str:
    """Add an approver to ``registry`` and return the private key (base64). The key belongs to the
    human side (``SimHumanApprover``); the agent side only ever sees the public key."""
    approver, private_key = load("maintenance_agent.gate").make_approver(approver_id, name, roles)
    registry.add(approver)
    return str(private_key)


def run_id_for(alert_id: str) -> str:
    """A legal run id for an alert (mismatch 11: lower-case tokens of at most 64 characters)."""
    return str(load("maintenance_agent.models").check_identifier(f"run-{alert_id}", "run id"))


def build_orchestrator(
    alerts: Any,
    cmms: Any,
    registry: Any,
    clock: Callable[[], datetime],
    *,
    planner: Any | None = None,
    store: Any | None = None,
    max_alert_age_days: int | None = None,
) -> Any:
    """The real ``Orchestrator`` over ``alerts`` and ``cmms``, with the simulation clock.

    ``max_alert_age_days`` defaults to the sibling's own limit (3 days, verifier rule V10).
    """
    orchestrator = load("maintenance_agent.orchestrator")
    verifier = load("maintenance_agent.verifier")
    rules = load("maintenance_agent.planners.rules")
    checkpoint = load("maintenance_agent.checkpoint")
    audit = load("maintenance_agent.audit")
    models = load("maintenance_agent.models")

    def stamp() -> str:
        return aware(clock()).isoformat()

    def audit_factory(run_id: str) -> Any:
        models.check_identifier(run_id, "run id")
        return audit.AuditLog(None, clock=stamp)

    return orchestrator.Orchestrator(
        alerts=alerts,
        store=store if store is not None else procedure_store(),
        cmms=cmms,
        planner=planner if planner is not None else rules.RulesPlanner(),
        checkpoints=checkpoint.MemoryCheckpointer(),
        audit_factory=audit_factory,
        clock=lambda: aware(clock()),
        registry=registry,
        max_alert_age_days=(
            verifier.DEFAULT_MAX_ALERT_AGE_DAYS
            if max_alert_age_days is None
            else max_alert_age_days
        ),
    )


def new_mock_cmms(registry: Any, clock: Callable[[], datetime]) -> Any:
    """One ``MockCMMS`` per episode (mismatch 11), with the simulation clock for its ledger and for
    the expiry of approvals."""
    return load("maintenance_agent.cmms").MockCMMS.from_fixtures(
        registry, None, clock=lambda: aware(clock())
    )
