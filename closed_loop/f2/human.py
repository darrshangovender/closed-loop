"""The simulated human approver (packet P11b). Everything is SYNTHETIC.

This is the human-side module: it imports ``maintenance_agent.gate`` (the signing half, which no
agent-side module may import, mismatch 16) and drives the real ``ApprovalGate`` with the simulation
clock, so an approval's issue and expiry times are simulation time. The orchestrator, the verifier
and the CMMS see an ordinary signed ``Approval``; nothing is faked on their side.

The attention model is the one of the plant (0001 section 2) carried over to the real agent:

* a **latency** from the moment the draft reaches the approver: log-normal, median
  ``approver_median_h`` hours, spread ``approver_sigma`` (drawn once per run, so it does not depend
  on how often the runner looks);
* a **blocked** draft is rejected, with the failed rule ids as the reason (the gate refuses to sign
  it anyway);
* a draft that passes the sixteen rules but names the **wrong failure mode** is caught with
  probability ``approver_catch_p`` and rejected, otherwise signed. This is maintenance-agent's
  documented gap ``wrong_diagnosis_fully_compliant``: the verifier does not judge the diagnosis,
  only the human can, and a wrong draft that gets a signature is the counted metric of 0003 A3.
  The true mode is sim-only knowledge supplied by ``truth_mode``; an inspection-first draft claims
  no diagnosis and is never counted wrong;
* a job that needs a **safety officer** is signed only by an approver whose roles include one
  (mismatch 17), and only while ``on_shift`` says the person is at work (the roster has one safety
  officer, on the day shift). An approver who is not at work, or whose rank is too low, leaves the
  run waiting; ``signs_above_rank`` models the planner who signs anyway, so that the last mile
  (the CMMS) is what refuses.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from closed_loop.f2._siblings import load
from closed_loop.f2.alert_source import aware
from closed_loop.plant.params import PlantParams

PAUSED = ("AWAITING_APPROVAL", "BLOCKED")
DAY_SHIFT = (6, 14)  # hours of the day: shift 0 runs 06:00 to 14:00 (0002 section 3.2)


def on_day_shift(instant: datetime) -> bool:
    """True while the simulation clock is in the day shift (the safety officer's hours)."""
    return DAY_SHIFT[0] <= aware(instant).hour < DAY_SHIFT[1]


@dataclass(frozen=True)
class Review:
    """One decision, for the metrics. ``kind`` is ``approve`` or ``reject``."""

    run_id: str
    asset_id: str
    at: datetime
    kind: str
    reason: str
    top_mode: str | None
    inspection_first: bool
    true_mode: str | None
    wrong_mode: bool
    caught: bool
    latency_h: float
    required_role: str


@dataclass
class _Pending:
    ready_at: datetime
    latency_h: float
    u_catch: float


class SimHumanApprover:
    """One approver: a signing key, a role, a working pattern and an attention model."""

    def __init__(
        self,
        approver_id: str,
        private_key_b64: str,
        clock: Callable[[], datetime],
        params: PlantParams,
        rng: np.random.Generator,
        *,
        roles: tuple[str, ...] = ("PLANNER",),
        truth_mode: Callable[[str], str | None] | None = None,
        on_shift: Callable[[datetime], bool] | None = None,
        ttl: timedelta = timedelta(hours=8),
        signs_above_rank: bool = False,
    ) -> None:
        gate = load("maintenance_agent.gate")
        models = load("maintenance_agent.models")
        self._gate = gate.ApprovalGate(
            approver_id, private_key_b64, ttl=ttl, clock=lambda: aware(clock())
        )
        self._rank = max((models.ROLE_RANK.get(r, 0) for r in roles), default=0)
        self._role_rank = models.ROLE_RANK
        self.approver_id = self._gate.approver_id
        self.roles = roles
        self._clock = clock
        self._params = params
        self._rng = rng
        self._truth_mode = truth_mode
        self._on_shift = on_shift
        self._above_rank = signs_above_rank
        self._pending: dict[str, _Pending] = {}
        self._signed: set[tuple[str, str, str]] = set()
        self._refused: set[tuple[str, str]] = set()  # signed, then not released: do not sign again
        self.reasons: dict[str, str] = {}
        self.log: list[Review] = []
        self.deferred_role = 0
        self.deferred_shift = 0

    # ------------------------------------------------------------------ the decision

    def review(self, run_state: Any) -> Any | None:
        """An ``Approval`` (sign), a ``SignedDecision`` (reject) or ``None`` (not yet).

        Pure with respect to the orchestrator: it reads the run state it is given. The caller should
        re-verify just before (``act`` does), because the approval commits to the verification
        digest it signs.
        """
        if run_state.status.value not in PAUSED:
            return None
        pending = self._open(run_state)
        now = aware(self._clock())
        if now < pending.ready_at:
            return None
        if self._on_shift is not None and not self._on_shift(now):
            self.deferred_shift += 1
            return None
        approvals = load("maintenance_agent.approval")
        models = load("maintenance_agent.models")
        run_id = run_state.run_id
        draft_hash = run_state.draft_hash
        if run_state.status.value == "BLOCKED":
            rules = sorted(v.rule_id for v in run_state.verification.violations)
            reason = "verification failed: " + ", ".join(rules)
            return self._reject(run_state, pending, reason, None, False, "n/a")
        draft = models.WorkOrderDraft.model_validate(run_state.draft)
        required = approvals.required_role(draft)
        if self._role_rank[required] > self._rank and not self._above_rank:
            self.deferred_role += 1
            return None  # left for a colleague with the role (mismatch 17)
        key = (run_id, draft_hash, run_state.verification.digest)
        if key in self._signed or (run_id, draft_hash) in self._refused:
            return None  # one signature per verified draft: a refusal at the last mile stands
        top = draft.top_mode()
        top_mode = top.value if top is not None else None
        true_mode = self._truth_mode(draft.asset_id) if self._truth_mode is not None else None
        wrong = bool(
            true_mode is not None
            and not draft.inspection_first
            and top_mode is not None
            and top_mode != true_mode
        )
        if wrong and pending.u_catch < self._params.approver_catch_p:
            return self._reject(
                run_state,
                pending,
                "the diagnosis is not supported by the readings",
                true_mode,
                True,
                required,
            )
        self._signed.add(key)
        approval = self._gate.approve(run_id=run_id, draft=draft, report=run_state.verification)
        self.log.append(
            Review(
                run_id,
                draft.asset_id,
                aware(self._clock()),
                "approve",
                "",
                top_mode,
                draft.inspection_first,
                true_mode,
                wrong,
                False,
                pending.latency_h,
                required,
            )
        )
        return approval

    def act(self, orchestrator: Any, run_id: str) -> Any:
        """One look at a run, as the command-line approver does it: wait for the latency, re-verify
        against the live store and alert, then sign or reject and submit. Returns the run state."""
        state = orchestrator.get(run_id)
        if state.status.value not in PAUSED:
            return state
        pending = self._open(state)
        if aware(self._clock()) < pending.ready_at:
            return state
        if self._on_shift is not None and not self._on_shift(aware(self._clock())):
            self.deferred_shift += 1
            return state
        state = orchestrator.refresh_verification(run_id)
        outcome = self.review(state)
        if outcome is None:
            return state
        if getattr(outcome, "kind", "approve") == "reject":
            return orchestrator.submit_rejection(run_id, outcome, self.reasons[run_id])
        state = orchestrator.submit_approval(run_id, outcome)
        if state.status.value != "COMPLETED":
            self._refused.add((run_id, state.draft_hash))
        return state

    # ------------------------------------------------------------------ helpers

    def _open(self, run_state: Any) -> _Pending:
        """Start the clock for a run the first time it is seen (one latency and one catch draw)."""
        pending = self._pending.get(run_state.run_id)
        if pending is None:
            p = self._params
            hours = float(
                self._rng.lognormal(np.log(max(p.approver_median_h, 1e-9)), p.approver_sigma)
            )
            catch = float(self._rng.random())
            pending = _Pending(aware(self._clock()) + timedelta(hours=hours), hours, catch)
            self._pending[run_state.run_id] = pending
        return pending

    def _reject(
        self,
        run_state: Any,
        pending: _Pending,
        reason: str,
        true_mode: str | None,
        caught: bool,
        required: str,
    ) -> Any:
        decision = self._gate.sign_reject(
            run_id=run_state.run_id, draft_hash=run_state.draft_hash, reason=reason
        )
        self.reasons[run_state.run_id] = reason
        draft = run_state.draft or {}
        hypotheses = draft.get("hypotheses") or [{}]
        self.log.append(
            Review(
                run_state.run_id,
                str(draft.get("asset_id", "")),
                aware(self._clock()),
                "reject",
                reason,
                hypotheses[0].get("mode"),
                bool(draft.get("inspection_first", False)),
                true_mode,
                caught,
                caught,
                pending.latency_h,
                required,
            )
        )
        return decision
