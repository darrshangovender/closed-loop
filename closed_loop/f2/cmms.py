"""The plant's CMMS for maintenance-agent (packet P11b). Everything is SYNTHETIC.

``PlantCMMS`` implements ``maintenance_agent.ports.CMMS``. It adds three things to the sibling's
``MockCMMS`` and nothing else: it serves part stock from the plant instead of the fixture
(mismatch 13), it queues a plant job when a work order is released (mismatch 14), and it keeps the
sibling's own approval checks, idempotency and duplicate refusal by delegating every check to
``MockCMMS.release_work_order``. It never inspects an approval itself.

A work order is released at most once and its job is queued exactly once: a repeat release of the
same run and draft returns the existing work order and queues nothing (the sibling's idempotency),
and a job whose ``queue`` call failed is queued again when the release is next looked up, because
the orchestrator adopts an existing release without calling ``release_work_order`` again.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from closed_loop.f2._siblings import load
from closed_loop.plant.params import PARTS


def inventory_of(stock: Callable[[], Mapping[str, int]]) -> Callable[[str], Any | None]:
    """A ``plant_inventory`` lookup over the plant's live stock (part number to quantity on hand).

    Lead times are the plant's catalogue (``PARTS``, copied from the agent's inventory fixture); a
    part the plant does not hold gives ``None``, which ``PlantCMMS`` reads from the inner mock.
    """
    models = load("maintenance_agent.models")

    def lookup(part_no: str) -> Any | None:
        spec = PARTS.get(part_no)
        if spec is None:
            return None
        return models.PartRecord(
            part_no=part_no,
            description=f"SYNTHETIC plant spare {part_no}",
            stock_qty=int(stock().get(part_no, 0)),
            lead_time_days=int(round(spec.lead_days)),
        )

    return lookup


class PlantCMMS:
    """``ports.CMMS`` over a ``MockCMMS`` (one per episode, mismatch 11), with plant stock and a
    job queue.

    ``plant_inventory(part_no)`` returns the plant's current ``PartRecord`` for a part it tracks,
    or ``None``; a part it does not track is read from the inner mock and counted in
    ``fallback_lookups``. ``queue(draft, wo_number)`` receives each newly released work order once.
    """

    def __init__(
        self,
        inner: Any,
        plant_inventory: Callable[[str], Any | None],
        queue: Callable[[Any, str], None],
    ) -> None:
        self._inner = inner
        self._inventory = plant_inventory
        self._queue = queue
        self._drafts: dict[str, Any] = {}
        self._queued: set[str] = set()
        self.fallback_lookups = 0
        self.plant_lookups = 0

    # ------------------------------------------------------------------ reads (mismatch 13)

    def get_part(self, part_no: str) -> Any | None:
        record = self._inventory(part_no)
        if record is not None:
            self.plant_lookups += 1
            return record.model_copy(deep=True)
        self.fallback_lookups += 1
        return self._inner.get_part(part_no)

    # ------------------------------------------------------------------ dry run

    def draft_work_order(self, draft: Any) -> Any:
        """The sibling's dry run, over the plant's current stock.

        ``MockCMMS.draft_work_order`` reads a private fixed stock table, so a throw-away mock is
        built from the plant's stock for the parts in this draft and the sibling's own code runs on
        it. Nothing is created.
        """
        mock = load("maintenance_agent.cmms")
        parts = {}
        for line in draft.parts:
            record = self.get_part(line.part_no)
            if record is not None:
                parts[line.part_no] = record
        return mock.MockCMMS(parts, self._inner.registry).draft_work_order(draft)

    # ------------------------------------------------------------------ the one side effect

    def release_work_order(self, draft: Any, approval: Any) -> Any:
        """Delegate every check to the sibling; queue the plant job once for a new order.

        A repeat release returns the existing order from the sibling (its idempotency) and
        ``_queue_once`` makes the second queue a no-op, unless the first one failed.
        """
        released = self._inner.release_work_order(draft, approval)
        self._drafts.setdefault(released.wo_number, draft)
        self._queue_once(released.wo_number)
        return released

    def find_release(self, approval_id: str) -> Any | None:
        released = self._inner.find_release(approval_id)
        if released is not None:
            self._queue_once(released.wo_number)
        return released

    def find_release_for(self, run_id: str, draft_hash: str) -> Any | None:
        released = self._inner.find_release_for(run_id, draft_hash)
        if released is not None:
            self._queue_once(released.wo_number)
        return released

    # ------------------------------------------------------------------ ours

    @property
    def released_orders(self) -> list[str]:
        """Work-order numbers whose plant job has been queued, in release order."""
        return [r["wo_number"] for r in self._inner.released() if r["wo_number"] in self._queued]

    def draft_of(self, wo_number: str) -> Any:
        return self._drafts[wo_number]

    def _queue_once(self, wo_number: str) -> None:
        if wo_number in self._queued:
            return
        draft = self._drafts.get(wo_number)
        if (
            draft is None
        ):  # released before this object saw it (a resumed episode): nothing to queue
            return
        self._queue(draft, wo_number)  # an exception leaves it unqueued, to be retried on lookup
        self._queued.add(wo_number)
