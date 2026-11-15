"""Splice the pieces of an asset's history into one record for the real pipeline (packet P11a).

``splice`` concatenates the rendered pieces of one plant asset; ``maintenance_events`` writes the
CMMS repair record at every join; ``run_segments`` runs the real ``asset_health.pipeline.run_asset``
on the two. The spike ``docs/adr/spikes/splice_lives.py`` (seeds 3, 5, 8 and 11) showed that the
pipeline cuts a segment at the repair record, re-learns the baseline (no decision for 23 days),
opens
nothing in that window, and gives scores before the join that equal a single-life run bit for bit.
So the fallback of 0001 section 15 (one life per asset per episode) is not needed.

What a splice cannot hide, and the pipeline does not mind: the gap between the stop and the restart
has no rows, so it reads as a communications outage; and the data-quality masks look back across the
join (a 48 h spike window, a 14 d roughness reference), so the first days of a new life can mask a
few rows differently from a run on that life alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from closed_loop.f2._siblings import load
from closed_loop.f2.render import RenderedTelemetry


class SpliceError(ValueError):
    """The pieces do not form one asset history."""


def check_order(segments: Sequence[RenderedTelemetry]) -> None:
    """Raise ``SpliceError`` unless the pieces are one asset's history, in order.

    A piece must start where or after the previous one stopped; a new life must come with a repair
    record (without it the pipeline would take a fresh component for the old one and raise
    alarms against a baseline learned on the old signature); the position labels must be 0, 1, ...
    """
    if not segments:
        raise SpliceError("nothing to splice")
    first = segments[0]
    for k, seg in enumerate(segments):
        if seg.latent.asset_id != first.latent.asset_id or seg.ah_asset_id != first.ah_asset_id:
            raise SpliceError(
                f"piece {k} is for {seg.latent.asset_id}, not {first.latent.asset_id}"
            )
        if seg.segment != k:
            raise SpliceError(f"piece {k} is labelled segment {seg.segment}")
        if seg.n_rows == 0:
            raise SpliceError(f"piece {k} has no rows (start {seg.start}, stop {seg.stop})")
        if k == 0:
            continue
        prev = segments[k - 1]
        if seg.start < prev.stop:
            raise SpliceError(
                f"piece {k} starts at {seg.start}, before piece {k - 1} stops at {prev.stop}"
            )
        if seg.latent.life_id < prev.latent.life_id:
            raise SpliceError(f"piece {k} goes back from life {prev.latent.life_id}")
        if seg.latent.life_id != prev.latent.life_id and not seg.latent.repair_at_start:
            raise SpliceError(
                f"piece {k} is a new life ({seg.latent.life_id}) but carries no repair record"
            )


def splice(segments: Sequence[RenderedTelemetry]) -> Any:
    """One historian export (a ``pandas.DataFrame``) for the whole history of the asset."""
    check_order(segments)
    pd = load("pandas")
    return pd.concat([s.frame for s in segments], ignore_index=True)


def maintenance_events(segments: Sequence[RenderedTelemetry]) -> list[Any]:
    """The CMMS log the pipeline may read: a repair record at the start of every later piece that
    asks for one (a new life, or a job completed on the same component)."""
    check_order(segments)
    models = load("asset_health.models")
    events = []
    for seg in segments[1:]:
        if seg.latent.repair_at_start:
            events.append(
                models.MaintenanceEvent(
                    asset_id=seg.ah_asset_id,
                    timestamp=seg.start,
                    kind="repair",
                    note="SYNTHETIC: component work completed",
                )
            )
    return events


def run_segments(
    segments: Sequence[RenderedTelemetry], cfg: Any | None = None, as_of: datetime | None = None
) -> Any:
    """The real ``run_asset`` on the spliced history. ``as_of`` replays that instant: nothing later
    is read into any stage (the pipeline's own rule)."""
    pd = load("pandas")
    pipeline = load("asset_health.pipeline")
    raw = splice(segments)
    stamp = None if as_of is None else pd.Timestamp(as_of)
    return pipeline.run_asset(raw, segments[0].spec, maintenance_events(segments), cfg, stamp)
