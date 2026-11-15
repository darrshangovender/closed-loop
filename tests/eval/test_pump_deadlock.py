"""0003 A18, R8: the pump deadlock, kept as a measured number and shown to be gone.

Both pumps' sets reach the top bin around operating day 50. The game stops one as a reserve; its job
asks for ``LOCK``, the proof refuses it while the other pump runs, the shield substitutes a ``STOP``
that does nothing to a stopped pump, and the stateless runner asks again the next quarter. Before
R8 that stood for the rest of the episode (PU-201 stopped 3,455 quarters on seed 1102, and 3,346
overridden commands). ``P1-nopair`` is P1 as it was before R8, so the deadlock stays a generated
number; ``P1`` has the operator's pair stop.

The episodes use the full prepared context (the deadlock is a property of the real sets, which the
small context's sets reproduce only in part) and development seed 1102.
"""

from __future__ import annotations

from collections import Counter

import pytest

from closed_loop.core.types import Cmd, Op
from closed_loop.eval.arms import ARMS
from closed_loop.eval.context import PreparedContext
from closed_loop.eval.episode import EpisodeRecord, scenario_for, simulate_episode
from closed_loop.proposers.restart import OPERATOR

pytestmark = pytest.mark.slow

SEED = 1102
OPS = list(Op)
PUMPS = ("PU-201", "PU-202")


def quarters_out_of_run(rec: EpisodeRecord, aid: str) -> int:
    return sum(1 for v in rec.trace.op[aid] if OPS[int(v)] is not Op.RUN)


def closes(rec: EpisodeRecord, aid: str) -> int:
    return sum(
        1
        for _q, c, ex, _ov in rec.trace.commands
        if c.asset_id == aid and c.cmd is Cmd.CLOSE and ex
    )


@pytest.fixture(scope="module")
def nopair(full_context: PreparedContext) -> EpisodeRecord:
    return simulate_episode(ARMS["P1-nopair"], SEED, scenario_for("nominal"), context=full_context)


@pytest.fixture(scope="module")
def paired(full_context: PreparedContext) -> EpisodeRecord:
    return simulate_episode(ARMS["P1"], SEED, scenario_for("nominal"), context=full_context)


class TestTheDeadlockIsMeasuredAndGone:
    def test_without_the_pair_stop_a_pump_stands_stopped_to_the_end(
        self, nopair: EpisodeRecord
    ) -> None:
        assert OPS[int(nopair.trace.op["PU-201"][-1])] is Op.STOP
        assert quarters_out_of_run(nopair, "PU-201") > 3000
        assert closes(nopair, "PU-201") == 0, "its job never began"
        replaced = Counter(
            f"{c.asset_id}:{c.cmd.name}" for _q, c, _ex, ov in nopair.trace.commands if ov
        )
        assert replaced["PU-201:STOP"] > 3000, "the shield answered the job's LOCK with a STOP"
        assert not [
            1
            for _q, c, ex, _ov in nopair.trace.commands
            if c.source == OPERATOR and c.cmd is Cmd.STOP and ex
        ]

    def test_with_the_pair_stop_both_pumps_end_running_and_the_job_is_done(
        self, paired: EpisodeRecord
    ) -> None:
        for aid in PUMPS:
            assert OPS[int(paired.trace.op[aid][-1])] is Op.RUN, aid
        assert closes(paired, "PU-201") >= 1, "the reserve pump was renewed"
        assert quarters_out_of_run(paired, "PU-201") < 300
        assert quarters_out_of_run(paired, "PU-202") < 300
        pair = [
            1
            for _q, c, ex, ov in paired.trace.commands
            if c.source == OPERATOR and c.cmd is Cmd.STOP and ex and not ov
        ]
        assert len(pair) >= 1, "the operator's pair stop was admitted"

    def test_the_pair_stop_removes_the_overrides_and_the_lost_throughput(
        self, nopair: EpisodeRecord, paired: EpisodeRecord
    ) -> None:
        assert paired.result.shield_overrides < nopair.result.shield_overrides / 10
        assert paired.result.throughput > nopair.result.throughput + 0.05
        assert paired.result.cost < nopair.result.cost
