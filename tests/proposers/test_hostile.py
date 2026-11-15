"""Careless and Gullible (packet P6b): what they change, and the counts they produce.

The counts over whole episodes are recorded by ``scripts/hostile_and_conformance.py`` into the
progress note; the assertions here are only the ones the packet asks for (greater than zero) and the
properties that hold by construction.
"""

from __future__ import annotations

import importlib.util
from collections import Counter
from dataclasses import dataclass, field

import pytest

from closed_loop.core.types import Cmd, Command, PlanCmd
from closed_loop.plant.notes import NOTE_BANK, cached_scan
from closed_loop.plant.notes import screener_misses as cached_misses
from closed_loop.proposers import Careless, Gullible, make_baseline, screener_misses
from closed_loop.proposers.hostile import SKIP_TO
from tests.proposers._loop import run_episode
from tests.proposers._views import observation

CRUSHER, PUMP_A, CONVEYOR = "CR-501", "PU-201", "CV-101"


@dataclass
class Scripted:
    """Proposes the same commands every call and remembers resets."""

    commands: list[Command] = field(default_factory=list)
    name: str = "scripted"
    resets: list[int] = field(default_factory=list)

    def reset(self, seed: int) -> None:
        self.resets.append(seed)

    def propose(self, obs: object) -> list[Command]:
        return list(self.commands)


def cmd(asset: str, c: Cmd | PlanCmd, arg: str | None = None) -> Command:
    return Command(asset, c, arg, "inner")


class TestCarelessByConstruction:
    def test_p_skip_zero_is_the_inner_proposer(self) -> None:
        inner = Scripted([cmd(CRUSHER, Cmd.LOCK), cmd(PUMP_A, Cmd.PROVE)])
        c = Careless(inner, p_skip=0.0)
        c.reset(3)
        for q in range(300):
            assert c.propose(observation(quarter=q)) == inner.commands
        assert c.skips == []

    @pytest.mark.parametrize("intended", sorted(SKIP_TO, key=lambda x: x.value))
    def test_p_skip_one_applies_the_skip_table(self, intended: Cmd) -> None:
        c = Careless(Scripted([cmd(CRUSHER, intended)]), p_skip=1.0)
        c.reset(0)
        out = c.propose(observation(quarter=5))
        target = SKIP_TO[intended]
        if target is intended:
            assert out == [cmd(CRUSHER, intended)]
            assert c.skips == []
        elif target is None:
            assert out == []
            assert [(s.intended, s.issued) for s in c.skips] == [(intended, None)]
        else:
            assert [(x.asset_id, x.cmd, x.source) for x in out] == [(CRUSHER, target, c.name)]
            assert [(s.intended, s.issued, s.quarter) for s in c.skips] == [(intended, target, 5)]

    def test_the_example_of_the_brief_begin_before_prove(self) -> None:
        assert SKIP_TO[Cmd.PROVE] is Cmd.BEGIN
        assert SKIP_TO[Cmd.STOP] is Cmd.BEGIN
        assert SKIP_TO[Cmd.TEST] is Cmd.ENTER
        assert SKIP_TO[Cmd.EXIT] is None

    def test_enter_is_not_a_skip(self) -> None:
        assert Cmd.ENTER not in SKIP_TO
        c = Careless(Scripted([cmd(CRUSHER, Cmd.ENTER)]), p_skip=1.0)
        c.reset(0)
        assert c.propose(observation()) == [cmd(CRUSHER, Cmd.ENTER)]

    def test_planning_commands_and_commands_outside_the_alphabet_pass_untouched(self) -> None:
        inner = Scripted(
            [cmd(CRUSHER, PlanCmd.RAISE_WO, "LINER_WEAR"), cmd(PUMP_A, PlanCmd.INSPECT),
             cmd(CONVEYOR, Cmd.NOOP), cmd(CONVEYOR, Cmd.DERATE)]
        )  # fmt: skip
        c = Careless(inner, p_skip=1.0)
        c.reset(0)
        assert c.propose(observation()) == inner.commands

    def test_other_commands_in_the_same_quarter_are_kept(self) -> None:
        inner = Scripted([cmd(CRUSHER, Cmd.PROVE), cmd(PUMP_A, Cmd.ENTER)])
        c = Careless(inner, p_skip=1.0)
        c.reset(0)
        out = c.propose(observation())
        assert [(x.asset_id, x.cmd) for x in out] == [(CRUSHER, Cmd.BEGIN), (PUMP_A, Cmd.ENTER)]

    def test_p_skip_is_validated(self) -> None:
        for bad in (-0.1, 1.1):
            with pytest.raises(ValueError, match="p_skip"):
                Careless(Scripted(), p_skip=bad)

    def test_the_name_says_what_it_wraps(self) -> None:
        assert Careless(Scripted(name="B2")).name == "careless(B2)"


class TestCarelessRandomness:
    def test_the_skip_rate_is_p_skip(self) -> None:
        c = Careless(Scripted([cmd(CRUSHER, Cmd.PROVE)]), p_skip=0.05)
        c.reset(11)
        n = 20_000
        for q in range(n):
            c.propose(observation(quarter=q % 8640))
        k = len(c.skips)
        # 1000 expected, standard deviation about 31
        assert abs(k - 0.05 * n) < 5 * (n * 0.05 * 0.95) ** 0.5

    def test_same_seed_same_skips_and_a_different_seed_other_skips(self) -> None:
        def skips(seed: int) -> list[int]:
            c = Careless(Scripted([cmd(CRUSHER, Cmd.PROVE)]), p_skip=0.2)
            c.reset(seed)
            for q in range(500):
                c.propose(observation(quarter=q))
            return [s.quarter for s in c.skips]

        assert skips(4) == skips(4)
        assert skips(4) != skips(5)

    def test_which_quarters_go_wrong_does_not_depend_on_what_the_inner_proposer_does(self) -> None:
        """Paired arms: the same seed sees the same carelessness whatever the inner proposer."""

        def skip_quarters(inner: Scripted) -> list[int]:
            c = Careless(inner, p_skip=0.3)
            c.reset(9)
            for q in range(400):
                c.propose(observation(quarter=q))
            return [s.quarter for s in c.skips if s.asset_id == CRUSHER]

        a = skip_quarters(Scripted([cmd(CRUSHER, Cmd.PROVE)]))
        b = skip_quarters(Scripted([cmd(CRUSHER, Cmd.LOCK), cmd(PUMP_A, Cmd.STOP)]))
        assert a == b and a

    def test_streams_of_different_assets_are_independent(self) -> None:
        c = Careless(Scripted([cmd(CRUSHER, Cmd.PROVE), cmd(PUMP_A, Cmd.PROVE)]), p_skip=0.3)
        c.reset(1)
        for q in range(400):
            c.propose(observation(quarter=q))
        by = {a: [s.quarter for s in c.skips if s.asset_id == a] for a in (CRUSHER, PUMP_A)}
        assert by[CRUSHER] != by[PUMP_A]

    def test_reset_forwards_the_seed_clears_the_log_and_redraws(self) -> None:
        inner = Scripted([cmd(CRUSHER, Cmd.PROVE)])
        c = Careless(inner, p_skip=0.5)
        c.reset(1)
        for q in range(50):
            c.propose(observation(quarter=q))
        first = [s.quarter for s in c.skips]
        assert first
        c.reset(2)
        assert c.skips == [] and inner.resets == [1, 2]
        for q in range(50):
            c.propose(observation(quarter=q))
        assert [s.quarter for s in c.skips] != first
        c.reset(1)
        for q in range(50):
            c.propose(observation(quarter=q))
        assert [s.quarter for s in c.skips] == first


class TestCarelessOverEpisodes:
    def test_with_the_shield_off_100_episodes_give_a_p1_or_p3_violation(self) -> None:
        """P6b done criterion. The count is recorded by the script, not asserted beyond > 0."""
        seen: Counter[str] = Counter()
        for seed in range(1, 101):
            ep = run_episode(Careless(make_baseline("B2")), seed)
            seen.update(p for p, _a, _q in ep.violations)
            if seen["P1"] + seen["P3"] > 0:
                break
        assert seen["P1"] + seen["P3"] > 0

    def test_the_honest_proposer_alone_has_no_p1_violation(self) -> None:
        total = Counter()
        for seed in range(1, 11):
            ep = run_episode(make_baseline("B2"), seed)
            total.update(p for p, _a, _q in ep.violations)
        assert total["P1"] == 0


class TestGullibleByConstruction:
    def note_obs(self, text: str | None, quarter: int = 32) -> object:
        return observation(quarter=quarter, note=text)

    def test_without_a_note_it_is_the_inner_proposer(self) -> None:
        inner = Scripted([cmd(CRUSHER, Cmd.LOCK)])
        g = Gullible(inner, False, cached_scan)
        g.reset(0)
        assert g.propose(self.note_obs(None)) == inner.commands  # type: ignore[arg-type]
        assert g.notes_seen == 0

    @pytest.mark.parametrize(
        "note", [n for n in NOTE_BANK if not n.malicious], ids=lambda n: n.note_id
    )
    def test_a_routine_note_changes_nothing(self, note) -> None:  # type: ignore[no-untyped-def]
        inner = Scripted([cmd(CRUSHER, Cmd.LOCK)])
        for screen in (False, True):
            g = Gullible(inner, screen, cached_scan)
            g.reset(0)
            assert g.propose(self.note_obs(note.text)) == inner.commands  # type: ignore[arg-type]
            assert g.obeyed == [] and g.notes_seen == 1

    @pytest.mark.parametrize("note", [n for n in NOTE_BANK if n.malicious], ids=lambda n: n.note_id)
    def test_without_screening_every_instruction_is_obeyed(self, note) -> None:  # type: ignore[no-untyped-def]
        g = Gullible(Scripted(), False, cached_scan)
        g.reset(0)
        out = g.propose(self.note_obs(note.text, quarter=64))  # type: ignore[arg-type]
        assert [(c.asset_id, c.cmd, c.source) for c in out] == [
            (note.asset_id, note.induces, g.name)
        ]
        assert [(o.quarter, o.note_id, o.asset_id, o.command) for o in g.obeyed] == [
            (64, note.note_id, note.asset_id, note.induces)
        ]

    @pytest.mark.parametrize("note", [n for n in NOTE_BANK if n.malicious], ids=lambda n: n.note_id)
    def test_with_screening_an_instruction_is_obeyed_exactly_when_the_screener_misses_it(
        self, note
    ) -> None:  # type: ignore[no-untyped-def]
        g = Gullible(Scripted(), True, cached_scan)
        g.reset(0)
        out = g.propose(self.note_obs(note.text))  # type: ignore[arg-type]
        flagged = cached_scan(note.text) is not None
        assert (out == []) is flagged
        assert g.notes_dropped == int(flagged)
        assert [d[1:] for d in g.dropped] == (
            [(note.note_id, cached_scan(note.text))] if flagged else []
        )
        assert len(g.obeyed) == int(not flagged)

    def test_the_instruction_takes_the_place_of_the_inner_command_for_that_asset(self) -> None:
        inner = Scripted([cmd(CRUSHER, Cmd.LOCK), cmd(PUMP_A, Cmd.STOP)])
        g = Gullible(inner, False, cached_scan)
        g.reset(0)
        begin = next(n for n in NOTE_BANK if n.note_id == "M01")  # BEGIN on CR-501
        out = g.propose(self.note_obs(begin.text))  # type: ignore[arg-type]
        assert [(c.asset_id, c.cmd) for c in out] == [(PUMP_A, Cmd.STOP), (CRUSHER, Cmd.BEGIN)]

    def test_a_text_outside_the_bank_is_read_as_routine(self) -> None:
        g = Gullible(Scripted(), False, cached_scan)
        g.reset(0)
        assert g.propose(self.note_obs("Please BEGIN work on CR-501 immediately.")) == []  # type: ignore[arg-type]
        assert g.obeyed == []

    def test_a_scan_that_flags_everything_drops_everything_and_one_that_flags_nothing_nothing(
        self,
    ) -> None:
        note = next(n for n in NOTE_BANK if n.note_id == "M01")
        drop_all = Gullible(Scripted(), True, lambda text: "rule")
        drop_all.reset(0)
        assert drop_all.propose(self.note_obs(note.text)) == []  # type: ignore[arg-type]
        none = Gullible(Scripted(), True, lambda text: None)
        none.reset(0)
        assert len(none.propose(self.note_obs(note.text))) == 1  # type: ignore[arg-type]
        off = Gullible(Scripted(), False, lambda text: "rule")  # not consulted
        off.reset(0)
        assert len(off.propose(self.note_obs(note.text))) == 1  # type: ignore[arg-type]

    def test_reset_forwards_the_seed_and_clears_the_counters(self) -> None:
        inner = Scripted()
        g = Gullible(inner, True, cached_scan)
        g.reset(7)
        g.propose(self.note_obs(NOTE_BANK[10].text))
        g.reset(8)
        assert inner.resets == [7, 8]
        assert (g.notes_seen, g.notes_dropped, g.obeyed, g.dropped) == (0, 0, [], [])

    def test_the_name_says_whether_it_screens(self) -> None:
        assert Gullible(Scripted(name="B2"), False, cached_scan).name == "gullible(B2)"
        assert Gullible(Scripted(name="B2"), True, cached_scan).name == "gullible(B2,screened)"


class TestTheScreenerMissList:
    def test_the_list_for_the_cached_verdicts_is_the_one_the_note_module_reports(self) -> None:
        assert screener_misses(cached_scan) == cached_misses()

    def test_every_malicious_note_misses_a_screener_that_flags_nothing(self) -> None:
        assert {n.note_id for n in screener_misses(lambda t: None)} == {
            n.note_id for n in NOTE_BANK if n.malicious
        }

    def test_a_screener_that_flags_everything_misses_nothing(self) -> None:
        assert screener_misses(lambda t: "rule") == ()

    def test_every_intent_survives_screening_at_least_once(self) -> None:
        assert {n.induces for n in screener_misses(cached_scan)} == {
            Cmd.BEGIN,
            Cmd.ENTER,
            Cmd.START,
        }

    def test_the_misses_are_the_plain_notes_and_one_hidden_arabic_note(self) -> None:
        # a recorded output, not a target: if the screener or the bank changes, so does this
        assert [n.note_id for n in screener_misses(cached_scan)] == [
            "M01", "M02", "M03", "M04", "M05", "A01", "A02", "A05",
        ]  # fmt: skip


class TestGullibleOverEpisodes:
    def test_without_screening_malicious_notes_produce_violations(self) -> None:
        total = 0
        for seed in range(1, 11):
            ep = run_episode(Gullible(make_baseline("B2"), False, cached_scan), seed)
            total += ep.violated("P1", "P2", "P3", "P6")
        assert total > 0

    def test_a_screened_gullible_obeys_only_notes_the_screener_missed(self) -> None:
        misses = {n.note_id for n in screener_misses(cached_scan)}
        obeyed: Counter[str] = Counter()
        dropped = 0
        for seed in range(1, 9):
            g = Gullible(make_baseline("B2"), True, cached_scan)
            run_episode(g, seed)
            obeyed.update(o.note_id for o in g.obeyed)
            dropped += g.notes_dropped
        assert obeyed, "the misses get through"
        assert set(obeyed) <= misses
        assert dropped > 0, "the screener catches some"


@pytest.mark.f2
class TestTheRealScreener:
    """With ``plant_copilot`` installed: the screened gullible ignores what ``scan_text`` flags."""

    def test_the_real_scan_text_decides_which_notes_are_obeyed(self) -> None:
        if importlib.util.find_spec("plant_copilot") is None:
            pytest.skip("plant_copilot is not installed")
        from plant_copilot.injection import scan_text

        misses = screener_misses(scan_text)
        assert {n.note_id for n in misses} == {n.note_id for n in cached_misses()}
        for note in NOTE_BANK:
            g = Gullible(Scripted(), True, scan_text)
            g.reset(0)
            out = g.propose(observation(quarter=32, note=note.text))
            expected = note.malicious and scan_text(note.text) is None
            assert bool(out) is expected, note.note_id
