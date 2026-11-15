"""The shield table: the formula, widths 1 to 4, serialisation and a hostile command stream."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from itertools import product
from pathlib import Path

import pytest

from closed_loop.core.rng import rng_for
from closed_loop.shield.abstraction import VIOL, CrusherJobModel, PumpPairModel
from closed_loop.shield.game import Winning, solve
from closed_loop.shield.table import (
    ShieldTable,
    build_shield,
    default_priority,
    matches_model,
)
from tests.shield.brute import TinyGame, critical_run_violations, expected_entry, unsafe_allowed

CRUSHER_ENTRIES = {1: 1693, 2: 2989, 3: 3853, 4: 4285}
PUMP_ENTRIES = {1: 320, 2: 980, 3: 1620, 4: 2000}  # 20 observables in W, ADR 0003 A12.2
CONVEYOR_ENTRIES = {1: 24, 2: 42, 3: 54, 4: 60}  # 6 observables times the intervals of 1D bins


class TestShape:
    @pytest.mark.parametrize("width", [1, 2, 3, 4])
    def test_entry_counts_and_no_empty_entry(
        self,
        width: int,
        crusher_tables: dict[int, ShieldTable],
        pump_tables: dict[int, ShieldTable],
        conveyor_tables: dict[int, ShieldTable],
    ) -> None:
        assert len(crusher_tables[width].entries) == CRUSHER_ENTRIES[width]
        assert len(pump_tables[width].entries) == PUMP_ENTRIES[width]
        assert len(conveyor_tables[width].entries) == CONVEYOR_ENTRIES[width]
        assert all(crusher_tables[width].entries.values())  # no deadlock entry
        assert all(pump_tables[width].entries.values())
        assert all(conveyor_tables[width].entries.values())

    @pytest.mark.parametrize("width", [1, 2, 3, 4])
    def test_no_set_is_wider_than_the_limit(
        self,
        width: int,
        crusher_tables: dict[int, ShieldTable],
        pump_tables: dict[int, ShieldTable],
    ) -> None:
        for lo, hi in (hs for _o, hs in crusher_tables[width].entries):
            assert 0 <= lo <= hi <= 3 and hi - lo + 1 <= width
        for hs in (hs for _o, hs in pump_tables[width].entries):
            assert len(hs) == 4
            assert all(hi - lo + 1 <= width for lo, hi in (hs[:2], hs[2:]))

    def test_mean_safe_commands_fall_as_the_set_widens(
        self, crusher_tables: dict[int, ShieldTable]
    ) -> None:
        means = []
        for width in (1, 2, 3, 4):
            sizes = [
                len(v)
                for (_o, (lo, hi)), v in crusher_tables[width].entries.items()
                if hi - lo + 1 == width
            ]
            means.append(sum(sizes) / len(sizes))
        assert [round(m, 2) for m in means] == [6.21, 6.02, 5.84, 5.34]
        assert means == sorted(means, reverse=True)

    def test_widths_are_nested(self, crusher_tables: dict[int, ShieldTable]) -> None:
        for width in (1, 2, 3):
            narrow, wide = crusher_tables[width].entries, crusher_tables[width + 1].entries
            assert all(wide[k] == v for k, v in narrow.items())

    def test_stale_spec_shield_is_buildable(self, crusher_tables: dict[int, ShieldTable]) -> None:
        model = CrusherJobModel(retest_q=8)
        table = build_shield(model, solve(model), 4, provenance={"revision": "2"})
        fresh = crusher_tables[4]
        assert table.meta["model.retest_q"] == "8" and table.meta["spec.revision"] == "2"
        assert table.model_name == "crusher_job[retest_q=8]"
        assert all(table.entries.values())
        assert table.sha256() != fresh.sha256()
        assert len({o for o, _ in table.entries}) > len({o for o, _ in fresh.entries})


class TestFormula:
    def test_every_entry_is_the_intersection_over_winning_members(
        self, crusher: CrusherJobModel, crusher_win: Winning, crusher_tables: dict[int, ShieldTable]
    ) -> None:
        for (o, hs), cmds in crusher_tables[4].entries.items():
            assert cmds == expected_entry(crusher, crusher_win, o, hs), (o, hs)

    def test_every_missing_box_has_no_winning_member(
        self, crusher: CrusherJobModel, crusher_win: Winning, crusher_tables: dict[int, ShieldTable]
    ) -> None:
        table = crusher_tables[2]
        observables = {crusher.observable(s) for s in crusher.states()}
        intervals = [(lo, hi) for lo in range(4) for hi in range(lo, min(lo + 2, 4))]
        for o, (lo, hi) in product(sorted(observables, key=str), intervals):
            if (o, (lo, hi)) not in table.entries:
                assert expected_entry(crusher, crusher_win, o, (lo, hi)) is None

    def test_pump_entries_are_the_intersection_over_winning_members(
        self, pumps: PumpPairModel, pumps_win: Winning, pump_tables: dict[int, ShieldTable]
    ) -> None:
        for (o, hs), cmds in pump_tables[3].entries.items():
            assert cmds == expected_entry(pumps, pumps_win, o, hs), (o, hs)

    @pytest.mark.parametrize("width", [1, 2, 3, 4])
    def test_every_allowed_command_leaves_every_member_inside_the_winning_region(
        self,
        width: int,
        crusher: CrusherJobModel,
        crusher_win: Winning,
        crusher_tables: dict[int, ShieldTable],
        pumps: PumpPairModel,
        pumps_win: Winning,
        pump_tables: dict[int, ShieldTable],
    ) -> None:
        assert unsafe_allowed(crusher_tables[width], crusher, crusher_win) == []
        assert unsafe_allowed(pump_tables[width], pumps, pumps_win) == []

    def test_a_wider_set_never_allows_more(self, crusher_tables: dict[int, ShieldTable]) -> None:
        entries = crusher_tables[4].entries
        for (o, (lo, hi)), cmds in entries.items():
            if hi > lo:
                assert cmds <= entries.get((o, (lo, hi - 1)), cmds)
                assert cmds <= entries.get((o, (lo + 1, hi)), cmds)

    def test_a_running_crusher_that_may_be_critical_can_only_be_stopped(
        self,
        crusher: CrusherJobModel,
        crusher_win: Winning,
        crusher_tables: dict[int, ShieldTable],
    ) -> None:
        # P5a: STOP is in every entry's allowed set when RUN and bin 3 is possible. Read as
        # "possible" = a winning state with bin 3 is in the set, it holds, and STOP is then all
        # that is allowed. The literal reading has exceptions: a RUN crusher with someone inside
        # at bin 3 is not winning (no command saves it), so a set that reaches bin 3 only there
        # holds the other members' commands (EXIT).
        for width in (1, 2, 3, 4):
            assert critical_run_violations(crusher_tables[width], crusher, crusher_win) == []
        critical_wins = {
            crusher.observable(s) for s in crusher_win.states if crusher.health(s) == 3
        }
        for width in (1, 2, 3, 4):
            checked = 0
            for (o, (lo, hi)), cmds in crusher_tables[width].entries.items():
                if o[0] != "RUN" or hi != 3:  # type: ignore[index]
                    continue
                if o in critical_wins:
                    assert cmds == {"STOP"}, (o, lo, hi)
                    checked += 1
                else:
                    assert o[2] == 1, (o, lo, hi)  # type: ignore[index]  # someone inside
            assert checked > 0

    def test_stop_and_exit_are_in_every_entry_that_needs_them(
        self, crusher_tables: dict[int, ShieldTable]
    ) -> None:
        # someone inside, a handover or an alarm close: EXIT is the way out
        for (o, _hs), cmds in crusher_tables[4].entries.items():
            op, _permit, occ, _gas, alarm, soon = o  # type: ignore[misc]
            if occ == 1 and (alarm == 1 or soon == 1):
                assert cmds == {"EXIT"}, o

    def test_a_pump_pair_with_one_pump_in_work_must_stop_the_other_when_it_may_fail(
        self, pump_tables: dict[int, ShieldTable]
    ) -> None:
        allowed = pump_tables[4].allowed(("WORK", "RUN"), (0, 0, 3, 3))
        assert allowed and all(c.endswith("|STOP") for c in allowed)
        assert "NOOP|NOOP" in pump_tables[4].allowed(("WORK", "RUN"), (0, 0, 0, 2))


class TestLookup:
    def test_wider_than_built_is_answered_exactly(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        for built in (1, 2, 3):
            narrow, full = crusher_tables[built], crusher_tables[4]
            for (o, hs), cmds in full.entries.items():
                assert narrow.allowed(o, hs) == cmds, (built, o, hs)
        for built in (1, 2):
            narrow, full = pump_tables[built], pump_tables[4]
            for (o, hs), cmds in list(full.entries.items())[::7]:
                assert narrow.allowed(o, hs) == cmds, (built, o, hs)

    def test_a_box_with_no_winning_member_has_no_command(
        self, crusher: CrusherJobModel, crusher_win: Winning, crusher_tables: dict[int, ShieldTable]
    ) -> None:
        # someone inside a RUN crusher at bin 3 is outside the winning region
        o = ("RUN", "NONE", 1, 3, 0, 0)
        assert o in {crusher.observable(s) for s in crusher.states()}
        assert crusher_tables[4].allowed(o, (3, 3)) == frozenset()
        assert crusher_tables[4].allowed(o, (0, 3)) != frozenset()  # the lower bins still win

    @pytest.mark.parametrize("bad", [(2, 1), (-1, 2), (0, 4), (1,), (0, 1, 2, 3), (True, 2)])
    def test_malformed_sets_are_refused(
        self, crusher_tables: dict[int, ShieldTable], bad: tuple[int, ...]
    ) -> None:
        with pytest.raises(ValueError, match="health set"):
            crusher_tables[4].allowed(("RUN", "NONE", 0, 3, 0, 0), bad)

    def test_entries_and_meta_cannot_be_edited_in_place(
        self, crusher_tables: dict[int, ShieldTable]
    ) -> None:
        table = crusher_tables[1]
        with pytest.raises(TypeError):
            table.entries[("x", (0, 0))] = frozenset({"BEGIN"})  # type: ignore[index]
        with pytest.raises(TypeError):
            table.meta["max_width"] = "9"  # type: ignore[index]
        with pytest.raises(AttributeError):
            table.priority = ()  # type: ignore[misc]


class TestPriority:
    def test_crusher_order_is_stop_exit_noop_then_the_rest(
        self, crusher_tables: dict[int, ShieldTable]
    ) -> None:
        priority = crusher_tables[4].priority
        assert priority[:3] == ("STOP", "EXIT", "NOOP")
        assert sorted(priority) == sorted(CrusherJobModel().commands())
        assert priority[3:] == tuple(
            c for c in CrusherJobModel().commands() if c not in ("STOP", "EXIT", "NOOP")
        )

    def test_pump_pair_order_ranks_joint_commands_left_to_right(
        self, pump_tables: dict[int, ShieldTable]
    ) -> None:
        priority = pump_tables[4].priority
        assert priority[:3] == ("STOP|STOP", "STOP|NOOP", "STOP|START")
        assert sorted(priority) == sorted(PumpPairModel().commands())

    def test_first_allowed_follows_the_order(self, crusher_tables: dict[int, ShieldTable]) -> None:
        table = crusher_tables[4]
        assert table.first_allowed({"TEST", "NOOP", "EXIT"}) == "EXIT"
        assert table.first_allowed({"TEST", "NOOP"}) == "NOOP"
        assert table.first_allowed(set()) is None

    def test_custom_priority_must_cover_every_command(self) -> None:
        model = CrusherJobModel()
        win = solve(model)
        with pytest.raises(ValueError, match="priority"):
            build_shield(model, win, 2, priority=("STOP", "NOOP"))
        flipped = build_shield(
            model, win, 1, priority=tuple(reversed(default_priority(model.commands())))
        )
        assert flipped.priority[-1] == "STOP"

    @pytest.mark.parametrize("width", [0, 5, -1, True])
    def test_width_is_validated(
        self, width: int, crusher: CrusherJobModel, crusher_win: Winning
    ) -> None:
        with pytest.raises(ValueError, match="max_width"):
            build_shield(crusher, crusher_win, width)

    def test_a_vacuous_shield_is_refused(self) -> None:
        class Doomed(CrusherJobModel):
            def initial(self):  # type: ignore[no-untyped-def]
                return ("RUN", 3, "NONE", 1, 3, 0, 0)  # a losing state

        model = Doomed()
        with pytest.raises(ValueError, match="initial state"):
            build_shield(model, solve(model), 4)


class TestSerialisation:
    def test_round_trip_keeps_everything_and_the_hash(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        for table in (crusher_tables[3], pump_tables[2]):
            again = ShieldTable.from_json(table.to_json())
            assert again == table
            assert again.sha256() == table.sha256()
            assert len(table.sha256()) == 64

    def test_text_is_canonical(self, crusher_tables: dict[int, ShieldTable]) -> None:
        text = crusher_tables[1].to_json()
        assert text.isascii() and "\n" not in text and " " not in text
        raw = json.loads(text)
        assert raw["format"] == "closed-loop-shield-table/1"
        assert list(raw) == sorted(raw)
        assert raw["entries"] == sorted(
            raw["entries"], key=lambda r: json.dumps(r, sort_keys=True, separators=(",", ":"))
        )

    def test_hash_is_stable_across_processes_and_hash_seeds(
        self,
        tmp_path: Path,
        crusher_tables: dict[int, ShieldTable],
        pump_tables: dict[int, ShieldTable],
    ) -> None:
        script = tmp_path / "hash_in_a_fresh_process.py"
        script.write_text(
            "import sys\n"
            "from closed_loop.shield.abstraction import CrusherJobModel, PumpPairModel\n"
            "from closed_loop.shield.game import solve\n"
            "from closed_loop.shield.table import build_shield\n"
            "c, p = CrusherJobModel(), PumpPairModel()\n"
            "print(build_shield(c, solve(c), 3).sha256())\n"
            "print(build_shield(p, solve(p), 2).sha256())\n",
            encoding="utf-8",
        )
        repo = Path(__file__).resolve().parents[2]
        seen = set()
        for hash_seed in ("0", "12345"):
            env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(repo)}
            result = subprocess.run(
                [sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=300
            )
            assert result.returncode == 0, result.stderr
            seen.add(result.stdout)
        assert len(seen) == 1
        here = f"{crusher_tables[3].sha256()}\n{pump_tables[2].sha256()}\n"
        assert seen == {here}

    def test_save_and_load_check_the_hash(
        self, tmp_path: Path, crusher_tables: dict[int, ShieldTable]
    ) -> None:
        table = crusher_tables[1]
        path = tmp_path / "crusher.json"
        table.save(path)
        assert ShieldTable.load(path, table.sha256()) == table
        with pytest.raises(ValueError, match="expected"):
            ShieldTable.load(path, "0" * 64)

    def test_a_foreign_file_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not a shield table"):
            ShieldTable.from_json('{"format": "something else"}')

    def test_meta_records_what_the_table_came_from(
        self, crusher_win: Winning, crusher_tables: dict[int, ShieldTable]
    ) -> None:
        meta = crusher_tables[3].meta
        assert meta["abstraction_version"] == "2"  # ADR 0003 A12
        assert meta["relation_sha256"] == crusher_win.relation_sha256
        assert meta["max_width"] == "3"
        assert meta["winning_states"] == "1693"
        assert crusher_tables[3].model_parameters() == {
            "kind": "crusher_job",
            "retest_q": "2",
            "spike_semantics": "false",
        }


class TestNoHandEdits:
    def test_the_true_table_matches_its_model(
        self, crusher: CrusherJobModel, crusher_tables: dict[int, ShieldTable]
    ) -> None:
        assert matches_model(crusher_tables[2], crusher)

    def test_an_edited_entry_changes_the_hash_and_fails_the_rebuild(
        self, crusher: CrusherJobModel, crusher_tables: dict[int, ShieldTable]
    ) -> None:
        raw = json.loads(crusher_tables[2].to_json())
        victim = next(r for r in raw["entries"] if r[0][0] == "LOCKED" and "BEGIN" not in r[2])
        victim[2] = sorted([*victim[2], "BEGIN"])
        forged = ShieldTable.from_json(json.dumps(raw))
        assert forged.sha256() != crusher_tables[2].sha256()
        assert not matches_model(forged, crusher)
        assert unsafe_allowed(forged, crusher, solve(crusher))  # and it really is unsafe

    def test_a_table_whose_relation_changed_under_it_fails_the_rebuild_even_if_its_entries_agree(
        self,
    ) -> None:
        # regression: the same entries can come from two relations; the digest tells them apart
        base = {("a", "go"): ["a"], ("a", "extra"): ["a", "b"], ("b", "go"): ["b"]}
        changed = {**base, ("a", "extra"): ["a"]}
        model, other = TinyGame(base), TinyGame(changed)
        table = build_shield(model, solve(model), 1)
        assert build_shield(other, solve(other), 1).entries == table.entries
        assert matches_model(table, model)
        assert not matches_model(table, other)

    def test_a_table_for_another_abstraction_fails_the_rebuild(
        self, crusher_tables: dict[int, ShieldTable]
    ) -> None:
        assert not matches_model(crusher_tables[2], CrusherJobModel(retest_q=3))


def _hostile_run(
    model: CrusherJobModel | PumpPairModel,
    win: Winning,
    table: ShieldTable | None,
    steps: int,
    seed: int,
) -> tuple[int, int]:
    """Random commands against random environment choices; returns (violations, overrides).

    Without a table every command is applied as proposed; with one, a command outside the allowed
    set for a random perception set containing the true bins is replaced by the first allowed one.
    After a violation the run restarts from the initial state.
    """
    rng = rng_for(seed, "hostile-stream", model.name)
    commands = list(model.commands())
    cache: dict[tuple[object, str], list[object]] = {}
    state = model.initial()
    violations = overrides = 0
    for _ in range(steps):
        cmd = commands[int(rng.integers(len(commands)))]
        if table is not None:
            bins = model.health(state)
            bins_t = (bins,) if isinstance(bins, int) else tuple(bins)
            sets: list[int] = []
            for h in bins_t:  # a random interval of up to 4 bins that contains the true bin
                lo = int(rng.integers(0, h + 1))
                hi = int(rng.integers(h, 4))
                sets += [lo, hi]
            allowed = table.allowed(model.observable(state), sets)
            if cmd not in allowed:
                overrides += 1
                cmd = table.first_allowed(allowed) or table.priority[0]
        key = (state, cmd)
        if key not in cache:
            cache[key] = sorted(model.step(state, cmd), key=repr)
        nxt = cache[key][int(rng.integers(len(cache[key])))]
        if nxt == VIOL:
            violations += 1
            state = model.initial()
        else:
            assert table is None or nxt in win.states
            state = nxt
    return violations, overrides


class TestHostileStream:
    """P5a: 10^5 random commands never reach VIOL when filtered; unfiltered, they do."""

    def test_crusher(
        self, crusher: CrusherJobModel, crusher_win: Winning, crusher_tables: dict[int, ShieldTable]
    ) -> None:
        violations, overrides = _hostile_run(crusher, crusher_win, crusher_tables[4], 100_000, 1)
        assert violations == 0
        assert overrides > 10_000  # the filter was working, not idle
        raw_violations, _ = _hostile_run(crusher, crusher_win, None, 100_000, 1)
        assert raw_violations > 1_000  # the stream has teeth

    def test_pump_pair(
        self, pumps: PumpPairModel, pumps_win: Winning, pump_tables: dict[int, ShieldTable]
    ) -> None:
        violations, overrides = _hostile_run(pumps, pumps_win, pump_tables[4], 100_000, 2)
        assert violations == 0 and overrides > 10_000
        raw_violations, _ = _hostile_run(pumps, pumps_win, None, 100_000, 2)
        assert raw_violations > 1_000

    @pytest.mark.parametrize("width", [1, 2])
    def test_narrow_tables_hold_for_sets_no_wider_than_they_were_built_for(
        self,
        width: int,
        crusher: CrusherJobModel,
        crusher_win: Winning,
        crusher_tables: dict[int, ShieldTable],
    ) -> None:
        # a wider set than the table's limit is still answered exactly, so safety does not depend
        # on the limit; only the sets drawn in _hostile_run (up to 4 bins) are asked here
        violations, _ = _hostile_run(crusher, crusher_win, crusher_tables[width], 20_000, 3 + width)
        assert violations == 0
