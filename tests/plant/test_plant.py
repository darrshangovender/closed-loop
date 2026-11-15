"""P3b: the Plant protocol: observations, determinism, daily readings, alerts, notes, the Trace."""

import dataclasses
import os
import subprocess
import sys
import textwrap
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binomtest

from closed_loop.core.timegrid import DAY_Q, SHIFT_Q
from closed_loop.core.trace import Trace
from closed_loop.core.types import Cmd, Fidelity, Op, Permit, PlanCmd
from closed_loop.plant import (
    AssetView,
    JobView,
    Observation,
    PlantParams,
    Scenario,
    StepResult,
    TruthView,
    make_plant,
)
from closed_loop.plant.jobs import JobRunner
from closed_loop.plant.notes import NOTE_BANK, bank_index, note_template
from closed_loop.plant.observe import DailyObs, simulate_lives
from closed_loop.plant.params import ASSET_IDS, PARTS
from closed_loop.plant.plant import InspectResult

from ._helpers import theta_of, trace_digest
from ._scripts import CR, CV, P1, P2, C, advance, confined_job, fresh, scripted_params

REPO = Path(__file__).resolve().parents[2]


def run_all(plant, commands=()):
    """Step to the end of the episode; return all StepResults."""
    out = []
    while True:
        r = plant.step(list(commands))
        out.append(r)
        if r.done:
            return out


class TestInterfaceShape:
    def test_frozen_fields_are_present_by_name_and_order(self):
        def names(cls):
            return [f.name for f in dataclasses.fields(cls)]

        assert names(Observation) == [
            "quarter", "day", "shift", "assets", "note", "alerts", "crew_free",
            "safety_officer_on_shift", "monitor_fired",
        ]  # fmt: skip
        assert names(StepResult) == ["obs", "executed", "refused", "violations", "done"]
        assert names(AssetView)[:15] == [
            "asset_id", "op", "permit", "occupied", "gas_age_q", "alarm", "soon", "set_lo",
            "set_hi", "rul_lo", "rul_hi", "prior_only", "hi_last", "job_open", "spares",
        ]  # fmt: skip
        assert names(TruthView)[:4] == ["true_rul_days", "health_bin", "life_id", "failure_time_q"]

    def test_the_value_types_are_frozen(self):
        plant = fresh()
        obs = plant.initial_observation
        with pytest.raises(dataclasses.FrozenInstanceError):
            obs.quarter = 5  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            obs.assets[CR].op = Op.STOP  # type: ignore[misc]
        with pytest.raises(TypeError):
            obs.assets[CR] = None  # type: ignore[index]  # the mapping is read-only

    def test_make_plant_builds_f1_and_f1d_only(self):
        assert make_plant(fidelity=Fidelity.F1D).params.derate_enabled
        for bad in (Fidelity.F0, Fidelity.F2):
            with pytest.raises(ValueError):
                make_plant(fidelity=bad)
        with pytest.raises(ValueError):
            make_plant(PlantParams(horizon_q=100))

    def test_step_before_reset_and_after_the_end_raise(self):
        with pytest.raises(RuntimeError):
            make_plant().step(())
        short = make_plant(scripted_params(horizon_q=5 * DAY_Q, first_onset_min_day=1.0))
        short.reset(1, Scenario())
        results = run_all(short)
        assert len(results) == 5 * DAY_Q and results[-1].done and not results[-2].done
        with pytest.raises(RuntimeError):
            short.step(())

    def test_twin_before_reset_raises(self):
        with pytest.raises(RuntimeError):
            make_plant().twin([])


class TestNoTruthLeaks:
    FORBIDDEN = (
        "true", "truth", "health_bin", "life_id", "severity", "failure", "cause", "rate_rul", "tau",
        "latent",
    )  # fmt: skip

    def check(self, cls, seen):
        if cls in seen:
            return
        seen.add(cls)
        for f in dataclasses.fields(cls):
            for word in self.FORBIDDEN:
                assert word not in f.name.lower(), f"{cls.__name__}.{f.name}"

    def test_no_observation_field_is_a_truth_field(self):
        seen: set[type] = set()
        for cls in (Observation, AssetView, JobView, DailyObs, InspectResult, StepResult):
            self.check(cls, seen)

    def test_the_values_in_a_real_observation_are_only_the_documented_types(self):
        plant = fresh(7, p_fail_in_episode=0.6)
        allowed = (str, int, float, bool, type(None), JobView, DailyObs, InspectResult, Mapping)
        for r in run_all(plant)[:700]:
            for view in r.obs.assets.values():
                for f in dataclasses.fields(view):
                    v = getattr(view, f.name)
                    assert isinstance(v, allowed), (f.name, type(v))


class TestReset:
    def test_initial_observation(self):
        plant = make_plant(PlantParams())
        obs = plant.reset(9, Scenario())
        assert (obs.quarter, obs.day, obs.shift) == (0, 0, 0)
        assert obs.note is None and obs.alerts == () and not obs.monitor_fired
        assert obs.crew_free == plant.params.crew_per_shift and obs.safety_officer_on_shift
        assert set(obs.assets) == set(ASSET_IDS)
        for v in obs.assets.values():
            assert (v.op, v.permit, v.occupied, v.alarm, v.job_open) == (
                Op.RUN,
                Permit.NONE,
                False,
                False,
                False,
            )
            assert (v.set_lo, v.set_hi, v.rul_lo, v.rul_hi, v.prior_only) == (
                0,
                3,
                None,
                None,
                False,
            )
            assert v.last_obs is not None and v.last_obs.day == 0
            assert v.hi_last == v.last_obs.hi
        assert obs.assets[CR].spares == {"SYN-LNR-5001": 1, "SYN-BRG-5001": 1}
        assert obs.assets[P1].spares == {
            p: PARTS[p].stock for p in ("SYN-BRG-2001", "SYN-IMP-2001")
        }

    def test_a_part_taken_for_one_pump_shows_in_the_view_of_the_other_at_once(self):
        # The pumps share their spares. The view of the idle pump used to keep the old stock until
        # something else touched that pump, so a controller could plan on a part that had gone.
        plant = fresh(5, approver_catch_p=0.0)
        part = "SYN-BRG-2001"
        runner = JobRunner()
        obs = plant.step([C(PlanCmd.RAISE_WO, P1, "BEARING_DEFECT")]).obs
        for _ in range(200):
            view = obs.assets[P1]
            cmd = runner.next_command(view, view.job, obs.quarter) if view.job else None
            obs = plant.step([cmd] if cmd is not None else []).obs
            if obs.assets[P1].spares[part] < PARTS[part].stock:
                break
        assert obs.assets[P1].spares[part] == PARTS[part].stock - 1
        assert obs.assets[P2].spares[part] == obs.assets[P1].spares[part]

    def test_reset_twice_gives_the_same_world(self):
        plant = make_plant(scripted_params())
        plant.reset(4, Scenario())
        run_all(plant)
        d1 = trace_digest(plant.trace())
        plant.reset(4, Scenario())
        run_all(plant)
        assert trace_digest(plant.trace()) == d1
        plant.reset(5, Scenario())
        run_all(plant)
        assert trace_digest(plant.trace()) != d1


class TestDeterminism:
    def test_same_seed_same_trace_bit_for_bit(self):
        traces = []
        for _ in range(2):
            p = fresh(11, p_fail_in_episode=0.6)
            advance(p, confined_job("LINER_WEAR"), until=200)
            traces.append(p.trace())
        assert traces[0].identical_to(traces[1])

    def test_identical_across_processes_and_hash_seeds(self, tmp_path):
        script = tmp_path / "digest.py"
        script.write_text(
            textwrap.dedent(
                f"""
                import sys
                sys.path.insert(0, {str(REPO)!r})
                from tests.plant._helpers import trace_digest
                from tests.plant._scripts import advance, confined_job, fresh
                p = fresh(11, p_fail_in_episode=0.6)
                advance(p, confined_job("LINER_WEAR"), until=400)
                print(trace_digest(p.trace()))
                """
            ),
            encoding="utf-8",
        )
        local = fresh(11, p_fail_in_episode=0.6)
        advance(local, confined_job("LINER_WEAR"), until=400)
        want = trace_digest(local.trace())
        for hash_seed in ("1", "2"):
            done = subprocess.run(
                [sys.executable, "-I", "-B", str(script)],
                capture_output=True, text=True, timeout=180,
                env={**os.environ, "PYTHONHASHSEED": hash_seed},
            )  # fmt: skip
            assert done.returncode == 0, done.stderr
            assert done.stdout.strip() == want


class TestDailyReadings:
    def test_a_reading_appears_at_the_first_quarter_of_its_day_and_holds_all_day(self):
        plant = fresh(3)
        seen = {}
        for _ in range(DAY_Q * 4):
            r = plant.step(())
            obs = r.obs.assets[CV].last_obs
            seen.setdefault(r.obs.quarter, obs)
        for q in range(1, DAY_Q * 4):
            now, prev = seen[q], seen[q - 1] if q - 1 in seen else None
            assert now is not None and now.day == q // DAY_Q
            if prev is not None and q % DAY_Q != 0:
                assert now is prev

    def test_with_no_commands_the_plant_reads_exactly_what_the_life_table_says(self):
        params = PlantParams()
        for seed in (0, 1, 2):
            table = simulate_lives(params, seed, Scenario())
            plant = make_plant(params)
            plant.reset(seed, Scenario())
            for day in range(params.horizon_days):
                if day > 0:
                    for _ in range(DAY_Q - 1):
                        plant.step(())
                    r = plant.step(())
                    assert r.obs.quarter == day * DAY_Q
                    views = r.obs.assets
                else:
                    views = plant.initial_observation.assets
                for aid in ASSET_IDS:
                    o = views[aid].last_obs
                    assert o is not None and o.day == day
                    want_hi = table.hi[aid][day]
                    assert (o.hi is None and np.isnan(want_hi)) or o.hi == want_hi
                    assert (o.sig_bearing, o.sig_proc, o.sig_hyd) == tuple(table.sig[aid][:, day])
                    assert o.load == table.load[aid][day]
                    assert plant.truth.true_rul_days[aid] == pytest.approx(
                        table.true_rul_days[aid][day], abs=1e-9
                    )

    def test_a_failed_asset_keeps_reading_its_failed_severity(self):
        params = scripted_params(p_fail_in_episode=1.0)
        plant = make_plant(params)
        plant.reset(2, Scenario())
        table = simulate_lives(params, 2, Scenario())
        last = run_all(plant)[-1].obs.assets
        for aid in ASSET_IDS:
            o = last[aid].last_obs
            assert o is not None and o.day == 89
            want = table.hi[aid][-1]
            assert (o.hi is None and np.isnan(want)) or o.hi == want
            assert (o.sig_bearing, o.sig_proc, o.sig_hyd) == tuple(table.sig[aid][:, -1])


class TestAlerts:
    def test_alerts_open_after_the_blind_window_with_a_cause_and_never_twice(self):
        params = PlantParams(p_refuse=0.0)
        n_alerts = 0
        causes = set()
        for seed in range(12):
            plant = make_plant(params)
            plant.reset(seed, Scenario())
            ids = []
            for r in run_all(plant):
                for a in r.obs.alerts:
                    t = plant.alert_truth(a)
                    ids.append(a)
                    causes.add(t.cause)
                    assert t.day >= params.blind_days  # the estimator is blind for 23 days
                    assert r.obs.day == t.day
                    assert t.cause in ("degradation", "sensor_fault", "process_upset")
                    assert t.mode in ("LINER_WEAR", "BEARING_DEFECT", "IMPELLER_WEAR")
            assert len(ids) == len(set(ids))
            n_alerts += len(ids)
            per_life = {}
            for a in ids:
                key = a.rsplit("-A", 1)[0]
                per_life[key] = per_life.get(key, 0) + 1
            assert all(v <= 2 for v in per_life.values())
        assert n_alerts > 10
        assert "degradation" in causes

    def test_a_longer_blind_window_silences_the_alerts(self):
        plant = make_plant(PlantParams(blind_days=1000.0, p_fail_in_episode=1.0))
        plant.reset(1, Scenario())
        assert all(not r.obs.alerts for r in run_all(plant))

    def test_process_upsets_and_sensor_faults_raise_alerts_too(self):
        upset = PlantParams(
            upset_p_per_circuit=1.0,
            upset_magnitude=(0.6, 0.6),
            p_fail_in_episode=0.0,
            p_gain_fault=0.0,
            hi_noise_sd=0.05,
        )
        causes = set()
        for seed in range(6):
            plant = make_plant(upset)
            plant.reset(seed, Scenario())
            for r in run_all(plant):
                causes.update(plant.alert_truth(a).cause for a in r.obs.alerts)
        assert "process_upset" in causes
        drift = PlantParams(p_gain_fault=1.0, gain_rate=(0.008, 0.008), p_fail_in_episode=0.0,
                            upset_p_per_circuit=0.0, hi_noise_sd=0.05)  # fmt: skip
        causes = set()
        for seed in range(6):
            plant = make_plant(drift)
            plant.reset(seed, Scenario())
            for r in run_all(plant):
                causes.update(plant.alert_truth(a).cause for a in r.obs.alerts)
        assert "sensor_fault" in causes

    def test_alert_ids_are_stable_across_policies(self):
        a = fresh(2, p_fail_in_episode=0.6)
        b = fresh(2, p_fail_in_episode=0.6)
        ids_a = [x for r in run_all(a) for x in r.obs.alerts]
        ids_b = [x for r in run_all(b, [C(Cmd.NOOP, CV)]) for x in r.obs.alerts]
        assert ids_a == ids_b


class TestNotes:
    def test_one_note_at_every_shift_start_and_none_elsewhere(self):
        plant = fresh(1)
        notes = {}
        for r in run_all(plant):
            if r.obs.note is not None:
                notes[r.obs.quarter] = r.obs.note
        assert sorted(notes) == list(range(SHIFT_Q, plant.params.horizon_q, SHIFT_Q))
        assert all(note_template(t) is not None for t in notes.values())

    def test_the_malicious_share_follows_the_parameter(self):
        for p, lo, hi in ((0.0, 0, 0), (1.0, 1, 1), (0.05, 0.02, 0.09)):
            plant = fresh(2, p_malicious_note=p)
            kinds = [note_template(r.obs.note).malicious for r in run_all(plant) if r.obs.note]
            share = sum(kinds) / len(kinds)
            assert lo <= share <= hi, (p, share)

    def test_notes_do_not_depend_on_what_the_proposer_does(self):
        a, b = fresh(6), fresh(6)
        na = [r.obs.note for r in run_all(a)]
        sched = confined_job("LINER_WEAR")
        nb = []
        while True:
            r = b.step(sched.get(b.des.q, []))
            nb.append(r.obs.note)
            if r.done:
                break
        assert na == nb

    def test_the_adversary_replaces_the_note_at_the_nearest_handover(self):
        sc = Scenario.decode(theta_of(c25=0.99, c26=0.5), "in")
        plant = fresh(1, sc)
        want = NOTE_BANK[bank_index(0.99)].text
        k = max(round(int(0.5 * 8640) / SHIFT_Q), 1)
        got = {r.obs.quarter: r.obs.note for r in run_all(plant) if r.obs.note}
        assert got[k * SHIFT_Q] == want
        assert (
            sum(1 for t in got.values() if t == want) <= 3
        )  # the routine draw may repeat it by chance


class TestPerceptionAndMonitor:
    def test_set_perception_reaches_the_view_and_the_trace_from_the_next_quarter(self):
        plant = fresh()
        plant.step(())
        plant.set_perception(CR, set_lo=1, set_hi=2, rul_lo=12.0, rul_hi=40.0, prior_only=True)
        r = plant.step(())
        v = r.obs.assets[CR]
        assert (v.set_lo, v.set_hi, v.rul_lo, v.rul_hi, v.prior_only) == (1, 2, 12.0, 40.0, True)
        assert r.obs.assets[CV].set_lo == 0 and r.obs.assets[CV].set_hi == 3
        plant.step(())
        tr = plant.trace()
        assert list(tr.set_lo[CR][:3]) == [0, 1, 1] and list(tr.set_hi[CR][:3]) == [3, 2, 2]
        assert tr.set_lo[CV].max() == 0

    def test_a_bad_set_is_refused(self):
        plant = fresh()
        for lo, hi in ((2, 1), (-1, 2), (0, 4)):
            with pytest.raises(ValueError):
                plant.set_perception(CR, set_lo=lo, set_hi=hi)

    def test_monitor_flag(self):
        plant = fresh()
        assert not plant.step(()).obs.monitor_fired
        plant.set_monitor_fired(True)
        assert plant.step(()).obs.monitor_fired


class TestTrace:
    def test_shapes_dtypes_and_defaults(self):
        plant = fresh()
        for _ in range(50):
            plant.step(())
        tr = plant.trace()
        assert isinstance(tr, Trace) and tr.n_q == 50 and tr.asset_ids == ASSET_IDS
        tr.validate()
        assert tr.throughput.dtype == np.float32 and tr.throughput.shape == (50,)
        assert (tr.op[CR] == list(Op).index(Op.RUN)).all() and (tr.set_hi[CR] == 3).all()
        assert (tr.gas_age_min[CR] == 1.0e6).all()

    def test_a_trace_is_a_prefix_of_any_later_trace(self):
        plant = fresh(4, p_fail_in_episode=0.6)
        sched = confined_job("LINER_WEAR")
        advance(plant, sched, until=99)
        early = plant.trace()
        advance(plant, sched, until=199)
        late = plant.trace()
        for aid in ASSET_IDS:
            for f in ("op", "permit", "occupied", "gas_age_min", "alarm", "iso_proven",
                      "permit_margin_min", "true_rul_days", "health_bin", "set_lo", "set_hi",
                      "uncontrolled_failure"):  # fmt: skip
                assert getattr(early, f)[aid].tobytes() == getattr(late, f)[aid][:100].tobytes(), (
                    aid,
                    f,
                )
        assert early.throughput.tobytes() == late.throughput[:100].tobytes()
        assert late.commands[: len(early.commands)] == early.commands

    def test_the_command_log_records_executed_refused_and_shield_overrides(self):
        plant = fresh(p_refuse=0.0)
        keep = C(Cmd.STOP, CV)
        swapped = C(Cmd.STOP, P1)
        bad = C(Cmd.LOCK, CR)  # a running crusher cannot be locked
        r = plant.step([keep, swapped, bad], overridden=[swapped])
        assert r.refused == (bad,)
        assert plant.trace().commands == [
            (0, keep, True, False), (0, swapped, True, True), (0, bad, False, False),
        ]  # fmt: skip

    def test_the_quiet_fast_path_gives_the_same_trace_as_the_slow_path(self):
        a, b = fresh(8, p_fail_in_episode=0.6), fresh(8, p_fail_in_episode=0.6)
        for _ in range(8500):  # past the earliest failure (day 44)
            a.step(())
            b.step([C(Cmd.NOOP, CV)])  # any command takes the full path
        ta, tb = a.trace(), b.trace()
        assert ta.commands == [] and len(tb.commands) == 8500
        assert (
            ta.uncontrolled_failure[CR].any()
            or ta.uncontrolled_failure[CV].any()
            or any(ta.uncontrolled_failure[p].any() for p in (P1, P2))
        )
        for aid in ASSET_IDS:
            for f in (
                "op", "permit", "occupied", "gas_age_min", "alarm", "iso_proven",
                "permit_margin_min", "true_rul_days", "health_bin", "uncontrolled_failure",
            ):  # fmt: skip
                assert getattr(ta, f)[aid].tobytes() == getattr(tb, f)[aid].tobytes(), (aid, f)
        assert ta.throughput.tobytes() == tb.throughput.tobytes()

    def test_true_rul_in_the_trace_matches_the_life_table_without_commands(self):
        params = scripted_params(p_fail_in_episode=0.6)
        plant = make_plant(params)
        plant.reset(3, Scenario())
        run_all(plant)
        tr = plant.trace()
        table = simulate_lives(params, 3, Scenario())
        for aid in ASSET_IDS:
            for d in (1, 10, 40, 70, 89):
                # row r is the end of step r: the truth is read at the start of quarter r + 1
                assert tr.true_rul_days[aid][d * DAY_Q - 1] == pytest.approx(
                    table.true_rul_days[aid][d], abs=1e-3
                ), (aid, d)

    def test_health_bin_falls_through_the_four_bins_before_a_failure(self):
        params = scripted_params(p_fail_in_episode=1.0)
        plant = make_plant(params)
        plant.reset(3, Scenario())
        run_all(plant)
        tr = plant.trace()
        for aid in ASSET_IDS:
            bins = tr.health_bin[aid]
            if tr.uncontrolled_failure[aid].any():
                seen = [int(b) for i, b in enumerate(bins) if i == 0 or bins[i - 1] != b]
                assert seen[-1] == 3 and set(seen) <= {0, 1, 2, 3}
                # no bin is ever skipped in the in-assumption regime (A_phys), though a bin edge can
                # be crossed back and forth by the day-to-day load wander
                assert all(abs(b - a) == 1 for a, b in zip(seen, seen[1:], strict=False)), seen


class TestAPhysAtThePlantLevel:
    """A_phys (0001 section 4.2): the health bin rises by at most one per tick.

    In regime ``in`` that holds by construction; regime ``out`` can break it with a large load step,
    which is the violation the conformance test of P5c is meant to measure.
    """

    @staticmethod
    def biggest_bin_jump(regime: str, seed: int) -> int:
        theta = theta_of(c30=1.0, c31=0.4)  # the largest load shift of the regime, on day 35
        plant = fresh(seed, Scenario.decode(theta, regime), p_fail_in_episode=1.0)
        run_all(plant)
        jumps = [
            int(np.abs(np.diff(plant.trace().health_bin[a].astype(int))).max()) for a in ASSET_IDS
        ]
        return max(jumps)

    def test_regime_in_never_skips_a_bin(self):
        assert max(self.biggest_bin_jump("in", s) for s in range(20)) <= 1

    def test_regime_out_can_skip_a_bin_so_the_check_above_has_teeth(self):
        assert max(self.biggest_bin_jump("out", s) for s in range(20)) >= 2


def test_truth_view_is_complete_and_consistent():
    plant = fresh(5, p_fail_in_episode=0.6)
    for _ in range(300):
        plant.step(())
    t = plant.truth
    assert isinstance(t, TruthView)
    for aid in ASSET_IDS:
        assert t.life_id[aid] == 0
        assert t.failure_time_q[aid] == pytest.approx(plant.des.q + t.true_rul_days[aid] * DAY_Q)
        assert 0 <= t.health_bin[aid] <= 3
        assert t.life_mode[aid] in ("LINER_WEAR", "BEARING_DEFECT", "IMPELLER_WEAR")


def test_binomial_helper_is_available_for_the_suite():
    # guards the import used by sibling test modules
    assert binomtest(5, 10, 0.5).pvalue > 0.5
