"""P3a: PlantParams and the SYNTHETIC catalogue."""

import ast
import dataclasses
from pathlib import Path

import pytest

from closed_loop.core.timegrid import HORIZON_Q
from closed_loop.plant import params as params_module
from closed_loop.plant.params import (
    ASSET_CLASS,
    ASSET_IDS,
    CONFINED_JOBS,
    JOB_WORK_Q,
    MODES_BY_CLASS,
    PARTS,
    PlantParams,
)


class TestDefaults:
    def test_interface_defaults_are_the_0002_values(self):
        p = PlantParams()
        assert p.horizon_q == 8640 == HORIZON_Q
        assert p.beta_load == 0.8
        assert p.p_fail_in_episode == 0.6
        assert p.holder_present_p == 0.9
        assert p.gas_excursion_rate == 0.01
        assert p.unplanned_cost_ratio == 8.0
        assert p.approver_median_h == 2.0
        assert p.approver_catch_p == 0.5
        assert p.blind_days == 23.0
        assert p.single_pump_throughput == 0.6
        assert p.derate_enabled is False
        assert p.horizon_days == 90

    def test_defaults_validate(self):
        PlantParams().validate()

    def test_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            PlantParams().beta_load = 1.0  # type: ignore[misc]

    def test_the_a3_signature_coefficients_are_stated_and_mode_specific(self):
        p = PlantParams()
        # bearing severity mostly moves sig_bearing; wear mostly moves sig_proc and sig_hyd
        assert p.sig_loading_bearing[0] > 5 * max(p.sig_loading_bearing[1:])
        assert p.sig_loading_wear[0] * 4 < min(p.sig_loading_wear[1:])
        assert all(c > 0 for c in p.sig_load_coef) and p.hi_load_coef > 0


class TestEveryFieldIsDocumented:
    def test_each_field_has_an_attribute_docstring_naming_a_source(self):
        src = Path(params_module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PlantParams")
        docs: dict[str, str] = {}
        body = cls.body
        for node, nxt in zip(body, body[1:], strict=False):
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and isinstance(nxt, ast.Expr)
                and isinstance(nxt.value, ast.Constant)
                and isinstance(nxt.value.value, str)
            ):
                docs[node.target.id] = nxt.value.value
        fields = {f.name for f in dataclasses.fields(PlantParams)}
        assert fields - set(docs) == set(), "fields without an attribute docstring"
        # every doc names where the number comes from: a source tag, a section or "A"
        tags = ("(S", "(A", "0001", "0003", "PREREGISTRATION", "(0002", "CORP-HSE", "S:")
        weak = [n for n, d in docs.items() if not any(t in d for t in tags)]
        assert weak == [], f"docstrings that name no source: {weak}"


class TestValidate:
    @pytest.mark.parametrize(
        "bad",
        [
            {"horizon_q": 0},
            {"horizon_q": 100},
            {"p_fail_in_episode": 1.5},
            {"p_refuse": -0.1},
            {"holder_present_p": 2.0},
            {"beta_load": -1.0},
            {"load_phi": 1.0},
            {"hi_fail": 0.5},
            {"hi_alert": 0.5},
            {"first_onset_min_day": 89.0},
            {"gain_rate": (0.01, 0.001)},
            {"upset_days": (0, 3)},
            {"gas_retest_q": 0},
            {"crew_per_shift": 0},
            {"permit_max_h": 0.0},
            {"downtime_hours": (96.0, 36.0)},
        ],
    )
    def test_refuses(self, bad):
        with pytest.raises(ValueError):
            PlantParams(**bad).validate()


class TestCatalogue:
    def test_every_asset_has_a_class_with_modes(self):
        for aid in ASSET_IDS:
            assert ASSET_CLASS[aid] in MODES_BY_CLASS

    def test_weights_sum_to_one_and_ranges_are_ordered(self):
        for cls, modes in MODES_BY_CLASS.items():
            assert abs(sum(m.weight for m in modes) - 1.0) < 1e-12, cls
            for m in modes:
                assert m.duration_days[0] < m.duration_days[1]
                assert m.shape[0] < m.shape[1]
                assert m.signature in ("bearing", "wear")

    def test_catalogue_matches_0001_section_2(self):
        crush = {m.mode: m for m in MODES_BY_CLASS["CRUSHER"]}
        assert crush["LINER_WEAR"].weight == 0.7 and crush["LINER_WEAR"].duration_days == (45, 85)
        assert crush["BEARING_DEFECT"].duration_days == (25, 55)
        pump = {m.mode: m for m in MODES_BY_CLASS["SLURRY_PUMP"]}
        assert pump["IMPELLER_WEAR"].duration_days == (40, 75)
        assert pump["BEARING_DEFECT"].duration_days == (20, 45)
        conv = MODES_BY_CLASS["BELT_CONVEYOR"]
        assert [m.mode for m in conv] == ["BEARING_DEFECT"] and conv[0].duration_days == (25, 55)
        for cls, modes in MODES_BY_CLASS.items():
            for m in modes:
                want = (2.0, 4.5) if m.signature == "bearing" else (1.2, 3.0)
                assert m.shape == want, (cls, m.mode)

    def test_parts_and_work_tables_cover_every_mode(self):
        for cls, modes in MODES_BY_CLASS.items():
            for m in modes:
                assert m.part_no in PARTS, (cls, m.mode)
                assert JOB_WORK_Q[(cls, m.mode)] > 0
        assert CONFINED_JOBS <= set(JOB_WORK_Q)

    def test_the_crusher_liner_job_spans_a_shift(self):
        # it must cross a handover so that the REVAL path is exercised (see params.py)
        assert JOB_WORK_Q[("CRUSHER", "LINER_WEAR")] > 32 // 2

    def test_no_sibling_import_in_the_module(self):
        text = Path(params_module.__file__).read_text(encoding="utf-8")
        for name in ("asset_health", "maintenance_agent", "plant_copilot", "pandas", "sklearn"):
            assert f"import {name}" not in text and f"from {name}" not in text
