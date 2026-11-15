"""P11a: rendering a plant life as asset-health telemetry. Real asset-health code, no mocks."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from closed_loop.f2 import F2Unavailable, _siblings, render
from closed_loop.f2.render import (
    ASSET_TYPE,
    PLANT_START,
    LifeLatent,
    RenderError,
    asset_health_id,
    asset_health_shape,
    render_life,
    sim_day,
    sim_time,
)
from closed_loop.plant.params import ASSET_CLASS, ASSET_IDS, PlantParams

pd = pytest.importorskip("pandas", reason="f2 test needs the sibling repositories (pandas)")
pytestmark = pytest.mark.f2

VALID_MODE = {
    "CR-501": "LINER_WEAR",
    "CV-101": "BEARING_DEFECT",
    "PU-201": "IMPELLER_WEAR",
    "PU-202": "BEARING_DEFECT",
}


def latent(asset: str = "PU-201", **kw) -> LifeLatent:
    base = dict(
        asset_id=asset,
        life_id=0,
        mode=VALID_MODE[asset],
        onset_day=24.0,
        failure_day=60.0,
        shape=2.0,
        downtime_hours=48.0,
    )
    base.update(kw)
    return LifeLatent(**base)


class TestIds:
    def test_asset_health_ids_cover_the_crusher_and_the_pulley(self) -> None:
        """0001 section 15 item 3: ID_PREFIX has entries for the crusher and the pulley."""
        assert {a: asset_health_id(a) for a in ASSET_IDS} == {
            "CR-501": "CRU-01",
            "CV-101": "PUL-01",
            "PU-201": "PMP-01",
            "PU-202": "PMP-02",
        }

    def test_the_prefix_table_is_the_siblings_own(self) -> None:
        from asset_health.models import AssetType
        from asset_health.simulate.fleet import ID_PREFIX

        assert ID_PREFIX[AssetType.PRIMARY_CRUSHER] == "CRU"
        assert ID_PREFIX[AssetType.CONVEYOR_PULLEY] == "PUL"
        for plant_id, type_name in ASSET_TYPE.items():
            assert asset_health_id(plant_id).startswith(ID_PREFIX[AssetType(type_name)] + "-")

    def test_an_unknown_plant_asset_is_refused(self) -> None:
        with pytest.raises(RenderError, match="unknown plant asset"):
            asset_health_id("CR-999")

    def test_the_type_table_covers_every_plant_asset(self) -> None:
        assert set(ASSET_TYPE) == set(ASSET_IDS) == set(ASSET_CLASS)


class TestShape:
    @pytest.mark.parametrize("k", [1.2, 1.8, 2.5, 3.0, 4.5])
    def test_the_generators_curve_follows_the_plants_power_law(self, k: float) -> None:
        kappa = asset_health_shape(k)
        u = np.linspace(0.0, 1.0, 201)
        gap = np.max(np.abs(np.expm1(kappa * u) / np.expm1(kappa) - u**k))
        assert gap < 0.04

    def test_a_steeper_plant_curve_gives_a_steeper_generator_curve(self) -> None:
        kappas = [asset_health_shape(k) for k in (1.2, 1.8, 2.5, 3.0, 4.5)]
        assert kappas == sorted(kappas)

    def test_a_non_positive_shape_is_refused(self) -> None:
        with pytest.raises(RenderError):
            asset_health_shape(0.0)


class TestLifeLatent:
    def test_a_mode_the_asset_class_does_not_have_is_refused(self) -> None:
        with pytest.raises(RenderError, match="has no failure mode"):
            latent("CR-501", mode="IMPELLER_WEAR")

    def test_failure_must_follow_onset(self) -> None:
        with pytest.raises(RenderError, match="after onset_day"):
            latent(onset_day=30.0, failure_day=30.0)

    def test_the_piece_must_have_a_positive_length(self) -> None:
        with pytest.raises(RenderError, match="end_day"):
            latent(start_day=10.0, end_day=10.0)

    def test_a_piece_may_not_start_after_the_failure(self) -> None:
        with pytest.raises(RenderError, match="after the life has failed"):
            latent(failure_day=45.0, start_day=50.0, end_day=60.0)

    def test_the_stop_is_never_past_the_failure(self) -> None:
        assert latent(end_day=70.0).stop_day == 60.0
        assert latent(end_day=40.0).stop_day == 40.0
        assert latent().stop_day == 60.0

    def test_an_unknown_asset_is_refused(self) -> None:
        with pytest.raises(RenderError, match="unknown plant asset"):
            LifeLatent("XX-1", 0, "LINER_WEAR", 1.0, 2.0, 2.0)


class TestRender:
    def test_every_plant_asset_renders(self, params: PlantParams) -> None:
        for asset in ASSET_IDS:
            r = render_life(latent(asset), params, 7, 0)
            assert r.ah_asset_id == asset_health_id(asset)
            assert r.n_rows > 24 * 50
            assert r.spec.asset_id == r.ah_asset_id
            assert r.failure.asset_id == r.ah_asset_id

    def test_rendering_is_deterministic_for_a_seed(self, params: PlantParams) -> None:
        a = render_life(latent(), params, 7, 0)
        render._full_record.cache_clear()
        render._spec_for.cache_clear()
        b = render_life(latent(), params, 7, 0)
        pd.testing.assert_frame_equal(a.frame, b.frame)
        assert a.spec == b.spec

    def test_a_different_seed_or_life_gives_different_telemetry(self, params: PlantParams) -> None:
        base = render_life(latent(), params, 7, 0).frame
        assert not base.equals(render_life(latent(), params, 8, 0).frame)
        assert not base.equals(render_life(latent(life_id=1), params, 7, 0).frame)

    def test_the_columns_are_asset_healths_own_schema(self, params: PlantParams) -> None:
        from asset_health.config import SimConfig
        from asset_health.models import AssetType
        from asset_health.simulate.fleet import generate_fleet

        types = {
            AssetType.SLURRY_PUMP: 1,
            AssetType.PRIMARY_CRUSHER: 1,
            AssetType.CONVEYOR_PULLEY: 1,
        }
        fleet = generate_fleet(SimConfig(seed=1, days=61, fleet=types), keep_latent=False)
        native = {s.asset_type.value: fleet.raw[s.asset_id] for s in fleet.specs}
        for asset in ("PU-201", "CR-501", "CV-101"):
            ours = render_life(latent(asset), params, 7, 0)
            theirs = native[ASSET_TYPE[asset]]
            assert list(ours.frame.columns) == list(theirs.columns)
            assert (ours.frame.dtypes == theirs.dtypes).all()

    def test_a_piece_is_a_cut_of_the_same_record(self, params: PlantParams) -> None:
        """Telemetry already shown never depends on when the policy stops (0001 section 3)."""
        short = render_life(latent(end_day=30.0), params, 7, 0).frame
        long = render_life(latent(end_day=50.0), params, 7, 0).frame
        pd.testing.assert_frame_equal(short, long.iloc[: len(short)].reset_index(drop=True))
        assert len(long) > len(short)

    def test_a_restart_piece_is_the_tail_of_the_same_record(self, params: PlantParams) -> None:
        whole = render_life(latent(end_day=50.0), params, 7, 0).frame
        tail = render_life(latent(start_day=20.0, end_day=50.0), params, 7, 0).frame
        pd.testing.assert_frame_equal(
            tail, whole.iloc[len(whole) - len(tail) :].reset_index(drop=True)
        )
        assert len(tail) < len(whole)

    def test_pieces_start_and_stop_on_the_hour_and_rows_sit_on_the_grid(
        self, params: PlantParams
    ) -> None:
        r = render_life(latent(start_day=20.37, end_day=45.61), params, 7, 0)
        assert r.start.minute == r.start.second == 0 and r.stop.minute == r.stop.second == 0
        assert r.start >= sim_time(20.37) and r.start - sim_time(20.37) < timedelta(hours=1)
        assert r.stop >= sim_time(45.61) and r.stop - sim_time(45.61) < timedelta(hours=1)
        first = pd.Timestamp(r.frame["timestamp"].iloc[0])
        last = pd.Timestamp(r.frame["timestamp"].iloc[-1])
        assert abs(first - pd.Timestamp(r.start)) < pd.Timedelta(minutes=1)
        assert last < pd.Timestamp(r.stop)

    def test_no_logger_clock_is_rendered(self, params: PlantParams) -> None:
        """Reported timestamps sit within seconds of their slot, so cutting by them is safe."""
        r = render_life(latent(), params, 11, 0)
        assert not [e for e in r.events if e.kind == "clock_drift"]
        stamps = pd.DatetimeIndex(r.frame["timestamp"])
        off = (stamps - stamps.round("h")).total_seconds()
        assert np.abs(off).max() < 15

    def test_the_failure_event_carries_the_latent(self, params: PlantParams) -> None:
        lat = latent("PU-202", onset_day=30.0, failure_day=70.0, downtime_hours=61.0)
        r = render_life(lat, params, 7, 0)
        assert r.failure.mode.value == "bearing_spall"
        assert r.failure.onset == sim_time(30.0) and r.failure.failure == sim_time(70.0)
        assert r.failure.downtime_hours == 61.0
        assert r.failure.shape_k == pytest.approx(asset_health_shape(2.0))
        assert r.failure.race in ("outer", "inner")
        liner = render_life(latent("CR-501"), params, 7, 0)
        assert liner.failure.mode.value == "liner_wear" and liner.failure.race is None

    def test_a_piece_cannot_run_past_the_failure(self, params: PlantParams) -> None:
        r = render_life(latent(failure_day=45.0, end_day=60.0), params, 7, 0)
        assert r.stop - sim_time(45.0) < timedelta(hours=1)

    def test_a_failure_beyond_the_horizon_is_renderable(self, params: PlantParams) -> None:
        r = render_life(latent(onset_day=70.0, failure_day=120.0), params, 7, 0)
        assert r.stop == sim_time(params.horizon_days)

    def test_no_planned_service_is_written_into_the_telemetry(self, params: PlantParams) -> None:
        # the plant has no lubrication round; the log holds repairs at joins only (splice.py)
        r = render_life(latent(), params, 7, 0)
        assert r.frame["timestamp"].is_monotonic_increasing

    def test_the_degradation_is_visible_in_the_signature_channels(
        self, params: PlantParams
    ) -> None:
        """The worn impeller raises the vane-pass band, which is what the rule layer reads."""
        r = render_life(latent(onset_day=24.0, failure_day=45.0), params, 7, 0)
        stamps = pd.DatetimeIndex(r.frame["timestamp"])
        early = r.frame.loc[stamps < PLANT_START + timedelta(days=20), "vib_band_proc"]
        late = r.frame.loc[stamps >= PLANT_START + timedelta(days=42), "vib_band_proc"]
        assert late.median() > 2.0 * early.median()

    def test_a_negative_segment_is_refused(self, params: PlantParams) -> None:
        with pytest.raises(RenderError, match="segment"):
            render_life(latent(), params, 7, -1)


class TestTimeHelpers:
    def test_plant_time_starts_at_six(self) -> None:
        assert PLANT_START.hour == 6 and PLANT_START.date().isoformat() == "2026-10-01"
        assert sim_day(sim_time(12.25)) == pytest.approx(12.25)


class TestWithoutTheSiblings:
    def test_every_public_function_raises_f2_unavailable(
        self, monkeypatch: pytest.MonkeyPatch, params: PlantParams
    ) -> None:
        def refuse(name: str):
            raise ImportError(f"No module named {name!r}")

        render._full_record.cache_clear()
        render._spec_for.cache_clear()
        monkeypatch.setattr(_siblings.importlib, "import_module", refuse)
        with pytest.raises(F2Unavailable, match="never a bare package name"):
            asset_health_id("PU-201")
        with pytest.raises(F2Unavailable):
            render_life(latent(), params, 1, 0)
        render._full_record.cache_clear()
        render._spec_for.cache_clear()
