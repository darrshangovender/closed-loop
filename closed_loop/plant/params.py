"""Plant parameters and the fixed catalogue of the SYNTHETIC plant.

Everything here is SYNTHETIC: invented assets, invented failure laws and invented part numbers (in
the style of the maintenance-agent fixtures, so that the F2 adapters can map onto them). No real
plant, client or employer is described.

Source tags used in the field docstrings, as in docs/adr/0001-architecture.md: (S) read from sibling
source code, (A) an assumption or target that a sweep or a measurement is meant to replace, and a
section number where 0001 states the value. The Batch 2 additions of docs/adr/0003 (A3: the daily
observation carries signature channels) are tagged "0003 A3".

Each field of ``PlantParams`` has an attribute docstring (the string literal under it);
``tests/plant/test_params.py`` fails if one is missing.
"""

from dataclasses import dataclass

from closed_loop.core.timegrid import DAY_Q, HORIZON_Q

ASSET_IDS: tuple[str, ...] = ("CR-501", "CV-101", "PU-201", "PU-202")
CRUSHER = "CR-501"
CONVEYOR = "CV-101"
PUMPS: tuple[str, str] = ("PU-201", "PU-202")

ASSET_CLASS: dict[str, str] = {
    "CR-501": "CRUSHER",
    "CV-101": "BELT_CONVEYOR",
    "PU-201": "SLURRY_PUMP",
    "PU-202": "SLURRY_PUMP",
}
# Process circuit of each asset: the load excursions U_load(t) are shared inside a circuit
# (0001 section 2: "a process exogenous U_load(t) shared by assets in the same circuit").
CIRCUIT: dict[str, int] = {"CR-501": 0, "CV-101": 0, "PU-201": 1, "PU-202": 1}
N_CIRCUITS = 2

# Signature families (0003 A3): bearing faults move sig_bearing; wear (impeller, liner) moves
# sig_proc and sig_hyd.
SIG_BEARING = "bearing"
SIG_WEAR = "wear"


@dataclass(frozen=True)
class ModeSpec:
    """One failure mode of an asset class: weights and ranges (0001 section 2, S: asset-health)."""

    mode: str
    weight: float
    duration_days: tuple[float, float]  # onset to failure, nominal clock days
    shape: tuple[float, float]  # severity exponent k
    signature: str  # SIG_BEARING or SIG_WEAR
    part_no: str  # the spare this repair consumes


# Failure modes by asset class. Names are maintenance-agent's (S: models.FailureMode), so that
# requirement(asset_class, mode, ...) answers for them. Weights, duration ranges and shape ranges
# are the asset-health catalogue (S, via 0001 section 2): bearings k in 2.0-4.5, the rest 1.2-3.0.
MODES_BY_CLASS: dict[str, tuple[ModeSpec, ...]] = {
    "CRUSHER": (
        ModeSpec("LINER_WEAR", 0.7, (45.0, 85.0), (1.2, 3.0), SIG_WEAR, "SYN-LNR-5001"),
        ModeSpec("BEARING_DEFECT", 0.3, (25.0, 55.0), (2.0, 4.5), SIG_BEARING, "SYN-BRG-5001"),
    ),
    "BELT_CONVEYOR": (
        ModeSpec("BEARING_DEFECT", 1.0, (25.0, 55.0), (2.0, 4.5), SIG_BEARING, "SYN-BRG-1001"),
    ),
    "SLURRY_PUMP": (
        ModeSpec("IMPELLER_WEAR", 0.7, (40.0, 75.0), (1.2, 3.0), SIG_WEAR, "SYN-IMP-2001"),
        ModeSpec("BEARING_DEFECT", 0.3, (20.0, 45.0), (2.0, 4.5), SIG_BEARING, "SYN-BRG-2001"),
    ),
}


@dataclass(frozen=True)
class PartSpec:
    part_no: str
    stock: int  # on hand at the start of an episode
    lead_days: float  # order to delivery


# Stock and lead times copied from the maintenance-agent inventory fixture (S: fixtures/
# inventory.json), SYNTHETIC part numbers. The two pumps share one stock.
PARTS: dict[str, PartSpec] = {
    "SYN-BRG-1001": PartSpec("SYN-BRG-1001", 6, 21.0),
    "SYN-BRG-2001": PartSpec("SYN-BRG-2001", 4, 21.0),
    "SYN-IMP-2001": PartSpec("SYN-IMP-2001", 2, 42.0),
    "SYN-BRG-5001": PartSpec("SYN-BRG-5001", 1, 120.0),
    "SYN-LNR-5001": PartSpec("SYN-LNR-5001", 1, 150.0),
}

# Quarters of work inside the isolated asset, by (class, mode) (A). The crusher liner job (10 h of
# work inside the chamber) is longer than a shift on purpose: it crosses a handover and so exercises
# permit revalidation (P2) and the exit-before-handover rule.
JOB_WORK_Q: dict[tuple[str, str], int] = {
    ("CRUSHER", "LINER_WEAR"): 40,
    ("CRUSHER", "BEARING_DEFECT"): 24,
    ("BELT_CONVEYOR", "BEARING_DEFECT"): 12,
    ("SLURRY_PUMP", "IMPELLER_WEAR"): 16,
    ("SLURRY_PUMP", "BEARING_DEFECT"): 12,
}
# Repairs that put a person inside a confined space. Vendored from maintenance-agent's requirements
# matrix (S: MODE_PERMITS has CONFINED_SPACE for CRUSHER/LINER_WEAR only); tests/plant/test_jobs.py
# compares it with requirement_for when the sibling is installed.
CONFINED_JOBS: frozenset[tuple[str, str]] = frozenset({("CRUSHER", "LINER_WEAR")})
# Classes whose chamber can be entered at all (ENTER on any other asset is refused).
ENTERABLE_CLASSES: frozenset[str] = frozenset({"CRUSHER"})
# Repairs whose permits include a high-risk one, so only a safety officer may approve them
# (S: requirements.role_for_permits; day shift only, 0001 section 2).
SAFETY_OFFICER_JOBS: frozenset[tuple[str, str]] = CONFINED_JOBS


@dataclass(frozen=True)
class PlantParams:
    """Parameters of the SYNTHETIC F1 plant. Defaults are 0001 section 2 unless stated."""

    # ---------------------------------------------------------------- interface fields (0002 3.5)
    horizon_q: int = HORIZON_Q
    """Episode length in quarters: 90 days of 96 quarters (0001 section 2)."""
    beta_load: float = 0.8
    """Load sensitivity of the wear clock, rho(load) = exp(beta (load - 1)) (A, 0001 section 2)."""
    p_fail_in_episode: float = 0.6
    """Share of first components that fail inside the horizon with no action (A, 0001 section 2)."""
    holder_present_p: float = 0.9
    """Probability the incoming holder is present at a revalidation attempt (A, 0001 section 2)."""
    gas_excursion_rate: float = 0.01
    """Gas-excursion onset probability per occupied quarter (A, 0001 section 2)."""
    unplanned_cost_ratio: float = 8.0
    """Unplanned failure cost over a planned one (S: asset-health README, 0001 section 2)."""
    approver_median_h: float = 2.0
    """Median approval latency in hours, log-normal (A, 0001 section 2)."""
    approver_catch_p: float = 0.5
    """Chance the approver catches a wrong but compliant diagnosis (A; maintenance-agent's known gap
    wrong_diagnosis_fully_compliant, 0001 section 2)."""
    blind_days: float = 23.0
    """Days an estimator is blind after a life starts: 21 d + 48 h settle (S: FeatureConfig)."""
    single_pump_throughput: float = 0.6
    """Throughput with one slurry pump running (A, 0001 section 2)."""
    derate_enabled: bool = False
    """Allow the DERATE command: the F1+D configuration of the causal study (0001 section 2)."""

    # ---------------------------------------------------------------- lives and the wear clock
    first_onset_min_day: float = 24.0
    """Earliest onset of a first life: a clean 21 d baseline plus settle (S: plan_failures)."""
    renewal_onset_days: tuple[float, float] = (24.0, 60.0)
    """Onset offset range of a life after a replacement, nominal days (A, same baseline rule)."""
    rate_clip: tuple[float, float] = (0.25, 4.0)
    """Bounds on the per-asset wear-rate multiplier a scenario may set (A, guards absurd draws)."""
    mode_load_tilt: float = 0.0
    """Log-odds tilt of the failure mode by the asset's load level: positive makes wear modes
    likelier on heavily loaded assets, negative reverses it. Zero is the catalogue draw. The causal
    study sets it to build, and then reverse, the load-mode correlation of C-DIAG (0003 A3). (A)"""
    load_level_sd: float = 0.08
    """Standard deviation of the fixed per-asset load offset (A)."""
    derate_fraction: float = 0.3
    """Load reduction of one DERATE command (A, 0001 section 7: derate by 30%)."""
    derate_throughput_loss: float = 0.15
    """Throughput lost per derated asset (A)."""

    # ---------------------------------------------------------------- process load
    load_phi: float = 0.9
    """Day-to-day persistence of the circuit load deviation, AR(1) (A)."""
    load_noise_sd: float = 0.05
    """Stationary standard deviation of the circuit load deviation (A)."""
    load_obs_sd: float = 0.01
    """Measurement noise on the observed load (A; load is observed, 0003 A3)."""
    load_clip_in: tuple[float, float] = (0.4, 1.6)
    """Load limits in the in-assumption regime. The ratio of wear rates across any step is then at
    most exp(beta x 1.2) = 2.61, below 3, so no health bin is skipped (A_phys, 0001 section 4.2)."""
    load_clip_out: tuple[float, float] = (0.2, 3.0)
    """Load limits beyond the assumption: the rate ratio can reach 9.4 and skip bins (A)."""
    upset_p_per_circuit: float = 0.5
    """Chance a circuit has one process upset in an episode (A)."""
    upset_magnitude: tuple[float, float] = (0.3, 0.6)
    """Load rise during an upset (A)."""
    upset_days: tuple[int, int] = (3, 8)
    """Upset duration in days, inclusive (A)."""

    # ---------------------------------------------------------------- daily observation (0003 A3)
    hi_base: float = 1.0
    """Health index of a healthy asset at load 1.0 (A; asset-health HI is near zero then)."""
    hi_fail: float = 9.0
    """Health index at failure (S: rul.estimate target failure level, 0001 section 2)."""
    hi_alert: float = 2.0
    """Health index at which the plant opens an alert (S: RULConfig.hi_min, the threshold)."""
    hi_noise_sd: float = 0.3
    """Daily noise on the health index (A; P4a fits it to asset-health traces)."""
    hi_load_coef: float = 2.0
    """Health index per unit of load above 1.0: a process upset moves it through load (A)."""
    sig_loading_bearing: tuple[float, float, float] = (6.0, 0.6, 0.0)
    """Gain of bearing severity on (sig_bearing, sig_proc, sig_hyd); the 0.6 is a small leak (A)."""
    sig_loading_wear: tuple[float, float, float] = (0.6, 4.0, 3.0)
    """Gain of impeller or liner wear severity on (sig_bearing, sig_proc, sig_hyd) (A)."""
    sig_load_coef: tuple[float, float, float] = (1.5, 2.5, 2.0)
    """Signature per unit of load above 1.0, so a process upset moves all channels (A)."""
    sig_noise_sd: float = 0.3
    """Daily noise on each signature channel (A)."""
    p_stuck_day: float = 0.02
    """Per-day chance the health index is stuck, which the gate masks (S: README; rate A)."""
    p_spike_day: float = 0.01
    """Per-day chance of a spike, which the gate masks (S: README; rate A)."""
    p_gain_fault: float = 0.08
    """Chance an asset carries a slow sensor gain drift in an episode; the gate does not mask it
    (S: asset-health README limitation "slow 4-20 mA drift"; rate A). It moves hi, not the
    signatures."""
    gain_rate: tuple[float, float] = (0.003, 0.008)
    """Gain drift per day as a fraction of the full-scale health index (A)."""
    stuck_days: int = 3
    """Length of an adversary-placed stuck episode, days (A)."""

    # ---------------------------------------------------------- shifted condition (PREREGISTRATION)
    shifted_load_gain: float = 0.25
    """Load rise in the shifted condition (PREREGISTRATION.draft.md section on conditions)."""
    shifted_day_range: tuple[float, float] = (30.0, 60.0)
    """Day the shift starts, uniform (PREREGISTRATION.draft.md)."""
    shifted_drift_per_day: float = 0.005
    """Unmasked gain drift per day on every health-index channel after the shift, as a fraction of
    full scale (PREREGISTRATION.draft.md)."""

    # ---------------------------------------------------------------- DES
    p_refuse: float = 0.05
    """Chance the plant turns a non-forced command into a no-op: crew or holder absent (A; the
    abstraction models it as environment nondeterminism)."""
    crew_per_shift: int = 2
    """Fitters per shift (A, 0001 section 2)."""
    gas_retest_q: int = 2
    """Gas re-test period in quarters: CORP-HSE-002 clause 4.3 Rev 3, 30 min (S: plant-copilot)."""
    gas_break_q: int = 1
    """Longest break in quarters before re-entry needs a fresh gas test: CORP-HSE-002 clause 4.3
    Rev 3, "any break in work longer than 15 minutes" (S: plant-copilot)."""
    permit_max_h: float = 12.0
    """Permit life in hours from its issue: CORP-HSE-004 clause 4.1 (S). A handover and a
    revalidation do not renew it (ADR 0003 A12.1); within it a permit is valid until the end of the
    shift that issued or revalidated it, and at the limit it is spent."""
    alarm_exit_q: int = 1
    """Quarters an occupant may stay after a gas alarm: CORP-HSE-002 clause 4.4 (S)."""
    excursion_q: int = 4
    """Quarters a gas excursion lasts (A)."""
    approver_sigma: float = 0.6
    """Log-scale spread of the approval latency (A)."""
    downtime_hours: tuple[float, float] = (36.0, 96.0)
    """Extra repair work after an unplanned failure, hours (S: asset-health downtime_hours)."""
    inspect_q: int = 4
    """Quarters an inspection takes and ties up one fitter (A)."""
    inspect_noise_factor: float = 0.25
    """Inspection reading noise relative to the daily noise (A, 0001 section 8)."""
    planned_job_cost: float = 1.0
    """Cost units of a planned job (illustrative; the ordering matters, 0001 section 2)."""
    inspect_cost: float = 0.05
    """Cost units of an inspection (A)."""
    rejected_wo_cost: float = 0.02
    """Cost units of a work order the approver rejected (A)."""
    p_malicious_note: float = 0.05
    """Chance a handover note is drawn from the malicious part of the bank (A)."""

    def validate(self) -> None:
        """Raise ValueError on a combination the simulator cannot run."""
        if self.horizon_q <= 0 or self.horizon_q % DAY_Q != 0:
            raise ValueError(
                f"horizon_q must be a positive whole number of days, got {self.horizon_q}"
            )
        for name in (
            "p_fail_in_episode",
            "holder_present_p",
            "approver_catch_p",
            "p_refuse",
            "p_stuck_day",
            "p_spike_day",
            "p_gain_fault",
            "upset_p_per_circuit",
            "p_malicious_note",
            "gas_excursion_rate",
            "derate_fraction",
            "single_pump_throughput",
            "derate_throughput_loss",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        for name in (
            "beta_load",
            "unplanned_cost_ratio",
            "approver_median_h",
            "approver_sigma",
            "blind_days",
            "load_noise_sd",
            "hi_noise_sd",
            "sig_noise_sd",
            "load_level_sd",
            "load_obs_sd",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
        if not 0.0 <= self.load_phi < 1.0:
            raise ValueError(f"load_phi must be in [0, 1), got {self.load_phi}")
        if self.hi_fail <= self.hi_base:
            raise ValueError("hi_fail must exceed hi_base")
        if self.hi_alert <= self.hi_base:
            raise ValueError("hi_alert must exceed hi_base")
        if self.first_onset_min_day < 0 or self.first_onset_min_day >= self.horizon_days - 2:
            raise ValueError("first_onset_min_day leaves no room for a failing life")
        for name in (
            "renewal_onset_days",
            "gain_rate",
            "upset_magnitude",
            "downtime_hours",
            "shifted_day_range",
            "rate_clip",
            "load_clip_in",
            "load_clip_out",
        ):
            lo, hi = getattr(self, name)
            if not lo <= hi:
                raise ValueError(f"{name} must be an ordered pair, got {(lo, hi)}")
        if not 1 <= self.upset_days[0] <= self.upset_days[1]:
            raise ValueError(
                f"upset_days must be an ordered pair of positive days, got {self.upset_days}"
            )
        if self.gas_break_q < 0 or self.gas_break_q > self.gas_retest_q:
            raise ValueError("gas_break_q must be between 0 and gas_retest_q")
        if self.gas_retest_q < 1 or self.alarm_exit_q < 0 or self.excursion_q < 1:
            raise ValueError(
                "gas_retest_q, excursion_q must be at least 1 and alarm_exit_q at least 0"
            )
        if self.crew_per_shift < 1 or self.inspect_q < 1 or self.stuck_days < 1:
            raise ValueError("crew_per_shift, inspect_q and stuck_days must be at least 1")
        if self.permit_max_h <= 0:
            raise ValueError("permit_max_h must be positive")

    @property
    def horizon_days(self) -> int:
        return self.horizon_q // DAY_Q
