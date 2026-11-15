# ADR 0002: Frozen interfaces, package layout and work packets

Status: proposed, 31 October 2026. Read `0001-architecture.md` first; section numbers below refer to it. The interfaces in section 3 are **frozen**: a packet may add private helpers and extra optional arguments with defaults, but may not rename, remove or change the meaning of anything listed. A change needs an amendment ADR (`0003-...`) agreed by the owner of every packet that touches the symbol.

## 1. Rules every packet follows

1. **Importable at every packet boundary.** When a packet is done, `python -c "import closed_loop"` succeeds in under 3 seconds with only numpy, scipy and pydantic installed, and `pytest -q -m "not f2"` passes. A packet that cannot finish leaves its files complete-but-unregistered (not imported by `__init__`) rather than half-wired.
2. **Progress note.** Each engineer keeps `docs/adr/progress/<packet>-<slug>.md` (template in section 6), updated after every sub-step (file written, test passing), so a stopped run resumes from the note. Delete nothing; mark `DONE` at the end.
3. **Own your files.** The table in section 2 gives one owner per file. Never edit another packet's file; ask for an interface amendment instead. Tests live in `tests/<area>/` and are owned by the packet that owns the code under test.
4. **No sibling imports outside `closed_loop/f2/` and `closed_loop/spec/clauses.py`** (the latter imports `plant_copilot` lazily, inside functions). Importing `asset_health` anywhere else is a defect: it costs 44 s (0001, section 3) and pulls scikit-learn.
5. **Determinism.** All randomness comes from `rng_for(root_seed, stream, *ids)`. No global `np.random`, no `random`, no wall-clock reads in simulation code (the clock is passed in). Same seed, same result, bit for bit, in one process and across processes.
6. **Typed and linted.** Type hints everywhere, `ruff check .` clean (line length 100, rules E, F, W, I, B, UP), frozen dataclasses for values crossing a module boundary, pydantic only for files read from disk.
7. **Every number has a script.** Nothing is typed into a README or doc by hand; metrics are `ResultRecord`s written by the code that computed them, with fidelity and what they were measured on.
8. **Synthetic labelling.** Every data file and every report header says SYNTHETIC. No employer, client or customer is named anywhere.
9. **Mutation honesty.** For each safety-relevant function (shield table, checker, monitors, label resolution, conformal updates) the packet includes at least one test that fails when the function is sabotaged, and says which mutation it killed in its progress note.
10. **Tooling environment.** `UV_CACHE_DIR` and `PYTHONPYCACHEPREFIX` outside the repository; no `__pycache__` or `.pytest_cache` left in the tree; tests run in the foreground; the heavy-run lock and 88% commit-charge rule of 0001 section 13 apply to anything longer than a minute.

## 2. Package layout and file ownership

```
closed-loop/
  pyproject.toml  Makefile  LICENSE  CHANGELOG.md  .gitattributes  .gitignore        P0
  README.md                                                                          P13
  src/closed_loop/
    __init__.py                                                                      P0 (version only)
    core/   types.py  rng.py  timegrid.py  results.py  trace.py                      P0
    stats/  confseq.py  eprocess.py  intervals.py                                    P1
    spec/   stl.py  monitor.py                                                       P2
            properties.py                                                            P2
            clauses.py                                                               P11a (lazy plant_copilot)
    plant/  params.py  scenario.py  degradation.py  observe.py  plant.py             P3a
            des.py  jobs.py  notes.py                                                P3b, P3c
    perception/ estimator.py  labels.py  coverage.py                                 P4a
                conformal.py                                                         P4b
                contract.py  monitor.py                                              P4c
    shield/ abstraction.py  game.py  table.py  runtime.py                            P5a
            reference.py  checker.py                                                 P5b
            conformance.py                                                           P5c
    proposers/ base.py  baselines.py                                                 P6a
               hostile.py  llm.py                                                    P6b
    causal/ scm.py  correlational.py  twin.py  queries.py                            P7
    planning/ worldmodel.py  belief.py                                               P8a
              planner.py  exactdp.py                                                 P8b
    search/ cmaes.py  random_search.py                                               P9a
            falsify.py  shrink.py                                                    P9b
    audit/  seeds.py  prereg.py  ledger.py  guard.py                                 P10
    eval/   arms.py  episode.py  metrics.py                                          P12a
            runner.py  report.py                                                     P12b
    f2/     render.py  splice.py                                                     P11a
            alert_source.py  cmms.py  human.py                                       P11b
            episode.py  replay.py  mismatch.py                                       P11c
  benchmarks/run.py  (--quick, --check, --full)                                      P10 skeleton, P12b fills
  integration/siblings.lock  scripts/verify_siblings.py                              P0, P11c
  tests/  core/ stats/ spec/ plant/ perception/ shield/ proposers/ causal/ planning/
          search/ audit/ eval/ f2/ (marked f2)                                       owner of the code under test
  docs/  adr/ (this folder)  PREREGISTRATION.draft.md  FINDINGS.md (P11c)  WHAT-GETS-THROUGH.md (P13, generated)
```

## 3. Frozen interfaces

All in Python 3.11 syntax. `Mapping`, `Sequence` from `collections.abc`; `NDArray = numpy.typing.NDArray`.

### 3.1 `core/types.py`

```python
class Fidelity(StrEnum):  F0 = "F0"; F1 = "F1"; F1D = "F1+D"; F2 = "F2"

class Op(StrEnum):        RUN = "RUN"; STOP = "STOP"; LOCKED = "LOCKED"; ISO = "ISO"; WORK = "WORK"; FAIL = "FAIL"
class Permit(StrEnum):    NONE = "NONE"; VALID = "VALID"; REVAL = "REVAL"

class Cmd(StrEnum):
    NOOP="NOOP"; START="START"; STOP="STOP"; LOCK="LOCK"; PROVE="PROVE"; ISSUE="ISSUE"; REVAL="REVAL"
    BEGIN="BEGIN"; TEST="TEST"; ENTER="ENTER"; EXIT="EXIT"; COMPLETE="COMPLETE"; CLOSE="CLOSE"; UNISO="UNISO"
    DERATE="DERATE"                                   # F1+D only; F2 masks it
FORCED_CMDS: frozenset[Cmd] = frozenset({Cmd.STOP, Cmd.EXIT})   # assumption A_exec

class PlanCmd(StrEnum):   INSPECT="INSPECT"; ORDER="ORDER"; RAISE_WO="RAISE_WO"

N_BINS = 4                                           # health bins 0..3, 3 = critical (0001 section 4.1)
BIN_EDGES_DAYS: tuple[float, float, float] = (30.0, 10.0, 3.0)   # bin 0 > 30 d, 1: 10-30, 2: 3-10, 3: < 3

@dataclass(frozen=True)
class Command:
    asset_id: str
    cmd: Cmd | PlanCmd
    arg: str | None = None                           # part number, failure mode, target shift index ...
    source: str = ""                                 # proposer name, for audit

def bin_of_rul(rul_days: float) -> int: ...          # 0..3 using BIN_EDGES_DAYS; rul <= 0 -> 3
def bins_of_interval(lo: float, hi: float) -> tuple[int, int]: ...   # (lowest, highest) bin touched; lo may be < 0
```

### 3.2 `core/rng.py`, `core/timegrid.py`, `core/results.py`, `core/trace.py`

```python
def rng_for(root_seed: int, stream: str, *ids: int | str) -> np.random.Generator:
    """Keyed generator: PCG64(SeedSequence(root_seed, spawn_key=(crc32(stream), *int_ids)));
    str ids are crc32'd. Same arguments, same stream, regardless of any other draw."""

QUARTER_MIN = 15; SHIFT_Q = 32; DAY_Q = 96; HORIZON_DAYS = 90; HORIZON_Q = 8640
def shift_of(q: int) -> int: ...                     # 0 = 06:00-14:00, 1 = 14:00-22:00, 2 = 22:00-06:00; day starts 06:00
def day_of(q: int) -> int: ...
def is_handover_next(q: int) -> bool: ...            # next quarter starts a new shift

@dataclass(frozen=True)
class ResultRecord:
    name: str; value: float; n: int; fidelity: Fidelity
    measured_on: str                                 # e.g. "held-out seeds, nominal condition, 500 episodes"
    script: str                                      # module that produced it
    ci_lo: float | None = None; ci_hi: float | None = None; ci_kind: str = ""   # "confseq-0.05"
    extra: Mapping[str, float | int | str] = field(default_factory=dict)
    def __post_init__(self): ...                     # raises ValueError on empty measured_on / script / NaN value

@dataclass
class Trace:
    """Dense per-quarter record of one episode (numpy arrays of length n_q). Used by spec and eval."""
    n_q: int
    asset_ids: tuple[str, ...]
    op: dict[str, NDArray[np.int8]]                  # index into list(Op)
    permit: dict[str, NDArray[np.int8]]; occupied: dict[str, NDArray[np.bool_]]
    gas_age_min: dict[str, NDArray[np.float32]]      # minutes since last pass; large when never
    alarm: dict[str, NDArray[np.bool_]]; iso_proven: dict[str, NDArray[np.bool_]]
    permit_margin_min: dict[str, NDArray[np.float32]]
    true_rul_days: dict[str, NDArray[np.float32]]    # sim-only truth, no-intervention twin
    health_bin: dict[str, NDArray[np.int8]]          # sim-only
    set_lo: dict[str, NDArray[np.int8]]; set_hi: dict[str, NDArray[np.int8]]    # perception sets given to the shield
    throughput: NDArray[np.float32]; uncontrolled_failure: dict[str, NDArray[np.bool_]]
    commands: list[tuple[int, Command, bool, bool]]  # (quarter, command, executed, overridden_by_shield)
```

### 3.3 `stats/`

```python
# confseq.py
def betting_cs(x: Sequence[float] | NDArray, alpha: float = 0.05, grid: int = 1001) -> tuple[NDArray, NDArray]:
    """Anytime-valid confidence sequence for the mean of x in [0,1]: returns (lower[t], upper[t]) for every prefix."""
class CSAccumulator:
    def __init__(self, alpha: float = 0.05, grid: int = 1001): ...
    def update(self, x: float) -> None: ...          # raises ValueError if x not in [0,1]
    def bounds(self) -> tuple[float, float]: ...
    n: int
def paired_cs(a: NDArray, b: NDArray, lo: float, hi: float, alpha: float = 0.05) -> tuple[float, float]:
    """CS for mean(a - b): the spread-scaled sequence of 0003 A16 with at least 100 pairs and sample kurtosis at most n / 50, else paired_betting_cs."""
def paired_betting_cs(a: NDArray, b: NDArray, lo: float, hi: float, alpha: float = 0.05) -> tuple[float, float]:
    """CS for mean(a - b) where a, b in [lo, hi]: rescales (a - b) to [0,1], returns bounds on the original scale."""
def zero_event_upper(n: int, alpha: float = 0.05) -> float: ...   # upper bound after n events with none observed
# eprocess.py
class BernoulliEProcess:
    def __init__(self, p0: float, delta: float = 0.01): ...    # H0: P(miss) <= p0
    def update(self, miss: bool) -> float: ...        # returns current e-value
    fired: bool; n: int; e_value: float
# intervals.py
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]: ...
def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]: ...   # exact, two-sided
```

### 3.4 `spec/`

```python
# stl.py  (bounded-past fragment)
class Formula: ...                                   # AST nodes: Atom, Not, And, Or, Implies, Once, Hist, Always
def Atom(name: str) -> Formula: ...                  # margin signal looked up by name in the signal dict
def Once(f: Formula, a: int, b: int) -> Formula: ... # a, b in quarters, inclusive
def Hist(f: Formula, a: int, b: int) -> Formula: ...
def Always(f: Formula) -> Formula: ...
def robustness(f: Formula, signals: Mapping[str, NDArray[np.float32]]) -> NDArray[np.float32]:
    """Vectorised robustness at every quarter. Margin > 0 means satisfied."""
def min_robustness(f: Formula, signals: Mapping[str, NDArray]) -> float: ...
# monitor.py
class StreamingMonitor:
    def __init__(self, f: Formula): ...
    def push(self, signals_now: Mapping[str, float]) -> float: ...   # robustness at this quarter; O(window) memory
# properties.py
@dataclass(frozen=True)
class SpecParams:
    retest_q: int = 2; break_q: int = 1; permit_max_h: float = 12.0; alarm_exit_q: int = 1
    provenance: Mapping[str, str] = field(default_factory=dict)   # id -> verbatim clause + doc + revision
def property_formulas(p: SpecParams) -> dict[str, Formula]: ...      # keys "P1".."P6"
def signals_from_trace(tr: Trace, p: SpecParams, crusher: str, pumps: tuple[str, str]) -> dict[str, NDArray]: ...
def evaluate(tr: Trace, p: SpecParams, crusher: str, pumps: tuple[str, str]) -> dict[str, float]:
    """min robustness per property; negative = violation."""
# clauses.py  (lazy plant_copilot import inside functions)
def spec_from_copilot(as_of: date) -> SpecParams: ...                 # raises ClauseError if a clause or revision is missing
```

Contract tested: `StreamingMonitor` fed the signals one quarter at a time returns exactly `robustness(...)` (to float tolerance) for every formula built in `properties.py`.

### 3.5 `plant/`

```python
@dataclass(frozen=True)
class PlantParams:                                   # defaults = 0001 section 2; every field has a docstring naming its source
    horizon_q: int = 8640; beta_load: float = 0.8; p_fail_in_episode: float = 0.6
    holder_present_p: float = 0.9; gas_excursion_rate: float = 0.01; unplanned_cost_ratio: float = 8.0
    approver_median_h: float = 2.0; approver_catch_p: float = 0.5; blind_days: float = 23.0
    single_pump_throughput: float = 0.6; derate_enabled: bool = False
    def validate(self) -> None: ...

@dataclass(frozen=True)
class Scenario:                                      # decoded adversary / shift description
    theta: NDArray[np.float64] | None = None         # in [0,1]^32 or None for nominal
    condition: str = "nominal"                       # "nominal" | "shifted"
    regime: str = "in"                               # "in" | "out" (0001 section 9)
    @staticmethod
    def decode(theta: NDArray, regime: str) -> "Scenario": ...        # deterministic; theta clipped to [0,1]

@dataclass(frozen=True)
class AssetView:
    asset_id: str; op: Op; permit: Permit; occupied: bool; gas_age_q: int; alarm: bool; soon: bool
    set_lo: int; set_hi: int                         # perception set (health bins)
    rul_lo: float | None; rul_hi: float | None; prior_only: bool
    hi_last: float | None; job_open: bool; spares: Mapping[str, int]

@dataclass(frozen=True)
class Observation:
    quarter: int; day: int; shift: int
    assets: Mapping[str, AssetView]
    note: str | None                                 # shift-handover note, if one was posted this quarter
    alerts: tuple[str, ...]                          # alert ids opened since the last call
    crew_free: int; safety_officer_on_shift: bool; monitor_fired: bool

@dataclass(frozen=True)
class StepResult:
    obs: Observation; executed: tuple[Command, ...]; refused: tuple[Command, ...]
    violations: tuple[tuple[str, str, int], ...]     # (property id, asset id, quarter): measured by the plant's own checks
    done: bool

class Plant(Protocol):
    def reset(self, seed: int, scenario: Scenario) -> Observation: ...
    def step(self, commands: Sequence[Command]) -> StepResult: ...   # advances exactly one quarter
    def trace(self) -> Trace: ...
    def twin(self, intervention: Sequence[tuple[int, Command]]) -> "Plant": ...   # F1 only: same exogenous streams, replayed with do()
    truth: "TruthView"                               # sim-only; proposers never receive it

@dataclass(frozen=True)
class TruthView:
    true_rul_days: Mapping[str, float]; health_bin: Mapping[str, int]; life_id: Mapping[str, int]
    failure_time_q: Mapping[str, float]              # T in the no-intervention continuation
def make_plant(params: PlantParams | None = None, *, fidelity: Fidelity = Fidelity.F1,
               spec: SpecParams | None = None) -> Plant: ...
```

`plant/jobs.py`: `def requirement(asset_class: str, mode: str, inspection_first: bool) -> JobRequirement` wraps `maintenance_agent.requirements.requirement_for` **when installed**, else a vendored copy of the matrix with a test that compares the two when both exist (`JobRequirement(permits: frozenset[str], work_procedure: str | None, role: str, known: bool)`). The job runner `JobRunner.next_command(view, job) -> Command | None` emits the correct job script (LOCK, PROVE, ISSUE, BEGIN, TEST, ENTER, ... EXIT, COMPLETE, CLOSE, UNISO, START), one command per quarter, honouring `shield` pauses (it re-plans after a refused or overridden command).

### 3.6 `perception/`

```python
@dataclass(frozen=True)
class EstimatorOutput:
    asset_id: str; day: int; status: str             # "estimated" | "prior" | "not_degrading" | "insufficient"
    m_days: float; sigma_days: float                 # point and scale; prior values when status != "estimated"
    hi: float | None

class StreamingEstimator(Protocol):
    def update(self, asset_id: str, day: int, hi: float | None, life_id: int) -> EstimatorOutput: ...
    def reset_life(self, asset_id: str, day: int) -> None: ...      # blind window starts

class LabelStatus(StrEnum): HIT="HIT"; MISS="MISS"; LOWER_HIT="LOWER_HIT"; UNKNOWN="UNKNOWN"
@dataclass(frozen=True)
class Label:
    life_id: int; asset_id: str; checkpoint_day: int; lo: float; hi: float; level: float
    resolved_day: int; status: LabelStatus; end_day: float; censored: bool; audit: bool; prior: bool
    true_rul: float | None = None                    # sim-only; None in learner-visible copies
def resolve_labels(checkpoints: Sequence["Checkpoint"], end_day: float, censored: bool, audit: bool) -> list[Label]:
    """0001 section 6.2 status rules, exactly."""

@dataclass(frozen=True)
class RulInterval:
    lo: float; m: float; hi: float; level: float; prior: bool

class IntervalProvider(Protocol):
    name: str                                        # "V1".."V7"
    def predict(self, est: EstimatorOutput, level: float) -> RulInterval: ...
    def observe(self, labels: Sequence[Label]) -> None: ...         # called at resolution events, in time order
    def state(self) -> Mapping[str, float]: ...      # alpha_t, window size, ...: logged
def make_provider(name: str, *, alpha: float, development_scores: NDArray, seed: int, **kw) -> IntervalProvider: ...

# coverage.py
def oracle_coverage(intervals, true_rul, mask=None) -> float: ...
def learner_coverage(labels: Sequence[Label]) -> dict[str, float]: ...   # hit, miss, lower_hit, unknown counts and rates
def manski_bounds(labels: Sequence[Label]) -> tuple[float, float]: ...
def per_life_any_miss(...) -> float: ...
def gaps(open_loop: ..., closed_loop: ...) -> dict[str, float]: ...      # Gap_A, Gap_B

# contract.py / monitor.py
def health_set(interval: RulInterval) -> tuple[int, int]: ...             # bins_of_interval, clipped to 0..3
class CoverageMonitor:
    def __init__(self, mode: str, p0: float = 0.01, delta: float = 0.01): ...   # mode "opt" | "pess" | "inf"
    def observe(self, labels: Sequence[Label]) -> None: ...
    fired: bool; fired_at_label: int | None
```

### 3.7 `shield/`

```python
VIOL = "VIOL"
class AbstractModel(Protocol):
    name: str
    def states(self) -> Iterable[Hashable]: ...
    def commands(self) -> Sequence[str]: ...
    def step(self, s: Hashable, c: str) -> frozenset[Hashable | str]: ...    # all environment outcomes; may contain VIOL
    def initial(self) -> Hashable: ...
    def observable(self, s: Hashable) -> Hashable: ...                       # drops the hidden health bin(s)
    def health(self, s: Hashable) -> int | tuple[int, ...]: ...

class CrusherJobModel:   ...  # 1,728 raw states (0001 section 4.1)
class PumpPairModel:     ...  # 400 raw states, 49 joint commands

@dataclass(frozen=True)
class Winning:
    states: frozenset[Hashable]; safe: Mapping[Hashable, frozenset[str]]; iterations: int
def solve(model: AbstractModel) -> Winning: ...

@dataclass(frozen=True)
class ShieldTable:
    model_name: str
    entries: Mapping[tuple[Hashable, tuple[int, int]], frozenset[str]]       # (observable, (lo, hi) set) -> allowed commands
    priority: tuple[str, ...]                                                # substitution order
    meta: Mapping[str, str]                                                  # spec provenance, abstraction version, hash
    def allowed(self, o: Hashable, health_set: tuple[int, int]) -> frozenset[str]: ...
    def sha256(self) -> str: ...
def build_shield(model: AbstractModel, win: Winning, max_width: int = 4) -> ShieldTable: ...

class ShieldedProposer:                              # runtime.py
    def __init__(self, inner: Proposer, crusher: ShieldTable, pumps: ShieldTable,
                 mapper: "AbstractionMap", monitor_fired: Callable[[], bool] | None = None): ...
    def propose(self, obs: Observation) -> list[Command]: ...      # filters and substitutes; counts overrides
    overrides: int; admitted: int
    def reset(self) -> None: ...

class AbstractionMap(Protocol):                      # concrete -> abstract
    def crusher_state(self, view: AssetView, health_bin: int) -> Hashable: ...
    def pump_pair_state(self, a: AssetView, b: AssetView, bins: tuple[int, int]) -> Hashable: ...
    def command(self, cmd: Command) -> str: ...      # pump LOCK -> NOOP, pump PROVE -> ISOLATE (0001 section 4.1)

# reference.py (independent)   checker.py
class ReferenceModel(Protocol):                      # written from the English spec in 0001 section 4.1 and 5, not from abstraction.py
    def outcomes(self, s: Hashable, c: str) -> frozenset[Hashable | str]: ...
@dataclass(frozen=True)
class CheckResult:
    reachable: int; findings: tuple[tuple[str, Hashable, object], ...]       # ("VIOLATION"|"DEADLOCK", state, detail)
def check(ref: ReferenceModel, table: ShieldTable, initial: Hashable, max_width: int,
          drop: tuple[Hashable, str] | None = None) -> CheckResult: ...
def differential(model: AbstractModel, ref: ReferenceModel) -> list[tuple[Hashable, str, frozenset, frozenset]]: ...   # [] when identical

# conformance.py
@dataclass(frozen=True)
class ConformanceReport:
    ticks: int; unmatched: int; by_variable: Mapping[str, int]; a_phys_violations: int; wilson: tuple[float, float]
def conformance(traces: Sequence[Trace], model_map: AbstractionMap, models: Mapping[str, AbstractModel]) -> ConformanceReport: ...
```

### 3.8 `proposers/`

```python
class Proposer(Protocol):
    name: str
    def reset(self, seed: int) -> None: ...
    def propose(self, obs: Observation) -> list[Command]: ...

def make_baseline(name: str, **kw) -> Proposer: ...  # "B0" run-to-failure, "B1" calendar, "B2" fixed limit, "B3" bound rule
class Careless:  def __init__(self, inner: Proposer, p_skip: float = 0.05): ...
class Gullible:  def __init__(self, inner: Proposer, screen: bool, scan: Callable[[str], str | None]): ...
class LLMProposer:  def __init__(self, *args, **kw): ...     # raises NotConfigured unless CLOSED_LOOP_LLM=1 and the extra is installed
NOTE_BANK: tuple[NoteTemplate, ...]                  # SYNTHETIC handover notes, benign and malicious, English and Arabic
```

### 3.9 `causal/`

```python
@dataclass(frozen=True)
class LifeRecord:  asset_id: str; hi_days: NDArray; hi: NDArray; derate_days: NDArray; fail_day: float | None; censored_day: float | None
class EffectModel(Protocol):
    def fit(self, lives: Sequence[LifeRecord]) -> None: ...
    def counterfactual_failure_day(self, life: LifeRecord, derate_from_day: float, load_cap: float) -> float: ...
    def p_fail_before(self, life: LifeRecord, replace_day: float | None, horizon_day: float) -> float: ...
def make_effect_model(kind: str) -> EffectModel: ...   # "correlational" | "scm" | "scm_misspecified"
def generate_observational(n: int, seed: int, policy: str = "pi_obs") -> list[LifeRecord]: ...
def twin_truth(life_seed: int, query: "Query") -> "TruthAnswer": ...  # reruns the F1+D plant with shared streams
def score(model: EffectModel, test_seeds: Sequence[int]) -> Mapping[str, ResultRecord]: ...
```

### 3.10 `planning/`

```python
class WorldModel(Protocol):
    def prior(self, rng: np.random.Generator, n: int) -> Particles: ...
    def step(self, p: Particles, action: int, rng: np.random.Generator) -> Particles: ...
    def loglik(self, p: Particles, obs: float | None, noise_scale: float = 1.0) -> NDArray: ...
    def failed(self, p: Particles) -> NDArray[np.bool_]: ...
class PlantWorldModel(WorldModel): def __init__(self, fitted: Mapping[str, float], mismatch: float = 0.0): ...
class TabularWorldModel(WorldModel): def __init__(self, T: NDArray, O: NDArray, cost: Mapping[str, float]): ...
class BeliefTracker:  def update(self, asset_id: str, day: int, hi: float | None, inspected: bool) -> None; def particles(self, asset_id) -> Particles
class BeliefPlanner:
    def __init__(self, model: WorldModel, *, n_particles: int = 128, horizon_days: int = 14,
                 delta_plan: float = 0.02, seed: int = 0): ...
    def plan(self, obs: Observation, tracker: BeliefTracker, constraints: "PlanConstraints") -> list[Command]: ...
class PlannerProposer(Proposer): ...
def exact_value(model: TabularWorldModel, belief: NDArray, horizon: int) -> tuple[float, str]: ...   # exactdp.py
```

### 3.11 `search/`

```python
class CMAES:
    def __init__(self, dim: int, seed: int, popsize: int | None = None, sigma0: float = 0.3, restarts: int = 3): ...
    def ask(self) -> NDArray: ...                    # (popsize, dim) in [0,1]
    def tell(self, x: NDArray, f: NDArray) -> None: ...
    best_x: NDArray; best_f: float; evals: int
def random_search(f: Callable[[NDArray], float], dim: int, budget: int, seed: int) -> "SearchRun": ...
@dataclass(frozen=True)
class SearchRun:  algo: str; regime: str; seed: int; best_f: float; first_violation_eval: int | None; evals: int; theta_best: NDArray; history: NDArray
def falsify(arm_factory: Callable[[], "Arm"], regime: str, algo: str, budget: int, seed: int) -> SearchRun: ...
def shrink(theta: NDArray, still_violates: Callable[[NDArray], bool], tol: float = 0.02) -> NDArray: ...   # coordinate bisection to nominal
```

### 3.12 `audit/`

```python
def normalised_bytes(path: Path) -> bytes: ...       # CRLF -> LF, trailing spaces stripped
def prereg_sha256(path: Path) -> str: ...
def heldout_seed(prereg_sha: str, i: int) -> int: ...  # int.from_bytes(sha256(prereg_sha || b"heldout" || i)[:8], "big")
def f2_seed(prereg_sha: str, i: int) -> int: ...        # label b"f2"
class Ledger:
    def __init__(self, path: Path): ...
    def verify(self) -> None: ...                    # raises LedgerError on a broken chain
    def append(self, kind: str, **fields) -> dict: ...
    def head(self) -> str: ...
    def heldout_evaluations(self) -> int: ...
class HeldoutGuard:
    def __init__(self, ledger: Ledger, prereg_sha: str): ...
    def seeds(self, n: int, start: int = 0) -> list[int]: ...   # raises unless a freeze entry for prereg_sha exists; records the use
def code_sha256(roots: Sequence[Path]) -> str: ...
```

### 3.13 `eval/` and `f2/`

```python
@dataclass(frozen=True)
class Arm:
    name: str; proposer: Callable[[], Proposer]; shield: bool; monitor: bool
    perception: str = "V3"; screen: bool = False; spec: str = "clause"; planner_mismatch: float = 0.0
@dataclass
class EpisodeResult:
    arm: str; seed: int; condition: str; fidelity: Fidelity
    violations: Mapping[str, int]; min_robustness: Mapping[str, float]
    throughput: float; cost: float; unplanned_failures: int; planned_jobs: int
    shield_overrides: int; shield_admitted: int; monitor_fired_day: float | None
    labels: tuple[Label, ...]; wall_s: float
def run_episode(arm: Arm, seed: int, scenario: Scenario, fidelity: Fidelity = Fidelity.F1) -> EpisodeResult: ...
def aggregate(results: Sequence[EpisodeResult], metric: str, alpha: float = 0.05) -> ResultRecord: ...
def run_cell(arm: Arm, seeds: Sequence[int], scenario: Scenario, batch: int = 50, target_halfwidth: float = 0.05,
             n_max: int = 500, workers: int = 3) -> list[EpisodeResult]: ...   # sequential stopping on the first listed primary metric

# f2 (imports asset_health, maintenance_agent, plant_copilot lazily; every public function raises F2Unavailable if missing)
def render_life(latent: "LifeLatent", params: PlantParams, seed: int, segment: int) -> "RenderedTelemetry": ...
def splice(segments: Sequence["RenderedTelemetry"]) -> "pd.DataFrame": ...
class AssetHealthAlertSource:    # implements maintenance_agent.ports.AlertSource
    def __init__(self, runs: "RunCache", clock: Callable[[], datetime], asset_map: Mapping[str, str]): ...
class PlantCMMS:                 # implements maintenance_agent.ports.CMMS
    def __init__(self, inner: "MockCMMS", plant_inventory: Callable[[str], "PartRecord | None"], queue: Callable[["WorkOrderDraft", str], None]): ...
class SimHumanApprover:          # human-side only; imports maintenance_agent.gate
    def __init__(self, approver_id: str, private_key_b64: str, clock: Callable[[], datetime], params: PlantParams, rng: np.random.Generator): ...
    def review(self, run_state) -> "Approval | SignedDecision | None": ...   # None = not yet (latency)
def run_f2_episode(arm: Arm, seed: int, scenario: Scenario) -> EpisodeResult: ...
def replay_theta(arm: Arm, theta: NDArray, regime: str) -> "ReplayOutcome": ...    # reason codes of 0001 section 9
```

## 4. Work packets

Sized for one engineer agent in one sitting (about 150-400 lines of code plus tests). "Done" always includes: ruff clean, `pytest -q -m "not f2"` green, `import closed_loop` under 3 s, the progress note marked DONE, and the listed tests. Dependencies are by packet id.

| Packet | Title | Depends | Owns | Size |
|---|---|---|---|---|
| P0 | Scaffold, core types, RNG | none | pyproject, Makefile, `.gitattributes`, `core/*`, `integration/siblings.lock` stub, CI file (written, not pushed), progress template | S |
| P1 | Statistics: confidence sequences, e-process, Wilson | P0 | `stats/*` | S |
| P2 | STL fragment, streaming monitor, the six properties | P0 | `spec/stl.py`, `monitor.py`, `properties.py` | M |
| P3a | Plant: parameters, degradation SCM, exogenous streams, observation chain | P0 | `plant/params.py`, `scenario.py`, `degradation.py`, `observe.py` | M |
| P3b | Plant: DES (shifts, permits, gas, crew, spares, approvals) and `Plant.step` | P3a | `plant/des.py`, `plant/plant.py` | M |
| P3c | Plant: job runner, requirements adapter, note bank | P3b | `plant/jobs.py`, `plant/notes.py` | S |
| P4a | Perception: estimator, label resolution, coverage metrics | P3a | `perception/estimator.py`, `labels.py`, `coverage.py` | M |
| P4b | Perception: seven conformal variants | P4a, P1 | `perception/conformal.py` | M |
| P4c | Perception: contract (sets to bins) and coverage monitors | P4a, P1 | `perception/contract.py`, `monitor.py` | S |
| P5a | Shield: abstraction models, solver, table, runtime wrapper | P0 | `shield/abstraction.py`, `game.py`, `table.py`, `runtime.py` | M |
| P5b | Shield: independent reference model, checker, differential, sabotage | P5a (interface only; must not read its code) | `shield/reference.py`, `checker.py` | M |
| P5c | Shield: abstraction map and conformance test | P5a, P3b | `shield/conformance.py` | S |
| P6a | Proposers: Protocol and baselines B0-B3 | P3c, P4c | `proposers/base.py`, `baselines.py` | S |
| P6b | Proposers: careless, gullible, LLM stub | P6a, P3c | `proposers/hostile.py`, `llm.py` | S |
| P7 | Causal: SCM, correlational model, twin queries, scoring | P3a | `causal/*` | M |
| P8a | Planner: world models, belief tracker | P3a | `planning/worldmodel.py`, `belief.py` | M |
| P8b | Planner: BeliefPlanner, Lagrangian, exact DP validation | P8a, P6a | `planning/planner.py`, `exactdp.py` | M |
| P9a | Search: CMA-ES and random search | P0 | `search/cmaes.py`, `random_search.py` | S |
| P9b | Search: falsify loop and shrink | P9a, P2, P5a, P6b, P12a | `search/falsify.py`, `shrink.py` | S |
| P10 | Audit: seeds, prereg hash, ledger, guard, `run.py` skeleton | P0 | `audit/*`, `benchmarks/run.py` skeleton | S |
| P11a | F2: spike on splicing lives, rendering, clause spec | P3a | `f2/render.py`, `splice.py`, `spec/clauses.py` | M |
| P11b | F2: alert source, PlantCMMS, simulated human | P11a, P3b | `f2/alert_source.py`, `cmms.py`, `human.py` | M |
| P11c | F2: episode runner, replay, mismatch tests, FINDINGS | P11b, P12a | `f2/episode.py`, `replay.py`, `mismatch.py`, `docs/FINDINGS.md`, `scripts/verify_siblings.py` | M |
| P12a | Eval: arms, episode loop, metrics | P3c, P4c, P5a, P6a, P8b | `eval/arms.py`, `episode.py`, `metrics.py` | M |
| P12b | Eval: runner (cells, stopping), report, `run.py` wiring | P12a, P10, P1 | `eval/runner.py`, `report.py`, `benchmarks/run.py` body | M |
| P13 | Docs: README, WHAT-GETS-THROUGH generator, results tables | P12b | `README.md`, `docs/WHAT-GETS-THROUGH.md` generator | S |

Parallel plan with three engineers: wave 1 = P0; wave 2 = P1, P2, P3a, P5a, P9a, P10 (six independent; run three at a time); wave 3 = P3b, P4a, P5b, P7, P8a, P11a; wave 4 = P3c, P4b, P4c, P5c, P11b; wave 5 = P6a, P6b, P8b, P12a; wave 6 = P9b, P11c, P12b; wave 7 = P13.

### P0 Scaffold, core types, RNG
Deliver: `pyproject.toml` (hatchling; core deps numpy, scipy, pydantic>=2; extras `dev`, `integration` with direct URLs only, `f2` marker), `core/*` exactly per 3.1-3.2, `.gitattributes` (`* text=auto eol=lf`), Makefile (`install test lint bench quick clean`), LICENSE (MIT 2026 Darrshan Govender), CHANGELOG, CI workflow file.
Done when: interfaces 3.1-3.2 present; `Trace` constructible empty and from arrays.
Tests: `rng_for` same args same stream; different `stream` or `ids` independent (correlation test over 10^4 draws); drawing from one keyed stream does not change another; stable across processes (spawn a subprocess and compare); `bin_of_rul`/`bins_of_interval` edge cases (exactly 30, 10, 3, negative); `ResultRecord` rejects empty `measured_on`, NaN; `test_no_bare_sibling_names` (parses pyproject and any `requirements*.txt`: every sibling reference must contain `@ git+` or a local path); a test that `import closed_loop` does not import `asset_health`, `maintenance_agent`, `plant_copilot`, `pandas`, `sklearn` (inspect `sys.modules` in a subprocess).

### P1 Statistics
Deliver 3.3. Betting CS vectorised over the grid; hedged capital in log space to avoid overflow.
Done when: coverage verified by simulation inside the test (10,000 Bernoulli streams of length 500 at means 0.01, 0.1, 0.5: fraction of streams whose CS ever excludes the true mean is at most 0.05, stated with a Wilson interval); CS width shrinks like `1/sqrt(n)`; `zero_event_upper(2000)` is reported and compared with the rule-of-three value `3/2000`.
Tests: bounds nested (CS at t+1 need not be nested, but the running intersection must be a valid CS: test `bounds()` returns the intersection); inputs outside [0,1] rejected; the e-process fires on a stream with miss rate 0.05 against `p0 = 0.01` and, in 2,000 streams at miss rate exactly 0.01, fires in at most `delta` of them; mutation: change `1/delta` to `1/(2 delta)` and the type-I test fails.

### P2 STL fragment and the six properties
Deliver 3.4 except `clauses.py`. Robustness by sliding-window max/min using monotonic deques or `scipy.ndimage`-free numpy cumulative tricks; streaming monitor with ring buffers.
Done when: P1-P6 formulas built; synthetic traces (hand-written, each violating exactly one property) give negative robustness only for the intended property and positive for the clean trace.
Tests: streaming equals offline for 200 random traces and every formula; boundary quarters (window shorter than history at the start); margin units (P3: 30 minus minutes since test); a property-based test (hypothesis) that robustness of `Implies(a, b)` equals `max(-a, b)`; mutation: off-by-one in a window bound is caught.

### P3a Plant: degradation SCM and observation chain
Deliver `PlantParams`, `Scenario.decode` (32-dimensional layout of 0001 section 9, documented index by index), life draws per 0001 section 2, severity curve, HI chain with data-quality events, blind-window bookkeeping. No commands yet: a function `simulate_lives(params, seed, scenario) -> LifeTable` returns per-asset lives with onset, failure time, shape, mode, daily HI (with NaN for masked days), and the no-intervention true RUL series.
Done when: life statistics match 0001 section 2 (duration ranges, shape ranges, 60% failing inside the horizon within a stated tolerance over 2,000 seeds, reported by the test).
Tests: twin property (a life table is bit-identical when only an unrelated stream is changed); `Scenario.decode` deterministic and total on `[0,1]^32` including corners; regime `in` never produces a load shift above the A_phys cap; HI is monotone in severity before noise; the first onset is never before day 24 (property-based).

### P3b Plant: DES and `Plant.step`
Deliver `des.py` (event queue or tick loop; shifts, handover announcements and permit REVAL, holder presence, gas test results and alarms, crew, spares with lead times, approvals as delayed events, uncontrolled failure with downtime) and `plant.py` implementing the `Plant` protocol and `Trace` recording, `twin()`. Violations are recorded by the plant's own checks written directly from the English properties of 0001 section 5 (they must not import `spec`).
Done when: a hand-written scripted episode performs a complete correct crusher job (LOCK, PROVE, ISSUE, BEGIN, TEST, ENTER, TEST every 2 quarters, EXIT before the handover, REVAL, ...) with zero violations, and 10 scripted wrong sequences each produce exactly the expected violation.
Tests: A_exec (STOP and EXIT never refused); refusal of other commands at the stated probability (binomial test); `twin([])` reproduces the factual trace bit for bit; `twin([(q, replace)])` leaves other assets' traces identical; throughput accounting; a violation is recorded at the quarter it occurs; `Observation` never exposes truth fields (reflection test over dataclass fields).

### P3c Plant: job runner, requirements adapter, notes
Deliver `jobs.py` (requirement matrix access, vendored fallback with a comparison test, `JobRunner`) and `notes.py` (a bank of at least 24 SYNTHETIC handover notes: benign, malicious-in-English, malicious-in-Arabic, hidden-formatting; each tagged with the command it tries to induce).
Done when: for every (class, mode) in the plant, the job's permits, role and procedure equal `requirement_for` (when maintenance_agent is installed; otherwise compared to the vendored table with a clear skip message).
Tests: the runner never emits a command out of order; re-plans after refusal; the note bank covers each malicious intent at least twice; Arabic notes are valid UTF-8 and survive a subprocess round trip with raw bytes (the owner's earlier mojibake lesson).

### P4a Perception: estimator, labels, coverage
Deliver `TheilSenEstimator` (window 8 daily points, minimum 5, minimum slope 0.05 per day, failure level and scale from `PlantParams`, prior fallback, blind window), `resolve_labels` exactly per 0001 section 6.2, coverage functions, Manski bounds, per-life any-miss, `gaps`.
Done when: label statuses match a hand-built table of 20 cases (failure vs censored, `tau` before, inside and after `[t+lo, t+hi]`); Manski bounds bracket the oracle coverage in every simulated run.
Tests: causality (estimates unchanged when future HI is truncated); blind window respected (status `prior` for 23 days after `reset_life`); label status for censoring at exact boundaries; mutation: swap `>=` and `>` in the censored-hit rule and the table test fails; the learner-visible label copies carry `true_rul = None`.

### P4b Perception: conformal variants
Deliver `make_provider` for V1-V7 per 0001 section 6.3 (ACI with `gamma = 0.02`, window 500; PID quantile tracker plus integrator; audit assignment by `rng_for(seed, "audit", life_id)`; per-life max-score band; oracle ACI taking true labels).
Done when: on an exchangeable synthetic score stream V1 and V6 hit their nominal coverage within a Wilson interval; ACI long-run miss frequency within 0.01 of alpha on a shifted stream (the Gibbs and Candes bound is `(max(alpha_1, 1 - alpha_1) + gamma) / (gamma T)`; assert it); PID tracks a step shift in fewer steps than split.
Tests: online updates occur only in `observe` and in the order given; V2 counts censored as hit, V3 skips UNKNOWN, V4 updates only on audit labels; mutation of each rule caught; providers are deterministic given the seed; state logged.

### P4c Perception: contract and coverage monitor
Deliver `health_set` and `CoverageMonitor` (three modes) on top of `BernoulliEProcess`.
Done when: monitor mode `opt` never counts UNKNOWN as a miss and `pess` always does; detection delay on an injected shift (miss rate 0.01 to 0.10) is measured in a test and written to the progress note; false-alarm rate on 500 unshifted streams at most `delta` (with Wilson).
Tests: sets always contain the point estimate bin; `health_set` of an interval wider than the range clips to (0, 3); mutation: counting UNKNOWN as covered in `pess` mode is caught.

### P5a Shield: abstraction, solver, table, runtime
Deliver the two models exactly as 0001 section 4.1 (the spike `docs/adr/spikes/shield_feasibility.py` is the reference behaviour: the same counts must come out: 1,728 / 1,692 / 432 / 6.08 mean safe commands; 400 / 256), `solve`, `build_shield`, `ShieldedProposer`, serialisation with a SHA-256.
Done when: spike numbers reproduced exactly; the shield table for widths 1-4 has no empty entry; `ShieldedProposer` substitutes by priority and counts overrides; a shield for a stale spec (`retest = 8`) is buildable (parameterised model).
Tests: winning region is a fixed point (re-applying the operator changes nothing); every allowed command's outcomes lie in the winning region (brute force); STOP is in every entry's allowed set when the crusher is RUN and bin 3 is possible; table hash stable across processes; a hostile command stream (random commands for 10^5 steps against the abstract model) never reaches VIOL when filtered; unfiltered, it does (so the test has teeth).

### P5b Shield: independent checker
Deliver, from the English description only (0001 sections 4.1, 4.4, 5), `ReferenceModel` implementations for both games, `check`, `differential`, and the sabotage suite. The engineer must not open `shield/abstraction.py` before the differential test passes; any disagreement is resolved by reading the spec and recording which side was wrong in the progress note (this is the point of the packet).
Done when: `differential` returns `[]` for both models; `check` finds no finding on the true table for widths 1-4 and reachable counts equal the spike's (200, 496, 792, 992) or the difference is explained in the note.
Tests: sabotage matrix, each must turn the checker red: (1) add `BEGIN` on a reachable LOCKED entry; (2) remove the `soon` guard from ENTER; (3) allow `START` under lock; (4) drop the refusal branch from the model (differential red); (5) let ENTER ignore the alarm in the model (differential red); (6) remove STOP from the pump table when the other pump is in WORK. Also a test that a clean shield passes.

### P5c Shield: conformance
Deliver `AbstractionMap` for the F1 plant (pump LOCK maps to NOOP, pump PROVE to ISOLATE, 0001 section 4.1) and `conformance`.
Done when: on 200 F1 episodes with the Careless proposer and shield off, the report lists unmatched ticks per variable and A_phys violations with Wilson intervals (the numbers are whatever they are; they are recorded in the progress note and in a `ResultRecord`).
Tests: a synthetic trace with one injected impossible transition is counted exactly once under the right variable; a trace with a bin skip counts one A_phys violation; reports are deterministic.

### P6a Proposers: Protocol and baselines
Deliver B0-B3 (B3 parameter `k` days; B2 threshold; B1 interval as a fraction of the class characteristic life).
Done when: each baseline completes an F1 episode without error and issues only commands from the job runner's alphabet.
Tests: B0 raises a work order only after a failure; B3 raises one when `lo < k` and not before; determinism; each baseline's `reset(seed)` clears state.

### P6b Proposers: hostile and LLM stub
Deliver `Careless`, `Gullible`, `LLMProposer`.
Done when: with the shield off, Careless produces at least one P1 or P3 violation in 100 episodes at the default `p_skip` and Gullible produces violations from malicious notes (counts recorded, not asserted beyond "greater than zero"); with the real `scan_text` supplied (when plant_copilot is installed) the screened Gullible ignores notes that `scan_text` flags, and the unflagged malicious notes are listed (the screener's miss list is an output).
Tests: `LLMProposer` raises `NotConfigured` by default and **no test constructs a client**; a socket-blocking fixture (as plant-copilot does) proves no network access in the whole suite.

### P7 Causal
Deliver 3.9 in numpy and scipy only. `generate_observational` uses the F1+D plant with the operator policy of 0001 section 7.
Done when: on 300 training lives and 300 test lives the three models run and `score` returns Q1-Q4 records, with paired confidence sequences, whichever way they come out.
Tests: abduction recovers the true exogenous terms of a noiseless life; the SCM counterfactual equals the plant twin exactly on a noise-free configuration; the correlational model's bias is positive on a confounded dataset and about zero on a randomised one (this shows the confounding is real); mutation: removing the abduction step makes the noise-free test fail.

### P8a Planner: world models and belief
Deliver `PlantWorldModel`, `TabularWorldModel`, `BeliefTracker` (importance weights, systematic resampling, jitter).
Done when: on simulated lives the posterior mean of remaining life covers the truth at least 80% of the time at the 90% credible level (recorded, not tuned to hit); blind-window days leave the prior unchanged.
Tests: weights normalised; effective sample size triggers resampling; missing HI skips the update; `TabularWorldModel` rows sum to one; determinism.

### P8b Planner and exact DP
Deliver `BeliefPlanner`, `PlannerProposer`, `exactdp.exact_value`, the validation test of 0001 section 8.
Done when: the exact-DP test passes at its stated targets (5% value, 90% agreement) or fails honestly and the miss is written in the progress note (do not widen the targets); one episode of F1 with the planner measured: seconds per episode recorded in the note against the 1.0 s target.
Tests: the Lagrangian raises `lambda` until the estimated failure probability is at most `delta_plan` when achievable and reports infeasibility when not; plan respects crew, pump-pair and one-confined-space-per-shift constraints; INSPECT chosen when a preposterior value of information exceeds its cost on a hand-built belief; determinism; mutation: dropping the constraint term changes the chosen plan on a hand-built case.

### P9a Search: CMA-ES and random search
Deliver `CMAES` (weights, evolution paths, step-size control, covariance update, IPOP restarts, box repair) and `random_search`.
Done when: sphere (d = 10) reaches `f < 1e-8` within 6,000 evaluations on 10 of 10 seeds; Rosenbrock (d = 5) reaches `f < 1e-6` within 20,000 on at least 8 of 10 seeds (stated honestly if not); box constraint never violated.
Tests: ask/tell shape contracts; determinism; restarts double the population; random search uniform (KS test).

### P9b Search: falsification loop and shrinking
Deliver `falsify` (objective = min robustness from `spec.evaluate` on the F1 trace of the given arm and decoded scenario) and `shrink`.
Done when: against the arm "Careless, shield off" the loop finds a violation in the in-assumption regime within 300 evaluations on at least 2 of 3 seeds (a sanity check that the objective can be minimised, not a result); against the shielded arm in `R_in` it reports no violation (if it reports one, that is a defect to be written to `docs/FINDINGS.md` with the shrunk scenario).
Tests: `shrink` returns a scenario that still violates and has no more non-nominal coordinates than the input; evaluation is deterministic; the budget is respected.

### P10 Audit
Deliver 3.12 and `benchmarks/run.py` skeleton (`--quick`, `--full`, `--check`; commit-charge guard via `ctypes`; lock-file option).
Done when: a ledger can be created, appended to and verified; tampering with any byte of any line fails `verify`; `run.py --check` recomputes the prereg and code hashes and compares them with the ledger.
Tests: CRLF and LF copies of the same prereg give the same hash; trailing spaces ignored; changing one character changes the hash and the held-out seeds; `HeldoutGuard.seeds` raises before a freeze entry and works after; every use increments the counter; deleting the last ledger line is detected when the head is supplied; the guard refuses to start above 88% commit charge (the reader is injected so the test does not depend on the machine).

### P11a F2 spike: rendering, splicing, clause spec
Deliver `render_life`, `splice`, `spec_from_copilot`. **Start with a spike** (`docs/adr/spikes/splice_lives.py` plus its `.out.txt`): render two lives for one pump (the first ends in a replacement at day 40), concatenate with `MaintenanceEvent(kind="repair")` at the join, run `run_asset`, and report whether segments, baselines and scores behave (blind window, no phantom alarm at the join, scores before the join identical to a single-life run). If this fails, stop and write the outcome to the progress note and `docs/FINDINGS.md`; the fallback in 0001 section 15 item 2 applies.
Done when: spike output committed; `spec_from_copilot(date(2026,10,1))` returns `retest_q = 2` with the verbatim clause, doc id, clause number and revision 3, and `date(2026,1,1)` returns `retest_q = 8` (Rev 2).
Tests (marked `f2`): rendering is deterministic for a seed; rendered columns equal `asset_health`'s schema; the clause regex fails loudly (raises `ClauseError`) on a text without a number; Arabic clause text is ignored with a recorded reason.

### P11b F2 adapters
Deliver `AssetHealthAlertSource`, `PlantCMMS`, `SimHumanApprover` per 0001 section 10 (mismatches 4-8, 11-17 handled and each handling commented with its mismatch number).
Done when: one alert produced from a rendered degrading pump goes through the real `Orchestrator.run`, reaches `AWAITING_APPROVAL`, is signed by `SimHumanApprover` through the real gate, and `PlantCMMS.release_work_order` returns a work order and queues a plant job; the same alert with a clock 8 days later is `BLOCKED` by V10 (this reproduces the owner's observation).
Tests (`f2`): a draft the verifier blocks cannot be signed; a wrong-role approver is refused (`ROLE_INSUFFICIENT`); the alert's `as_of` equals the sim date; part stock comes from the plant; duplicate alert id is refused.

### P11c F2 episode, replay, mismatch tests, findings
Deliver `run_f2_episode` (causal caching rule of 0001 section 3), `replay_theta`, `mismatch.py` (one test per mismatch in 0001 section 10.3 that can be mechanised), `docs/FINDINGS.md` (table: id, finding, file and line, severity, status "recorded, not fixed"), `scripts/verify_siblings.py`.
Done when: three F2 episodes complete on the owner's machine within the 30 s per episode budget or the measured time is recorded; `FINDINGS.md` lists all 21 mismatches with evidence.
Tests (`f2`): matched-seed F1 and F2 episodes share the plant's exogenous streams (compare latent failure times); `run_asset` call count per episode is logged and equals the caching rule; the estimator mismatch (F1 vs F2 RUL error) is computed for 20 lives and written as a `ResultRecord`.

### P12a Eval: arms, episode loop, metrics
Deliver `Arm` definitions for every arm in 0001 section 11, `run_episode` for F1 (wiring plant, estimator, provider, monitor, proposer, shield, spec evaluation), `metrics.py`.
Done when: every arm runs one nominal and one shifted episode; `EpisodeResult` fields filled; a regression test pins one episode's summary for a fixed seed (golden file, regenerated only by an explicit flag).
Tests: shield off vs on differ only in `shield_*` fields for a proposer that never violates; Careless with shield on has zero P1-P5 violations in 200 episodes (if not: a defect for P5/P3 to fix); arms are paired (same seed, same exogenous streams: compare life tables); wall time per episode recorded per arm (written to the progress note, compared with the 0.3 s and 1.0 s targets).

### P12b Eval: runner, report, run.py wiring
Deliver `run_cell` with batches and sequential stopping, multiprocessing with persistent workers (`spawn`, initializer), `report.py` writing `results/results.json` (list of `ResultRecord`) and `results/RESULTS.md` (generated tables, each row with `k/n`, fidelity, measured-on), and the `run.py` body: freeze check, guard, ledger append, `--quick` (no ledger, no held-out seeds: uses `dev` seeds and a banner "QUICK RUN, NOT A RESULT").
Done when: `python benchmarks/run.py --quick` completes in at most 8 minutes (excluding first imports) on the owner's machine, or the measured time and the cut taken are recorded.
Tests: sequential stopping stops when the half-width target is met and not before; results are identical for 1 and 3 workers; the runner refuses a full run without a freeze entry; `RESULTS.md` is exactly what `results.json` generates (the sibling repositories test this too); worker count capped at 3.

### P13 Docs
Deliver `README.md` in the owner's standard shape (one H1; four badges; blockquote; Why this exists; Quick start; How it works with a mermaid diagram; Results copied by script; Design decisions; Limitations; Project layout; Tests; author footer), the `WHAT-GETS-THROUGH.md` generator (violations found beyond assumptions, conformance gap, unconfirmed F2 replays, known gaps in the spec and abstraction), and `CITATION.cff`.
Done when: every number in `README.md` is produced by `run.py` (a test greps README numbers against `results.json`); the Limitations section lists items from 0001 section 15; no employer, client or customer name appears (a test greps the tree for a deny list kept outside the repository).

## 5. Integration checkpoints (owner: whoever finishes the later packet)

- **After wave 3:** one F1 episode with B3, Careless and no shield produces violations; with the shield produces none; `Trace` feeds `spec.evaluate`.
- **After wave 5:** all arms run; paired design verified; timing table filled with measurements replacing the (A) rows of 0001 section 13 in `docs/adr/progress/TIMINGS.md`.
- **Before freezing the pre-registration:** the packets' recorded timings and the conformance numbers are reviewed; any pre-registered `n_max` is adjusted once, before the freeze, and the change logged.

## 6. Progress note template (`docs/adr/progress/<packet>-<slug>.md`)

```
# <packet> <title>
Engineer: <agent or person>      Started: <date time>      Status: IN PROGRESS | DONE | BLOCKED
Depends on: <packets>            Files owned: <paths>
## Done so far (append one line per step, newest last)
- <time> <file or test> <what is true now>
## Tests run (command, result, count)
## Mutations killed (function, mutation, test name)
## Measurements recorded (name, value, what it was measured on)
## Next step
## Blockers and questions (for the owner of the interface concerned)
```
