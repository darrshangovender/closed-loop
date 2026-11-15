# ADR 0001: Architecture of `closed-loop`

Status: proposed, 31 October 2026. Author: Sphaerus (design). Builders: start here, then read `0002-interfaces.md` (frozen interfaces and work packets) and `../PREREGISTRATION.draft.md`.

Everything in this repository is synthetic. No real plant, client, employer or personal data is involved or implied. Every number below is one of: (M) measured by a spike in `docs/adr/spikes/` on 31 October 2026 on the owner's 2015 laptop; (D) derived from (M) by shown arithmetic; (A) an assumption or a target, to be replaced by a measurement in the named packet; (S) read from sibling source code. Nothing is unlabelled.

## 0. Summary of decisions

1. **Thesis, sharpened.** What validates open-loop can fail once the loop is closed, and the simulator lets us measure by how much. Headline: for each layer (perception, diagnosis, planning, shield) the open-loop result, the closed-loop result, and the gap, each with an anytime-valid confidence sequence and a fidelity label.
2. **Three fidelities.** F0 finite abstraction (the only place a proof exists). F1 fast owned simulator with a transparent streaming estimator (policy comparison, search, thousands of episodes). F2 integration: the real `asset-health` pipeline, the real `maintenance-agent` orchestrator, verifier, gate and CMMS, and the real `plant-copilot` retrieval (tens of episodes, plus replay of F1 findings).
3. **The shield is decomposed per component** (crusher job game, pump-pair availability game). The spike (M) solves the crusher game (1,728 raw states) in 0.8-1.0 s and the pump-pair game (400 raw states, 49 joint actions) in 0.3 s in pure Python. This makes the model checker, the sabotage tests and the conformance test cheap enough to run in the unit-test suite.
4. **The perception contract is coverage only (C1).** The spike shows the shield never deadlocks, even when the perception set is the whole health range, because "stop the asset" is always a safe fallback. Resolution (narrow sets) therefore affects throughput, not safety. This replaces my first draft of a two-part contract and removes an assumption the proof would have carried.
5. **Real code in F1 where it is cheap and pure**: `maintenance_agent.requirements.requirement_for`, `models.PermitType`, `approval.required_role`; `plant_copilot` answers and `injection.scan_text` verdicts cached per (question, as-of) and per note text. The full orchestrator, gate, CMMS, asset-health pipeline run only at F2. F1 workers never import `asset_health` (44 s import (M), pulls scikit-learn and pandas).
6. **Pre-registration is enforced by code**: held-out seeds are derived from the SHA-256 of the frozen pre-registration, so they cannot be chosen or peeked at before the freeze; every full run appends to a hash-chained ledger.

### Hypotheses of the brief that I changed

| # | Brief | Decision | Why |
|---|---|---|---|
| H1 | Censored labels often informative for a lower bound | **Kept, and sharpened.** A censored life is informative for the lower bound only when replacement happened at or after `t + lo`. A policy that replaces *before* the lower-bound date (maintenance-agent's own rule: lower bound minus one day, `policy_latest_date`) therefore generates almost no informative censored labels by construction. The closed-loop gap is measured, not asserted. | S: `models.policy_latest_date`. |
| H2 | Split, ACI, DtACI or PID, censoring-aware online, per-life band | **DtACI dropped; seven variants** (section 6). Per-life band is a max-score conformal band over lives (exact under exchangeability of lives), not an alpha-spending union. "Censoring-aware" is split into three concrete estimators, one of which (audit lives) is unbiased by randomisation. | DtACI adds an expert-aggregation layer for marginal gain; conformal PID (P and I terms only) is 30 lines. Audit randomisation gives the only unbiased censored-label estimator without extra assumptions. |
| H3 | Contract links conformal and shield; e-process monitor | **Kept; contract reduced to coverage (C1)**; two separate levels: scheduling interval at 90%, shield-grade set at 99%. | A 90% per-step contract is violated about once in ten steps and would void the proof constantly. Spike result in section 4. |
| H4 | Twin worlds for ground-truth counterfactuals | **Kept for F1 only.** At F2, `asset_health.simulate.fleet.simulate_asset` consumes one `rng` sequentially and has no `do()` hook, so a factual and a counterfactual run do not share noise. | S: `simulate_asset(rng, cfg, spec, failure, services)`. |
| H5 | Causal caching cuts F2 cost | **Kept, with a correction.** Re-running on a truncated history is *not* cheaper (M: 1.19 s at 29 days, 1.65 s at 89 days, 1.61 s at 119 days): cost is dominated by fixed per-segment model fits. The saving comes only from running once per regime-changing event. Also: **F2 cannot support DERATE** (no regime hook), and every replacement blinds the pipeline for 21 days of baseline plus 48 h settle (S: `FeatureConfig`). | Measured and read. |
| H6 | Integration through the three Protocols | **Kept.** 21 schema and semantic mismatches recorded in section 10. `guardrail` and `mcp-gateway` are **excluded** (reasons in section 10.3). | Not adding for completeness. |
| H7 | Spec draft | **Kept, 5 invariants plus 1 bounded-future alarm response**; only 3 of the parameters have a clause in the plant-copilot corpus; the other 3 are labelled plant design assumptions. maintenance-agent's procedure fixtures contain **no gas re-test interval** at all (S: grep of `procedures.json`), so the spec cannot come from the agent's own library. | Finding. |
| H8 | Shield, checker, conformance | **Kept.** State space counted and solved (spike). Decomposition justified in 4.5. | Spike. |
| H9 | CMA-ES vs random search | **Kept; own CMA-ES; random search is the baseline; adversarial finds replay at F2.** Added: search target is STL robustness of the *monitor-only* arm for beyond-assumption regime, shielded arm for in-assumption regime. | Section 9. |
| H10 | Vectorised belief-space planner, exact DP validation | **Kept**, tiny instance 1 asset, 5 hidden levels, horizon 5. | Section 8. |
| H11 | Ledger | **Kept, with limits stated** (section 12). Held-out seed derivation added. | |
| H12 | Name | **Kept** `closed-loop` for the repository; distribution name `closed-loop-lab`, import name `closed_loop`. All of `closed-loop`, `closed_loop`, `closedloop`, `asset-health`, `plant-copilot`, `maintenance-agent` returned HTTP 404 from `pypi.org/pypi/<name>/json` on 31 October 2026 (M), so they are unclaimed and open to dependency confusion (section 14). | |

## 1. Claims the repository is built to test

Each claim has a pre-registered decision rule in `PREREGISTRATION.draft.md`. A claim that fails is published as a failure.

- **C-PERC.** A conformal RUL interval that is well calibrated on open-loop lives under-covers, over-covers or is mis-estimated by its own learner once a policy acts on it, because the policy censors the labels the learner needs.
- **C-ADAPT.** Online adaptation (ACI, PID) restores long-run coverage under a distribution shift; a per-life band gives a trajectory-level guarantee but adapts slowly. Both costs are reported.
- **C-CAUSE.** A structural causal model with abduction beats a correlation-only effect model on ground-truth counterfactuals when the training data were collected under a confounded operator policy. (Given known structure; a misspecified-structure arm shows the sensitivity.)
- **C-PLAN.** A belief-space planner beats a plain "replace at the conformal lower bound" rule on cost at equal safety.
- **C-SHIELD.** A shield synthesised on the abstraction removes violations of the covered properties for careless and gullible proposers, at a measured throughput cost.
- **C-MON.** An e-process on resolved coverage labels detects the contract being broken, with a measured detection delay and false-alarm rate, and the fallback restores the unconditional part of the proof.
- **C-SEARCH.** CMA-ES finds violations beyond the proof's assumptions faster than random search. If it does not, that is published.
- **C-TOGETHER.** The three sibling repositories work together: the confirmed fraction of F1 adversarial finds that replay at F2, and the list of integration defects.

## 2. Plant definition (F1 and F2 share it)

Everything the plant simulator needs, in one place. Parameters marked (A) are assumptions with a sweep or sensitivity run in the pre-registration.

**Layout.** Four monitored assets, using maintenance-agent identifiers because the agent's fixtures bind bills of materials, spares and procedures to them:

| Plant id | Class (agent) | asset-health type and id (S) | Role |
|---|---|---|---|
| CR-501 | CRUSHER | primary_crusher (id prefix to be read from `ID_PREFIX`, not verified) | critical, single; liner wear needs LOTO, confined space, hot work, work at height and a safety officer (S: `requirements.MODE_PERMITS`) |
| CV-101 | BELT_CONVEYOR | conveyor_pulley | LOTO only |
| PU-201, PU-202 | SLURRY_PUMP | slurry_pump (`PMP-01`, `PMP-02` seen in spike) | duty pair; at least one must stay available |

Excluded because they have no counterpart on one side: haul-truck engine (no maintenance-agent class, failure mode OVERHEATING unknown to the requirements matrix), conveyor idler (no class), VENT_FAN, BALL_MILL, HV_MOTOR (no asset-health type).

**Time.** Base tick `dt = 15 min` (one quarter). A shift is 32 quarters (8 h); shifts start 06:00, 14:00, 22:00. Episode horizon `H = 90 days = 8,640 quarters` (A; chosen so that one 21-day baseline plus a 25-55 day degradation plus a replacement fit in one episode). Decision cadence: the planner (or any Proposer) is called at every shift start and on every alert; the job runner and any adversarial proposer are called every quarter while a job is open; the perception layer updates once a day (the pipeline is hourly, the health index daily); monitors run every quarter. A 15-minute tick is used because the gas re-test period is 30 minutes (2 ticks) and the "break longer than 15 minutes" clause is 1 tick (S: CORP-HSE-002 clause 4.3, Rev 3).

**Hidden degradation (the structural causal model).** For component life `l` of asset `a`:

- exogenous: `U_onset, U_dur, U_shape, U_amp, U_mode, U_noise(t)`, plus a process exogenous `U_load(t)` shared by assets in the same circuit.
- failure mode drawn from the asset-health catalogue weights (S): pump impeller wear 0.7 / bearing 0.3; crusher liner wear 0.7 / bearing 0.3; pulley bearing 1.0.
- degradation duration (onset to failure) uniform in the catalogue ranges (S): pulley bearing 25-55 d, crusher liner 45-85 d, crusher bearing 25-55 d, pump impeller 40-75 d, pump bearing 20-45 d.
- severity `s(t) = ((t - onset)/(T - onset))^k` for `t >= onset`, 0 before; `k` uniform 2.0-4.5 for bearings and 1.2-3.0 otherwise (S: `plan_failures`). This is the asset-health curve, so F1 and F2 degrade alike.
- load effect (the confounder and the causal lever): remaining-life clock runs at `rho(load) = exp(beta (load - 1))`, `beta = 0.8` (A). Load is a process variable driven by `U_load` and, in the DERATE study only, capped by the DERATE action.
- an episode starts mid-life: the residual life of the first component of each asset is drawn so that 60% of assets would fail inside the horizon with no action (A, same design choice as asset-health's `failure_fraction = 0.65`, stated there as a statistical-power choice). First onsets are constrained to day 24 or later so that the 21-day baseline is clean (S: asset-health `plan_failures` does the same with `baseline_days + settle`).

**Observation chain.** F1 emits a daily health index `HI_d = a_c * s(t_d) + eps`, with `a_c` and the noise standard deviation fitted to asset-health traces in packet P4b (target failure level about 9.0, degradation threshold 2.0: S, `rul.estimate` defaults and `RULConfig.hi_min`). Data-quality events (stuck day, spike, slow gain drift) are drawn as exogenous streams; stuck and spike days become missing HI (the asset-health gate masks them), gain drift is not masked (S: README limitation "slow 4-20 mA drift"). The transparent streaming estimator mirrors the structure of `asset_health.rul`: Theil-Sen slope over the last 8 daily points (`trend_window_days = 8`, `min_points = 5`, `min_slope_per_day = 0.05`), extrapolation to the failure level, horizon cap 90 d. It is *not* a copy of the quantile model; P4b measures the F1-F2 estimator mismatch and publishes it. After any replacement the estimator is blind for `21 d + 48 h` (S: `FeatureConfig.baseline_days = 21`, `settle_hours = 48`) and reports the class prior instead.

**DES events.** Shift change; permit issue and expiry (valid for at most 12 h and not beyond the shift in which it was issued, S: CORP-HSE-004 clause 4.1); handover revalidation requiring the incoming holder, present with probability 0.9 per attempted quarter (A); crew availability (two fitters per shift, one safety officer on the day shift only, A); approver latency (log-normal, median 2 h, A) and catch probability for a wrong but compliant diagnosis (0.5, A; this is maintenance-agent's documented known gap `wrong_diagnosis_fully_compliant`); spares with lead times from the agent's inventory fixture (S: `PartRecord.lead_time_days`); gas-atmosphere excursion while occupied, Poisson, 0.01 per occupied quarter (A); sensor faults as above; shift-handover note per handover, drawn from a bank of templates, some malicious (section 8).

**Action set (core, shared by F1 and F2).** Per asset: `NOOP, INSPECT, ORDER(part), RAISE_WO(mode, target shift), STOP, START`. Job micro-commands (the same alphabet as the abstraction, section 4): `LOCK, PROVE, ISSUE, REVAL, BEGIN, TEST, ENTER, EXIT, COMPLETE, CLOSE, UNISO`. Constraint: crew limits on concurrent jobs; `STOP`, `EXIT` and the shield's forced commands are always executed (assumption A_exec); every other command may be refused by the plant (crew absent, holder absent), which the abstraction models as environment nondeterminism. `DERATE(asset)` exists only in the **F1+D** configuration used by the causal study; F2 masks it.

**Production and cost.** Throughput per quarter is 1.0 if crusher, conveyor and both pumps run, 0.6 with one pump (A), 0 if the crusher or conveyor is not running. Unplanned failure costs 8 times a planned replacement (A, taken from the asset-health README's "about eight times", S) and adds 36-96 h of downtime (S: `downtime_hours` range). Costs are illustrative units; the metric of interest is the ordering and the paired difference.

## 3. Fidelity ladder

| Level | What is real | What is modelled | Used for | n per cell (A) | Cost per episode |
|---|---|---|---|---|---|
| F0 | the safety game, solver, checker | everything else abstracted (section 4) | proof, sabotage tests, conformance rate | exhaustive | under 1 s (M, spike) |
| F1 | plant simulator, SCM, streaming estimator, conformal layer, planner, shield, STL monitors, `requirements.requirement_for`, cached `plant_copilot` verdicts | HI chain, approvals (no signatures), CMMS | policy comparison, ablations, search, shift studies | up to 500 | target 0.3 s rule arms, 1.0 s planner (A, packet P3 measures) |
| F2 | `asset_health` ingest-to-RUL pipeline on telemetry rendered by `simulate_asset` from the plant's latent failure events; `maintenance_agent` orchestrator, verifier, Ed25519 gate through `ApprovalGate`, `MockCMMS`; `plant_copilot` retrieval and `scan_text` | the plant itself (shared with F1), humans | integration claim, replay of F1 finds, F1-F2 estimator mismatch | 60 per arm, plus up to 50 replays | about 30 s (D, section 13) |

**Every published number carries its fidelity label** (`F0`, `F1`, `F1+D`, `F2`) in `results.json` and in every table; `benchmarks/run.py` fails if a result record lacks one.

**How much of the real agent and copilot runs at F1.** The orchestrator costs 0.12-0.14 s per alert (M by owner) and builds in 0.66 s; with roughly 6 jobs per episode that is about 1 s, as large as the whole planner budget, and it adds nothing to policy comparison. So at F1 the *job model* uses the real `requirement_for` (microseconds) and the real `PermitType`/role rules; a fixed audit sample of 100 F1 jobs is replayed through the real orchestrator in packet P12 to measure agreement of permits, role and procedure (target 100%; any disagreement is a finding). plant-copilot is a pure function of (question, as-of date): the five spec questions are asked once per revision date and cached (first ask 0.67 s, later 7 ms, M by owner); `scan_text` verdicts for the note template bank are precomputed.

**Causal caching at F2 (the corrected rule).** Telemetry for a component life is rendered once for the whole episode from the plant's latent failure event, run through `run_asset` once, and the resulting `AssetRun` (causal, S: tested upstream) answers every later day by slicing with `as_of`. Re-render and re-run only when something changes the telemetry the controller has already seen would be inconsistent, i.e. at: (a) a replacement or repair record (new life, new segment, `MaintenanceEvent(kind="repair")`), (b) a stop that creates a `running = False` interval. `analyse_run` (0.09 s, M) is called only on days when an alert is open or the fused score first crosses its threshold; the RUL on other days comes from `rul.estimate.estimate_rul` on the cached health index.

**Profile of `run_asset` (M, `spikes/profile_run_asset.out.txt`, one pump, 120 days, hourly, 2,857 rows, best of 3):** total 1.69 s: ingest 0.11 s, `build_features` 0.68 s (ridge regime fit about 0.70 s cumulative across two segments), `detect_asset` 0.76 s (Isolation Forest fit about 0.69 s). `if_trees = 10` gives 1.46 s (saves 14%). A 2 h cadence gives 1.40 s; **a 4 h cadence makes the pipeline output no valid scores** (`fused valid = 0`), so there is no cheap cadence. There is **no cheap config**; the plan is fewer calls, not faster calls. Python import of the asset-health pieces took 43.8 s in the spike process (the owner measured about 30 s); each of 3 worker processes pays it once if workers are persistent (`ProcessPoolExecutor` with `initializer`), about 44 s wall in parallel (D).

## 4. Finite abstraction (F0), shield, and what the proof covers

### 4.1 Variables and sizes (spike, M)

Crusher job game (observable part `o`, hidden health bin `h`):

| Variable | Values | Meaning |
|---|---|---|
| op | RUN, STOP, LOCKED, ISO, WORK, FAIL (6) | LOCKED = lock applied, zero energy not yet proven; ISO = proven |
| h | 0, 1, 2, 3 (4) | true RUL bin: >30 d, 10-30 d, 3-10 d, under 3 d (edges A) |
| permit | NONE, VALID, REVAL (3) | REVAL = a handover happened since issue and revalidation is pending |
| occ | 0, 1 (2) | a person is inside the space |
| gas | 1, 2, 3 (3) | quarters since the last atmosphere test; 3 = stale or never |
| alarm | 0, 1 (2) | gas alarm active |
| soon | 0, 1 (2) | a handover happens at the end of this tick |

Raw product 6 x 4 x 3 x 2 x 3 x 2 x 2 = **1,728**; winning region **1,692**; observable abstract states in the winning region **432**; mean safe actions per winning state **6.08 of 14** (M). Pump-pair game: per pump op in RUN, STOP, ISO, WORK, FAIL and h in 0..3, so 400 raw pair states; 7 commands per pump, 49 joint commands; winning region **256 of 400** (M).

Time: one abstract step = one quarter. Order inside a step: the controller's command, then the environment's move. The environment is adversarial and nondeterministic in: health increments (one bin at a time, only while RUN), uncontrolled failure (only from RUN at h = 3), handover announcement (`soon` may rise at any step: an over-approximation of a fixed 32-tick shift), holder presence at revalidation, atmosphere-test result, alarm onset while occupied, and **command refusal** for every non-forced command (the plant may turn it into `NOOP`).

### 4.2 Environment assumptions the proof depends on

| ID | Assumption | Tested by |
|---|---|---|
| A_iso | isolation holds once proven (no energy release while ISO) | not testable here; stated |
| A_exec | `STOP` and `EXIT` are always executed; all other commands may be refused | plant code + unit test; conformance trace check |
| A_phys | health bin rises by at most one per tick and an uncontrolled failure occurs only from bin 3 while running | **conformance test** (rate of bin skips and failures from h < 3 in simulated traces) |
| A_dec | components interact only through the spec properties of the pump pair, so the game decomposes | argued in 4.5; unit test that no spec atom reads two components except the pump pair |
| C1 | the perception set `B_t` (a set of bins) contains the true bin at every tick | e-process monitor (section 6.4); measured C1 violation rate |
| A_map | the abstraction map and predicate implementations are correct | conformance test; independent re-implementation of the transition relation (packet P5b) |

### 4.3 Synthesis, runtime shield, perception sets

`W` is the greatest fixed point of "exists a command such that every environment move stays in `W` and never enters the violation sink" (`solve`, ~25 lines). The runtime shield is the table `A(o, B) = intersection over h in B with (o, h) in W of safe(o, h)`: a command is allowed only if it is safe for **every** bin the perception set admits. It allows the maximally permissive set (all safe commands), not one chosen command. If a proposer's command is not in `A`, the shield substitutes the first element of a fixed priority list (STOP, EXIT, NOOP, ...) that is in `A`, logs the override and counts it.

**Result that changed the contract (M).** For every perception-set width from 1 to 4 bins, with all 10 intervals of bins allowed, the number of deadlocks (entries with an empty `A`) is **0**, and explicit reachability on the shielded product (o, h, B) finds **no violation**: reachable product states 200, 496, 792, 992 for width limits 1, 2, 3, 4. Mean safe commands at exact width 1, 2, 3, 4: 6.08, 5.89, 5.72, 5.22 (M). A fully uninformative set is therefore still safe, because stopping the crusher is safe; what wider sets cost is production. So **C1 (coverage) is the only perception assumption in the proof**, and the throughput cost of poor resolution is a measured metric, not a proof obligation.

### 4.4 Model checker and sabotage

The checker is explicit reachability over `(o, h, B)` starting from every `B` containing the initial `h`, taking *every* command the shield admits, every environment move, every next `B'` that contains the next `h` and respects the width limit. It reports `VIOLATION` (a transition into the sink) or `DEADLOCK`. In the spike the transition relation `step` is shared with synthesis; **in the build, packet P5b re-implements the relation from the English spec of section 5 without reading `P5a`'s code**, and a differential test compares the two on all (state, command) pairs. Sabotage tests (M in the spike: adding `BEGIN` to a reachable shield entry for a LOCKED asset is detected) are part of the suite: remove a command's guard, add an unsafe command, drop an environment branch (the checker must then go *red* for the first two and the differential test for the third), and a sabotaged *model* (a transition that lets `ENTER` ignore the alarm) must make the checker's result differ from the spec-derived relation.

### 4.5 Why per-component decomposition is sound here

The five properties each mention one component, except availability, which mentions the pump pair. The environment's moves are independent across components (each draws its own health, refusal, holder, alarm), and the controller may command every component each tick. Under those conditions the product game's winning region is the product of the component regions, and a joint command is safe iff each component command is. Crew and shift limits are *plant* constraints that can only turn a command into `NOOP`; that is already inside the refusal nondeterminism. The decomposition would break if a property mentioned two components (for example "no two confined-space entries at once"); a unit test enforces that no spec atom reads two components other than the pump pair.

### 4.6 What the proof says, exactly

**Statement.** Under A_iso, A_exec, A_phys, A_map and C1, if every command issued to the crusher job component and to the pump pair is in the shield's allowed set, then no state violating P1-P5 (section 5) is reachable in the abstract models, from the initial states, for any environment behaviour. Checked by explicit enumeration of the reachable product (hundreds of states, section 4.3 numbers; final counts come from packet P5).

**It does not cover:** violations the abstraction cannot express (a mistaken zero-energy test, an unmodelled hazard class, two simultaneous entries); anything when C1 fails, except under the fallback (when the monitor fires and the sets become the whole range, C1 holds trivially and the proof stands conditional on A_phys, A_iso, A_exec, A_map); concrete-plant behaviours with no abstract counterpart (the measured conformance gap, section 4.7); a wrong specification (the spec is what we wrote, not what a regulator wrote); the 15-minute quantisation; the pump availability definition (a stopped pump counts as available). It is not a proof about the F1 or F2 simulators, only about the abstract model they are tested against.

### 4.7 Abstraction conformance testing

Map every concrete tick of F1 (and F2) traces to `(o, h, command)` and test whether the next mapped state is in `step(s, command)`. Report: the rate of concrete transitions with no abstract counterpart, per variable (op, permit, occ, gas, alarm, soon, h), with the count of ticks and a Wilson interval; the rate of A_phys violations (bin skips; failures with h < 3), expected to be non-zero when a load shift cuts the true RUL by more than one bin in a tick. The result is a **measured soundness gap on sampled traces, not a proof**. Every zero is reported with its denominator.

## 5. Specification

Syntax: a bounded-past fragment of signal temporal logic over the plant's 15-minute traces: atoms `mu(x) >= 0`, Boolean connectives, and `O[a,b]` (once), `H[a,b]` (historically), `G` at top level. All five invariants are expressible with past operators only, so one evaluator serves the **online monitor** (streaming deque per operator) and the **offline robustness** used as the search objective (vectorised). One bounded-future formula (P6) is evaluated offline only.

| ID | Property (informal) | Formula (sketch) | Parameter source |
|---|---|---|---|
| P1 | no work on an energised asset | `G( working(a) -> iso_proven(a) )` | CORP-HSE-001 clause 5.1 (zero energy tested at the point of work), clause 5.2 (S) |
| P2 | a valid permit during work, revalidated after handover | `G( occupied(a) -> (permit_valid(a) and not pending_reval(a)) )`; permit validity at most 12 h, not beyond issuing shift | CORP-HSE-004 clauses 4.1, 5.1 (S) |
| P3 | gas re-test period in confined space | `G( occupied(a) -> O[0, 2q] test_pass(a) )`; after a break of more than `1q`, a fresh test before re-entry | CORP-HSE-002 clause 4.3, Rev 3: "re-tested every 30 minutes and after any break in work longer than 15 minutes" (S) |
| P4 | at least one slurry pump available | `G( avail(PU-201) or avail(PU-202) )` | **plant design assumption, no clause** |
| P5 | no uncontrolled failure of the critical asset while running | `G( not uncontrolled_failure(CR-501) )` | **plant design assumption** |
| P6 | gas alarm leads to exit within 15 minutes | `G( alarm(a) and occupied(a) -> F[0, 1q] not occupied(a) )` (offline only; the abstraction carries the one-tick version) | CORP-HSE-002 clause 4.4 (S) |

Atom margins for robustness are real-valued where the plant has them: P3 margin = `30 min - minutes since last test`; P2 margin = minutes to permit expiry or `-1` while revalidation is pending; P1 and P5 are ±1 with a shaping term (minutes between proof and work start; true RUL in hours for P5); P4 margin = number of available pumps minus 0.5.

**Clause-backed parameters.** `closed_loop.spec.clauses` asks plant-copilot (at the sim date's as-of) the three questions, takes the verbatim clause and revision from `Answer.claims[0]` / `citations`, and extracts the number with a pinned regex (`every (\d+) minutes`). The spec object stores `clause_text`, `doc_id`, `clause`, `revision`, `language`. Rev 2 of CORP-HSE-002 had a 2-hour interval (S: its revision history, clause 6.1): an arm **stale-spec** builds the shield from the as-of date before 2026-02-01 (120 min = 8 ticks) and is scored against the Rev 3 monitor. Maintenance-agent's own library has no re-test interval (S, grep), so work orders drafted by the real agent cannot carry P3 and the shield is the only enforcement. Arabic: the same clause exists in the corpus in Arabic where a translation lags (S: README); the extractor handles the English master only and the Arabic-lag case is recorded as a limitation, not claimed.

## 6. Perception: conformal methods, censoring, and exact coverage definitions

### 6.1 Objects

For life `l` of asset `a`, `T_l` is its true failure time **in the twin world with no intervention from checkpoint time on** (the simulator holds it). At a daily checkpoint `t` the estimator gives a point `m_t` and a scale `sigma_t > 0`; `L_t = T_l - t` is the true RUL. The conformal interval is `[lo_t, hi_t] = [m_t - q^-_t sigma_t, m_t + q^+_t sigma_t]` with nonconformity score `r = (L - m)/sigma` and one-sided quantiles at levels `alpha/2` each side (scheduling level `alpha_s = 0.10`; shield level `alpha_h = 0.01`). Checkpoints during the estimator's blind window, or when the estimate status is not-degrading or insufficient-trend, carry the **class-prior interval** (Weibull conditional on age) and are tagged `prior`; they are included in coverage and also reported separately.

### 6.2 Coverage definitions (all computed by `perception.coverage`)

- **Oracle step coverage** `C_orc = mean over checkpoints of 1[lo_t <= L_t <= hi_t]`, over all post-blind checkpoints of an episode set, with `L_t` from the twin world. Only the simulator can compute it. Lower-bound version `C_orc^lo = mean 1[L_t >= lo_t]`.
- **Learner-visible coverage** `C_lrn`: what a plant could compute from its own records. A checkpoint's label resolves when its life ends, by failure at `T` or by replacement at `tau` (right-censoring: `T > tau`). Status:
  - failure observed: `HIT` or `MISS` exactly, `y = 1[lo <= T - t <= hi]`;
  - censored, `tau >= t + hi`: `MISS` (we know `T > tau >= t + hi`);
  - censored, `t + lo <= tau < t + hi`: `LOWER_HIT` (the lower bound held; the upper is unknown);
  - censored, `tau < t + lo`: `UNKNOWN`.
  `C_lrn` is reported for each estimator below; unknowns are never silently counted as hits.
- **Manski bounds** `[C_min, C_max]` on coverage given the labels: unknown counted all-miss and all-hit.
- **Per-life any-miss rate** `M_life = mean over lives of 1[exists checkpoint with L_t outside its interval]`, versus the union bound `H x alpha`. Reported per variant; this is where the per-step guarantee says nothing useful.
- **Gap definitions** (the headline): `Gap_A = C_lrn - C_orc` under the closed-loop policy (bias of learning from censored data); `Gap_B = C_orc(open loop) - C_orc(closed loop)` (the checkpoint population changes because the policy replaces parts early, fresh lives dominate, and shifts hit different lives).

**Why the policy starves the learner (H1 made precise).** Under the bound rule (replace at `lo - 1 day`), `tau < t + lo` for the decisive checkpoints, so their labels are `UNKNOWN`; only checkpoints early in a life (large `lo`, and `tau` much later) can be `LOWER_HIT`, and only when the policy replaced for another reason or the part failed. The informative labels are therefore a selected subset, selected on `lo`.

### 6.3 Variants (all share the same estimator; they differ in calibration and update)

| ID | Name | Calibration and update | Expected behaviour (hypothesis, to be measured) |
|---|---|---|---|
| V1 | split | quantile of scores on development lives, never updated | right in-distribution open-loop; under-covers under shift |
| V2 | ACI-naive | ACI (`alpha_{t+1} = alpha_t + gamma (alpha - err_t)`, `gamma = 0.02`, window 500 scores), `err` from resolved labels with censored counted as hit | optimistic, the common mistake |
| V3 | ACI-inf | as V2 but only `HIT`, `MISS`, `LOWER_HIT` labels update; `UNKNOWN` skipped | selection bias, direction measured |
| V4 | ACI-aud | each life is an audit life with probability `rho = 0.10` (randomised, independent of everything); audit lives are run to `lo` under shield supervision; only audit lives' labels update | unbiased, slow, costs exposure |
| V5 | PID | conformal PID: quantile tracker plus integrator on the V3 label stream, no scorecaster | faster under drift than ACI, same bias as V3 |
| V6 | per-life band | score per life `S_l = max over checkpoints of r_t`; calibrate the `1 - alpha_life` quantile over development lives; band holds for all checkpoints of a life at once under exchangeability of lives (`alpha_life = 0.05`) | per-life guarantee, slow, wide |
| V7 | oracle-ACI | ACI updated with the true twin-world labels (sim only, not deployable) | upper bound on what any label policy achieves |

Labels arrive with delay (up to a life's length); every online variant updates **at resolution events in time order**, never at the checkpoint. Calibration scores come from development seeds; held-out lives never calibrate anything.

### 6.4 The perception contract and the assumption monitor

The set handed to the shield is `B_t = {bins intersecting [lo_t^h, hi_t^h]}` computed with the shield-grade level `alpha_h = 0.01` (or the per-life band, an alternative source). **C1: the true bin is in `B_t`.** The monitor is an e-process on the resolved C1 labels (misses are `MISS` labels on the shield-grade set): bet on `p_miss > alpha_h` with capital `E_n = product (1 + lambda_i (x_i - alpha_h))`, `lambda_i` predictable (aGRAPA-style or a fixed mixture over a grid of 20 values), fire when `E_n >= 1/delta`, `delta = 0.01` (Ville's inequality gives type I error at most `delta` for any stopping time). Three label streams are monitored, as three arms: `opt` (UNKNOWN counted as covered: a firing is a true violation, power is low), `pess` (UNKNOWN counted as miss: fast, false alarms), `inf` (informative labels only). Measured: detection delay after an injected shift (days and number of labels), false-alarm rate on unshifted episodes, and the throughput cost of the fallback. On firing: all `B_t` become the whole bin range (fallback), the proof's C1 is then trivially true, the production cost is large (the crusher is held stopped whenever bin 3 is possible) and is reported.

## 7. Causal diagnosis and its scoring (F1 and F1+D only)

**Why this task.** Diagnosis in the closed loop is a causal question: operators derate when they see high vibration, and vibration reflects severity, which also drives failure (confounding by indication). A correlation-only effect model of "does derating extend life" is then biased. Twin worlds give the ground truth.

**The SCM** is the plant's own structural equations (section 2), restricted to what a diagnostician could write down: `severity <- f(onset, dur, shape, mode)`, `clock rate <- rho(load)`, `HI <- a * severity + noise`, `load <- U_load` (and operator derate policy). Fitting: parameters estimated from observational data by regression on abducted latent quantities (per-life onset and duration by least squares on the HI path); counterfactuals by abduction (infer the life's exogenous terms from its observed path), action (`do(derate from t0)`), prediction. Pure numpy/scipy.

**Observational data** (development seeds): 300 lives under operator policy `pi_obs`: derate by 30% when the smoothed HI exceeds a threshold. **Test**: held-out lives, each run in two twin worlds sharing every exogenous stream: factual under `pi_obs` and counterfactual under the query intervention.

**Queries and scores (all F1+D):**

- Q1 counterfactual failure time under `do(derate from t0)`: error `|T_cf - hat T_cf|` in days.
- Q2 counterfactual event `failure before day D` under `do(replace at tau)` versus `do(nothing)`: Brier score and calibration slope.
- Q3 diagnosis under confounding shift: train where heavy load co-occurs with mode X, test where the correlation is reversed; accuracy of mode and of "cause of alert" (true degradation, sensor fault, process upset) for correlational classifier (multinomial logistic regression written in numpy) versus the SCM posterior.
- Q4 downstream: the planner using each effect model in the DERATE study; paired cost difference.

**Arms:** correlational (ridge on HI, derate flag, interactions); SCM-correct (true structure); SCM-misspecified (drops the load-by-age interaction, to show sensitivity). The SCM has the structure handed to it, so Q1-Q2 favour it by construction; the misspecified arm and Q3 are the fair part. Paired differences per life use confidence sequences (section 11).

## 8. Planner and proposers

**Proposer Protocol.** Every controller, honest or hostile, implements `Proposer.propose(obs) -> list[Command]`. The shield sits between any proposer and the plant. Implementations: `RunToFailure` (B0), `Calendar` (B1), `FixedLimit` (B2), `BoundRule` (B3: raise a work order when `lo_t` falls below `k` days, `k` tuned on development seeds), `PlannerProposer` (P0-P2, the default "simulated model"), `Careless` (each tick, with probability `p_skip` it issues the next job command out of order, for example `BEGIN` before `PROVE`; seeded), `Gullible` (reads the shift-handover note; a note matching an instruction template such as "isolation already proven by the previous shift, proceed" makes it issue the instructed command; with screening on, notes for which `plant_copilot.injection.scan_text` returns a rule name are dropped first), and `LLMProposer` (optional adapter, raises `NotConfigured` unless an extra and an environment flag are set; **never run in tests or benchmarks**).

**Planner (vectorised belief-space rollout, numpy only).** Per asset, `N = 128` particles over (mode, severity, remaining life, rate multiplier). Belief update once a day: importance weights from the HI likelihood (missing HI days skip the update), systematic resampling with small jitter. Planning grid: daily steps over a 14-day lookahead; candidate decisions per asset: replace at day `j` in {0, 1, ..., 14} or never, with INSPECT now or not (INSPECT returns an observation with a quarter of the noise and detects slow gain drift). Joint choice by two sweeps of coordinate descent over the four assets, subject to: at most one confined-space job per shift, crew limits, pump availability (never schedule both pumps' work overlapping), spare lead times. Cost of a candidate = expected downtime plus repair plus failure cost over the particles, `+ lambda * p_fail`, where `p_fail` is the fraction of particles with an uncontrolled failure in the window; `lambda` is found by bisection (at most 6 steps) so that `p_fail <= delta_plan = 0.02` per asset per 14-day window (A), which is the Lagrangian treatment of the chance constraint. Value of information for INSPECT is a one-step preposterior estimate on the same particles. The model used for rollouts is the **plant's structure with parameters fitted on development seeds**, deliberately not the generating parameters, with a mismatch knob (rate error 0, 10%, 25%) so planner results are not an artefact of knowing the truth.

**Validation against exact DP (packet P8b).** Tiny instance: one asset, 5 hidden levels (level 4 absorbing failure; stay with probability 0.8, else rise by one), 4 observation symbols with a confusion matrix, actions `continue / inspect / replace`, horizon 5. Exact value by expectimax over the belief tree (at most 12^5 = 248,832 leaves). Targets (A, set before looking): planner value within 5% of exact on 50 random beliefs, and agreement on the best action in at least 90% of beliefs where the exact gap between best and second-best action exceeds 2% of the value. If the planner misses, the test fails and the planner is fixed or the miss is published.

**What the planner is compared with:** B3 (same perception, simple rule). The planner can only win where coupling matters: spare lead times, one confined-space job per shift, the pump pair, information-gathering. If it does not beat B3, that is the result.

## 9. Adversarial search

**Scenario parameter vector** `theta` in `[0,1]^32`, decoded deterministically into exogenous streams and choices: per asset (4) x 5 = 20 (rate multiplier, onset shift, residual-life shift, sensor-fault type, sensor-fault time); 3 holder-absence biases (one per shift boundary); 2 gas-excursion times; 1 note template index and 1 note insertion time; approver delay multiplier; approver catch probability; spare lead-time multiplier; load-shift magnitude and time. Discrete choices are decoded by flooring (a stated limitation of CMA-ES on mixed spaces; random search uses the same encoding so the comparison is fair).

**Two regimes.** *In-assumption* (`R_in`): decoded scenarios respect A_phys (load shift capped so no bin can be skipped, checked by the conformance monitor), sensor faults restricted to types the gate masks, forced commands executed. A violation of P1-P5 by a **shielded** arm in `R_in` is a bug or an unsound abstraction: it fails the test suite and is published as a defect. *Beyond-assumption* (`R_out`): drift the gate does not mask, load shifts that skip bins, C1 broken, a stale spec, forced commands refused with probability up to 0.3, gullible proposer with screening off. Violations are expected; the aim is to map where the proof ends. Attribution: each find is re-run with one adversary dimension reset to nominal at a time (an ablation of the adversary), and labelled with the assumption it broke.

**Objective.** Minimise `rho(theta)`, the minimum over properties and time of the normalised STL robustness (section 5). A negative value is a violation. **Optimiser.** Own (mu/mu_w, lambda)-CMA-ES: `lambda = 4 + floor(3 ln 32) = 14`, box handled by clipping with a quadratic repair penalty, IPOP restarts with doubling population up to 3 restarts, ~120 lines, tested on sphere (d = 10) and Rosenbrock (d = 5) against stated evaluation budgets; no `cma` dependency (the extra supply-chain surface is not worth 120 lines). **Baseline.** Uniform random search over the same box and the same budget. Budget (A): 300 evaluations per run, 10 independent runs per (regime, algorithm), target arm B3 + Careless + Gullible with shield at about 0.3 s per evaluation (the planner arms are too slow to search; three runs per cell, my first plan, cannot support any comparison). Reported: fraction of runs that find a violation within budget, evaluations to first violation (right-censored at 300), best robustness at budget; paired by run index with confidence sequences. If CMA-ES does not beat random search, that is published.

**Multi-fidelity falsification.** Distinct violating `theta` from F1 (up to 50, ranked by robustness, de-duplicated by decoded-scenario distance) are replayed at F2: same exogenous plant streams, real pipeline, real agent. The **confirmed fraction** is (F2 shows a violation of the same property) / (replayed), with an interval, and every unconfirmed replay gets a reason code (`estimator_mismatch`, `plant_logic`, `replay_nondeterminism`, `agent_blocked_earlier`). Both numbers and the reason histogram are published.

## 10. Integration map

### 10.1 Wiring at F2

```
asset-health: simulate_asset(rng, SimConfig, AssetSpec, FailureEvent, services) -> raw telemetry (plant latent failure events)
   -> run_asset (cached per life) -> hi / z24 -> estimate_rul, analyse_run -> WorkOrder, RULEstimate
   -> AssetHealthAlertSource (ours, implements ports.AlertSource) -> maintenance_agent.Orchestrator
        (constructed directly; planner = RulesPlanner; store = SyntheticProcedureStore; clock = sim clock)
   -> SimHumanApprover (ours; uses gate.ApprovalGate with the sim clock, attention model)
   -> PlantCMMS (ours; implements ports.CMMS; parts from plant inventory; release_work_order delegates the signature
      and role checks to MockCMMS, then queues a plant job)  ->  plant job runner  ->  shield  ->  plant
plant-copilot: spec clauses (cached per as-of), handover-note screening (scan_text)
```

### 10.2 What maps

| asset-health | maintenance-agent | plant |
|---|---|---|
| `conveyor_pulley` | `BELT_CONVEYOR` | CV-101 |
| `primary_crusher` | `CRUSHER` | CR-501 |
| `slurry_pump` | `SLURRY_PUMP` | PU-201, PU-202 |
| `bearing_spall` | `BEARING_DEFECT` | all three classes |
| `impeller_wear` | `IMPELLER_WEAR` | pumps |
| `liner_wear` | `LINER_WEAR` | crusher (permits LOTO + confined space + hot work + height, safety officer) |
| `RULEstimate.p10/p50/p90` | `RULInterval.lower/median/upper` (+ `as_of` date, `confidence`) | conformal interval replaces the quantile interval |
| `WorkOrder` severity urgent/critical | `Alert.severity = ALARM`; watch/warning to `WARN` | alert raised |

### 10.3 Mismatches found (each is a finding; none is fixed in the siblings)

All entries are (S) read from source unless marked. Recorded in `docs/FINDINGS.md` by packet P13 with file and line.

1. **Asset ids.** asset-health `PMP-01`-style (seen in spike); maintenance-agent `PU-201`, `CR-501`, `CV-101`, bound to its BOMs and procedures. The plant uses the agent's ids and a mapper to asset-health ids. asset-health's `ID_PREFIX` for crusher and pulley was not read (unverified).
2. **Class vocabularies.** 5 asset-health types vs 6 agent classes; overlap 3. Idler, haul-truck engine on one side; VENT_FAN, BALL_MILL, HV_MOTOR on the other.
3. **Failure modes.** `overheating` has no agent mode (the requirements matrix fails closed for unknown modes: `known = False`, blocks the draft); outer/inner race is dropped; agent modes SEAL_LEAK, GEAR_WEAR, MISALIGNMENT, IMBALANCE, WINDING_INSULATION, LUBE_DEGRADATION are never produced by asset-health.
4. **RUL schema.** `RULEstimate` has `status` and nullable quantiles (not-degrading and insufficient-trend carry no interval) and a `datetime`; `RULInterval` needs three finite floats within ±3650 and a `date`. A non-estimated status cannot become an alert. Our policy: no alert; counted as a missed-alert opportunity.
5. **Interval level.** asset-health's interval is p10-p90 (80%); `RULInterval.confidence` defaults to 0.8. We publish 0.9 intervals and set the field; whether any agent rule reads `confidence` is unverified.
6. **Alert shape.** asset-health emits a `WorkOrder` (severity watch/warning/urgent/critical, hypothesis enum, narrative); the agent expects `Alert` (severity WARN/ALARM, `anomaly_score` in [0, 1], summary, operator notes, work history). The fused score's scale is not bounded to [0, 1] (clamped by our adapter, recorded).
7. **Evidence.** The agent's `EvidenceItem` needs `baseline`, `alarm`, `direction`, `trend_per_day`; asset-health supplies observed and expected feature values with no alarm level. The adapter defines `alarm` as baseline plus the detection threshold in robust-sigma units (our definition, not an upstream one).
8. **Latest safe date.** Two policies: asset-health's cost-optimal date (never later than the pessimistic end) and the agent's `policy_latest_date` (lower bound minus one day, or the bound itself under three days). The agent's verifier enforces its own; asset-health's date is ignored.
9. **Dates and staleness.** asset-health uses naive-UTC hourly datetimes; the agent uses dates, a clock, and rule V10 blocks if `verified_on - as_of > max_alert_age_days (3)`. **This explains the BLOCKED alert in the owner's fixed-clock test:** the bundled alerts are dated 2026-10-12 (`FIXTURE_DATE`) and a clock of 2026-10-20 makes the age 8 days. It is the stale-alert rule working, not a defect; our alert source stamps `as_of` with the sim date and the clock is the sim clock.
10. **Bundled alerts are static.** Re-dating exists only for the bundled fixture (`SyntheticAlertSource.redated`); there is no generic alert generator.
11. **CMMS uniqueness.** `MockCMMS` refuses a second work order for an alert id or draft hash (`DUPLICATE`); a repeat alert after a replacement needs a fresh id; run ids are lower-cased tokens of at most 64 characters; one `MockCMMS` instance per episode.
12. **No CMMS injection in `build_system`.** `wiring.build_system` constructs `MockCMMS` internally and takes no `cmms` argument; `Orchestrator` must be constructed directly.
13. **Inventory is static.** No stock decrement on release, fixed lead times; `PlantCMMS.get_part` serves the plant's inventory so V11 and V12 see real stock.
14. **Release is not execution.** No scheduling, no crew, no completion callback (the agent states scheduling is out of scope); the plant owns start time within the latest safe date and reports completion to nobody on the agent side.
15. **Part numbers.** Plant spares must use the agent's `SYN-...` numbers and the BOM keys in `AssetInfo.bom`; asset-health has no parts concept.
16. **Gate import tripwire.** A test forbids agent-side modules importing `gate`; the simulated approver lives in our human-side module `closed_loop.f2.human`, and the sim clock is passed to `ApprovalGate(clock=...)` (default time-to-live 8 hours of sim time).
17. **Roles.** Crusher liner repairs need `SAFETY_OFFICER` (confined space, hot work); the roster has one on the day shift only (A), so approvals queue overnight.
18. **Two procedure libraries.** plant-copilot `CORP-HSE-*` and the agent's `PRC-*` are independent synthetic sets; isolation steps are defined twice and the agent's set has **no gas re-test interval** (grep of `procedures.json`: no match for re-test or minutes except a 60-minute fire watch). The spec uses plant-copilot only.
19. **Parameters from prose.** plant-copilot returns verbatim text and revision, no structured value; the 30-minute interval is extracted by regex from English text (fragile; Rev 2 says 2 hours, Rev 3 30 minutes). Sites and roles are needed for `User.of(id, role, *sites)`; the corp clause is visible at every site.
20. **asset-health simulation limits.** One failure per asset per record, records of at least 60 days (`SimConfig.days >= 60`, and `days >= baseline_days + 40`), no stop/derate hook, and a 21-day baseline plus 48 h settle after every repair record during which nothing is detected.
21. **Install and provenance.** asset-health pulls scikit-learn and pandas (the 44 s import); the siblings have no git history locally, so there is no commit to pin; their names are unclaimed on PyPI (M, 404).

### 10.4 guardrail and mcp-gateway: excluded

`guardrail` offers injection detection, PII redaction, groundedness and structured-output repair. Injection screening is already in plant-copilot (`scan_text`, with Arabic and folding rules) and is the screener the Gullible arms use; PII is irrelevant (no personal data); groundedness is irrelevant (no model is run). `mcp-gateway` hardens the boundary between a model and tools; our boundary between a proposer and the plant is the shield, and no MCP server is needed. Adding either would add a dependency and a build surface without a measurement they would change.

## 11. Evaluation design

**Arms** (full definitions and counts in the pre-registration): B0 run-to-failure; B1 calendar (age-based); B2 fixed limit on the health index; B3 bound rule (V-variant as a factor); P0 planner without shield; P1 planner with shield; P2 planner with shield and monitor fallback. Hostile-proposer arms: Careless and Gullible, each with shield off and on, Gullible also with screening on and off. Spec arms: clause-backed versus stale spec. Perception arms: V1-V7 under B3. Causal arms: correlational, SCM-correct, SCM-misspecified. Search arms: CMA-ES and random, `R_in` and `R_out`. Conditions: nominal and shifted (a load and sensor-gain shift starting at a random day in 30-60; magnitudes in the pre-registration).

**Ablations** (each layer removed, the rest unchanged): perception (V1 vs the best variant), planner (B3 vs P-arm), shield (off/on), monitor (off/on), screening (off/on), spec provenance (stale/clause), information-gathering (INSPECT disabled), planner model mismatch (0, 10%, 25%).

**Seed roles.** `dev` (1000-1999, tuning of `gamma`, `k`, thresholds, planner `delta_plan`), `calib` (2000-2999, conformal calibration lives), `search` (3000-3999), `heldout` (derived, section 12), `f2` (derived with a separate label). Counts are in the pre-registration. Every tuning decision is made on `dev` and written into a file `tuned.json` that is hashed into the ledger.

**Paired design.** All arms of one condition see the same exogenous streams for the same seed (per-stream keyed generators, below), so differences are paired. **Randomness.** `rng_for(root_seed, stream, *ids)` returns `numpy.random.Generator(PCG64(SeedSequence(root_seed, spawn_key=(crc32(stream), *ids))))`: every stream (per asset and life: onset, duration, shape, noise; process load; handover; gas; notes; approver) is keyed, not drawn in sequence, so an intervention in one stream cannot shift another. A test runs the factual episode and an episode with a no-op intervention and requires bit-identical traces; a second test applies `do(replace)` and requires all other assets' traces to be identical.

**Statistics.** Anytime-valid confidence sequences (section 11.1) for every proportion and bounded mean; paired differences via the same machinery on the paired per-seed difference rescaled to [0, 1]. No p-values. Every table shows `k/n`.

### 11.1 Confidence sequences

Betting confidence sequences for a mean in [0, 1] (Waudby-Smith and Ramdas style): for each candidate `m` on a grid of 1,001 points, hedged capital `K_t(m) = 0.5 prod (1 + lambda_i (x_i - m)) + 0.5 prod (1 - lambda_i (x_i - m))` with predictable `lambda_i = min(c, sqrt(2 ln(2/alpha) / (sigma_hat_{i-1}^2 i ln(i+1))))`, `c = 0.5`; the set `{m : K_t(m) < 1/alpha}` is a `1 - alpha` confidence sequence valid at all times simultaneously (Ville). Used with `alpha = 0.05`. Zero-event upper bounds (for example "0 violations in n shielded episodes") come from the same sequence. Bounded non-proportion metrics (throughput fraction, cost normalised by a stated cap) are rescaled to [0, 1] with the cap published; a clipped value is counted and reported. **Sequential stopping:** episodes are run in batches of 50 per cell; a cell stops when the CS half-width falls below its pre-registered target or at `n_max`. Because the sequences are anytime-valid, stopping early does not invalidate them. Packet P2 verifies coverage of the implementation by simulation (target: miscoverage at most 0.05 over 10,000 Bernoulli streams at three means).

## 12. Audit of the research process

- **Pre-registration.** `docs/PREREGISTRATION.md` (this draft, frozen) is hashed (SHA-256 over bytes after normalising CRLF to LF and stripping trailing spaces, with `.gitattributes` `eol=lf`: the owner's earlier line-ending trouble shows why). The freeze is recorded as the first ledger entry.
- **Ledger.** `results/ledger.jsonl`, append-only; each line holds `seq, utc, kind (freeze | full | full-failed), prereg_sha256, code_sha256, tuned_sha256, sibling versions and tree hashes, seed-role counts, heldout_evaluations (cumulative), headline values (also hashed), prev_hash, hash`. `benchmarks/run.py` verifies the chain, the prereg hash and the code hash relation before and after a full run, refuses a full run if the prereg hash differs from the freeze entry, and appends even on failure. `--quick` runs write nothing to the ledger and publish nothing.
- **Held-out seeds.** `heldout_seed(i) = int.from_bytes(sha256(prereg_sha256 || b"heldout" || i)[:8])`. They do not exist before the freeze, and any change to the pre-registration changes them. `HeldoutGuard` raises unless the ledger contains a freeze entry for the current hash, and increments `heldout_evaluations` on every use, so overuse is visible.
- **What this proves.** A published number came from a run whose pre-registration and code hashes are recorded; the number of full runs and any drift between them are visible; a later edit to a past entry breaks the chain.
- **What it does not prove.** There is no git history locally, so someone with write access can rewrite the whole ledger consistently; the machine clock could be wrong; runs outside `run.py` are not logged; analysis forks and choice of datasets are invisible; a pre-registration cannot make the model class well chosen. Mitigation: publish the ledger head hash in the release notes and README as an external anchor (and, as an owner action, timestamp it with a third-party service). The benefit is deterrence and visibility, not a proof of honesty.

## 13. Runtime budget

Machine: 4 logical CPUs, 2015 laptop shared with the owner; 3 workers; owner rules: take the owner's heavy-run lock file (path from the environment variable `CLOSED_LOOP_LOCK`) before a heavy run and stop at 88% commit charge (`benchmarks/run.py` reads commit charge through `ctypes` `GlobalMemoryStatusEx`, refuses to start above 88% and aborts a batch boundary above it; the lock path is configurable by environment variable). Episode counts are maxima; sequential stopping will usually use fewer.

| # | Stage | Units | Seconds per unit | CPU-s | Wall at 3 workers (min) | Basis |
|---|---|---|---|---|---|---|
| 1 | F0 solve, checker, differential, sabotage | 1 | 30 | 30 | 0.2 | solve 1.0 s (M), rest (A) |
| 2 | Open-loop perception, 7 variants | 4,000 lives | 0.14 | 560 | 3.1 | (A) |
| 3 | Closed-loop perception under B3, 7 variants x 2 conditions | 7,000 episodes | 0.3 | 2,100 | 11.7 | (A) |
| 4 | Policy arms B0, B1, B2 (0.3 s) and P0, P1, P2 (1.0 s), 2 conditions | 500 x 2 x (3 x 0.3 + 3 x 1.0) | | 3,900 | 21.7 | (A), (D) |
| 5 | Hostile proposers, 6 arms, nominal | 3,000 episodes | 0.3 | 900 | 5.0 | (A) |
| 6 | Causal study, F1+D | 600 lives x 4 worlds | 0.15 | 420 | 2.3 | (A) |
| 7 | Search: 2 regimes x 2 algorithms x 10 runs x 300 | 12,000 evaluations | 0.3 | 3,600 | 20.0 | (A) |
| 8 | F2 arms: 3 arms x 60 episodes | 180 | 30 | 5,400 | 30.0 | (D) below |
| 9 | F2 replay of F1 finds | 50 | 30 | 1,500 | 8.3 | (D) |
| 10 | Conformance, estimator mismatch fit, exact DP, 100-job audit via real orchestrator (100 x 0.15 s) | | | 300 | 1.7 | (A) |
| | **Total** | | | **18,710** | **104** | (D) |

**F2 episode cost (D).** Calls of `run_asset`: 4 initial lives + about 4 replacements + about 3 stops = 11, at 2.0 s each (M: 1.6-1.7 s at 120 days, nearly flat in record length; allowance for a 90-day record with 2 segments) = 22 s; `analyse_run` on event days, about 40 calls x 0.09 s (M) = 4 s; agent: about 4 alerts x 0.15 s (M, 0.12-0.14 s) = 0.6 s; plant, shield and Ed25519 signing about 1 s (A); sum 27.6 s, rounded to 30 s. Worker import 44 s (M) is paid once per worker, in parallel, about 0.7 min wall.

**Over budget? Cut in this order:** (1) F2 replays 50 to 25; (2) search 300 to 200 evaluations per run; (3) planner-arm `n_max` 500 to 300; (4) F2 arms 60 to 40 episodes (CS widens); (5) V4 and V5 closed-loop variants dropped, kept open-loop; (6) planner `N` 128 to 64. A cut changes the pre-registered `n_max` only before the freeze. **`--quick`:** all counts divided by 25 (minimum 4), F2 limited to 3 episodes and 3 replays, search to 30 evaluations; target at most 8 minutes after imports (A), writes no ledger entry.

## 14. Packaging, dependencies, install

- `pyproject.toml` (hatchling), `requires-python >= 3.11`, tested on 3.12. Core dependencies: `numpy`, `scipy`, `pydantic>=2` only (all already pulled by the siblings). `pandas` is imported only in `closed_loop.f2`. Dev extras: `pytest`, `pytest-cov`, `ruff`, `hypothesis` (the sibling repositories already use it).
- **Siblings are optional (`integration` extra) and referenced only by direct URL**: `asset-health @ git+https://github.com/darrshangovender/asset-health@<tag>` and likewise for `maintenance-agent` and `plant-copilot`, with `<tag>` filled when the owner tags releases (no tags or commits exist locally: unverified). Local development: `pip install ../asset-health ../maintenance-agent ../plant-copilot` into the same virtual environment.
- **Dependency confusion.** Bare names would be resolved from PyPI, where `asset-health`, `plant-copilot`, `maintenance-agent` and `closed-loop` were unclaimed on 31 October 2026 (M: HTTP 404 from the PyPI JSON API). An attacker could register one and have a reader's `pip install closed-loop[integration]` fetch it. Rules: never a bare sibling name in any requirements file or `pyproject.toml`; a test (`test_no_bare_sibling_names`) parses every dependency specifier; `scripts/verify_siblings.py` prints installed versions and a tree hash of each sibling and compares them with `integration/siblings.lock`; the ledger's code hash includes them. Owner action (not done by me): register placeholder projects on PyPI for the four names, or never publish and say so in the README.
- F1 and F0 (the whole proof and the policy comparison) run with the core dependencies only; F2 tests are marked `f2` and skipped with a clear message when the siblings are absent.

## 15. Risks, unverified items, blocking issues

1. **F1 timings are targets, not measurements** (0.3 s and 1.0 s per episode). If the planner costs more than 1.5 s, reduce `N`, then replan every other day; the budget has 16 minutes of headroom.
2. **Splicing lives at F2 is unverified.** Rendering a replacement as a fresh `simulate_asset` record concatenated onto the previous one, with a `MaintenanceEvent(kind="repair")` at the join, has not been run. Packet P11 starts with a spike; if the pipeline mishandles the join, F2 falls back to "one life per asset per episode, replacement ends the asset's monitoring" and the replacement analysis moves to F1 only. **This is the largest integration risk.**
3. **asset-health's `ID_PREFIX` for crusher and pulley** and the usage of `RULInterval.confidence` were not read.
4. **The spike abstraction is a toy**: the final counts will differ when P5 adds the spec's exact guards. The structural findings (decomposition, no deadlock, C1-only) depend on "stop is always safe"; a future property that stopping violates (for example a minimum production) would reopen them.
5. **C1 at 99% is a demanding contract**; the e-process has little power against small excess miss rates. The detection delay is reported as measured, not hoped.
6. **The planner may not beat the rule; the SCM may not beat the correlational model in the fair arms; CMA-ES may not beat random search.** All are pre-registered as possible outcomes.
7. **F2 has n = 60 per arm**; the confidence sequences will be wide. That is reported.
8. **F1-F2 estimator mismatch** may be large enough that F1 policy rankings do not transfer; P4b and the F2 arms measure it and the claim is limited accordingly.
9. **No clean-venv install of the whole stack was run** (I measured with the owner's pre-built venv only).
10. **Sibling versions are unpinned** until the owner tags releases.

## 16. Rejected alternatives

- **DtACI** (expert aggregation over step sizes): extra machinery for a small gain; conformal PID covers "faster than ACI under drift". Revisit as an exploratory arm.
- **One global product game** over all components: roughly 40 thousand to 880 thousand raw states depending on the encoding (hand counts, not run), slow in pure Python, and unnecessary given section 4.5.
- **Two-part perception contract (coverage and resolution)**: unnecessary after the spike; removed.
- **Pure-Python POMCP**: too slow for the episode budget.
- **The `cma` package**: not needed; 120 lines, tested.
- **Gymnasium or an RL framework**: an extra dependency and a worse fit than a `Proposer` Protocol.
- **Reusing asset-health's `generate_fleet` as the F1 plant**: whole-record generator with one failure per asset, no controllable regime, 1.4 s for 4 assets over 120 days (M by owner): neither fast enough nor controllable.
- **Running the real orchestrator at F1**: about 1 s per episode for no policy-comparison value; replaced by the real requirements matrix plus a 100-job audit.
- **Alpha-spending union for the per-life band**: needs a spending function with no principled choice; the max-score band is exact.
- **Alpha 0.1 for the shield contract**: breaks about once per ten steps and voids the proof.
- **Adding guardrail or mcp-gateway**: section 10.4.
- **Publishing to PyPI**: a direct-URL dependency cannot be uploaded and the name risk above.
