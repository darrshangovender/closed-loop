"""The integration findings as data: 21 mismatches and 12 sibling defects (packet P11c).

``FINDINGS`` is the table of the findings document, one row per finding, each with the file and line
that show it. It is data on purpose: ``tests/f2/test_findings.py`` checks that every row has a
mechanised check in ``closed_loop.f2.mismatch``, that every cited line exists in the installed
sibling and contains the text the row claims, and that the wording avoids absolutes and internal
references. ``render_document`` turns the rows into the markdown of ``docs/FINDINGS.md`` and
``scripts/render_findings.py`` prints it, so the document and the checked rows cannot drift apart.

Everything is SYNTHETIC. The siblings were read and run, not modified; the status of every row is
"recorded, not fixed". Numbers appear only in the generated sections, copied from the checks and the
measurement files, not typed here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "FINDINGS",
    "STATUS",
    "Finding",
    "Location",
    "render_document",
    "render_measured",
    "render_table",
]

STATUS = "recorded, not fixed"
RUNS = "runs the siblings"
READS = "reads their source"


@dataclass(frozen=True)
class Location:
    """A file and line range in one of the siblings, with a text the lines must contain.

    ``package`` locations are inside the installed Python package (``asset_health/...``);
    the others are at the root of the repository (``README.md``, ``tests/...``, ``corpus/...``).
    ``needle`` is None where the row cites a range whose text was not pinned.
    """

    repo: str  # "asset-health" | "maintenance-agent" | "plant-copilot"
    path: str
    lines: str  # "232" or "47-53"
    needle: str | None = None

    @property
    def first(self) -> int:
        return int(self.lines.split("-")[0])

    @property
    def last(self) -> int:
        return int(self.lines.split("-")[-1])

    @property
    def in_package(self) -> bool:
        return self.path.split("/")[0] in ("asset_health", "maintenance_agent", "plant_copilot")


@dataclass(frozen=True)
class Finding:
    id: str
    title: str  # the bold lead of the row
    text: str
    locations: tuple[Location, ...]
    severity: str  # "low" | "medium" | "high"
    check: str  # RUNS | READS
    status: str = STATUS

    @property
    def kind(self) -> str:
        return "mismatch" if self.id.startswith("M") else "defect"


AH, MA, PC = "asset-health", "maintenance-agent", "plant-copilot"


def L(repo: str, path: str, lines: str, needle: str | None = None) -> Location:
    return Location(repo, path, lines, needle)


FINDINGS: tuple[Finding, ...] = (
    Finding(
        "M1",
        "Asset ids do not translate.",
        "asset-health builds an id from a three-letter type prefix and a counter (for example "
        "`PMP-01`). maintenance-agent uses plant-style ids (`CV-101`, `PU-201`, `CR-501`) that its "
        "fixtures tie to bills of materials and, for a few documents, to named assets. Neither "
        "repository mentions the other's scheme and neither holds a mapping, so whoever wires them "
        "has to add a translation table.",
        (
            L(AH, "asset_health/simulate/fleet.py", "47-53", "ID_PREFIX"),
            L(MA, "maintenance_agent/fixtures/alerts.json", "7", "CV-101"),
            L(MA, "maintenance_agent/fixtures/alerts.json", "88", "PU-201"),
            L(MA, "maintenance_agent/fixtures/alerts.json", "280", "CR-501"),
        ),
        "medium",
        RUNS,
    ),
    Finding(
        "M2",
        "The asset classes overlap only in part.",
        "asset-health has five asset types and maintenance-agent six asset classes. Three pairs "
        "can be matched by judgement (conveyor pulley with belt conveyor, primary crusher with "
        "crusher, slurry pump with slurry pump) and only the pump matches by name. The conveyor "
        "idler and the haul-truck engine have no agent class; the ventilation fan, ball mill and "
        "high-voltage motor have no asset-health type. An asset class the agent's requirements "
        "table does not know is blocked rather than guessed.",
        (
            L(AH, "asset_health/models.py", "37-42"),
            L(MA, "maintenance_agent/requirements.py", "42-49", "CLASS_PERMITS"),
            L(MA, "maintenance_agent/requirements.py", "126-127", "Requirement(False"),
        ),
        "medium",
        RUNS,
    ),
    Finding(
        "M3",
        "The failure modes do not line up.",
        "asset-health can name an `overheating` hypothesis, which the agent's closed list of "
        "failure modes rejects when the alert is built (rule V00, before the requirements table is "
        "consulted). It also splits bearing faults into outer-race, inner-race and generic "
        "variants "
        "where the agent has one bearing-defect mode. Six agent modes (seal leak, gear wear, "
        "misalignment, imbalance, winding insulation, lube degradation) are not produced by "
        "asset-health at all.",
        (
            L(AH, "asset_health/models.py", "45-64"),
            L(MA, "maintenance_agent/models.py", "102-111"),
            L(MA, "maintenance_agent/requirements.py", "118-140"),
            L(MA, "maintenance_agent/verifier.py", "440-446"),
            L(MA, "maintenance_agent/verifier.py", "833-845"),
        ),
        "medium",
        RUNS,
    ),
    Finding(
        "M4",
        "A reading without an interval cannot become an alert.",
        'asset-health\'s RUL estimate has a status, and only the "estimated" status carries the '
        '10th, 50th and 90th percentiles; "not degrading" and "insufficient trend" carry none. '
        "The agent's interval type needs three finite numbers within plus or minus 3,650 days and "
        "a date, and its alert requires one. Without a rule for this case a hand-off either fails "
        "or invents numbers. The case is common: a work order can be open while the status is one "
        "of the two (see S6).",
        (
            L(AH, "asset_health/models.py", "374-404"),
            L(MA, "maintenance_agent/models.py", "183"),
            L(MA, "maintenance_agent/models.py", "198-215"),
            L(MA, "maintenance_agent/models.py", "244"),
        ),
        "high",
        RUNS,
    ),
    Finding(
        "M5",
        "The interval level is a label the agent does not read.",
        "asset-health's interval is its 10th to 90th percentile, an 80% interval, and its estimate "
        "type has no field saying so. The agent's interval has a `confidence` field that defaults "
        "to 0.8; no rule, planner or verifier reads it. Its only effect is through the alert "
        "fingerprint (see S7).",
        (
            L(AH, "asset_health/config.py", "225", "quantiles"),
            L(MA, "maintenance_agent/models.py", "207", "confidence"),
            L(MA, "maintenance_agent/verifier.py", "174", "rul"),
        ),
        "low",
        RUNS,
    ),
    Finding(
        "M6",
        "asset-health's diagnosis does not reach the agent.",
        "A work order from asset-health carries a failure-mode hypothesis (one of seven values). "
        "The agent's alert has no field for it, and the agent's rule-based planner does not read "
        "the alert's summary text: it scores failure modes from the evidence items alone. Two "
        "alerts that differ only in summary and anomaly score give the same draft. So the "
        "diagnosis in a draft is the agent's own reading of the evidence, not asset-health's "
        "hypothesis.",
        (
            L(AH, "asset_health/models.py", "410-428"),
            L(MA, "maintenance_agent/models.py", "237-246"),
            L(MA, "maintenance_agent/planners/rules.py", "228", "score_modes"),
        ),
        "medium",
        RUNS,
    ),
    Finding(
        "M7",
        "Evidence has no alarm level.",
        "asset-health's evidence item is a feature name, a description, an observed value, a "
        "regime-expected value, a z-score and a unit, and a work order holds at most four of them, "
        "each at least 2.0 standard deviations out. The agent's evidence item requires a baseline "
        "and an alarm level, which asset-health does not supply; direction and trend are optional "
        'and default to "high" and zero. A mapping must invent the alarm level, and the agent\'s '
        "failure-mode score depends on it (see S11).",
        (
            L(AH, "asset_health/models.py", "355-363"),
            L(AH, "asset_health/alerting/workorder.py", "60", "2.0"),
            L(MA, "maintenance_agent/models.py", "161-180"),
        ),
        "high",
        RUNS,
    ),
    Finding(
        "M8",
        'There are two "latest safe date" rules.',
        "asset-health takes the smaller of a cost-optimal replacement day and the 10th percentile "
        "of the interval, and not less than zero. The agent uses today when the lower bound is "
        "under one day, the bound itself from one to under three days, and the bound minus one day "
        "from three days up. The agent's alert has no field for asset-health's date, its planner "
        "chooses its own, and its verifier blocks any later date. For a lower bound of about six "
        "days the two dates differ (the dates the check saw are listed in the evidence section).",
        (
            L(AH, "asset_health/alerting/cost.py", "112", "min(d_cost, p10)"),
            L(AH, "asset_health/alerting/workorder.py", "113-120"),
            L(MA, "maintenance_agent/models.py", "186-195"),
            L(MA, "maintenance_agent/verifier.py", "643-656"),
        ),
        "medium",
        RUNS,
    ),
    Finding(
        "M9",
        "Time conventions differ, and staleness blocks.",
        "asset-health works in naive UTC date-times on an hourly grid by default. The agent works "
        "in calendar dates with an injectable clock, and rule V10 blocks a draft when the RUL "
        "reading's date is more than 3 days before the check date (or more than a day after it). "
        "The agent's bundled alerts are all dated 12 October 2026, so under a fixed clock four "
        "days "
        "later they are stale (the check: 3 days passes, 4 days raises V10). A caller that "
        "supplies "
        "readings must stamp them with the date of the check.",
        (
            L(AH, "asset_health/timeutil.py", "1-13"),
            L(MA, "maintenance_agent/verifier.py", "97", "DEFAULT_MAX_ALERT_AGE_DAYS"),
            L(MA, "maintenance_agent/adapters/synthetic.py", "47", "FIXTURE_DATE"),
        ),
        "medium",
        RUNS,
    ),
    Finding(
        "M10",
        "Alerts are fixtures.",
        "The dates of the bundled alerts can be shifted (`SyntheticAlertSource.redated`) but "
        "nothing in the agent builds an alert from parameters; a live source has to implement the "
        "`AlertSource` protocol itself.",
        (
            L(MA, "maintenance_agent/adapters/synthetic.py", "78", "def redated"),
            L(MA, "maintenance_agent/ports.py", "74-84"),
        ),
        "low",
        RUNS,
    ),
    Finding(
        "M11",
        "The maintenance system refuses repeats and keeps state.",
        "`MockCMMS` refuses a second work order for the same alert id or the same draft hash "
        "(`DUPLICATE`), so a repeat alert after a repair needs a fresh id. Run ids must be "
        "letters, digits, dot, underscore or hyphen, at most 64 characters, and are lower-cased. "
        "Its ledger is held in memory unless a home folder is given, in which case it persists, so "
        "each independent episode needs a new instance or a new folder.",
        (
            L(MA, "maintenance_agent/cmms.py", "176", 'r["alert_id"] == draft.alert_id'),
            L(MA, "maintenance_agent/models.py", "28-47"),
            L(MA, "maintenance_agent/wiring.py", "109-111"),
        ),
        "medium",
        RUNS,
    ),
    Finding(
        "M12",
        "`build_system` cannot be given a different maintenance system.",
        "It creates its own `MockCMMS` from the bundled inventory on each call and has no `cmms` "
        "parameter; only the clock, the approver registry and the ledger folder can be set. A "
        "different implementation can be used only by building `Orchestrator` by hand.",
        (
            L(MA, "maintenance_agent/wiring.py", "81", "def build_system"),
            L(MA, "maintenance_agent/wiring.py", "109", "MockCMMS.from_fixtures"),
        ),
        "medium",
        READS,
    ),
    Finding(
        "M13",
        "Stock does not change.",
        "The mock keeps its parts table in a private dictionary loaded once from the bundled "
        "inventory; a release appends a ledger entry and does not reduce stock, and lead times are "
        "fixed numbers. A simulation with several work orders sees the same stock every time "
        "unless "
        "the caller supplies it.",
        (
            L(MA, "maintenance_agent/cmms.py", "75", "_parts"),
            L(MA, "maintenance_agent/cmms.py", "104-143"),
        ),
        "low",
        READS,
    ),
    Finding(
        "M14",
        "Release is the end of the agent's job.",
        "The maintenance-system interface has five calls (read a part, dry-run a draft, release, "
        "and two lookups of a release) and none to schedule, assign a crew, report progress or "
        "report completion. The project's own documentation lists scheduling as out of scope. "
        "Nothing flows back to asset-health about what happened to a work order.",
        (
            L(MA, "maintenance_agent/ports.py", "122-149"),
            L(MA, "README.md", "294", "Scheduling is out of scope"),
        ),
        "medium",
        READS,
    ),
    Finding(
        "M15",
        "Part numbers come from nowhere in asset-health.",
        "The agent's parts (invented `SYN-` numbers) are held in a bill of materials, a map from "
        "component to part number, which travels inside every alert; its verifier rejects any part "
        "that is not in the inventory and in that bill. asset-health's asset record has bearing "
        "geometry and a component age but no part numbers, so an integration must source the bill "
        "elsewhere.",
        (
            L(MA, "maintenance_agent/models.py", "158", "bom"),
            L(MA, "maintenance_agent/verifier.py", "271-275"),
            L(AH, "asset_health/models.py", "156-178"),
        ),
        "medium",
        READS,
    ),
    Finding(
        "M16",
        "The signing half is fenced off from the checking half.",
        "A test in maintenance-agent fails if any of twenty named agent-side modules imports the "
        "signing module or the private-key classes. In the installed package only the command-line "
        "module and the evaluation harness import it. For an integration this means a simulated "
        "human has to be written outside the agent's checking core. The test covers the listed "
        "modules and calls itself a tripwire, not a runtime control.",
        (
            L(MA, "tests/test_bypass.py", "66", "ApprovalGate"),
            L(MA, "tests/test_bypass.py", "22-73"),
        ),
        "low",
        READS,
    ),
    Finding(
        "M17",
        "Some repairs need a safety officer.",
        "In the agent's requirements table a crusher liner repair needs lock-out plus confined "
        "space, hot work and work at height; confined space and hot work are high-risk, so a "
        "planner's signature is refused (`ROLE_INSUFFICIENT`). An inspection-first draft on the "
        "same crusher needs lock-out only and a planner may sign it. If the safety officer is on "
        "shift for part of the day only, approvals for these repairs wait.",
        (
            L(MA, "maintenance_agent/requirements.py", "54", "CONFINED_SPACE"),
            L(MA, "maintenance_agent/requirements.py", "113-148"),
            L(MA, "maintenance_agent/models.py", "143-147"),
            L(MA, "maintenance_agent/approval.py", "192-205"),
        ),
        "medium",
        RUNS,
    ),
    Finding(
        "M18",
        "Two procedure libraries, and the agent's has no gas re-test interval.",
        "plant-copilot holds the plant's standards (`CORP-HSE-*`); the agent has its own invented "
        "procedures (`PRC-*`). Isolation is defined in both with different wording, and the "
        'agent\'s confined-space procedure says only that entry follows "atmospheric testing". '
        "No gas re-test interval appears in the agent's procedures; the only minutes-related text "
        "is a 60-minute fire watch after hot work. A draft can therefore quote isolation and "
        "confined-space wording that differs from the plant's current standard.",
        (
            L(MA, "maintenance_agent/fixtures/procedures.json", "501", "fire watch"),
            L(MA, "maintenance_agent/fixtures/procedures.json", "44-120"),
            L(PC, "corpus/en/CORP-HSE-001_r5.md", "37-57"),
        ),
        "high",
        READS,
    ),
    Finding(
        "M19",
        "A safety figure exists only as prose.",
        "plant-copilot returns clause text word for word with document, clause number, revision "
        "and language; its answer types have no field for a number or a unit. The gas re-test "
        'interval in clause 4.3 of `CORP-HSE-002` is "at least every 2 hours" in revision 2 and '
        '"every 30 minutes" in revision 3, with a new rule for breaks over 15 minutes, so a '
        "caller must parse English and pin the revision. Asking a question needs a user with one "
        "of seven roles and at least one site.",
        (
            L(PC, "corpus/en/CORP-HSE-002_r2.md", "39", "2 hours"),
            L(PC, "corpus/en/CORP-HSE-002_r3.md", "39", "30 minutes"),
            L(PC, "plant_copilot/models.py", "161-201"),
        ),
        "high",
        RUNS,
    ),
    Finding(
        "M20",
        "The simulator is narrower than a closed loop needs.",
        "asset-health's generator injects at most one failure per asset per record, needs a record "
        "of at least 60 days (61 with the default 21-day baseline), and `simulate_asset` has no "
        "way "
        "to stop or derate an asset on command. After every repair record the pipeline spends 48 "
        "hours plus 21 days learning a new baseline, during which it produces no score, so nothing "
        "is detected. A second fault in the same record, or a quick re-failure after a repair, "
        "cannot be tested.",
        (
            L(AH, "asset_health/config.py", "81", "baseline_days + 40"),
            L(AH, "asset_health/simulate/fleet.py", "264", "def simulate_asset"),
            L(AH, "asset_health/features/build.py", "83-84"),
        ),
        "medium",
        RUNS,
    ),
    Finding(
        "M21",
        "Install and provenance.",
        "asset-health requires scikit-learn and pandas (with numpy, scipy and pydantic) and "
        "imports scikit-learn at module level in its detector and RUL modules, so importing its "
        "pipeline is slow and pulls both in. None of the three repositories has tags or commits in "
        "what is installed (all declare version 0.1.1), so there is nothing to pin except a hash "
        "of "
        "the files. That the package names were unclaimed on a public package index was recorded "
        "online on 31 October 2026 and cannot be re-checked offline; the repository's own README "
        "says the same as of 14 October 2026.",
        (
            L(AH, "pyproject.toml", "38", "scikit-learn"),
            L(AH, "asset_health/detect/iforest.py", "9", "sklearn"),
            L(AH, "asset_health/rul/model.py", "18", "sklearn"),
        ),
        "low",
        READS,
    ),
    Finding(
        "S1",
        "A repair record hides a fault that was not repaired.",
        "The pipeline cuts an asset's history into segments at each repair record and treats the "
        "data after a cut as a new healthy life: it skips 48 hours, learns a baseline over the "
        "next 21 days from whatever data follows, and only then scores. The record has an asset "
        "id, a time, a kind and a free-text note the code does not read, so nothing says whether "
        "the part was renewed. In the check, the same worn impeller with a repair record and no "
        "renewal scores far lower and opens no alert, against an open alert without the record "
        "(numbers in the evidence section).",
        (
            L(AH, "asset_health/pipeline.py", "103", "segment_bounds"),
            L(AH, "asset_health/features/build.py", "84", "baseline_days"),
            L(AH, "asset_health/models.py", "208-217"),
        ),
        "high",
        RUNS,
    ),
    Finding(
        "S2",
        "The generator cannot be steered mid-record.",
        "`simulate_asset` takes a random generator, a config, an asset spec, one optional failure "
        "and planned services. It has no way to stop or derate the asset on command, to continue "
        "an operating regime from an earlier record, or to take a custom severity curve. The "
        "operating regime and the measurement noise come from one random stream, so a failure's "
        "own random draws shift the noise that follows.",
        (
            L(AH, "asset_health/simulate/fleet.py", "264", "def simulate_asset"),
            L(AH, "asset_health/simulate/fleet.py", "276", "generate_regime"),
            L(AH, "asset_health/simulate/regime.py", "59-66"),
        ),
        "medium",
        RUNS,
    ),
    Finding(
        "S3",
        "The logger clock cannot continue across a join.",
        "The data-fault injector draws the clock error once per call (whether the asset drifts, "
        "the rate, up to three steps) and measures it from the first row of the frame it is "
        "given. Two records joined end to end each restart the drift from zero.",
        (
            L(AH, "asset_health/simulate/faults.py", "144", "drift"),
            L(AH, "asset_health/simulate/faults.py", "137-154"),
        ),
        "low",
        RUNS,
    ),
    Finding(
        "S4",
        "No sensor gain drift exists, and no fault can be placed.",
        "The data-fault kinds are a fixed list of eight (dropout, stuck, sentinel, glitch, comm "
        "outage, missing row, duplicate row, clock drift). A slow change of scale is not among "
        "them, and fault times are drawn at random, so a fault cannot be placed at a chosen time.",
        (
            L(AH, "asset_health/models.py", "232", "Literal["),
            L(AH, "asset_health/models.py", "240", "clock_drift"),
        ),
        "low",
        RUNS,
    ),
    Finding(
        "S5",
        "The fused score is not a value between 0 and 1.",
        "It is 0.4 times the CUSUM score plus 0.3 times the isolation-forest score plus 0.3 times "
        "the rule score, each clipped at 3.0, and an alert opens when it stays at or above 1.0 for "
        "four valid hours with two detectors agreeing. Anything fed into a 0 to 1 field needs an "
        "explicit rescale; the agent's anomaly score has such a field.",
        (
            L(AH, "asset_health/config.py", "204", "clip"),
            L(AH, "asset_health/detect/ensemble.py", "98", "weight_rule"),
            L(MA, "maintenance_agent/models.py", "243"),
        ),
        "medium",
        RUNS,
    ),
    Finding(
        "S6",
        "A work order can be open without an interval, and carries at most four pieces of "
        "evidence.",
        "Evidence is up to four channels (`n = 4`) at least 2.0 standard deviations from expected "
        "in the harmful direction, strongest first; it can hold fewer, or none. A work order can "
        'also exist while the RUL status is "not degrading" or "insufficient trend": it then '
        "gets warning severity, no latest safe date and no interval. This ties directly to M4.",
        (
            L(AH, "asset_health/alerting/workorder.py", "54", "n: int = 4"),
            L(AH, "asset_health/alerting/workorder.py", "108-120"),
            L(AH, "asset_health/alerting/policy.py", "172-177"),
        ),
        "high",
        RUNS,
    ),
    Finding(
        "S7",
        "An unread field is not inert.",
        "`RULInterval.confidence` is read by no rule, but the alert fingerprint hashes the whole "
        "interval. The orchestrator compares the fingerprint recorded when it drafted with the "
        "live "
        "alert's at every re-verification, and any difference makes rule V14 block. Changing only "
        "`confidence` upstream therefore blocks an otherwise identical draft until it is "
        "redrafted; the anomaly score, summary and notes are not in the fingerprint.",
        (
            L(MA, "maintenance_agent/models.py", "207", "confidence"),
            L(MA, "maintenance_agent/verifier.py", "174", "rul"),
            L(MA, "maintenance_agent/orchestrator.py", "626-638"),
        ),
        "medium",
        RUNS,
    ),
    Finding(
        "S8",
        "Same defect as M12, seen from the signature.",
        "`build_system` is keyword-only and lists home, planner, store, alerts, registry, budget, "
        "tool wrapper, fault hook, retries, clock, `cmms_clock` and the maximum alert age; the "
        "nearest thing to a CMMS option is `cmms_clock`. Counted separately because it was found "
        "twice.",
        (L(MA, "maintenance_agent/wiring.py", "81-95", "def build_system"),),
        "medium",
        READS,
    ),
    Finding(
        "S9",
        "The dry run cannot see live stock.",
        "`MockCMMS.draft_work_order` reads its private parts table and does not go through the "
        "public `get_part`, so replacing `get_part` does not reach the dry run. The constructor "
        "accepts a parts table, but it is a snapshot, not live stock, and `build_system` does not "
        "let one be passed in.",
        (
            L(MA, "maintenance_agent/cmms.py", "108", "_parts"),
            L(MA, "maintenance_agent/cmms.py", "66-92"),
        ),
        "medium",
        READS,
    ),
    Finding(
        "S10",
        "An approval has a practical lifetime of one calendar day.",
        "The verification digest covers the check date. When an approval is released, the "
        "orchestrator re-verifies against the live sources and voids the approval if the digest "
        "differs. An approval signed late on one day and released after midnight is void and must "
        "be signed again (in the check, the run returns to awaiting approval with nothing "
        "released). This is documented, safe behaviour; it matters for a simulation that crosses "
        "midnight.",
        (
            L(MA, "maintenance_agent/orchestrator.py", "727", "verification_digest"),
            L(MA, "maintenance_agent/verifier.py", "144"),
            L(MA, "maintenance_agent/verifier.py", "812-830"),
        ),
        "medium",
        RUNS,
    ),
    Finding(
        "S11",
        "From asset-health's evidence the agent drafts an inspection, not a repair.",
        "The agent scores each failure mode as the weighted share of its listed signals that are "
        "over their alarm levels, and writes an inspection-first draft when the best score is "
        "under 0.55 (or the top two are within 0.15). Crusher liner wear is 75% liner thickness "
        "(an ultrasonic survey) and 25% power deviation; pump impeller wear is 50% flow deviation, "
        "30% discharge-pressure drop and 20% overall vibration; bearing defect is 45% "
        "high-frequency envelope, 25% bearing temperature, 15% overall vibration and 15% oil "
        "particle count. The scorer keeps one evidence item per signal, and asset-health's four "
        "strongest channels often map to the same agent signal, so the reachable score is one "
        "signal's weight: 0.25 for liner wear, 0.50 for impeller wear, 0.45 for a bearing, all "
        "under 0.55. Channels that map to different signals can score higher (a test builds such a "
        "case). The counts from live episodes are in the measurements section.",
        (
            L(MA, "maintenance_agent/planners/rules.py", "117", "exceedance"),
            L(MA, "maintenance_agent/fixtures/playbook.json", "3-4"),
            L(MA, "maintenance_agent/fixtures/playbook.json", "140-188"),
            L(MA, "maintenance_agent/planners/rules.py", "228-247"),
        ),
        "high",
        RUNS,
    ),
    Finding(
        "S12",
        "plant-copilot reads its corpus on every construction.",
        "`PlantCopilot.__init__` calls `load_corpus` whenever it is given a path or nothing, and "
        "the search index is built lazily inside each instance, so a new instance repeats the "
        "work. Passing an already loaded corpus skips the read.",
        (
            L(PC, "plant_copilot/copilot.py", "46", "load_corpus"),
            L(PC, "plant_copilot/retrieve.py", "174-191"),
        ),
        "low",
        RUNS,
    ),
)


# ----------------------------------------------------------------------------- rendering

_INTRO = """# Findings from wiring three repositories together

**SYNTHETIC.** Every repository, asset, part number, procedure, alert and number named here is invented for research. No real plant, person or organisation is described.

## What this document is

`closed-loop` is a research harness that runs three small, separate repositories as one system for the first time:

- **asset-health** reads sensor data from monitored equipment, detects anomalies, estimates the remaining useful life (RUL) of a component and writes a work order for an open alert. Its Python package is `asset_health`.
- **maintenance-agent** takes an alert, drafts a maintenance work order from it, checks the draft against sixteen verifier rules (V00 to V15), asks a human to sign it with an Ed25519 key, and releases it to a maintenance system. Its package is `maintenance_agent`.
- **plant-copilot** retrieves and quotes plant procedures and standards, word for word, with the document, clause and revision. Its package is `plant_copilot`.

Putting them in one loop exposed 33 places where they do not fit together or where one of them behaves in a way that limits the loop. Each is listed below with the file and line that show it, a severity and a status. **None has been fixed.** The three repositories were read and run, not modified; the status of every row is "recorded, not fixed".

The rows are of two kinds. **M1 to M21** are *mismatches*: two repositories describe the same thing in different ways, or leave a gap between them. **S1 to S12** are *sibling defects*: behaviour in one repository that surprised us or limits what it can be used for. Row S8 states the same fact as M12 from the side of the function signature; it has its own row because it was found twice.

## How to read the table

- **File and line** are relative to the root of the named repository, for example `asset-health: asset_health/models.py:232`. Line numbers are those of version 0.1.1 as installed on 1 November 2026; `python scripts/verify_siblings.py` checks the installed copies against `integration/siblings.lock` and against the repository sources. A different version can move a line without changing the finding.
- **Severity** is a judgement of how much the point matters for running the three together, not a measurement. *High*: the hand-off between two repositories stops working, or a fault is hidden. *Medium*: an adapter or a decision is needed, and a wrong choice gives wrong results. *Low*: a limit, a nuisance or a documented design choice.
- **Check** says how the row was verified. *runs the siblings*: a function in `closed_loop/f2/mismatch.py` with the same id runs the real code of the installed repositories and reports what it saw. *reads their source*: the check parses the installed source or reads a signature, because the point is the absence of something. `tests/f2/test_mismatch.py` runs all 33 and fails if one stops holding, which would mean a repository has changed; `tests/f2/test_findings.py` checks that every cited line exists and contains the text the row relies on. No check substitutes a stand-in for a sibling.
- Numbers come only from the generated sections at the end, which are copied from the checks and from the measurement script.
"""  # noqa: E501

_OUTRO = """## What these findings do not cover

- They were found by running the siblings on synthetic data, with the plant model of this harness (four monitored assets, 90-day episodes, 15-minute ticks) driving them. A different plant, a different alert mix or longer episodes could show other behaviour. Some points (S11 in particular) depend on how asset-health's channels are mapped onto the agent's signals, which is a choice made in this harness (`closed_loop/f2/alert_source.py`).
- A finding marked "runs the siblings" holds for the installed version 0.1.1 on the date of the check. The tests re-run them; none of them proves a property for another version.
- The numbers in the generated sections are development numbers on development seeds, taken to see whether the integration works and what it costs. They are not results, and no held-out or pre-registered seed was used.
- Nothing here says the siblings are wrong to behave as they do. Several rows (M12, M13, M16, S9, S10) are sensible choices for a demonstrator that become limits when the repositories are driven in a loop.
"""  # noqa: E501


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def _where(loc: Location) -> str:
    return f"{loc.repo}: `{loc.path}:{loc.lines}`"


def render_table() -> str:
    """The markdown table of all findings, one row each."""
    rows = [
        "| ID | Finding | File and line | Severity | Status | Check |",
        "|---|---|---|---|---|---|",
    ]
    for f in FINDINGS:
        where = "; ".join(_where(loc) for loc in f.locations)
        rows.append(
            f"| {f.id} | **{_cell(f.title)}** {_cell(f.text)} | {_cell(where)} | {f.severity} "
            f"| {f.status} | {f.check} |"
        )
    return "\n".join(rows)


def render_measured(
    episodes: Mapping[str, Any] | None,
    mismatch: Mapping[str, Any] | None,
    evidence: Sequence[tuple[str, bool, str]] | None = None,
) -> str:
    """The generated sections: what the checks saw, and the counts and timings of F2 episodes.

    Every number is copied from ``evidence`` (the ``Check`` results), ``episodes`` and ``mismatch``
    (the JSON of ``scripts/f2_measure.py``); nothing is rounded here beyond the stored precision.
    """
    out = ["## What the checks saw", ""]
    if evidence:
        out += ["| ID | Holds | Observed |", "|---|---|---|"]
        for check_id, holds, text in evidence:
            out.append(f"| {check_id} | {'yes' if holds else 'NO'} | {_cell(text)} |")
    else:
        out.append("Not generated: run `python scripts/render_findings.py --run-checks`.")
    out += ["", "## What the integration measured", ""]
    if episodes is None:
        out.append("Not generated: run `python scripts/f2_measure.py episodes`.")
    else:
        out += [
            f"SYNTHETIC development numbers. F2 episodes of the three arms on development seed "
            f"{episodes['seed']}, nominal condition, {episodes['context']} fitted context, one "
            "process, one core, below-normal priority on a shared laptop. The budget is "
            f"{episodes['budget_s_per_episode']} s per episode; a composition of separately timed "
            f"pieces estimated {episodes['composed_estimate_s']} s.",
            "",
            "| Arm | Wall s | CPU s | `run_asset` | By the caching rule | `analyse_run` | Alerts "
            "| Drafts naming a mode | Inspection-first | Without a draft "
            "| Repairs sent to the plant | Wrong-mode drafts signed | Unplanned failures |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for e in episodes["episodes"]:
            out.append(
                f"| {e['arm']} | {e['wall_s']} | {e['cpu_s']} | {e['run_asset_calls']} "
                f"| {e['run_asset_calls_by_the_caching_rule']} | {e['analyse_run_calls']} "
                f"| {e['alerts_raised']} | {e['drafts_naming_a_mode']} "
                f"| {e['drafts_inspection_first']} | {e.get('alerts_without_a_draft', 'n/a')} "
                f"| {e['repairs_issued']} | {e['wrong_compliant_signed']} "
                f"| {e['unplanned_failures']} |"
            )
    out += ["", "### Estimator mismatch between the fast simulator and asset-health", ""]
    if mismatch is None:
        out.append("Not generated: run `python scripts/f2_measure.py mismatch`.")
    else:
        rec = mismatch["record"]
        extra = rec["extra"]
        out += [
            f"SYNTHETIC development numbers. {rec['measured_on']}. Mean difference "
            f"{rec['value']} days (confidence sequence {rec['ci_lo']} to {rec['ci_hi']}, "
            f"{rec['ci_kind']}); mean absolute error {extra['mean_mae_f1']} days at F1 and "
            f"{extra['mean_mae_f2']} days at F2; F2 worse on {extra['lives_f2_worse']} of "
            f"{rec['n']} lives; {extra['checkpoints']} checkpoints.",
        ]
    return "\n".join(out)


def render_document(measured: str) -> str:
    """The whole markdown document."""
    return "\n".join([_INTRO, "## The table", "", render_table(), "", measured, "", _OUTRO])
