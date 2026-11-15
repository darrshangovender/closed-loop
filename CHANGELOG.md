# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed
- CI on Linux: three golden-run tests compared a Windows-recorded hash of the replay intervals bit
  for bit, and the last bit of exp and log can differ between platforms. The counts and the
  provider state are now compared on every platform and the hash only on Windows. No code under
  `closed_loop/`, no result and no number in the README changed.

### Known issue
- On Linux with Python 3.11 (numpy 2.4.6, scipy 1.17.1), `test_cma_es_reaches_it_within_200_evaluations_on_some_seeds_and_random_search_on_none`
  fails: CMA-ES reached the planted violation on 0 of 10 seeds, against 5 of 10 on Windows with
  the same versions. Under investigation. The README now says the two full runs were on one Windows
  laptop and have not been rerun elsewhere.

## [0.1.0] - 2026-11-15

First public release. Everything in this repository is SYNTHETIC.

### Added
- The plant, perception, causal, planning, shield, specification, search, statistics and audit
  packages, the F2 loop over asset-health and maintenance-agent with plant-copilot's output read from
  checked-in caches, and the evaluation in `benchmarks/run.py`.
- `docs/PREREGISTRATION.md`, frozen on 2 November 2026, with the ledger in `benchmarks/ledger.jsonl`.
- Two full runs on held-out seeds. The first found a crash in `f2/telemetry.py`, and its numbers stay
  in the report beside the second run's (`docs/adr/0003-review-amendments.md`, A20).
- `examples/hostile_episode.py`: two episodes on one seed, with the shield off and on.

### Fixed after the freeze
All recorded in A20. None changed a decision rule, a metric, a seed or a tuned value.
- A piece of a life that restarts past the life's failure day is dropped as intended, and counted.
- A plain string could pass the shield's gate. No proposer built one.
- A run wrote this machine's absolute path into its records, and into the keys of its cached cells.
- The report's failure section read the F1 cells only.
- `benchmarks/run.py` now writes `docs/WHAT-GETS-THROUGH.md` and the README's Results block after a
  full run, and `--check` compares both with `results.json`.
- New records: `shield.uncovered.<arm>.<condition>`, the shield's decisions with no allowed command,
  in the report; and `f2.dropped_pieces.<arm>`, the telemetry pieces the guard dropped, in
  `docs/WHAT-GETS-THROUGH.md`.
- The verdict tables print the secondary rows apart from the registered claims.
- After the second run, from a fact-check of the README: C-TOGETHER's level reads "no test", the
  confirmed fraction carries its exact interval, the C-PLAN table names the level of each column, and
  `docs/WHAT-GETS-THROUGH.md` says which tables come from search seeds.
- Two F2 test modules imported pandas at the top, so a clean install of `.[dev]` could not collect
  the suite. They now skip without it, and a packaging test guards against the same import.
- The closing line of each archived run log, which listed this machine's paths, was cut. In run 1's
  log the home-folder prefix of the one F2 traceback line now reads `<repo>\`. The run records and
  the ledger are unchanged.
- 59 test, script and spike files had Windows line endings (CRLF), against `.gitattributes`. They
  now end lines with LF. No other byte changed. The 31 files under `closed_loop/` with CRLF were
  left as they are.

### Scaffold (31 October 2026)

Scaffold. Everything in this repository is SYNTHETIC: no real plant, client or customer data is
used anywhere.

### Added
- Repository shape: `pyproject.toml` (hatchling, flat `closed_loop/` package), `Makefile`, CI
  workflow, `.gitattributes` and `.gitignore`.
- `closed_loop.core`: frozen types (`Fidelity`, `Op`, `Permit`, `Cmd`, `PlanCmd`, `Command`,
  health bins), keyed random streams (`rng_for`), the 15-minute time grid, `ResultRecord` and the
  dense per-quarter `Trace`.
- Empty packages for every other area, so that later work only adds files.
- `integration/siblings.lock` stub. The three sibling repositories are an optional extra and are
  referenced by direct URL only, never by a bare name.
