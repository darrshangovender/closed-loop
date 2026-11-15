.PHONY: install test lint bench quick clean

install:
	pip install -e ".[dev]"

# F2 tests are marked "f2" and need the three sibling repositories; they skip themselves when absent.
test:
	pytest tests/ -q

lint:
	ruff check .

# The full run: held-out seeds, ledger entry, regenerates benchmarks/results.json and RESULTS.md.
bench:
	python benchmarks/run.py

# Development seeds only, no ledger, banner "QUICK RUN, NOT A RESULT".
quick:
	python benchmarks/run.py --quick

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .hypothesis benchmarks/.cache *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
