.PHONY: all test lint format data fit simulate report clean

PYTHON ?= python

all: lint test

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

test:
	$(PYTHON) -m pytest

data:
	$(PYTHON) scripts/download_data.py
	$(PYTHON) scripts/build_bars.py

fit:
	$(PYTHON) scripts/fit_impact.py

simulate:
	$(PYTHON) scripts/run_simulation.py

report:
	$(PYTHON) scripts/generate_report.py

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
