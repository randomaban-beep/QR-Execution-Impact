# AGENTS.md — Working Rules & Engineering Standards

These rules are strictly binding for all agents working on this quantitative research codebase.

## 1. Environment & Tooling
- **Python Version**: Python 3.11.
- **Package Management**: `uv` or `pip` with exact pinned versions in `requirements.txt` / `pyproject.toml`.
- **Code Quality & Typing**:
  - `ruff` for linting and code formatting (`ruff check .`, `ruff format --check .`).
  - Strict type hints on all function signatures, return values, and module boundaries (`mypy` / `pyright` compliant).
  - `pytest` for unit, integration, and property testing.

## 2. Version Control & Git Hygiene
- **Commit Granularity**: Commit after each completed module or milestone.
- **Commit Messages**: Conventional commits format (`feat:`, `test:`, `docs:`, `fix:`, `refactor:`, `chore:`).
- **Pre-commit Gate**: Always run `ruff` and `pytest` before every commit. Never commit broken code or failing tests.
- **Gitignore Safety**:
  - NEVER commit raw data files, zips, checksum files, parquet caches, or artifacts in `data/`.
  - NEVER commit caches (`.pytest_cache`, `__pycache__`, `.ruff_cache`, `.venv`).
  - NEVER commit secrets, API keys, or local environment configurations.

## 3. Strict No-Lookahead Guarantee
- **No Future Information**: Under no circumstances should future data enter any estimation or decision logic.
- **Prior-Data-Only Estimators**:
  - Trailing volatility ($\sigma$) must use only returns strictly prior to decision timestamp $t_0$.
  - Intraday volume profiles ($V$) must be computed strictly from prior days (e.g. prior 7 days), excluding the trade day / forward intervals.
  - Market impact calibration parameters ($Y$, $\delta$, $\eta$, $\gamma$) must be calibrated on prior out-of-sample data windows.
- **Automated Anti-Lookahead Tests**: Every estimator and schedule generator must include dedicated tests asserting that perturbations to future data have zero effect on past outputs.

## 4. Determinism & Reproducibility
- **Random Seeds**: Explicitly fix random seeds across `numpy`, `random`, `scipy`, and bootstrap routines.
- **Reproducible Tables & Figures**: Every number, metric, and figure in `README.md` and `results/` must be bitwise or numerically reproducible with a single command (e.g. `make report`).

## 5. Research Integrity & Scientific Honesty
- **Unbiased Reporting**: If Almgren-Chriss fails to outperform TWAP/VWAP in certain regime cells (e.g., small order sizes, low volatility, or heavy tail regimes), report the exact empirical finding without post-hoc rationalization.
- **Model Assumption Violations**: Clearly document where empirical assumptions break down (e.g. non-linear vs linear impact, trade endogeneity, absence of order book depth / queue dynamics, fee frictions).
- **Oracle Benchmarks**: Clearly label hypothetical benchmarks (e.g., realized-volume oracle VWAP) as unattainable in live execution.

## 6. Session Continuity & Progress Tracking
- **State Persistence**: Update `PROGRESS.md` immediately at the conclusion of each stage.
- **Session Resumption**: Ensure any fresh session can inspect `PROGRESS.md` and `git log` to resume seamlessly without losing context or duplicating work.
