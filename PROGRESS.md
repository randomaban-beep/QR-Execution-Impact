# PROGRESS.md — Project Roadmap & Stage Checklist

This checklist tracks the execution stages for the `exec-impact` quantitative research project.
Status indicators:
- `[ ]` Not started
- `[/]` In progress
- `[x]` Completed

---

## Stages Checklist

### [x] Stage 0: Repository Scaffolding & CI Environment
- [x] Initialize git repository and configure `.gitignore` (ignore `data/`, caches, secrets, artifacts).
- [x] Configure `pyproject.toml` / `requirements.txt` with pinned dependencies (Python 3.11, numpy, pandas, polars, pyarrow, scipy, statsmodels, ruff, pytest, matplotlib, seaborn, pyyaml).
- [x] Configure `ruff` linting and formatting settings.
- [x] Set up GitHub Actions CI workflow (`.github/workflows/ci.yml`) running ruff and pytest.
- [x] Set up `config.yaml` specifying symbols (`BTCUSDT`, `ETHUSDT`), dates, directories, parameters, and random seed.
- [x] Set up base `Makefile` with entrypoints (`test`, `lint`, `data`, `fit`, `simulate`, `report`).
- **Verification**: Run `pytest` and `ruff check .` with clean exits. (PASSED)

### [x] Stage 1: Data Acquisition & Normalization Pipeline (`src/exec_impact/data.py`)
- [x] Implement `scripts/download_data.py` to fetch spot aggTrades daily zips and `.CHECKSUM` files from `data.binance.vision`.
- [x] Implement SHA256 checksum verification against downloaded `.CHECKSUM` files.
- [x] Build robust parser in `src/exec_impact/data.py`:
  - Detect headers vs headerless CSVs.
  - Detect millisecond vs microsecond timestamps and normalize to UTC datetime.
  - Parse aggressor side: `is_buyer_maker == True` -> Seller is aggressor (Sell), `False` -> Buyer is aggressor (Buy).
  - Clean data types and cast to Arrow/Polars/Parquet storage.
- [x] Write unit tests: checksum validation, timestamp parsing, schema detection, aggressor side mapping.
- **Verification**: Run `pytest tests/test_data.py` and verify test coverage. (PASSED - 7/7 tests passed)

### [x] Stage 2: Bar Construction & Microstructure Features (`src/exec_impact/bars.py`)
- [x] Implement aggregation engine to build 1-minute, 5-minute, and 15-minute bars:
  - `last_price`: last trade price in the interval.
  - `volume`: total trade quantity.
  - `signed_volume` ($Q$): aggressor buy volume minus aggressor sell volume.
  - `trade_count`: total trades in the interval.
  - `realized_volatility` ($\sigma$): trailing realized volatility strictly prior to bar close.
  - `trailing_volume` ($V$): trailing average volume strictly prior to bar close.
- [x] Anti-lookahead assertions in rolling bar calculations.
- [x] Unit tests for aggregation math, boundary conditions, zero-trade intervals, and sign alignment.
- **Verification**: Run `pytest tests/test_bars.py`. (PASSED - 11/11 tests passing)

### [ ] Stage 3: Market Impact Calibration & Decomposition (`src/exec_impact/impact.py`)
- [ ] Implement normalized return regression:
  $$\frac{r}{\sigma} = Y \cdot \text{sign}(Q) \cdot \left(\frac{|Q|}{V}\right)^\delta$$
- [ ] Log-log robust regression using Huber regressor; compare freely fitted $\delta$ against fixed square-root law ($\delta = 0.5$).
- [ ] Bootstrap resampling for 95% confidence intervals on $Y$ and $\delta$.
- [ ] Decompose impact into temporary vs permanent:
  - Permanent impact: price response persisting $k$ intervals forward.
  - Temporary impact: peak impact minus permanent impact.
- [ ] Weekly stability and parameter drift tracking (by symbol and week).
- [ ] Out-of-sample predictive $R^2$ evaluation (fit on weeks 1–2, evaluate on weeks 3–4).
- [ ] Synthetic parameter recovery tests (simulate flow with known $Y$ and $\delta$, verify recovery).
- **Verification**: Run `pytest tests/test_impact.py`.

### [ ] Stage 4: Execution Schedules & Optimization (`src/exec_impact/schedules.py`)
- [ ] TWAP strategy: uniform child slices $\frac{X}{N}$.
- [ ] VWAP strategy:
  - Estimated VWAP: intraday volume profile computed strictly from prior 7 days rolling volume.
  - Oracle VWAP: realized volume profile benchmark (clearly flagged as unattainable oracle).
- [ ] Almgren-Chriss optimal trajectory:
  $$x_k = X \cdot \frac{\sinh(\kappa(T - t_k))}{\sinh(\kappa T)}$$
  with $\kappa = \sqrt{\frac{\lambda \sigma^2}{\tilde{\eta}}}$ where $\tilde{\eta}$ linearizes temporary impact.
- [ ] Linearization of square-root law impact around reference order size.
- [ ] Numerical trajectory optimizer under power-law temporary impact (scipy/cvxpy) as stretch comparison.
- [ ] Risk-aversion parameter sweep ($\lambda$ across at least 6 values from near-zero to large).
- [ ] Unit tests: total order sum equals $X$, non-negativity, monotone holdings, $\lambda \to 0$ convergence to TWAP, strict anti-lookahead test on 7-day VWAP profile.
- **Verification**: Run `pytest tests/test_schedules.py`.

### [ ] Stage 5: Execution Simulation Engine (`src/exec_impact/simulate.py`)
- [ ] Simulation mode (A): Tape Replay
  - Sample 500+ paired random start times across the month.
  - Determine arrival price $S_0$ as the last trade before start.
  - Compute slice fill price = tape VWAP over slice interval + calibrated temporary impact + accumulated permanent impact.
  - Explicit documentation: order does not alter historical tape.
- [ ] Simulation mode (B): Synthetic Monte Carlo
  - Geometric Brownian Motion (GBM) price simulator with linear impact.
  - Correctness verification that Almgren-Chriss minimizes mean-variance utility $E[x] + \lambda \text{Var}(x)$.
- [ ] Parameter grid execution: sizes $\{0.01, 0.05, 0.1, 0.5, 1.0\} \times \text{ADV}$, horizons $\{30\text{m}, 1\text{h}, 4\text{h}\}$, symbols $\{BTC, ETH\}$, sides $\{Buy, Sell\}$.
- [ ] Unit tests: buy vs sell shortfall sign conventions, zero-impact benchmark tests, Monte Carlo optimality checks.
- **Verification**: Run `pytest tests/test_simulate.py`.

### [ ] Stage 6: Evaluation Metrics & Efficient Frontier (`src/exec_impact/metrics.py`)
- [ ] Calculate Implementation Shortfall (IS) in basis points (sign-adjusted: positive = cost).
- [ ] Compute summary statistics per grid cell: mean, std, 95th percentile, CVaR95.
- [ ] Paired difference tests vs TWAP: bootstrap 95% confidence intervals and Wilcoxon signed-rank p-values.
- [ ] Risk/cost efficient frontier analysis (mean cost vs std of cost across $\lambda$).
- [ ] Configurable exchange fee integration (taker fee modeled as separate line).
- [ ] Headline comparison table generator.
- [ ] Unit tests for metric formulas, shortfall edge cases, bootstrap CI coverage.
- **Verification**: Run `pytest tests/test_metrics.py`.

### [ ] Stage 7: Reporting, Figures & README (`scripts/generate_report.py`, `README.md`)
- [ ] Generate `results/metrics.json` capturing all grid results and statistics.
- [ ] Generate high-resolution figures in `results/figures/`:
  - `impact_curves.png`: fitted vs square-root impact curves with confidence bands.
  - `cost_vs_size.png`: implementation shortfall vs order size (% ADV) across strategies.
  - `efficient_frontier.png`: mean shortfall vs standard deviation across $\lambda$ values.
  - `volume_profiles.png`: 7-day prior estimated volume profiles vs actual profiles.
- [ ] Write comprehensive, honest `README.md`:
  - Mathematical problem formulation and execution model.
  - Empirical results table (highlighting where AC wins AND where it fails).
  - Limitations and assumptions (endogeneity, passive fills, no order book queue, single month, crypto vs equities).
  - Complete reproduction instructions (`make report`).
- **Verification**: Run `make report` and verify clean generation of all artifacts.

### [ ] Stage 8: Final Review & Quality Gate
- [ ] Full test suite execution (`pytest -v`).
- [ ] Full lint and format verification (`ruff check .`, `ruff format --check .`).
- [ ] Verify clean git state, no tracked binary data or caches.
