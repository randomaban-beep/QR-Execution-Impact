# PLAN.md — Repository Architecture, Mathematical Design & Test Specification

This plan outlines the architecture, data pipeline, mathematical modeling, simulation engine, test matrix, and deliverables for the `exec-impact` quantitative research project.

---

## 1. Repository Layout

```text
exec-impact/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions: ruff check/format, pytest on Python 3.11
├── config.yaml                  # Global configuration (symbols, dates, grid, seeds, fees)
├── Makefile                     # Top-level commands: data, fit, simulate, report, test, lint
├── pyproject.toml               # Build metadata, pinned tool settings (ruff, pytest)
├── requirements.txt             # Pinned production & research dependencies
├── requirements-dev.txt         # Pinned testing & linting dependencies
├── AGENTS.md                    # Pair-programming working rules & standards
├── PROGRESS.md                  # Stage execution checklist & status
├── SPEC.md                      # Verbatim problem specification
├── README.md                    # Research report, metrics table, figures, limitations
├── data/                        # Gitignored local data cache (zips, checksums, parquet)
│   ├── raw/                     # Binance aggTrades daily CSVs and .CHECKSUM files
│   └── processed/               # Cleaned trade and 1m/5m/15m bar parquet files
├── results/                     # Gitignored generated metrics and plots (committed figures as needed)
│   ├── metrics.json             # Key output metrics table across simulation grid
│   └── figures/                 # Generated plots (impact curves, shortfall vs size, frontiers)
├── src/
│   └── exec_impact/
│       ├── __init__.py          # Package entrypoint & version
│       ├── config.py            # Typed configuration dataclasses & YAML loader
│       ├── data.py              # Download, checksum verify, schema parse, aggTrade normalize
│       ├── bars.py              # 1m/5m/15m bar aggregator, realized vol, trailing volume
│       ├── impact.py            # Impact regressions (Huber, power-law, bootstrap, perm/temp)
│       ├── schedules.py         # TWAP, prior-7d VWAP, Oracle VWAP, Almgren-Chriss, numerical
│       ├── simulate.py          # Tape replay engine & Monte Carlo GBM validator
│       └── metrics.py           # Shortfall (bps), bootstrap CI, Wilcoxon tests, frontiers
├── scripts/
│   ├── download_data.py         # Standalone CLI to download month of aggTrades + checksums
│   ├── build_bars.py            # Process raw data into normalized bar parquets
│   ├── fit_impact.py            # Fit and calibrate impact models; produce parameter tables
│   ├── run_simulation.py        # Execute tape replay and synthetic Monte Carlo grid
│   └── generate_report.py       # Compute metrics.json and export publication figures
└── tests/
    ├── conftest.py              # Pytest fixtures, mock data generators, fixed seeds
    ├── test_data.py             # Checksum, timestamp parsing, schema detection, aggressor side
    ├── test_bars.py             # Bar aggregation, realized vol, volume profiles, no-lookahead
    ├── test_impact.py           # Huber fit, bootstrap CI, perm/temp decomposition, param recovery
    ├── test_schedules.py        # Monotonicity, non-negativity, sum to X, AC limits, 7d no-lookahead
    ├── test_simulate.py         # Tape replay fills, arrival price, Monte Carlo AC optimality
    └── test_metrics.py          # Shortfall bps sign conventions, bootstrap CI, Wilcoxon tests
```

---

## 2. Module Specifications & Responsibilities

### 2.1 Configuration (`src/exec_impact/config.py` & `config.yaml`)
- **Schema**: Typed dataclasses (`pydantic` or `@dataclass`) for type validation.
- **Configurable Settings**:
  - `symbols`: `["BTCUSDT", "ETHUSDT"]`
  - `month`: e.g. `2024-01` (one complete, fixed calendar month)
  - `data_dir`: `./data`
  - `results_dir`: `./results`
  - `random_seed`: `42`
  - `taker_fee_bps`: `4.5` (standard Binance VIP0 taker fee or configurable discount)
  - `horizons`: `[30, 60, 240]` (minutes)
  - `order_sizes_adv_pct`: `[0.01, 0.05, 0.1, 0.5, 1.0]`
  - `lambdas`: `[1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]`
  - `n_replay_samples`: `500`

### 2.2 Data Pipeline (`src/exec_impact/data.py` & `scripts/download_data.py`)
- **Source**: `https://data.binance.vision/data/spot/daily/aggTrades/{symbol}/{symbol}-aggTrades-{date}.zip` and `.CHECKSUM`.
- **Validation**:
  - Compute SHA256 of downloaded zip; assert match against `.CHECKSUM` content.
- **Parsing & Normalization**:
  - Robust header detection: check if line 1 contains column names or integer trade IDs.
  - Timestamp resolution detection:
    - If timestamp $> 10^{14}$, interpret as microseconds; else if $> 10^{11}$, milliseconds.
    - Convert to UTC `datetime64[ns]` or Polars `Datetime("ns", "UTC")`.
  - Aggressor direction:
    - Binance aggTrades column `is_buyer_maker`:
      - `is_buyer_maker == True`: Buyer was passive maker $\implies$ aggressive SELLER (flow $< 0$).
      - `is_buyer_maker == False`: Seller was passive maker $\implies$ aggressive BUYER (flow $> 0$).
  - Storage format: Write validated raw and parsed trades into daily Parquet files with ZSTD compression for speed and memory efficiency.

### 2.3 Bar Construction & Features (`src/exec_impact/bars.py`)
- **Resampling**: Aggregate trades into uniform 1-minute, 5-minute, and 15-minute bars.
- **Bar Fields**:
  - `open_time`, `close_time`: UTC timestamps.
  - `first_price`, `last_price`: Arrival/close prices.
  - `vwap`: $\frac{\sum p_i q_i}{\sum q_i}$.
  - `volume`: Total trade volume $\sum q_i$.
  - `signed_volume` ($Q$): Aggressive buy volume minus aggressive sell volume $\sum q_i \cdot \text{sign}_i$.
  - `trade_count`: Number of aggregate trades.
- **Trailing Estimators (Strictly Causal / No-Lookahead)**:
  - `trailing_volatility` ($\sigma_t$): Realized volatility computed over trailing rolling window $W_{\text{vol}}$ (e.g. 24 hours of 1m log returns) strictly *prior* to bar $t$:
    $$\sigma_t = \sqrt{\frac{1}{M}\sum_{i=1}^M (r_{t-i} - \bar{r})^2} \times \sqrt{\text{bars\_per\_day}}$$
  - `trailing_adv` ($V_t$): Trailing 7-day average daily or interval volume ending at $t - 1$.
  - `intraday_volume_profile`: Empirical fraction of daily volume in interval $k$ of the day, estimated strictly using prior 7 days.

### 2.4 Market Impact Calibration (`src/exec_impact/impact.py`)
- **Normalized Flow Return Model**:
  $$\frac{r_\tau}{\sigma_\tau} = Y \cdot \text{sign}(Q_\tau) \cdot \left(\frac{|Q_\tau|}{V_\tau}\right)^\delta$$
- **Estimation Techniques**:
  - Log-log transform for non-zero flow:
    $$\ln \left|\frac{r_\tau}{\sigma_\tau}\right| = \ln Y + \delta \ln \left(\frac{|Q_\tau|}{V_\tau}\right) + \epsilon_\tau$$
  - Huber Robust Regression: Minimizes Huber loss to mitigate heavy-tailed crypto price shock outliers.
  - Comparison against square-root benchmark ($\delta = 0.5$): estimate $Y$ under fixed $\delta = 0.5$ via robust slope.
  - Uncertainty quantification: 1,000 bootstrap iterations to calculate 95% confidence intervals on $\hat{Y}$ and $\hat{\delta}$.
- **Temporary vs Permanent Impact Decomposition**:
  - For lag $k \in \{1, 2, 5, 10, 15\}$ bars after interval $\tau$:
    $$r_{\tau \to \tau+k} = \ln(P_{\tau+k} / P_\tau)$$
  - Permanent impact parameter $\gamma$: regression of long-term residual price response $r_{\tau \to \tau+k}$ on flow $Q_\tau$.
  - Peak impact: immediate return $r_\tau$.
  - Temporary impact: $\eta = \text{Peak Impact} - \gamma$.
- **Stability & Out-of-Sample (OOS) Testing**:
  - Fit per week (Weeks 1, 2, 3, 4) and per symbol (`BTCUSDT`, `ETHUSDT`) to assess parameter drift.
  - Fit on Weeks 1–2; evaluate out-of-sample $R^2$ on Weeks 3–4:
    $$R^2_{\text{OOS}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y}_{\text{train}})^2}$$
  - Document endogeneity limitations: public trade flow reflects market-wide reaction, not exogenous child order fills.

### 2.5 Execution Strategies (`src/exec_impact/schedules.py`)
Parent order of size $X$ units over horizon $T$ divided into $N$ child slices at intervals $t_k = k \Delta t$:
1. **TWAP (Time-Weighted Average Price)**:
   $$n_k = \frac{X}{N}, \quad k = 1, \dots, N$$
2. **VWAP (Volume-Weighted Average Price)**:
   - Slices proportional to estimated volume profile:
     $$n_k = X \cdot \frac{\hat{v}(t_k)}{\sum_{j=1}^N \hat{v}(t_j)}$$
     where $\hat{v}(t)$ is the historical median/mean fraction of volume for that intraday bucket over the preceding 7 days.
   - **Oracle VWAP**: Proportional to realized volume $v_{\text{realized}}(t_k)$ (labeled unattainable benchmark).
3. **Almgren-Chriss (Linear Impact Closed-Form)**:
   - Trajectory of remaining shares $x_k$:
     $$x_k = X \cdot \frac{\sinh(\kappa(T - t_k))}{\sinh(\kappa T)}, \quad n_k = x_{k-1} - x_k$$
   - Urgency parameter:
     $$\kappa = \sqrt{\frac{\lambda \sigma^2}{\tilde{\eta}}}$$
   - **Square-Root Linearization**:
     The calibrated model has impact $I(n) = \eta \left(\frac{n}{V}\right)^{1/2}$. To map into linear impact $\tilde{\eta} n$:
     Linearize around average slice size $\bar{n} = X / N$:
     $$\tilde{\eta} = \left. \frac{d I(n)}{dn} \right|_{n = \bar{n}} = \frac{1}{2} \frac{\eta}{V^{1/2} \bar{n}^{1/2}}$$
     or effective secant impact $\tilde{\eta} = \frac{I(\bar{n})}{\bar{n}}$. Document both explicitly.
4. **Numerical Optimal Execution (Stretch Goal)**:
   - Solve constrained optimization under power-law temporary impact:
     $$\min_{\{n_k\}} \sum_{k=1}^N \left( \gamma X_k n_k + \eta \left(\frac{n_k}{V_k}\right)^\delta n_k \right) + \lambda \sigma^2 \Delta t \sum_{k=1}^N x_k^2$$
     subject to $\sum_{k=1}^N n_k = X, \quad n_k \ge 0, \quad x_k = X - \sum_{j=1}^k n_j$.
5. **Parameter Sweep**:
   - $\lambda \in \{10^{-8}, 10^{-6}, 10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}\}$ (from risk-neutral / near-TWAP to aggressive execution).

### 2.6 Simulation Engine (`src/exec_impact/simulate.py`)
- **Mode A: Tape Replay Simulator**:
  - Sample 500+ paired random starting timestamps $t_0$ across weeks 3–4 (strictly after calibration window).
  - All strategies (TWAP, VWAP, AC across all $\lambda$, Oracle VWAP) run on identical start times.
  - **Arrival Price** $S_0$: Last trade price immediately prior to $t_0$.
  - **Slice Fill Price**:
    For slice $k$ with size $n_k$ (signed: $+n_k$ for buy, $-n_k$ for sell):
    1. Base tape price: VWAP of raw tape trades in $[t_{k-1}, t_k]$: $P_k^{\text{tape}}$.
    2. Calibrated temporary impact: $\Delta P_{\text{temp}} = S_0 \cdot \eta \left(\frac{n_k}{V_k}\right)^\delta \cdot \text{sign}$.
    3. Accumulated permanent impact:
       $$\Delta P_{\text{perm}, k} = S_0 \cdot \gamma \sum_{j=1}^{k-1} \left(\frac{n_j}{V_j}\right)^\delta \cdot \text{sign} + \frac{1}{2} S_0 \cdot \gamma \left(\frac{n_k}{V_k}\right)^\delta \cdot \text{sign}$$
    4. Fill price: $\tilde{P}_k = P_k^{\text{tape}} + \Delta P_{\text{temp}} + \Delta P_{\text{perm}, k}$.
    5. Order fill does not alter future tape (explicitly documented assumption).
- **Mode B: Synthetic Monte Carlo Simulator (GBM)**:
  - Correctness verification: simulate $dS_t = \sigma S_t dW_t$.
  - Execute AC, TWAP, VWAP under idealized linear impact $I(n) = \eta n + \gamma n$.
  - Confirm AC achieves the empirical minimum of $E[\text{IS}] + \lambda \text{Var}(\text{IS})$ matching theoretical Almgren-Chriss derivation.

### 2.7 Evaluation Metrics & Statistics (`src/exec_impact/metrics.py`)
- **Implementation Shortfall (IS)**:
  $$\text{IS}_{\text{bps}} = \text{side} \cdot \left(\frac{\bar{P}_{\text{exec}} - S_0}{S_0}\right) \times 10^4$$
  where $\text{side} = +1$ for Buy, $-1$ for Sell, and $\bar{P}_{\text{exec}} = \frac{1}{X}\sum_{k=1}^N n_k \tilde{P}_k$.
- **Distributional Metrics**:
  - Mean, Standard Deviation, Median, 95th Percentile, CVaR95 (Expected Shortfall beyond 95th percentile).
- **Paired Hypothesis Testing vs TWAP**:
  - Difference $\Delta_i = \text{IS}_{\text{strat}, i} - \text{IS}_{\text{TWAP}, i}$ for each paired start time $i$.
  - Bootstrap 95% Confidence Interval of mean paired difference.
  - Wilcoxon signed-rank non-parametric test $p$-value.
- **Efficient Frontier**:
  - Plot Expected Cost $E[\text{IS}]$ vs Cost Volatility $\text{Std}(\text{IS})$ across the $\lambda$ spectrum.
  - Constant exchange taker fee line (e.g. 4.5 bps).

---

## 3. Comprehensive Test Specification (`tests/`)

Every test must run and pass under `pytest -v`.

| Test File | Test Name | Invariant / Property Tested |
|:---|:---|:---|
| `test_data.py` | `test_checksum_verification` | Correctly verifies valid SHA256; raises on corrupted file. |
| `test_data.py` | `test_schema_header_detection` | Automatically parses files both with and without CSV headers. |
| `test_data.py` | `test_timestamp_ms_us_detection` | Accurately converts ms ($10^{12}$) and $\mu$s ($10^{15}$) to identical UTC timestamps. |
| `test_data.py` | `test_aggressor_side_mapping` | `is_buyer_maker == True` maps to Sell ($-1$); `False` maps to Buy ($+1$). |
| `test_bars.py` | `test_bar_aggregation_quantities` | Bar volume equals sum of trade volumes; price within trade range; signed volume correct. |
| `test_bars.py` | `test_trailing_volatility_no_lookahead` | Changing bar $t+1$ trade price has zero effect on $\sigma_t$. |
| `test_bars.py` | `test_empty_intervals_handling` | Zero-trade minutes forward-fill last price with zero volume without NaN crashes. |
| `test_impact.py` | `test_parameter_recovery_synthetic` | Generate synthetic data with known $Y=0.45, \delta=0.55$; verify Huber fit recovers within tolerance. |
| `test_impact.py` | `test_bootstrap_ci_coverage` | Bootstrap 95% CI contains true parameter on synthetic benchmark. |
| `test_impact.py` | `test_impact_decomposition_signs` | Permanent impact $\gamma \ge 0$, temporary impact $\eta \ge 0$, total peak $\ge$ permanent. |
| `test_schedules.py` | `test_schedule_sum_and_nonnegativity` | $\sum n_k == X \pm 10^{-12}$ and $n_k \ge 0$ for all strategies. |
| `test_schedules.py` | `test_schedule_monotone_holdings` | Remaining holdings $x_k$ is strictly monotonically decreasing to 0. |
| `test_schedules.py` | `test_ac_lambda_limits` | $\lim_{\lambda \to 0} \text{AC} = \text{TWAP}$; larger $\lambda$ produces strictly more front-loaded schedule. |
| `test_schedules.py` | `test_vwap_weights_sum_to_one` | VWAP slice weights sum to 1.0. |
| `test_schedules.py` | `test_vwap_profile_no_lookahead` | Perturbing Day 8 volume does not alter the 7-day prior volume profile for Day 8. |
| `test_simulate.py` | `test_shortfall_sign_conventions` | Price rise hurts buyer (positive shortfall) and benefits seller (negative shortfall). |
| `test_simulate.py` | `test_monte_carlo_ac_optimality` | In linear GBM Monte Carlo, AC schedule minimizes $E[\text{IS}] + \lambda \text{Var}(\text{IS})$ vs arbitrary perturbations. |
| `test_metrics.py` | `test_paired_difference_wilcoxon` | Wilcoxon signed-rank test correctly detects significant difference when present and returns $p \approx 1$ for identical schedules. |
| `test_metrics.py` | `test_cvar95_formula` | CVaR95 is greater than or equal to 95th percentile shortfall. |

---

## 4. Dependencies & Toolchain

### Production Dependencies (`requirements.txt`)
- `numpy==1.26.4`
- `pandas==2.2.2`
- `polars==0.20.31`
- `pyarrow==16.1.0`
- `scipy==1.13.1`
- `statsmodels==0.14.2`
- `pyyaml==6.0.1`
- `requests==2.32.3`
- `tqdm==4.66.4`
- `matplotlib==3.8.4`
- `seaborn==0.13.2`

### Development & CI Dependencies (`requirements-dev.txt`)
- `pytest==8.2.2`
- `pytest-cov==5.0.0`
- `ruff==0.4.8`

---

## 5. Execution Stages & Workflow Gates

1. **Stage 0: Repository Scaffolding & CI Environment**
   - Setup project structure, `config.yaml`, `Makefile`, `.gitignore`, CI workflow.
2. **Stage 1: Data Acquisition & Normalization**
   - Download aggTrades, verify checksums, parse schemas, test normalization.
3. **Stage 2: Bar Construction & Microstructure Features**
   - Aggregate 1m bars, compute realized vol and 7d rolling volume profiles with strict anti-lookahead.
4. **Stage 3: Market Impact Calibration**
   - Huber robust log-log regressions, fixed $\delta=0.5$ comparison, bootstrap CIs, perm/temp decomposition, OOS test.
5. **Stage 4: Execution Schedules**
   - Implement TWAP, prior-7d VWAP, Oracle VWAP, Almgren-Chriss, numerical optimizer, lambda sweeps.
6. **Stage 5: Simulation Engine**
   - Tape replay over 500+ paired starts and synthetic Monte Carlo GBM validation.
7. **Stage 6: Metrics & Evaluation**
   - Implementation shortfall distributions, bootstrap CIs, Wilcoxon tests, efficient frontier.
8. **Stage 7: Reporting, Figures & README**
   - Generate `metrics.json`, publication plots, and exhaustive `README.md` with transparent limitations.
9. **Stage 8: Final Review & Quality Gate**
   - Full test run, linting, and reproduction verification.

---

## 6. Scientific Integrity & Model Limitations to Document

1. **Endogeneity & Public Flow vs Private Execution**:
   - In public trades, order flow and price changes are simultaneous and mutual. Regressing returns on public trade flow estimates market-wide response, not the exogenous price response to a private child execution.
2. **Order Book Mechanics**:
   - The simulation does not model limit order queue placement, cancellations, or hidden iceberg orders.
3. **Passive Replay Assumption**:
   - The tape replay assumes our child orders do not displace other participants or alter subsequent tape trades.
4. **Asset Class Specificity**:
   - Crypto spot markets trade 24/7 without opening/closing auctions, exhibiting distinct volatility clustering and tick dynamics relative to equity markets.
