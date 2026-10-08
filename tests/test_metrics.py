"""Unit tests for quantitative metrics, CVaR95, bootstrap CIs, and Wilcoxon tests."""

import numpy as np
import pandas as pd

from exec_impact.metrics import (
    analyze_grid_cell,
    build_headline_summary,
    compute_cvar95,
    compute_paired_bootstrap_ci,
    compute_paired_wilcoxon,
)


def test_cvar95_formula() -> None:
    # 100 values from 1 to 100
    vals = np.arange(1.0, 101.0)
    p95 = np.percentile(vals, 95.0)  # ~95.05
    cvar = compute_cvar95(vals)

    # CVaR95 must be strictly greater than or equal to 95th percentile
    assert cvar >= p95
    # The tail values are [96, 97, 98, 99, 100], mean is 98.0
    assert np.isclose(cvar, 98.0, atol=0.5)


def test_paired_bootstrap_ci() -> None:
    rng = np.random.default_rng(42)
    # Differences with true mean = 5.0
    diffs = rng.normal(loc=5.0, scale=2.0, size=500)
    ci_low, ci_high = compute_paired_bootstrap_ci(diffs, n_bootstrap=500, random_seed=42)

    assert ci_low <= 5.0 <= ci_high
    assert ci_high > ci_low


def test_paired_wilcoxon() -> None:
    # Identical series -> difference 0 -> p = 1.0
    zeros = np.zeros(100)
    assert compute_paired_wilcoxon(zeros) == 1.0

    # Distinct shift
    rng = np.random.default_rng(42)
    shift = rng.normal(loc=2.0, scale=0.5, size=100)
    p_val = compute_paired_wilcoxon(shift)
    assert p_val < 0.001


def test_analyze_grid_cell_and_headline() -> None:
    start_times = pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC")
    rows = []
    for st in start_times:
        # TWAP shortfall: ~10 bps
        rows.append(
            {
                "strategy_name": "TWAP",
                "symbol": "BTCUSDT",
                "horizon_minutes": 30,
                "order_size_adv_pct": 0.05,
                "start_time": st,
                "side": 1,
                "shortfall_bps": 10.0 + np.random.normal(0, 1),
            }
        )
        # AC shortfall: ~8 bps (beats TWAP by 2 bps)
        rows.append(
            {
                "strategy_name": "AlmgrenChriss_lambda_1.0e-04",
                "symbol": "BTCUSDT",
                "horizon_minutes": 30,
                "order_size_adv_pct": 0.05,
                "start_time": st,
                "side": 1,
                "shortfall_bps": 8.0 + np.random.normal(0, 1),
            }
        )
        # VWAP shortfall: ~9.5 bps
        rows.append(
            {
                "strategy_name": "VWAP",
                "symbol": "BTCUSDT",
                "horizon_minutes": 30,
                "order_size_adv_pct": 0.05,
                "start_time": st,
                "side": 1,
                "shortfall_bps": 9.5 + np.random.normal(0, 1),
            }
        )

    df = pd.DataFrame(rows)
    metrics = analyze_grid_cell(df, twap_strategy_name="TWAP")

    assert len(metrics) == 3
    ac_metric = [m for m in metrics if "AlmgrenChriss" in m.strategy][0]
    assert ac_metric.mean_diff_vs_twap_bps < 0.0  # AC had lower cost

    headline = build_headline_summary(metrics)
    assert len(headline) == 1
    cell = headline[0]
    assert cell["symbol"] == "BTCUSDT"
    assert cell["ac_savings_vs_twap_bps"] > 0.0
