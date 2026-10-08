"""Unit tests for simulation engine: sign conventions, replay fills, and Monte Carlo AC optimality."""

import numpy as np
import pandas as pd

from exec_impact.schedules import generate_twap_schedule
from exec_impact.simulate import (
    run_monte_carlo_gbm_validation,
    sample_replay_start_indices,
    simulate_single_order_replay,
)


def create_mock_bars(n_bars: int = 60, base_price: float = 40000.0) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "open_time": dates,
            "last_price": np.full(n_bars, base_price),
            "vwap": np.full(n_bars, base_price),
            "volume": np.full(n_bars, 100.0),
            "trailing_volatility": np.full(n_bars, 0.001),
            "trailing_volume": np.full(n_bars, 100.0),
        }
    )


def test_shortfall_sign_conventions() -> None:
    """CRITICAL SIGN CONVENTION TEST:

    - Buy order fills at higher price -> positive shortfall (cost).
    - Buy order fills at lower price -> negative shortfall (benefit).
    - Sell order fills at lower price -> positive shortfall (cost).
    - Sell order fills at higher price -> negative shortfall (benefit).
    """
    bars = create_mock_bars(n_bars=10, base_price=40000.0)
    sched = generate_twap_schedule(total_shares=10.0, num_slices=10)

    # 1. Buy order with positive price impact
    res_buy = simulate_single_order_replay(
        schedule=sched,
        bars_window=bars,
        arrival_price=40000.0,
        side=1,  # Buy
        eta_param=0.2,
        gamma_param=0.1,
        delta_param=0.5,
    )
    assert res_buy.exec_price > 40000.0, "Buy should incur positive price impact"
    assert res_buy.shortfall_bps > 0.0, "Buy execution above arrival must have positive shortfall"

    # 2. Sell order with impact
    res_sell = simulate_single_order_replay(
        schedule=sched,
        bars_window=bars,
        arrival_price=40000.0,
        side=-1,  # Sell
        eta_param=0.2,
        gamma_param=0.1,
        delta_param=0.5,
    )
    assert res_sell.exec_price < 40000.0, "Sell should incur downward price impact"
    assert res_sell.shortfall_bps > 0.0, "Sell execution below arrival must have positive shortfall"

    # 3. Market drift tests (zero impact, purely tape moving)
    drift_up_bars = create_mock_bars(n_bars=10, base_price=41000.0)
    drift_down_bars = create_mock_bars(n_bars=10, base_price=39000.0)

    # Buy on rising tape -> cost
    res_buy_drift_up = simulate_single_order_replay(
        schedule=sched,
        bars_window=drift_up_bars,
        arrival_price=40000.0,
        side=1,
        eta_param=0.0,
        gamma_param=0.0,
        delta_param=0.5,
    )
    assert res_buy_drift_up.shortfall_bps > 0.0

    # Sell on falling tape -> cost
    res_sell_drift_down = simulate_single_order_replay(
        schedule=sched,
        bars_window=drift_down_bars,
        arrival_price=40000.0,
        side=-1,
        eta_param=0.0,
        gamma_param=0.0,
        delta_param=0.5,
    )
    assert res_sell_drift_down.shortfall_bps > 0.0


def test_sample_replay_start_indices() -> None:
    total_bars = 20000
    warmup = 10000
    max_horizon = 240
    indices = sample_replay_start_indices(
        total_bars=total_bars,
        max_horizon_bars=max_horizon,
        n_samples=100,
        warmup_bars=warmup,
        random_seed=42,
    )

    assert len(indices) == 100
    assert (indices >= warmup).all()
    assert (indices + max_horizon < total_bars).all()


def test_monte_carlo_ac_optimality() -> None:
    """MONTE CARLO CORRECTNESS TEST:

    Under its own assumptions (linear impact + GBM), Almgren-Chriss must achieve
    lower mean-variance utility U = E[IS] + lambda * Var(IS) than a suboptimal schedule.
    """
    utils = run_monte_carlo_gbm_validation(
        total_shares=500.0,
        horizon_slices=30,
        risk_aversion=1e-3,
        sigma=0.003,
        eta_linear=0.08,
        gamma_linear=0.02,
        s0=40000.0,
        n_paths=2000,
        random_seed=42,
    )

    ac_utility = utils["AC_utility"]
    inv_utility = utils["Inverted_utility"]

    assert ac_utility < inv_utility, (
        f"AC utility ({ac_utility}) should be strictly lower than Inverted ({inv_utility})"
    )
