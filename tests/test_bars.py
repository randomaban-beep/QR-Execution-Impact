"""Unit tests for bar construction, microstructure features, and anti-lookahead guarantees."""

import numpy as np
import pandas as pd

from exec_impact.bars import (
    aggregate_trades_to_bars,
    compute_prior_volume_profiles,
    compute_trailing_features,
)


def create_synthetic_trades(n_trades: int = 1000) -> pd.DataFrame:
    """Generate synthetic trades over multiple hours for deterministic testing."""
    start_time = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    # Generates trades roughly every 10 seconds
    timestamps = [start_time + pd.Timedelta(seconds=i * 10) for i in range(n_trades)]
    prices = 40000.0 + np.cumsum(np.random.normal(0, 5, n_trades))
    quantities = np.random.exponential(scale=0.5, size=n_trades) + 0.01
    aggressors = np.random.choice([-1, 1], size=n_trades)
    signed_quantities = quantities * aggressors

    return pd.DataFrame(
        {
            "agg_trade_id": np.arange(n_trades),
            "price": prices,
            "quantity": quantities,
            "aggressor_side": aggressors,
            "signed_quantity": signed_quantities,
            "timestamp": timestamps,
        }
    )


def test_bar_aggregation_quantities() -> None:
    trades = create_synthetic_trades(n_trades=600)  # ~100 minutes
    bars_1m = aggregate_trades_to_bars(trades, interval_minutes=1)

    assert not bars_1m.empty
    assert "last_price" in bars_1m.columns
    assert "signed_volume" in bars_1m.columns
    assert "vwap" in bars_1m.columns

    # Volume conservation
    assert np.isclose(bars_1m["volume"].sum(), trades["quantity"].sum(), rtol=1e-5)
    # Signed volume conservation
    assert np.isclose(bars_1m["signed_volume"].sum(), trades["signed_quantity"].sum(), rtol=1e-5)
    # High >= Low
    assert (bars_1m["high_price"] >= bars_1m["low_price"]).all()
    # VWAP between Low and High for non-empty bars
    non_empty = bars_1m[bars_1m["trade_count"] > 0]
    assert (non_empty["vwap"] >= non_empty["low_price"] - 1e-6).all()
    assert (non_empty["vwap"] <= non_empty["high_price"] + 1e-6).all()


def test_empty_intervals_handling() -> None:
    # Two trades separated by a gap of 10 minutes
    t1 = pd.Timestamp("2024-01-01 00:00:10", tz="UTC")
    t2 = pd.Timestamp("2024-01-01 00:10:15", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": [t1, t2],
            "price": [42000.0, 42100.0],
            "quantity": [1.0, 2.0],
            "signed_quantity": [1.0, 2.0],
            "aggressor_side": [1, 1],
        }
    )

    bars = aggregate_trades_to_bars(df, interval_minutes=1)
    # Should have 11 one-minute bars (0 to 10)
    assert len(bars) == 11

    # Empty bars between minute 1 and 9 should have volume 0 and last_price forward-filled
    gap_bars = bars.iloc[1:10]
    assert (gap_bars["volume"] == 0.0).all()
    assert (gap_bars["last_price"] == 42000.0).all()
    assert (gap_bars["trade_count"] == 0).all()


def test_trailing_volatility_no_lookahead() -> None:
    """STRICT ANTI-LOOKAHEAD TEST:

    Changing future price at t+1 or t+5 MUST NOT alter trailing volatility or volume at t.
    """
    trades = create_synthetic_trades(n_trades=1000)
    bars_orig = aggregate_trades_to_bars(trades, interval_minutes=1)
    features_orig = compute_trailing_features(
        bars_orig, vol_window_bars=60, volume_window_bars=60, min_vol_periods=10
    )

    # Pick a test index t in the middle of the sample
    t = 40
    orig_vol_at_t = features_orig.loc[t, "trailing_volatility"]
    orig_volume_at_t = features_orig.loc[t, "trailing_volume"]

    # Perturb future bars (t+1, t+2, ..., end) dramatically
    bars_perturbed = bars_orig.copy()
    bars_perturbed.loc[t + 1 :, "last_price"] = bars_perturbed.loc[t + 1 :, "last_price"] * 10.0
    bars_perturbed.loc[t + 1 :, "volume"] = bars_perturbed.loc[t + 1 :, "volume"] * 100.0

    features_perturbed = compute_trailing_features(
        bars_perturbed, vol_window_bars=60, volume_window_bars=60, min_vol_periods=10
    )
    perturbed_vol_at_t = features_perturbed.loc[t, "trailing_volatility"]
    perturbed_volume_at_t = features_perturbed.loc[t, "trailing_volume"]

    # Trailing volatility and volume at t must be BITWISE identical
    assert np.isclose(orig_vol_at_t, perturbed_vol_at_t, atol=0.0, rtol=1e-12), (
        f"Lookahead violation! Volatility changed: {orig_vol_at_t} vs {perturbed_vol_at_t}"
    )
    assert np.isclose(orig_volume_at_t, perturbed_volume_at_t, atol=0.0, rtol=1e-12), (
        f"Lookahead violation! Volume changed: {orig_volume_at_t} vs {perturbed_volume_at_t}"
    )


def test_volume_profile_no_lookahead() -> None:
    """STRICT ANTI-LOOKAHEAD TEST FOR VOLUME PROFILES:

    Modifying trade volume on Day 8 or Day 9 MUST have ZERO effect on Day 8's prior profile.
    """
    # Create 10 days of synthetic bars
    days = 10
    dates = pd.date_range("2024-01-01", periods=days * 1440, freq="1min", tz="UTC")
    volumes = np.random.uniform(1.0, 10.0, size=len(dates))
    bars = pd.DataFrame(
        {
            "open_time": dates,
            "volume": volumes,
            "last_price": 40000.0,
            "high_price": 40000.0,
            "low_price": 40000.0,
            "open_price": 40000.0,
            "vwap": 40000.0,
            "signed_volume": 0.0,
            "trade_count": 1,
        }
    )

    profiles_orig = compute_prior_volume_profiles(bars, interval_minutes=1, lookback_days=7)
    day8_ts = pd.Timestamp("2024-01-08 00:00:00", tz="UTC")
    day8_profile_orig = profiles_orig[day8_ts].copy()

    # Assert profile sums to 1.0
    assert np.isclose(np.sum(day8_profile_orig), 1.0, rtol=1e-6)

    # Perturb Day 8 and Day 9 volumes heavily
    bars_perturbed = bars.copy()
    mask_day8_and_later = bars_perturbed["open_time"] >= day8_ts
    bars_perturbed.loc[mask_day8_and_later, "volume"] *= 1000.0

    profiles_perturbed = compute_prior_volume_profiles(
        bars_perturbed, interval_minutes=1, lookback_days=7
    )
    day8_profile_perturbed = profiles_perturbed[day8_ts]

    # Day 8 profile must be BITWISE identical
    assert np.allclose(day8_profile_orig, day8_profile_perturbed, atol=0.0, rtol=1e-12), (
        "Lookahead violation! Day 8 profile changed after modifying Day 8+ data."
    )
