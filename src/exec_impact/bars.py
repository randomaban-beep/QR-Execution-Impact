"""Bar aggregation and causal microstructure feature engineering."""

from datetime import timedelta

import numpy as np
import pandas as pd


def aggregate_trades_to_bars(
    trades_df: pd.DataFrame,
    interval_minutes: int = 1,
) -> pd.DataFrame:
    """Aggregate raw trades into uniform OHLCV bars with signed volume and VWAP.

    Parameters
    ----------
    trades_df : pd.DataFrame
        DataFrame with columns: ['timestamp', 'price', 'quantity', 'signed_quantity', 'aggressor_side']
    interval_minutes : int
        Bar interval width in minutes (e.g. 1, 5, 15).

    Returns
    -------
    pd.DataFrame
        Continuous time-indexed bars without gaps.
    """
    if trades_df.empty:
        return pd.DataFrame(
            columns=[
                "open_time",
                "close_time",
                "open_price",
                "high_price",
                "low_price",
                "last_price",
                "vwap",
                "volume",
                "signed_volume",
                "trade_count",
            ]
        )

    df = trades_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Floor timestamps to interval
    rule = f"{interval_minutes}min"
    df["bucket"] = df["timestamp"].dt.floor(rule)
    df["dollar_volume"] = df["price"] * df["quantity"]

    # Group by bucket
    grouped = df.groupby("bucket")
    bars = grouped.agg(
        open_price=("price", "first"),
        high_price=("price", "max"),
        low_price=("price", "min"),
        last_price=("price", "last"),
        volume=("quantity", "sum"),
        signed_volume=("signed_quantity", "sum"),
        dollar_volume=("dollar_volume", "sum"),
        trade_count=("quantity", "count"),
    ).reset_index()

    bars["vwap"] = np.where(
        bars["volume"] > 0,
        bars["dollar_volume"] / bars["volume"],
        bars["last_price"],
    )
    bars = bars.drop(columns=["dollar_volume"])

    # Reindex to full uniform timeline between min and max bucket to ensure continuous series
    start_time = bars["bucket"].min()
    end_time = bars["bucket"].max()
    full_index = pd.date_range(start_time, end_time, freq=rule, tz="UTC", name="bucket")

    bars = bars.set_index("bucket").reindex(full_index)

    # Fill empty intervals: forward-fill price, zero-fill volumes
    bars["last_price"] = bars["last_price"].ffill().bfill()
    bars["open_price"] = bars["open_price"].fillna(bars["last_price"])
    bars["high_price"] = bars["high_price"].fillna(bars["last_price"])
    bars["low_price"] = bars["low_price"].fillna(bars["last_price"])
    bars["vwap"] = bars["vwap"].fillna(bars["last_price"])
    bars["volume"] = bars["volume"].fillna(0.0)
    bars["signed_volume"] = bars["signed_volume"].fillna(0.0)
    bars["trade_count"] = bars["trade_count"].fillna(0).astype(int)

    bars = bars.reset_index()
    bars = bars.rename(columns={"bucket": "open_time"})
    delta = timedelta(minutes=interval_minutes)
    bars["close_time"] = bars["open_time"] + delta

    return bars


def compute_trailing_features(
    bars_df: pd.DataFrame,
    vol_window_bars: int = 1440,
    min_vol_periods: int = 60,
    volume_window_bars: int = 1440,
) -> pd.DataFrame:
    """Compute strictly prior (causal, no-lookahead) realized volatility and trailing volume.

    Strict No-Lookahead Guarantee:
    - Return r_t is ln(last_price_t / last_price_{t-1}).
    - trailing_volatility_t and trailing_volume_t are shifted by 1 so that bar t's own return
      and volume are EXCLUDED from the window.
    """
    df = bars_df.copy()
    if df.empty:
        df["log_return"] = pd.Series(dtype=float)
        df["trailing_volatility"] = pd.Series(dtype=float)
        df["trailing_volume"] = pd.Series(dtype=float)
        return df

    # Log return over the bar
    df["log_return"] = np.log(df["last_price"] / df["last_price"].shift(1)).fillna(0.0)

    # Trailing standard deviation of returns shifted by 1 (prior data strictly)
    # ddof=1 for sample standard deviation
    rolling_vol = (
        df["log_return"]
        .shift(1)
        .rolling(window=vol_window_bars, min_periods=min_vol_periods)
        .std(ddof=1)
    )
    # Fallback to expanding or non-zero min value to prevent division by zero
    min_observed_vol = rolling_vol[rolling_vol > 0].min()
    fallback_vol = 1e-4 if pd.isna(min_observed_vol) else float(min_observed_vol)
    df["trailing_volatility"] = rolling_vol.bfill().fillna(fallback_vol)
    # Ensure strictly positive
    df["trailing_volatility"] = df["trailing_volatility"].clip(lower=1e-6)

    # Trailing mean volume per interval shifted by 1 (prior data strictly)
    rolling_vol_mean = (
        df["volume"].shift(1).rolling(window=volume_window_bars, min_periods=min_vol_periods).mean()
    )
    min_obs_vol = rolling_vol_mean[rolling_vol_mean > 0].min()
    fallback_v = 1.0 if pd.isna(min_obs_vol) else float(min_obs_vol)
    df["trailing_volume"] = rolling_vol_mean.bfill().fillna(fallback_v)
    df["trailing_volume"] = df["trailing_volume"].clip(lower=1e-4)

    return df


def compute_prior_volume_profiles(
    bars_df: pd.DataFrame,
    interval_minutes: int = 1,
    lookback_days: int = 7,
) -> dict[pd.Timestamp, np.ndarray]:
    """Compute empirical intraday volume profile for each day strictly using prior lookback_days.

    For any date D, the profile is:
    profile_k = sum_{d in [D - lookback_days, D - 1]} volume_{d, k} / total_volume_{[D - lookback_days, D - 1]}
    sum(profile) == 1.0.

    Returns
    -------
    dict[pd.Timestamp, np.ndarray]
        Mapping from date (pd.Timestamp at midnight UTC) to normalized profile vector of length (1440 // interval_minutes).
    """
    df = bars_df.copy()
    if df.empty:
        return {}

    bars_per_day = 1440 // interval_minutes
    df["date"] = df["open_time"].dt.floor("1D")
    df["minute_of_day"] = (
        df["open_time"].dt.hour * 60 + df["open_time"].dt.minute
    ) // interval_minutes

    unique_dates = sorted(df["date"].unique())
    daily_matrix: dict[pd.Timestamp, np.ndarray] = {}

    for d in unique_dates:
        day_sub = df[df["date"] == d]
        vec = np.zeros(bars_per_day, dtype=float)
        for _, row in day_sub.iterrows():
            idx = int(row["minute_of_day"])
            if 0 <= idx < bars_per_day:
                vec[idx] = float(row["volume"])
        daily_matrix[d] = vec

    prior_profiles: dict[pd.Timestamp, np.ndarray] = {}
    uniform_profile = np.ones(bars_per_day, dtype=float) / bars_per_day

    for i, d in enumerate(unique_dates):
        # Look back strictly at preceding days: [max(0, i - lookback_days) : i]
        prior_indices = list(range(max(0, i - lookback_days), i))
        if not prior_indices:
            # No prior days available (e.g. first day in dataset) -> fallback to uniform
            prior_profiles[d] = uniform_profile.copy()
            continue

        accumulated = np.zeros(bars_per_day, dtype=float)
        for idx in prior_indices:
            prior_date = unique_dates[idx]
            accumulated += daily_matrix[prior_date]

        total = np.sum(accumulated)
        if total > 0:
            prior_profiles[d] = accumulated / total
        else:
            prior_profiles[d] = uniform_profile.copy()

    return prior_profiles
