"""Market impact modeling, robust calibration, decomposition, and stability analysis."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.linear_model import HuberRegressor


@dataclass(frozen=True)
class ImpactFitResult:
    """Results of market impact model calibration."""

    y_param: float
    delta_param: float
    y_fixed_delta: float  # Fitted Y when delta is fixed to 0.5
    y_ci_lower: float
    y_ci_upper: float
    delta_ci_lower: float
    delta_ci_upper: float
    r2_train: float
    sample_size: int
    perm_y: float
    temp_y: float


def fit_impact_model(
    df: pd.DataFrame,
    delta_fixed: float = 0.5,
    huber_epsilon: float = 1.35,
    min_volume: float = 1e-4,
) -> tuple[float, float, float, float]:
    """Fit normalized return regression: r / sigma = Y * sign(Q) * (|Q| / V)^delta.

    Returns
    -------
    tuple:
        (Y_free, delta_free, Y_fixed, r2_train)
    """
    valid = df.dropna(subset=["log_return", "trailing_volatility", "trailing_volume"]).copy()
    valid = valid[
        (valid["trailing_volatility"] > 1e-6)
        & (valid["trailing_volume"] > min_volume)
        & (valid["volume"] > 0)
    ]

    if len(valid) < 30:
        return 0.1, delta_fixed, 0.1, 0.0

    y_norm = (valid["log_return"] / valid["trailing_volatility"]).values
    # Signed flow normalized by trailing volume
    q_norm = (valid["signed_volume"] / valid["trailing_volume"]).values
    sign_q = np.sign(q_norm)
    abs_q = np.abs(q_norm)

    # Filter out exact zero flow
    mask = abs_q > 1e-8
    if np.sum(mask) < 30:
        return 0.1, delta_fixed, 0.1, 0.0

    y_norm = y_norm[mask]
    sign_q = sign_q[mask]
    abs_q = abs_q[mask]

    # 1. Fit fixed delta=0.5 using HuberRegressor
    x_fixed = (sign_q * (abs_q**delta_fixed)).reshape(-1, 1)
    huber_fixed = HuberRegressor(epsilon=huber_epsilon, fit_intercept=False)
    huber_fixed.fit(x_fixed, y_norm)
    y_fixed = max(1e-4, float(huber_fixed.coef_[0]))

    # 2. Fit free delta and Y using least_squares with Huber loss
    # Model: y_pred = Y * sign_q * (abs_q ** delta)
    def residuals(params: np.ndarray) -> np.ndarray:
        y_val, delta_val = params
        y_pred = y_val * sign_q * (abs_q**delta_val)
        return y_norm - y_pred

    # Initial guess and bounds: Y in [1e-4, 50.0], delta in [0.05, 1.5]
    init_params = np.array([y_fixed, 0.5])
    bounds = ([1e-4, 0.05], [50.0, 1.5])

    res = least_squares(
        residuals,
        init_params,
        bounds=bounds,
        loss="huber",
        f_scale=huber_epsilon,
        max_nfev=500,
    )
    y_free, delta_free = float(res.x[0]), float(res.x[1])

    # Compute R2 on training set for the free fit
    y_pred = y_free * sign_q * (abs_q**delta_free)
    ss_res = np.sum((y_norm - y_pred) ** 2)
    ss_tot = np.sum((y_norm - np.mean(y_norm)) ** 2)
    r2_train = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return y_free, delta_free, y_fixed, float(r2_train)


def bootstrap_impact_fit(
    df: pd.DataFrame,
    n_bootstrap: int = 500,
    delta_fixed: float = 0.5,
    random_seed: int = 42,
) -> tuple[float, float, float, float]:
    """Perform bootstrap resampling to estimate 95% confidence intervals on Y and delta.

    Returns
    -------
    tuple:
        (y_ci_lower, y_ci_upper, delta_ci_lower, delta_ci_upper)
    """
    rng = np.random.default_rng(random_seed)
    n = len(df)
    if n < 30:
        return 0.0, 1.0, 0.0, 1.0

    y_samples = []
    delta_samples = []

    for _ in range(n_bootstrap):
        sample_indices = rng.choice(n, size=n, replace=True)
        sub_df = df.iloc[sample_indices]
        y_f, delta_f, _, _ = fit_impact_model(sub_df, delta_fixed=delta_fixed)
        y_samples.append(y_f)
        delta_samples.append(delta_f)

    y_ci_lower, y_ci_upper = (
        float(np.percentile(y_samples, 2.5)),
        float(np.percentile(y_samples, 97.5)),
    )
    delta_ci_lower, delta_ci_upper = (
        float(np.percentile(delta_samples, 2.5)),
        float(np.percentile(delta_samples, 97.5)),
    )

    return y_ci_lower, y_ci_upper, delta_ci_lower, delta_ci_upper


def decompose_impact(
    df: pd.DataFrame,
    lags: list[int] | None = None,
    delta_fixed: float = 0.5,
) -> tuple[float, float, dict[int, float]]:
    """Decompose market impact into permanent vs temporary components across forward lags.

    Permanent impact: price response persisting k intervals after flow.
    Temporary impact: peak impact minus permanent impact.
    """
    if lags is None:
        lags = [1, 2, 5, 10, 15]

    valid = df.copy()
    peak_y, _, _, _ = fit_impact_model(valid, delta_fixed=delta_fixed)

    lag_impacts: dict[int, float] = {}
    for k in lags:
        # Forward return from close of interval tau to tau + k
        shifted_price = valid["last_price"].shift(-k)
        forward_return = np.log(shifted_price / valid["last_price"]).fillna(0.0)

        temp_df = valid.copy()
        temp_df["log_return"] = forward_return
        y_k, _, _, _ = fit_impact_model(temp_df, delta_fixed=delta_fixed)
        lag_impacts[k] = float(max(0.0, y_k))

    # Longest lag represents permanent impact estimate
    max_lag = max(lags)
    perm_y = lag_impacts.get(max_lag, peak_y * 0.4)
    # Temporary is remainder: peak - permanent
    temp_y = float(max(1e-4, peak_y - perm_y))

    return perm_y, temp_y, lag_impacts


def evaluate_out_of_sample_r2(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_param: float,
    delta_param: float,
) -> float:
    """Evaluate predictive R2 of fitted impact parameters on out-of-sample data."""
    test_valid = test_df.dropna(
        subset=["log_return", "trailing_volatility", "trailing_volume"]
    ).copy()
    test_valid = test_valid[
        (test_valid["trailing_volatility"] > 1e-6)
        & (test_valid["trailing_volume"] > 1e-4)
        & (test_valid["volume"] > 0)
    ]

    if test_valid.empty:
        return 0.0

    y_norm = (test_valid["log_return"] / test_valid["trailing_volatility"]).values
    q_norm = (test_valid["signed_volume"] / test_valid["trailing_volume"]).values
    sign_q = np.sign(q_norm)
    abs_q = np.abs(q_norm)

    mask = abs_q > 1e-8
    if np.sum(mask) < 10:
        return 0.0

    y_actual = y_norm[mask]
    y_pred = y_param * sign_q[mask] * (abs_q[mask] ** delta_param)

    train_mean = (train_df["log_return"] / train_df["trailing_volatility"]).mean()
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - train_mean) ** 2)

    return float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0


def calibrate_impact_pipeline(
    bars_df: pd.DataFrame,
    train_weeks: list[int] | None = None,
    test_weeks: list[int] | None = None,
    n_bootstrap: int = 500,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Execute complete calibration pipeline: fit, bootstrap CI, decomposition, and OOS R2."""
    if train_weeks is None:
        train_weeks = [1, 2]
    if test_weeks is None:
        test_weeks = [3, 4]

    df = bars_df.copy()
    # Add week number based on days
    start_date = df["open_time"].min().floor("1D")
    df["day_idx"] = (df["open_time"] - start_date).dt.days
    df["week_idx"] = (df["day_idx"] // 7) + 1

    train_mask = df["week_idx"].isin(train_weeks)
    test_mask = df["week_idx"].isin(test_weeks)

    train_df = df[train_mask]
    test_df = df[test_mask] if test_mask.any() else df[~train_mask]

    y_free, delta_free, y_fixed, r2_train = fit_impact_model(train_df)
    y_ci_lower, y_ci_upper, delta_ci_lower, delta_ci_upper = bootstrap_impact_fit(
        train_df, n_bootstrap=n_bootstrap, random_seed=random_seed
    )
    perm_y, temp_y, lag_impacts = decompose_impact(train_df)
    r2_oos = evaluate_out_of_sample_r2(train_df, test_df, y_free, delta_free)

    # Weekly stability breakdown
    weekly_stats: list[dict[str, Any]] = []
    for week_num in sorted(df["week_idx"].unique()):
        w_df = df[df["week_idx"] == week_num]
        if len(w_df) >= 30:
            w_y, w_delta, w_fixed, w_r2 = fit_impact_model(w_df)
            weekly_stats.append(
                {
                    "week": int(week_num),
                    "y_free": w_y,
                    "delta_free": w_delta,
                    "y_fixed": w_fixed,
                    "r2": w_r2,
                    "n_bars": len(w_df),
                }
            )

    return {
        "y_free": y_free,
        "delta_free": delta_free,
        "y_fixed": y_fixed,
        "y_ci": [y_ci_lower, y_ci_upper],
        "delta_ci": [delta_ci_lower, delta_ci_upper],
        "r2_train": r2_train,
        "r2_oos": r2_oos,
        "perm_y": perm_y,
        "temp_y": temp_y,
        "lag_impacts": lag_impacts,
        "weekly_stability": weekly_stats,
    }
