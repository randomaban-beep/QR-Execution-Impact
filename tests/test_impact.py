"""Unit tests for impact model calibration, parameter recovery, and decomposition."""

import numpy as np
import pandas as pd

from exec_impact.impact import (
    bootstrap_impact_fit,
    calibrate_impact_pipeline,
    decompose_impact,
    evaluate_out_of_sample_r2,
    fit_impact_model,
)


def generate_synthetic_impact_data(
    n_samples: int = 1500,
    y_true: float = 0.5,
    delta_true: float = 0.5,
    noise_sigma: float = 0.05,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic microstructure data with known ground-truth impact parameters."""
    rng = np.random.default_rng(random_seed)

    trailing_vol = rng.uniform(0.001, 0.003, size=n_samples)
    trailing_volume = rng.uniform(50.0, 150.0, size=n_samples)

    # Normalized signed flow
    q_norm = rng.normal(0.0, 0.3, size=n_samples)
    signed_volume = q_norm * trailing_volume

    sign_q = np.sign(q_norm)
    abs_q = np.abs(q_norm)

    # True normalized return: y = Y * sign(q) * |q|^delta + noise
    noise = rng.normal(0, noise_sigma, size=n_samples)
    y_norm = (y_true * sign_q * (abs_q**delta_true)) + noise
    log_return = y_norm * trailing_vol

    dates = pd.date_range("2024-01-01", periods=n_samples, freq="1min", tz="UTC")
    prices = 40000.0 * np.exp(np.cumsum(log_return))

    return pd.DataFrame(
        {
            "open_time": dates,
            "last_price": prices,
            "volume": np.abs(signed_volume) + 1.0,
            "signed_volume": signed_volume,
            "log_return": log_return,
            "trailing_volatility": trailing_vol,
            "trailing_volume": trailing_volume,
        }
    )


def test_parameter_recovery_synthetic() -> None:
    """PARAMETER RECOVERY TEST:

    Confirm the estimator accurately recovers known Y and delta parameters within tolerance.
    """
    y_true = 0.45
    delta_true = 0.50
    df = generate_synthetic_impact_data(
        n_samples=2500, y_true=y_true, delta_true=delta_true, noise_sigma=0.03
    )

    y_free, delta_free, y_fixed, r2_train = fit_impact_model(df, delta_fixed=0.5)

    # Assert recovery within tight statistical tolerance
    assert np.isclose(y_free, y_true, atol=0.08), f"Recovered Y={y_free} vs true={y_true}"
    assert np.isclose(delta_free, delta_true, atol=0.08), (
        f"Recovered delta={delta_free} vs true={delta_true}"
    )
    assert np.isclose(y_fixed, y_true, atol=0.08), f"Fixed delta=0.5 Y={y_fixed} vs true={y_true}"
    assert r2_train > 0.4, f"Training R2 should be high for low noise: {r2_train}"


def test_bootstrap_ci_coverage() -> None:
    """Confirm bootstrap 95% confidence intervals contain the true parameters."""
    y_true = 0.50
    delta_true = 0.50
    df = generate_synthetic_impact_data(
        n_samples=500, y_true=y_true, delta_true=delta_true, noise_sigma=0.04
    )

    y_lower, y_upper, d_lower, d_upper = bootstrap_impact_fit(df, n_bootstrap=100, random_seed=42)

    assert y_lower <= y_true <= y_upper, f"Y true {y_true} not in CI [{y_lower}, {y_upper}]"
    assert d_lower <= delta_true <= d_upper, (
        f"delta true {delta_true} not in CI [{d_lower}, {d_upper}]"
    )


def test_impact_decomposition_signs() -> None:
    """Verify permanent and temporary impact decomposition properties."""
    df = generate_synthetic_impact_data(n_samples=500, y_true=0.4, delta_true=0.5)
    perm_y, temp_y, lag_impacts = decompose_impact(df, lags=[1, 2, 5])

    assert perm_y >= 0.0, "Permanent impact must be non-negative"
    assert temp_y >= 0.0, "Temporary impact must be non-negative"
    assert len(lag_impacts) == 3


def test_out_of_sample_r2() -> None:
    train_df = generate_synthetic_impact_data(n_samples=1000, y_true=0.5, delta_true=0.5)
    test_df = generate_synthetic_impact_data(
        n_samples=1000, y_true=0.5, delta_true=0.5, random_seed=999
    )

    y_free, delta_free, _, _ = fit_impact_model(train_df)
    r2_oos = evaluate_out_of_sample_r2(train_df, test_df, y_free, delta_free)

    assert r2_oos > 0.0, f"Out of sample R2 should be positive on consistent data: {r2_oos}"


def test_calibrate_pipeline_structure() -> None:
    # 28 days of synthetic bars (4 weeks)
    n_days = 28
    dates = pd.date_range("2024-01-01", periods=n_days * 60, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "open_time": dates,
            "last_price": 40000.0 + np.random.normal(0, 10, len(dates)),
            "volume": np.random.uniform(10, 50, len(dates)),
            "signed_volume": np.random.normal(0, 10, len(dates)),
            "log_return": np.random.normal(0, 0.001, len(dates)),
            "trailing_volatility": 0.002,
            "trailing_volume": 30.0,
        }
    )

    res = calibrate_impact_pipeline(
        df, train_weeks=[1, 2], test_weeks=[3, 4], n_bootstrap=20, random_seed=42
    )

    assert "y_free" in res
    assert "delta_free" in res
    assert "y_ci" in res
    assert "delta_ci" in res
    assert "r2_oos" in res
    assert len(res["weekly_stability"]) > 0
