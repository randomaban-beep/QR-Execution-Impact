"""Unit tests for execution schedules: properties, invariants, and limiting behavior."""

import numpy as np

from exec_impact.schedules import (
    generate_almgren_chriss_schedule,
    generate_twap_schedule,
    generate_vwap_schedule,
    linearize_temporary_impact,
    optimize_numerical_trajectory,
)


def test_schedule_sum_and_nonnegativity() -> None:
    X = 100.0
    N = 30

    twap = generate_twap_schedule(X, N)
    assert np.isclose(np.sum(twap.slice_sizes), X, atol=1e-10)
    assert (twap.slice_sizes >= 0).all()
    assert np.isclose(twap.holdings[0], X)
    assert np.isclose(twap.holdings[-1], 0.0)

    # VWAP
    dummy_profile = np.random.uniform(1.0, 5.0, size=1440)
    vwap = generate_vwap_schedule(X, dummy_profile, start_interval_idx=100, num_slices=N)
    assert np.isclose(np.sum(vwap.slice_sizes), X, atol=1e-10)
    assert (vwap.slice_sizes >= 0).all()
    assert np.isclose(vwap.holdings[-1], 0.0)

    # Almgren-Chriss
    for lmbda in [1e-8, 1e-5, 1e-3, 1e-1]:
        ac = generate_almgren_chriss_schedule(
            total_shares=X,
            num_slices=N,
            risk_aversion=lmbda,
            volatility=0.002,
            linearized_eta=0.05,
            horizon_minutes=30.0,
        )
        assert np.isclose(np.sum(ac.slice_sizes), X, atol=1e-10)
        assert (ac.slice_sizes >= 0).all()
        assert np.isclose(ac.holdings[-1], 0.0)


def test_schedule_monotone_holdings() -> None:
    X = 50.0
    N = 60

    for lmbda in [1e-8, 1e-4, 1e-2]:
        ac = generate_almgren_chriss_schedule(
            total_shares=X,
            num_slices=N,
            risk_aversion=lmbda,
            volatility=0.001,
            linearized_eta=0.01,
            horizon_minutes=60.0,
        )
        # Holdings must be non-increasing
        diffs = np.diff(ac.holdings)
        assert (diffs <= 1e-12).all(), f"Holdings not monotonically decreasing for lambda={lmbda}"


def test_ac_lambda_limits() -> None:
    """TEST AC LIMITING BEHAVIOR:

    1. lambda -> 0 matches TWAP.
    2. Increasing lambda strictly increases front-loading (larger first slice).
    """
    X = 100.0
    N = 30
    T = 30.0

    twap = generate_twap_schedule(X, N)

    ac_low = generate_almgren_chriss_schedule(
        total_shares=X,
        num_slices=N,
        risk_aversion=1e-12,
        volatility=0.001,
        linearized_eta=0.05,
        horizon_minutes=T,
    )
    # AC near-zero lambda should match TWAP within float precision
    assert np.allclose(ac_low.slice_sizes, twap.slice_sizes, atol=1e-3)

    # Test front-loading progression across lambda sweep
    lambdas = [1e-8, 1e-6, 1e-4, 1e-2, 1.0]
    first_slices = []
    for lmbda in lambdas:
        ac = generate_almgren_chriss_schedule(
            total_shares=X,
            num_slices=N,
            risk_aversion=lmbda,
            volatility=0.002,
            linearized_eta=0.05,
            horizon_minutes=T,
        )
        first_slices.append(ac.slice_sizes[0])

    # First slice must be non-decreasing as risk aversion increases
    for i in range(len(first_slices) - 1):
        assert first_slices[i + 1] >= first_slices[i] - 1e-6, (
            f"Front-loading violation: lambda {lambdas[i + 1]} gave {first_slices[i + 1]} "
            f"which is less than {first_slices[i]}"
        )


def test_linearize_temporary_impact() -> None:
    tilde_secant = linearize_temporary_impact(
        total_shares=100.0,
        num_slices=10,
        eta_power=0.5,
        delta=0.5,
        trailing_volume=1000.0,
        method="secant",
    )
    tilde_tangent = linearize_temporary_impact(
        total_shares=100.0,
        num_slices=10,
        eta_power=0.5,
        delta=0.5,
        trailing_volume=1000.0,
        method="tangent",
    )
    assert tilde_secant > 0
    assert tilde_tangent > 0
    # For delta = 0.5, tangent is exactly 0.5 * secant
    assert np.isclose(tilde_tangent, 0.5 * tilde_secant)


def test_numerical_trajectory_optimization() -> None:
    """STRETCH GOAL TEST:

    Verify numerical optimizer under power-law impact converges and satisfies constraints.
    """
    X = 100.0
    N = 15
    sched = optimize_numerical_trajectory(
        total_shares=X,
        num_slices=N,
        risk_aversion=1e-4,
        volatility=0.002,
        eta_power=0.5,
        delta=0.5,
        gamma=0.01,
        trailing_volume=1000.0,
        interval_minutes=1.0,
    )

    assert np.isclose(np.sum(sched.slice_sizes), X, atol=1e-5)
    assert (sched.slice_sizes >= -1e-6).all()
    assert np.isclose(sched.holdings[-1], 0.0, atol=1e-5)
