"""Execution schedule generators: TWAP, prior-7d VWAP, Oracle VWAP, Almgren-Chriss, and numerical optimizer."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class ExecutionSchedule:
    """Represents a discrete execution schedule over N intervals."""

    strategy_name: str
    total_shares: float
    num_slices: int
    slice_sizes: np.ndarray  # Array of length N: shares to execute in each slice n_k
    holdings: np.ndarray  # Array of length N+1: remaining shares x_k, starting at X and ending at 0
    weights: np.ndarray  # Fraction of order executed in each slice (sums to 1.0)


def generate_twap_schedule(
    total_shares: float,
    num_slices: int,
) -> ExecutionSchedule:
    """Generate Time-Weighted Average Price (TWAP) schedule with uniform slices."""
    if num_slices <= 0:
        raise ValueError("num_slices must be positive")
    if total_shares < 0:
        raise ValueError("total_shares must be non-negative")

    slice_size = total_shares / num_slices
    slices = np.full(num_slices, slice_size, dtype=float)
    holdings = np.zeros(num_slices + 1, dtype=float)
    holdings[0] = total_shares
    for k in range(1, num_slices + 1):
        holdings[k] = holdings[k - 1] - slices[k - 1]
    holdings[-1] = 0.0  # Numerical clean-up

    weights = np.full(num_slices, 1.0 / num_slices, dtype=float)

    return ExecutionSchedule(
        strategy_name="TWAP",
        total_shares=total_shares,
        num_slices=num_slices,
        slice_sizes=slices,
        holdings=holdings,
        weights=weights,
    )


def generate_vwap_schedule(
    total_shares: float,
    volume_profile: np.ndarray,
    start_interval_idx: int = 0,
    num_slices: int | None = None,
    is_oracle: bool = False,
) -> ExecutionSchedule:
    """Generate Volume-Weighted Average Price (VWAP) schedule based on volume profile.

    Parameters
    ----------
    total_shares : float
        Total parent order size X.
    volume_profile : np.ndarray
        Array representing intraday volume distribution (e.g. 1440 1-minute bins or subset).
    start_interval_idx : int
        Start bucket index within the day.
    num_slices : int, optional
        Number of intervals to execute over. If None, uses remaining profile length.
    is_oracle : bool
        Whether this is the unattainable realized-volume oracle benchmark.
    """
    if num_slices is None:
        num_slices = len(volume_profile) - start_interval_idx
    if num_slices <= 0:
        raise ValueError("num_slices must be positive")

    profile_slice = volume_profile[start_interval_idx : start_interval_idx + num_slices]
    if len(profile_slice) < num_slices:
        # Wrap around or pad if horizon extends across midnight
        needed = num_slices - len(profile_slice)
        wrap_slice = volume_profile[:needed]
        profile_slice = np.concatenate([profile_slice, wrap_slice])

    # Normalize weights so sum is exactly 1.0
    total_p = np.sum(profile_slice)
    if total_p <= 0:
        weights = np.full(num_slices, 1.0 / num_slices, dtype=float)
    else:
        weights = profile_slice / total_p

    slices = total_shares * weights
    holdings = np.zeros(num_slices + 1, dtype=float)
    holdings[0] = total_shares
    for k in range(1, num_slices + 1):
        holdings[k] = holdings[k - 1] - slices[k - 1]
    holdings[-1] = 0.0

    name = "VWAP_Oracle" if is_oracle else "VWAP"
    return ExecutionSchedule(
        strategy_name=name,
        total_shares=total_shares,
        num_slices=num_slices,
        slice_sizes=slices,
        holdings=holdings,
        weights=weights,
    )


def linearize_temporary_impact(
    total_shares: float,
    num_slices: int,
    eta_power: float,
    delta: float,
    trailing_volume: float,
    volatility: float = 1.0,
    method: str = "secant",
) -> float:
    """Linearize power-law impact around average reference order slice n_bar = X / N.

    Impact model: Delta P / S_0 = sigma * eta_power * (n / V)^delta.
    Methods:
    - 'secant': tilde_eta = sigma * eta_power * n_bar^(delta - 1) / V^delta
    - 'tangent': tilde_eta = delta * sigma * eta_power * n_bar^(delta - 1) / V^delta
    """
    n_bar = max(1e-6, total_shares / num_slices)
    v_norm = max(1e-4, trailing_volume)
    base_factor = volatility * eta_power * (n_bar ** (delta - 1.0)) / (v_norm**delta)

    if method == "tangent":
        return float(delta * base_factor)
    # Default secant
    return float(base_factor)


def generate_almgren_chriss_schedule(
    total_shares: float,
    num_slices: int,
    risk_aversion: float,  # lambda
    volatility: float,  # sigma
    linearized_eta: float,  # tilde_eta
    horizon_minutes: float,  # T
) -> ExecutionSchedule:
    """Generate Almgren-Chriss closed-form optimal trajectory.

    x_k = X * sinh(kappa * (T - t_k)) / sinh(kappa * T)
    where kappa = sqrt(lambda * sigma^2 / tilde_eta).
    """
    if num_slices <= 0:
        raise ValueError("num_slices must be positive")
    if total_shares == 0:
        return generate_twap_schedule(0.0, num_slices)

    t_grid = np.linspace(0.0, horizon_minutes, num_slices + 1)
    t_end = horizon_minutes

    # Urgency parameter kappa
    tilde_eta = max(1e-8, linearized_eta)
    var_factor = max(1e-12, risk_aversion * (volatility**2))
    kappa = np.sqrt(var_factor / tilde_eta)

    # Limiting case: kappa -> 0 yields TWAP linear trajectory
    if kappa * t_end < 1e-4:
        twap = generate_twap_schedule(total_shares, num_slices)
        return ExecutionSchedule(
            strategy_name=f"AlmgrenChriss_lambda_{risk_aversion:.1e}",
            total_shares=total_shares,
            num_slices=num_slices,
            slice_sizes=twap.slice_sizes,
            holdings=twap.holdings,
            weights=twap.weights,
        )

    # Compute continuous holdings x_k
    sinh_total = np.sinh(kappa * t_end)
    holdings = total_shares * np.sinh(kappa * (t_end - t_grid)) / sinh_total
    holdings[0] = total_shares
    holdings[-1] = 0.0

    slices = -np.diff(holdings)
    # Ensure numerical non-negativity
    slices = np.clip(slices, 0.0, None)
    # Re-normalize to exact total_shares
    if np.sum(slices) > 0:
        slices = total_shares * (slices / np.sum(slices))
    weights = slices / total_shares if total_shares > 0 else np.full(num_slices, 1.0 / num_slices)

    # Re-accumulate clean holdings
    holdings = np.zeros(num_slices + 1, dtype=float)
    holdings[0] = total_shares
    for k in range(1, num_slices + 1):
        holdings[k] = holdings[k - 1] - slices[k - 1]
    holdings[-1] = 0.0

    return ExecutionSchedule(
        strategy_name=f"AlmgrenChriss_lambda_{risk_aversion:.1e}",
        total_shares=total_shares,
        num_slices=num_slices,
        slice_sizes=slices,
        holdings=holdings,
        weights=weights,
    )


def optimize_numerical_trajectory(
    total_shares: float,
    num_slices: int,
    risk_aversion: float,
    volatility: float,
    eta_power: float,
    delta: float,
    gamma: float,
    trailing_volume: float,
    interval_minutes: float = 1.0,
) -> ExecutionSchedule:
    """STRETCH GOAL: Numerically optimize execution trajectory under power-law temporary impact.

    Minimizes:
        Cost(n) + RiskPenalty(n)
        = sum_k [ eta * (n_k / V)^delta * n_k + gamma * x_k * n_k ] + lambda * sigma^2 * dt * sum_k x_k^2
    subject to:
        sum(n_k) == total_shares, n_k >= 0.
    """
    if total_shares == 0:
        return generate_twap_schedule(0.0, num_slices)

    v_norm = max(1e-4, trailing_volume)
    dt = interval_minutes

    def objective(n_vec: np.ndarray) -> float:
        x_vec = np.zeros(num_slices + 1)
        x_vec[0] = total_shares
        for k in range(num_slices):
            x_vec[k + 1] = x_vec[k] - n_vec[k]

        # Power law temporary impact cost
        temp_cost = np.sum(eta_power * ((n_vec / v_norm) ** delta) * n_vec)
        # Permanent impact cost
        perm_cost = np.sum(gamma * x_vec[:-1] * n_vec)
        # Risk variance penalty
        risk_penalty = risk_aversion * (volatility**2) * dt * np.sum(x_vec[:-1] ** 2)

        return float(temp_cost + perm_cost + risk_penalty)

    init_guess = np.full(num_slices, total_shares / num_slices)
    bounds = [(0.0, total_shares) for _ in range(num_slices)]
    constraints = [{"type": "eq", "fun": lambda n_vec: np.sum(n_vec) - total_shares}]

    res = minimize(
        objective,
        init_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-6},
    )

    slices = np.clip(res.x, 0.0, None)
    slices = total_shares * (slices / np.sum(slices))
    weights = slices / total_shares

    holdings = np.zeros(num_slices + 1, dtype=float)
    holdings[0] = total_shares
    for k in range(1, num_slices + 1):
        holdings[k] = holdings[k - 1] - slices[k - 1]
    holdings[-1] = 0.0

    return ExecutionSchedule(
        strategy_name="NumericalOptimal",
        total_shares=total_shares,
        num_slices=num_slices,
        slice_sizes=slices,
        holdings=holdings,
        weights=weights,
    )
