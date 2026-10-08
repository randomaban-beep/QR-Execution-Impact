"""Execution simulation engine: tape replay across paired start times and synthetic Monte Carlo."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from exec_impact.schedules import (
    ExecutionSchedule,
    generate_almgren_chriss_schedule,
    generate_twap_schedule,
    generate_vwap_schedule,
    linearize_temporary_impact,
)


@dataclass(frozen=True)
class OrderResult:
    """Detailed result of simulating a single order execution."""

    strategy_name: str
    symbol: str
    side: int  # +1 Buy, -1 Sell
    start_time: pd.Timestamp
    horizon_minutes: int
    order_size: float
    order_size_adv_pct: float
    arrival_price: float
    exec_price: float
    shortfall_bps: float
    temp_impact_bps: float
    perm_impact_bps: float
    tape_slippage_bps: float


def simulate_single_order_replay(
    schedule: ExecutionSchedule,
    bars_window: pd.DataFrame,
    arrival_price: float,
    side: int,  # +1 Buy, -1 Sell
    eta_param: float,
    gamma_param: float,
    delta_param: float,
    symbol: str = "BTCUSDT",
    adv_pct: float = 0.05,
) -> OrderResult:
    """Replay execution schedule against historical tape bars.

    Fill price for slice k:
        P_k = tape_vwap_k + Delta_temp_k + Delta_perm_k
    where:
        Delta_temp_k = S_0 * eta * (n_k / V_k)^delta * side
        Delta_perm_k = S_0 * gamma * [ sum_{j < k} (n_j / V_j)^delta + 0.5 * (n_k / V_k)^delta ] * side
    """
    N = schedule.num_slices
    if len(bars_window) < N:
        raise ValueError(f"bars_window has {len(bars_window)} rows, expected at least {N}")

    slices = schedule.slice_sizes
    tape_vwaps = bars_window["vwap"].values[:N]
    trailing_volumes = bars_window["trailing_volume"].values[:N]

    # Modeled temporary and permanent impact per slice
    # Relative flow per slice: (n_k / V_k)
    q_ratio = slices / np.maximum(trailing_volumes, 1e-4)
    flow_power = q_ratio**delta_param

    # Temporary impact per slice
    temp_impacts = arrival_price * eta_param * flow_power * side

    # Accumulated permanent impact
    cum_prev = np.zeros(N)
    if N > 1:
        cum_prev[1:] = np.cumsum(flow_power[:-1])
    perm_impacts = arrival_price * gamma_param * (cum_prev + 0.5 * flow_power) * side

    fill_prices = tape_vwaps + temp_impacts + perm_impacts

    total_shares = schedule.total_shares
    if total_shares > 0:
        exec_price = float(np.sum(slices * fill_prices) / total_shares)
        tape_price = float(np.sum(slices * tape_vwaps) / total_shares)
        total_temp = float(np.sum(slices * temp_impacts) / total_shares)
        total_perm = float(np.sum(slices * perm_impacts) / total_shares)
    else:
        exec_price = arrival_price
        tape_price = arrival_price
        total_temp = 0.0
        total_perm = 0.0

    # Implementation shortfall in bps (sign-adjusted: positive = cost)
    shortfall_bps = side * ((exec_price - arrival_price) / arrival_price) * 10_000.0
    tape_slippage_bps = side * ((tape_price - arrival_price) / arrival_price) * 10_000.0
    temp_impact_bps = side * (total_temp / arrival_price) * 10_000.0
    perm_impact_bps = side * (total_perm / arrival_price) * 10_000.0

    start_time = bars_window["open_time"].iloc[0]

    return OrderResult(
        strategy_name=schedule.strategy_name,
        symbol=symbol,
        side=side,
        start_time=start_time,
        horizon_minutes=N,
        order_size=total_shares,
        order_size_adv_pct=adv_pct,
        arrival_price=arrival_price,
        exec_price=exec_price,
        shortfall_bps=shortfall_bps,
        temp_impact_bps=temp_impact_bps,
        perm_impact_bps=perm_impact_bps,
        tape_slippage_bps=tape_slippage_bps,
    )


def sample_replay_start_indices(
    total_bars: int,
    max_horizon_bars: int,
    n_samples: int = 500,
    warmup_bars: int = 1440 * 7,  # 7 days warmup
    random_seed: int = 42,
) -> np.ndarray:
    """Sample valid random starting bar indices strictly after warmup period."""
    rng = np.random.default_rng(random_seed)
    max_start = total_bars - max_horizon_bars - 1
    if max_start <= warmup_bars:
        raise ValueError(
            f"Insufficient data: total_bars={total_bars}, warmup={warmup_bars}, "
            f"max_horizon={max_horizon_bars}"
        )

    return rng.choice(np.arange(warmup_bars, max_start), size=n_samples, replace=False)


def run_paired_replay_simulation(
    bars_1m: pd.DataFrame,
    prior_volume_profiles: dict[pd.Timestamp, np.ndarray],
    impact_params: dict[str, Any],
    symbol: str,
    horizons_minutes: list[int],
    order_sizes_adv_pct: list[float],
    lambdas: list[float],
    n_samples: int = 500,
    random_seed: int = 42,
) -> list[OrderResult]:
    """Execute paired tape replay simulation across all grid dimensions."""
    max_horizon = max(horizons_minutes)
    start_indices = sample_replay_start_indices(
        total_bars=len(bars_1m),
        max_horizon_bars=max_horizon,
        n_samples=n_samples,
        random_seed=random_seed,
    )

    eta = impact_params.get("temp_y", 0.3)
    gamma = impact_params.get("perm_y", 0.1)
    delta = impact_params.get("delta_free", 0.5)

    results: list[OrderResult] = []

    for start_idx in start_indices:
        # Arrival price: last price of bar strictly preceding start
        arrival_price = float(bars_1m.loc[start_idx - 1, "last_price"])
        start_row = bars_1m.iloc[start_idx]
        start_time = start_row["open_time"]
        sigma = float(start_row["trailing_volatility"])
        trailing_v = float(start_row["trailing_volume"])
        # Daily volume estimate (1440 1m bars)
        daily_adv = trailing_v * 1440.0

        day_date = start_time.floor("1D")
        volume_profile = prior_volume_profiles.get(day_date, np.ones(1440) / 1440.0)
        start_bucket = (start_time.hour * 60 + start_time.minute) % 1440

        for horizon in horizons_minutes:
            bars_window = bars_1m.iloc[start_idx : start_idx + horizon]
            realized_window_volume = bars_window["volume"].values
            realized_profile = (
                realized_window_volume / np.sum(realized_window_volume)
                if np.sum(realized_window_volume) > 0
                else np.ones(horizon) / horizon
            )

            for size_pct in order_sizes_adv_pct:
                total_shares = daily_adv * size_pct

                # Precompute linearized temporary impact for AC
                tilde_eta = linearize_temporary_impact(
                    total_shares=total_shares,
                    num_slices=horizon,
                    eta_power=eta,
                    delta=delta,
                    trailing_volume=trailing_v,
                    method="secant",
                )

                # Generate schedules
                schedules: list[ExecutionSchedule] = [
                    generate_twap_schedule(total_shares, horizon),
                    generate_vwap_schedule(
                        total_shares,
                        volume_profile,
                        start_interval_idx=start_bucket,
                        num_slices=horizon,
                        is_oracle=False,
                    ),
                    generate_vwap_schedule(
                        total_shares,
                        realized_profile,
                        start_interval_idx=0,
                        num_slices=horizon,
                        is_oracle=True,
                    ),
                ]

                # Almgren-Chriss schedules across lambda
                for lmbda in lambdas:
                    schedules.append(
                        generate_almgren_chriss_schedule(
                            total_shares=total_shares,
                            num_slices=horizon,
                            risk_aversion=lmbda,
                            volatility=sigma,
                            linearized_eta=tilde_eta,
                            horizon_minutes=float(horizon),
                        )
                    )

                # Evaluate for both sides: Buy (+1) and Sell (-1)
                for side in [1, -1]:
                    for sched in schedules:
                        res = simulate_single_order_replay(
                            schedule=sched,
                            bars_window=bars_window,
                            arrival_price=arrival_price,
                            side=side,
                            eta_param=eta,
                            gamma_param=gamma,
                            delta_param=delta,
                            symbol=symbol,
                            adv_pct=size_pct,
                        )
                        results.append(res)

    return results


def run_monte_carlo_gbm_validation(
    total_shares: float = 100.0,
    horizon_slices: int = 30,
    risk_aversion: float = 1e-4,
    sigma: float = 0.002,
    eta_linear: float = 0.05,
    gamma_linear: float = 0.01,
    s0: float = 40000.0,
    n_paths: int = 1000,
    random_seed: int = 42,
) -> dict[str, float]:
    """Synthetic Monte Carlo correctness check: verify AC achieves best mean-variance utility.

    Utility: U = E[IS] + lambda * Var(IS).
    """
    rng = np.random.default_rng(random_seed)
    N = horizon_slices

    ac_sched = generate_almgren_chriss_schedule(
        total_shares=total_shares,
        num_slices=N,
        risk_aversion=risk_aversion,
        volatility=sigma,
        linearized_eta=eta_linear,
        horizon_minutes=float(N),
    )
    twap_sched = generate_twap_schedule(total_shares, N)

    # Inverted schedule as a sub-optimal alternative
    inverted_slices = ac_sched.slice_sizes[::-1]
    inv_weights = inverted_slices / total_shares
    inv_sched = ExecutionSchedule(
        strategy_name="Inverted",
        total_shares=total_shares,
        num_slices=N,
        slice_sizes=inverted_slices,
        holdings=np.zeros(N + 1),
        weights=inv_weights,
    )

    schedules = {"AC": ac_sched, "TWAP": twap_sched, "Inverted": inv_sched}
    shortfalls: dict[str, list[float]] = {k: [] for k in schedules}

    # Simulate GBM paths
    for _ in range(n_paths):
        # Log returns
        shocks = rng.normal(0, sigma, size=N)
        prices = s0 * np.exp(np.cumsum(shocks))

        for name, sched in schedules.items():
            slices = sched.slice_sizes
            temp_impacts = eta_linear * slices
            perm_prev = np.zeros(N)
            if N > 1:
                perm_prev[1:] = np.cumsum(gamma_linear * slices[:-1])
            perm_impacts = perm_prev + 0.5 * gamma_linear * slices

            fill_prices = prices + temp_impacts + perm_impacts
            exec_p = np.sum(slices * fill_prices) / total_shares
            is_cost = exec_p - s0
            shortfalls[name].append(float(is_cost))

    utilities: dict[str, float] = {}
    for name, costs in shortfalls.items():
        mean_c = float(np.mean(costs))
        var_c = float(np.var(costs))
        utilities[f"{name}_mean"] = mean_c
        utilities[f"{name}_std"] = float(np.std(costs))
        utilities[f"{name}_utility"] = mean_c + risk_aversion * var_c

    return utilities
