"""Quantitative evaluation metrics: Implementation Shortfall, CVaR95, bootstrap CIs, and Wilcoxon tests."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


@dataclass(frozen=True)
class CellMetrics:
    """Summary metrics for a specific strategy within a grid cell."""

    strategy: str
    symbol: str
    horizon_minutes: int
    order_size_adv_pct: float
    side: str  # 'buy', 'sell', or 'both'
    sample_size: int
    mean_shortfall_bps: float
    std_shortfall_bps: float
    median_shortfall_bps: float
    p95_shortfall_bps: float
    cvar95_shortfall_bps: float
    mean_diff_vs_twap_bps: float
    diff_ci_lower_bps: float
    diff_ci_upper_bps: float
    wilcoxon_p_value: float


def compute_cvar95(shortfall_bps_array: np.ndarray) -> float:
    """Compute CVaR95 (Expected Shortfall beyond the 95th percentile)."""
    if len(shortfall_bps_array) == 0:
        return 0.0
    p95 = np.percentile(shortfall_bps_array, 95.0)
    tail = shortfall_bps_array[shortfall_bps_array >= p95]
    if len(tail) == 0:
        return float(p95)
    return float(np.mean(tail))


def compute_paired_bootstrap_ci(
    diff_array: np.ndarray,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
) -> tuple[float, float]:
    """Compute 95% bootstrap confidence interval of the mean paired difference."""
    if len(diff_array) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(random_seed)
    n = len(diff_array)

    # Vectorized bootstrap
    indices = rng.choice(n, size=(n_bootstrap, n), replace=True)
    resamples = diff_array[indices]
    boot_means = np.mean(resamples, axis=1)

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    return ci_lower, ci_upper


def compute_paired_wilcoxon(diff_array: np.ndarray) -> float:
    """Compute two-sided Wilcoxon signed-rank test p-value for paired differences."""
    non_zero = diff_array[~np.isclose(diff_array, 0.0, atol=1e-9)]
    if len(non_zero) < 10:
        return 1.0
    try:
        res = wilcoxon(non_zero, alternative="two-sided")
        return float(res.pvalue)
    except Exception:
        return 1.0


def analyze_grid_cell(
    cell_df: pd.DataFrame,
    twap_strategy_name: str = "TWAP",
    n_bootstrap: int = 1000,
    random_seed: int = 42,
) -> list[CellMetrics]:
    """Analyze all strategies within a single grid cell (symbol, horizon, size_pct)."""
    metrics_list: list[CellMetrics] = []

    # Get TWAP baseline
    twap_rows = cell_df[cell_df["strategy_name"] == twap_strategy_name]
    if twap_rows.empty:
        return []

    strategies = cell_df["strategy_name"].unique()
    symbol = str(cell_df["symbol"].iloc[0])
    horizon = int(cell_df["horizon_minutes"].iloc[0])
    size_pct = float(cell_df["order_size_adv_pct"].iloc[0])

    for strat in strategies:
        strat_rows = cell_df[cell_df["strategy_name"] == strat]

        # Robust paired alignment via merge on (start_time, side)
        merged = pd.merge(
            strat_rows[["start_time", "side", "shortfall_bps"]],
            twap_rows[["start_time", "side", "shortfall_bps"]],
            on=["start_time", "side"],
            suffixes=("_strat", "_twap"),
        )
        if merged.empty:
            continue

        strat_vals = merged["shortfall_bps_strat"].values
        twap_vals = merged["shortfall_bps_twap"].values
        diff_vals = strat_vals - twap_vals

        mean_shortfall = float(np.mean(strat_vals))
        std_shortfall = float(np.std(strat_vals, ddof=1)) if len(strat_vals) > 1 else 0.0
        median_shortfall = float(np.median(strat_vals))
        p95 = float(np.percentile(strat_vals, 95.0))
        cvar95 = compute_cvar95(strat_vals)

        mean_diff = float(np.mean(diff_vals))
        if strat == twap_strategy_name:
            ci_lower, ci_upper = 0.0, 0.0
            p_val = 1.0
        else:
            ci_lower, ci_upper = compute_paired_bootstrap_ci(
                diff_vals, n_bootstrap=n_bootstrap, random_seed=random_seed
            )
            p_val = compute_paired_wilcoxon(diff_vals)

        metrics_list.append(
            CellMetrics(
                strategy=strat,
                symbol=symbol,
                horizon_minutes=horizon,
                order_size_adv_pct=size_pct,
                side="both",
                sample_size=len(merged),
                mean_shortfall_bps=mean_shortfall,
                std_shortfall_bps=std_shortfall,
                median_shortfall_bps=median_shortfall,
                p95_shortfall_bps=p95,
                cvar95_shortfall_bps=cvar95,
                mean_diff_vs_twap_bps=mean_diff,
                diff_ci_lower_bps=ci_lower,
                diff_ci_upper_bps=ci_upper,
                wilcoxon_p_value=p_val,
            )
        )

    return metrics_list


def build_headline_summary(
    all_metrics: list[CellMetrics],
) -> list[dict[str, Any]]:
    """Generate headline summary comparing best AC vs TWAP across all grid cells."""
    df = pd.DataFrame([vars(m) for m in all_metrics])
    if df.empty:
        return []

    headline_rows: list[dict[str, Any]] = []

    group_cols = ["symbol", "horizon_minutes", "order_size_adv_pct"]
    for (sym, horiz, size_pct), group in df.groupby(group_cols):
        twap_row = group[group["strategy"] == "TWAP"]
        vwap_row = group[group["strategy"] == "VWAP"]
        oracle_row = group[group["strategy"] == "VWAP_Oracle"]
        ac_rows = group[group["strategy"].str.startswith("AlmgrenChriss")]

        if twap_row.empty or ac_rows.empty:
            continue

        twap_cost = float(twap_row["mean_shortfall_bps"].iloc[0])
        vwap_cost = (
            float(vwap_row["mean_shortfall_bps"].iloc[0]) if not vwap_row.empty else float("nan")
        )
        oracle_cost = (
            float(oracle_row["mean_shortfall_bps"].iloc[0])
            if not oracle_row.empty
            else float("nan")
        )

        # Find best lambda for AC (lowest mean shortfall)
        best_ac = ac_rows.sort_values("mean_shortfall_bps").iloc[0]
        best_ac_strat = best_ac["strategy"]
        best_ac_cost = float(best_ac["mean_shortfall_bps"])
        diff_vs_twap = float(best_ac["mean_diff_vs_twap_bps"])
        ci_low = float(best_ac["diff_ci_lower_bps"])
        ci_high = float(best_ac["diff_ci_upper_bps"])
        p_val = float(best_ac["wilcoxon_p_value"])

        # AC savings: positive savings means AC reduced cost vs TWAP
        ac_savings = -diff_vs_twap
        savings_ci = [-ci_high, -ci_low]
        ac_outperformed = diff_vs_twap < 0 and p_val < 0.05

        headline_rows.append(
            {
                "symbol": sym,
                "horizon_minutes": int(horiz),
                "order_size_adv_pct": float(size_pct),
                "twap_shortfall_bps": round(twap_cost, 2),
                "vwap_shortfall_bps": round(vwap_cost, 2),
                "oracle_vwap_bps": round(oracle_cost, 2),
                "best_ac_strategy": best_ac_strat,
                "best_ac_shortfall_bps": round(best_ac_cost, 2),
                "ac_savings_vs_twap_bps": round(ac_savings, 2),
                "savings_ci_95": [round(s, 2) for s in savings_ci],
                "p_value": round(p_val, 4),
                "ac_beats_twap": bool(ac_outperformed),
            }
        )

    return headline_rows
