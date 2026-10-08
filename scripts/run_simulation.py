"""CLI script to run execution simulations across the full parameter grid.

Usage:
    python scripts/run_simulation.py --config config.yaml
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from exec_impact.bars import compute_prior_volume_profiles
from exec_impact.config import Config
from exec_impact.simulate import (
    OrderResult,
    run_monte_carlo_gbm_validation,
    run_paired_replay_simulation,
)


def run_full_grid_simulation(config: Config) -> pd.DataFrame:
    processed_dir = Path(config.data.processed_dir)
    results_dir = Path(config.results.dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run synthetic Monte Carlo correctness check
    print("Running Synthetic Monte Carlo GBM verification...")
    mc_results = run_monte_carlo_gbm_validation(
        total_shares=100.0,
        horizon_slices=30,
        risk_aversion=1e-4,
        sigma=0.002,
        eta_linear=0.05,
        gamma_linear=0.01,
        n_paths=config.simulation.monte_carlo_paths,
        random_seed=config.project.random_seed,
    )
    with open(results_dir / "monte_carlo_validation.json", "w", encoding="utf-8") as f:
        json.dump(mc_results, f, indent=2)
    print("Monte Carlo verification complete.")

    # 2. Load calibrated impact parameters
    impact_param_file = results_dir / "impact_parameters.json"
    if not impact_param_file.exists():
        raise FileNotFoundError(
            f"Impact parameters not found at {impact_param_file}. Please run scripts/fit_impact.py first."
        )

    with open(impact_param_file, encoding="utf-8") as f:
        impact_params = json.load(f)

    all_order_results: list[OrderResult] = []

    for symbol in config.data.symbols:
        bars_1m_path = processed_dir / symbol / f"{symbol}-bars-1m.parquet"
        if not bars_1m_path.exists():
            print(f"Bars file missing for {symbol}: {bars_1m_path}. Skipping.")
            continue

        print(f"Loading 1m bars for {symbol}...")
        bars_1m = pd.read_parquet(bars_1m_path)
        print(f"Computing 7-day prior volume profiles for {symbol}...")
        profiles = compute_prior_volume_profiles(
            bars_1m,
            interval_minutes=1,
            lookback_days=config.execution.volume_profile_lookback_days,
        )

        symbol_impact = impact_params.get(symbol, {}).get("1m", {})
        print(f"Executing paired tape replay for {symbol} (500 start times)...")

        results = run_paired_replay_simulation(
            bars_1m=bars_1m,
            prior_volume_profiles=profiles,
            impact_params=symbol_impact,
            symbol=symbol,
            horizons_minutes=config.execution.horizons_minutes,
            order_sizes_adv_pct=config.execution.order_sizes_adv_pct,
            lambdas=config.execution.lambdas,
            n_samples=config.simulation.n_start_samples,
            random_seed=config.project.random_seed,
        )
        all_order_results.extend(results)
        print(f"Completed {len(results)} replay runs for {symbol}.")

    # Convert results to DataFrame
    df_results = pd.DataFrame([vars(r) for r in all_order_results])
    out_parquet = results_dir / "simulated_orders.parquet"
    df_results.to_parquet(out_parquet, compression="zstd", index=False)
    print(f"Saved {len(df_results)} simulated orders to {out_parquet}")

    return df_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run execution simulations.")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = Config.load(args.config)
    run_full_grid_simulation(config)


if __name__ == "__main__":
    main()
