"""CLI script to calibrate impact parameters across symbols and intervals.

Usage:
    python scripts/fit_impact.py --config config.yaml
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from exec_impact.config import Config
from exec_impact.impact import calibrate_impact_pipeline


def run_calibration(config: Config) -> dict[str, dict[str, object]]:
    processed_dir = Path(config.data.processed_dir)
    results_dir = Path(config.results.dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_results: dict[str, dict[str, object]] = {}

    for symbol in config.data.symbols:
        summary_results[symbol] = {}
        for interval in config.impact.intervals:
            bars_path = processed_dir / symbol / f"{symbol}-bars-{interval}m.parquet"
            if not bars_path.exists():
                print(f"Bars file not found: {bars_path}. Skipping.")
                continue

            print(f"Calibrating impact model for {symbol} ({interval}m bars)...")
            bars_df = pd.read_parquet(bars_path)
            res = calibrate_impact_pipeline(
                bars_df,
                train_weeks=config.impact.train_weeks,
                test_weeks=config.impact.test_weeks,
                n_bootstrap=config.impact.bootstrap_samples,
                random_seed=config.project.random_seed,
            )
            summary_results[symbol][f"{interval}m"] = res

            print(
                f"  [{symbol} {interval}m] Fitted: Y={res['y_free']:.4f} (95% CI: {res['y_ci'][0]:.4f}-{res['y_ci'][1]:.4f}), "
                f"delta={res['delta_free']:.4f} (CI: {res['delta_ci'][0]:.4f}-{res['delta_ci'][1]:.4f}) | "
                f"Fixed delta=0.5: Y={res['y_fixed']:.4f} | "
                f"Train R2={res['r2_train']:.4f}, OOS R2={res['r2_oos']:.4f}"
            )

    out_file = results_dir / "impact_parameters.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)
    print(f"Saved calibration parameters to {out_file}")

    return summary_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate market impact models.")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = Config.load(args.config)
    run_calibration(config)


if __name__ == "__main__":
    main()
