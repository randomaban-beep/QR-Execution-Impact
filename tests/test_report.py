"""Integration test for report and figures generation."""

from pathlib import Path

import numpy as np
import pandas as pd

from exec_impact.config import Config
from scripts.generate_report import generate_full_report


def test_generate_report_end_to_end(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True)

    # Create dummy simulated_orders.parquet
    start_times = pd.date_range("2024-01-01", periods=20, freq="1h", tz="UTC")
    rows = []
    for st in start_times:
        for strat in ["TWAP", "VWAP", "AlmgrenChriss_lambda_1.0e-04"]:
            for side in [1, -1]:
                rows.append(
                    {
                        "strategy_name": strat,
                        "symbol": "BTCUSDT",
                        "horizon_minutes": 60,
                        "order_size": 10.0,
                        "order_size_adv_pct": 0.05,
                        "start_time": st,
                        "side": side,
                        "arrival_price": 40000.0,
                        "exec_price": 40005.0,
                        "shortfall_bps": 5.0 + np.random.normal(0, 0.5),
                        "temp_impact_bps": 2.0,
                        "perm_impact_bps": 1.0,
                        "tape_slippage_bps": 2.0,
                    }
                )

    orders_df = pd.DataFrame(rows)
    orders_df.to_parquet(results_dir / "simulated_orders.parquet", index=False)

    # Mock custom config
    config_dict = {
        "results": {
            "dir": str(results_dir),
            "metrics_file": str(results_dir / "metrics.json"),
            "figures_dir": str(figures_dir),
        },
        "data": {
            "symbols": ["BTCUSDT"],
            "processed_dir": str(tmp_path / "data" / "processed"),
        },
    }
    custom_yaml = tmp_path / "test_config.yaml"
    import yaml

    custom_yaml.write_text(yaml.dump(config_dict))
    test_cfg = Config.load(custom_yaml)

    report_payload = generate_full_report(test_cfg)

    assert "headline_summary" in report_payload
    assert (results_dir / "metrics.json").exists()
    assert (figures_dir / "impact_curves.png").exists()
    assert (figures_dir / "cost_vs_size.png").exists()
    assert (figures_dir / "efficient_frontier.png").exists()
    assert (figures_dir / "volume_profiles.png").exists()
