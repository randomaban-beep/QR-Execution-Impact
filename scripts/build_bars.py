"""CLI script to build 1m, 5m, and 15m bars from raw trades Parquet files.

Usage:
    python scripts/build_bars.py --config config.yaml
"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from exec_impact.bars import aggregate_trades_to_bars, compute_trailing_features
from exec_impact.config import Config


def process_symbol_bars(
    symbol: str,
    processed_dir: Path,
    intervals: list[int],
) -> None:
    symbol_dir = processed_dir / symbol
    trade_files = sorted(symbol_dir.glob(f"{symbol}-trades-*.parquet"))
    if not trade_files:
        print(f"No trade files found for {symbol} in {symbol_dir}")
        return

    print(f"Loading {len(trade_files)} trade files for {symbol}...")
    dfs = [pd.read_parquet(f) for f in trade_files]
    all_trades = pd.concat(dfs, ignore_index=True).sort_values("timestamp")

    for interval in intervals:
        print(f"Building {interval}-minute bars for {symbol}...")
        bars = aggregate_trades_to_bars(all_trades, interval_minutes=interval)
        bars = compute_trailing_features(bars)
        out_path = symbol_dir / f"{symbol}-bars-{interval}m.parquet"
        bars.to_parquet(out_path, compression="zstd", index=False)
        print(f"Saved {len(bars)} bars to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bars from trades.")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = Config.load(args.config)
    processed_dir = Path(config.data.processed_dir)

    for symbol in tqdm(config.data.symbols, desc="Symbols"):
        process_symbol_bars(symbol, processed_dir, config.data.bar_intervals)


if __name__ == "__main__":
    main()
