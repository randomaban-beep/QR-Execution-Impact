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

    print(f"Processing {len(trade_files)} daily trade files for {symbol}...")
    for interval in intervals:
        daily_bars: list[pd.DataFrame] = []
        for f in trade_files:
            trades = pd.read_parquet(f)
            bars_day = aggregate_trades_to_bars(trades, interval_minutes=interval)
            daily_bars.append(bars_day)

        all_bars = pd.concat(daily_bars, ignore_index=True).sort_values("open_time")
        # Ensure uniform continuous timeline without gaps across days
        rule = f"{interval}min"
        full_idx = pd.date_range(
            all_bars["open_time"].min(),
            all_bars["open_time"].max(),
            freq=rule,
            tz="UTC",
            name="open_time",
        )
        all_bars = all_bars.set_index("open_time").reindex(full_idx)
        all_bars["last_price"] = all_bars["last_price"].ffill().bfill()
        all_bars["open_price"] = all_bars["open_price"].fillna(all_bars["last_price"])
        all_bars["high_price"] = all_bars["high_price"].fillna(all_bars["last_price"])
        all_bars["low_price"] = all_bars["low_price"].fillna(all_bars["last_price"])
        all_bars["vwap"] = all_bars["vwap"].fillna(all_bars["last_price"])
        all_bars["volume"] = all_bars["volume"].fillna(0.0)
        all_bars["signed_volume"] = all_bars["signed_volume"].fillna(0.0)
        all_bars["trade_count"] = all_bars["trade_count"].fillna(0).astype(int)
        all_bars = all_bars.reset_index()
        from datetime import timedelta

        all_bars["close_time"] = all_bars["open_time"] + timedelta(minutes=interval)

        # Compute trailing features on full continuous series
        all_bars = compute_trailing_features(all_bars)

        out_path = symbol_dir / f"{symbol}-bars-{interval}m.parquet"
        all_bars.to_parquet(out_path, compression="zstd", index=False)
        print(f"Saved {len(all_bars)} continuous {interval}m bars to {out_path}")


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
