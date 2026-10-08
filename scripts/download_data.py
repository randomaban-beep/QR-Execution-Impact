"""CLI script to download Binance aggTrades and verify checksums.

Usage:
    python scripts/download_data.py --symbol BTCUSDT --year 2024 --month 1
    python scripts/download_data.py --config config.yaml
"""

import argparse
import calendar
from datetime import date
from pathlib import Path

from tqdm import tqdm

from exec_impact.config import Config
from exec_impact.data import (
    download_daily_aggtrades,
    parse_aggtrades_zip,
    save_trades_to_parquet,
)


def download_and_process_month(
    symbol: str,
    year: int,
    month: int,
    base_url: str,
    raw_dir: Path,
    processed_dir: Path,
    verify: bool = True,
) -> None:
    """Download daily aggTrades for a month, verify checksums, and save to Parquet."""
    num_days = calendar.monthrange(year, month)[1]
    print(f"Downloading {symbol} for {year}-{month:02d} ({num_days} days)...")

    for day in tqdm(range(1, num_days + 1), desc=f"{symbol} {year}-{month:02d}"):
        target_date = date(year, month, day)
        date_str = target_date.strftime("%Y-%m-%d")
        parquet_path = processed_dir / symbol / f"{symbol}-trades-{date_str}.parquet"

        if parquet_path.exists():
            continue  # Already processed

        try:
            zip_path, _ = download_daily_aggtrades(
                symbol=symbol,
                target_date=target_date,
                base_url=base_url,
                output_dir=raw_dir,
                verify=verify,
            )
            df = parse_aggtrades_zip(zip_path)
            save_trades_to_parquet(df, parquet_path)
        except Exception as e:
            print(f"Error downloading/processing {symbol} on {date_str}: {e}")
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify Binance aggTrades data.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--symbol", type=str, default=None, help="Override symbol (e.g. BTCUSDT)")
    parser.add_argument("--year", type=int, default=None, help="Override year")
    parser.add_argument("--month", type=int, default=None, help="Override month")
    parser.add_argument("--no-verify", action="store_true", help="Skip checksum verification")

    args = parser.parse_args()
    config = Config.load(args.config)

    symbols = [args.symbol] if args.symbol else config.data.symbols
    year = args.year or config.data.year
    month = args.month or config.data.month
    raw_dir = Path(config.data.raw_dir)
    processed_dir = Path(config.data.processed_dir)

    for symbol in symbols:
        download_and_process_month(
            symbol=symbol,
            year=year,
            month=month,
            base_url=config.data.base_url,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            verify=not args.no_verify,
        )


if __name__ == "__main__":
    main()
