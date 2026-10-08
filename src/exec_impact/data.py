"""Data acquisition, checksum verification, schema normalization, and Parquet caching."""

import hashlib
import io
import zipfile
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

AGGTRADE_COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
    "is_best_match",
]


def compute_sha256(file_path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 hex digest for a given file."""
    sha = hashlib.sha256()
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha.update(chunk)
    return sha.hexdigest().lower()


def parse_checksum_file(checksum_path: Path | str) -> str:
    """Extract SHA256 hex digest string from a .CHECKSUM file."""
    path = Path(checksum_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checksum file not found: {path}")

    content = path.read_text(encoding="utf-8").strip()
    # Format is usually: "<hex_sha256>  <filename>"
    parts = content.split()
    if not parts:
        raise ValueError(f"Empty checksum file: {path}")
    return parts[0].strip().lower()


def verify_checksum(file_path: Path | str, checksum_path: Path | str) -> bool:
    """Verify that file's SHA256 matches the checksum in checksum_path.

    Raises ValueError if checksum mismatch is detected.
    """
    actual_sha = compute_sha256(file_path)
    expected_sha = parse_checksum_file(checksum_path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"Checksum mismatch for {file_path}:\n"
            f"  Expected: {expected_sha}\n"
            f"  Actual:   {actual_sha}"
        )
    return True


def download_file(url: str, dest_path: Path, timeout: int = 30) -> Path:
    """Download a file from url to dest_path using streaming."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    return dest_path


def format_binance_filename(symbol: str, target_date: date | str) -> str:
    """Return Binance daily aggTrades filename, e.g. BTCUSDT-aggTrades-2024-01-01.zip."""
    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, "%Y-%m-%d").date()
    return f"{symbol}-aggTrades-{target_date.strftime('%Y-%m-%d')}.zip"


def download_daily_aggtrades(
    symbol: str,
    target_date: date | str,
    base_url: str = "https://data.binance.vision/data/spot/daily/aggTrades",
    output_dir: Path | str = "data/raw",
    verify: bool = True,
    timeout: int = 30,
) -> tuple[Path, Path]:
    """Download daily aggTrades zip and .CHECKSUM file, optionally verifying."""
    out_dir = Path(output_dir) / symbol
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_filename = format_binance_filename(symbol, target_date)
    checksum_filename = f"{zip_filename}.CHECKSUM"

    zip_url = f"{base_url}/{symbol}/{zip_filename}"
    checksum_url = f"{base_url}/{symbol}/{checksum_filename}"

    local_zip = out_dir / zip_filename
    local_checksum = out_dir / checksum_filename

    # Download checksum first
    download_file(checksum_url, local_checksum, timeout=timeout)
    # Download zip
    download_file(zip_url, local_zip, timeout=timeout)

    if verify:
        verify_checksum(local_zip, local_checksum)

    return local_zip, local_checksum


def detect_timestamp_unit(sample_ts: float | int | np.ndarray) -> str:
    """Detect whether timestamp is in milliseconds ('ms'), microseconds ('us'), or nanoseconds ('ns')."""
    median_val = np.median(sample_ts)
    if median_val > 1e17:
        return "ns"
    if median_val > 1e14:
        return "us"
    return "ms"


def parse_aggtrades_stream(
    stream: io.IOBase | str | Path,
) -> pd.DataFrame:
    """Parse Binance aggTrades CSV from stream/path, normalizing schema and timestamps.

    Handles:
    - Files with or without header
    - Millisecond or microsecond timestamps
    - Aggressor side: is_buyer_maker == True -> Seller aggressive (-1), False -> Buyer aggressive (+1)
    """
    # Read first line to detect if header exists
    if isinstance(stream, (str, Path)):
        f = open(stream, encoding="utf-8")
        should_close = True
    else:
        f = stream
        should_close = False

    try:
        first_line = f.readline()
        if not first_line:
            # Empty stream
            return pd.DataFrame(
                columns=AGGTRADE_COLUMNS + ["timestamp", "aggressor_side", "signed_quantity"]
            )

        # Check if first token contains letters (indicating header)
        tokens = [t.strip() for t in first_line.split(",")]
        has_header = False
        try:
            int(tokens[0])
        except ValueError:
            has_header = True

        if hasattr(f, "seek"):
            f.seek(0)
        else:
            # If stream does not support seek, prepend first_line
            remaining = f.read()
            full_content = first_line + (
                remaining if isinstance(remaining, str) else remaining.decode("utf-8")
            )
            f = io.StringIO(full_content)

        if has_header:
            df = pd.read_csv(
                f,
                dtype={
                    "agg_trade_id": "int64",
                    "price": "float64",
                    "quantity": "float64",
                    "first_trade_id": "int64",
                    "last_trade_id": "int64",
                    "transact_time": "int64",
                    "is_buyer_maker": "bool",
                },
            )
        else:
            # Map up to 8 standard columns
            names = AGGTRADE_COLUMNS
            df = pd.read_csv(
                f,
                header=None,
                names=names,
                usecols=list(range(min(len(tokens), len(names)))),
                dtype={
                    0: "int64",
                    1: "float64",
                    2: "float64",
                    3: "int64",
                    4: "int64",
                    5: "int64",
                    6: "bool",
                },
            )
    finally:
        if should_close:
            f.close()

    if df.empty:
        return pd.DataFrame(
            columns=AGGTRADE_COLUMNS + ["timestamp", "aggressor_side", "signed_quantity"]
        )

    # Standardize column names if header was present with different casing/naming
    col_map = {c: c.lower().strip() for c in df.columns}
    df = df.rename(columns=col_map)

    # Identify transact_time column
    time_col = "transact_time"
    if time_col not in df.columns:
        # Fallback to column index 5
        time_col = df.columns[5]

    unit = detect_timestamp_unit(df[time_col].values)
    df["timestamp"] = pd.to_datetime(df[time_col], unit=unit, utc=True)

    # Aggressor side mapping:
    # is_buyer_maker == True -> Maker was buyer -> Taker (aggressor) was SELLER (-1)
    # is_buyer_maker == False -> Maker was seller -> Taker (aggressor) was BUYER (+1)
    is_buyer_maker_bool = df["is_buyer_maker"].astype(bool)
    df["aggressor_side"] = np.where(is_buyer_maker_bool, -1, 1).astype(np.int8)
    df["signed_quantity"] = df["quantity"] * df["aggressor_side"]

    return df


def parse_aggtrades_zip(zip_path: Path | str) -> pd.DataFrame:
    """Extract and parse aggTrades directly from a zip file into a normalized DataFrame."""
    zip_p = Path(zip_path)
    if not zip_p.is_file():
        raise FileNotFoundError(f"Zip file not found: {zip_p}")

    with zipfile.ZipFile(zip_p, "r") as z:
        file_list = z.namelist()
        csv_files = [f for f in file_list if f.endswith(".csv")]
        if not csv_files:
            raise ValueError(f"No CSV file found inside zip: {zip_p}")

        with z.open(csv_files[0]) as csv_file:
            # Wrap binary stream with TextIOWrapper
            text_stream = io.TextIOWrapper(csv_file, encoding="utf-8")
            return parse_aggtrades_stream(text_stream)


def save_trades_to_parquet(df: pd.DataFrame, out_path: Path | str) -> Path:
    """Save normalized trades DataFrame to Parquet format with ZSTD compression."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure types are optimal
    df.to_parquet(path, compression="zstd", index=False)
    return path


def load_trades_parquet(parquet_path: Path | str) -> pd.DataFrame:
    """Load normalized trades Parquet file."""
    path = Path(parquet_path)
    if not path.is_file():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return pd.read_parquet(path)
