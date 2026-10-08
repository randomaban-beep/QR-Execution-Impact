"""Unit tests for data acquisition, checksum verification, and schema parsing."""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from exec_impact.data import (
    compute_sha256,
    detect_timestamp_unit,
    load_trades_parquet,
    parse_aggtrades_stream,
    parse_aggtrades_zip,
    parse_checksum_file,
    save_trades_to_parquet,
    verify_checksum,
)


def test_checksum_verification(tmp_path: Path) -> None:
    test_file = tmp_path / "sample.zip"
    test_content = b"Mock zip archive payload for testing"
    test_file.write_bytes(test_content)

    actual_hash = compute_sha256(test_file)
    assert len(actual_hash) == 64

    # Create matching checksum file
    checksum_file = tmp_path / "sample.zip.CHECKSUM"
    checksum_file.write_text(f"{actual_hash}  sample.zip\n")

    assert parse_checksum_file(checksum_file) == actual_hash
    assert verify_checksum(test_file, checksum_file) is True

    # Test corrupted file
    corrupt_checksum = tmp_path / "corrupt.CHECKSUM"
    corrupt_checksum.write_text("0" * 64 + "  sample.zip\n")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_checksum(test_file, corrupt_checksum)


def test_timestamp_ms_us_detection() -> None:
    # 2024-01-01 00:00:00 in ms: 1704067200000 (13 digits)
    ms_timestamps = np.array([1704067200000, 1704067201000])
    assert detect_timestamp_unit(ms_timestamps) == "ms"

    # In us: 1704067200000000 (16 digits)
    us_timestamps = np.array([1704067200000000, 1704067201000000])
    assert detect_timestamp_unit(us_timestamps) == "us"

    # In ns: 1704067200000000000 (19 digits)
    ns_timestamps = np.array([1704067200000000000, 1704067201000000000])
    assert detect_timestamp_unit(ns_timestamps) == "ns"


def test_schema_with_header() -> None:
    csv_text = """agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker,is_best_match
1001,42000.5,0.25,2001,2002,1704067200000,true,true
1002,42001.0,0.50,2003,2003,1704067201000,false,true
"""
    stream = io.StringIO(csv_text)
    df = parse_aggtrades_stream(stream)

    assert len(df) == 2
    assert "timestamp" in df.columns
    assert "aggressor_side" in df.columns
    assert "signed_quantity" in df.columns

    # Trade 1: is_buyer_maker is True -> SELLER aggressor -> -1
    assert df.loc[0, "aggressor_side"] == -1
    assert np.isclose(df.loc[0, "signed_quantity"], -0.25)

    # Trade 2: is_buyer_maker is False -> BUYER aggressor -> +1
    assert df.loc[1, "aggressor_side"] == 1
    assert np.isclose(df.loc[1, "signed_quantity"], 0.50)

    # Verify UTC timestamp conversion
    assert df.loc[0, "timestamp"].tzname() == "UTC"
    assert df.loc[0, "timestamp"] == pd.Timestamp("2024-01-01 00:00:00", tz="UTC")


def test_schema_headerless_microsecond() -> None:
    # Headerless format with microsecond timestamp
    csv_text = """1001,42000.5,0.25,2001,2002,1704067200000000,True,True
1002,42001.0,0.50,2003,2003,1704067201000000,False,True
"""
    stream = io.StringIO(csv_text)
    df = parse_aggtrades_stream(stream)

    assert len(df) == 2
    assert df.loc[0, "agg_trade_id"] == 1001
    assert df.loc[0, "aggressor_side"] == -1
    assert df.loc[1, "aggressor_side"] == 1
    assert df.loc[0, "timestamp"] == pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    assert df.loc[1, "timestamp"] == pd.Timestamp("2024-01-01 00:00:01", tz="UTC")


def test_parse_aggtrades_zip_and_parquet(tmp_path: Path) -> None:
    csv_text = """agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker,is_best_match
1001,42000.5,0.25,2001,2002,1704067200000,true,true
1002,42001.0,0.75,2003,2003,1704067205000,false,true
"""
    zip_path = tmp_path / "BTCUSDT-aggTrades-2024-01-01.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("BTCUSDT-aggTrades-2024-01-01.csv", csv_text)

    df = parse_aggtrades_zip(zip_path)
    assert len(df) == 2
    assert df["quantity"].sum() == 1.0

    # Save to parquet and verify roundtrip
    parquet_path = tmp_path / "test.parquet"
    save_trades_to_parquet(df, parquet_path)
    assert parquet_path.is_file()

    loaded_df = load_trades_parquet(parquet_path)
    assert len(loaded_df) == 2
    assert np.allclose(loaded_df["quantity"], df["quantity"])
    assert np.array_equal(loaded_df["aggressor_side"], df["aggressor_side"])
