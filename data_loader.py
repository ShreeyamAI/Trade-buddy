"""
Data Loading and Validation Module for Trade Buddy.
Validates 1-minute historical candle data for NIFTY 50 index analysis.
Strictly verifies column presence, chronological order, duplicates, missing values, and price integrity.
"""

from typing import Tuple, List, Optional
import pandas as pd
import io

REQUIRED_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']


def validate_and_load_csv(source) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Loads and validates historical candle data from a file path or file-like object.
    
    Parameters:
        source: File path (str) or uploaded file buffer (io.BytesIO / io.StringIO)
        
    Returns:
        (df, errors):
            df: Validated DataFrame sorted chronologically, or None if validation failed.
            errors: List of human-readable error messages explaining any failure.
    """
    errors: List[str] = []

    try:
        if isinstance(source, (str, io.IOBase)):
            df = pd.read_csv(source)
        else:
            errors.append(f"Invalid input type for data source: {type(source).__name__}")
            return None, errors
    except Exception as e:
        errors.append(f"Failed to read CSV file: {str(e)}")
        return None, errors

    if df.empty:
        errors.append("The CSV file is empty.")
        return None, errors

    # Normalize column names to lowercase and strip whitespace
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Check for missing required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(
            f"Missing required column(s): {', '.join(missing_cols)}. "
            f"Expected columns: {', '.join(REQUIRED_COLUMNS)}."
        )
        return None, errors

    # Select only required columns (plus any optional ones kept)
    df = df[REQUIRED_COLUMNS].copy()

    # Check for empty / missing values
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        null_details = ", ".join([f"{col} ({count} rows)" for col, count in cols_with_nulls.items()])
        errors.append(f"Dataset contains missing/null values in: {null_details}. Clean or impute data before loading.")

    # Validate and parse timestamp column
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        invalid_ts_count = df['timestamp'].isnull().sum()
        if invalid_ts_count > 0:
            errors.append(f"Found {invalid_ts_count} rows with invalid or unparseable timestamps.")
    except Exception as e:
        errors.append(f"Timestamp parsing error: {str(e)}")

    # Check numeric types for open, high, low, close, volume
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        nan_numeric = df[col].isnull().sum()
        if nan_numeric > 0:
            errors.append(f"Column '{col}' has {nan_numeric} non-numeric values.")

    if errors:
        return None, errors

    # Check for duplicate timestamps
    duplicate_count = df['timestamp'].duplicated().sum()
    if duplicate_count > 0:
        sample_dupes = df[df['timestamp'].duplicated(keep=False)]['timestamp'].head(3).tolist()
        sample_str = ", ".join([str(t) for t in sample_dupes])
        errors.append(
            f"Found {duplicate_count} duplicate timestamps (e.g. {sample_str}). "
            "Each 1-minute candle must have a unique timestamp."
        )

    # Check price integrity / candle sanity
    invalid_prices = (df['open'] <= 0) | (df['high'] <= 0) | (df['low'] <= 0) | (df['close'] <= 0)
    if invalid_prices.any():
        errors.append(f"Found {invalid_prices.sum()} rows with zero or negative price values.")

    invalid_high = (df['high'] < df['open']) | (df['high'] < df['close']) | (df['high'] < df['low'])
    if invalid_high.any():
        errors.append(
            f"Found {invalid_high.sum()} candles where High is lower than Open, Close, or Low."
        )

    invalid_low = (df['low'] > df['open']) | (df['low'] > df['close'])
    if invalid_low.any():
        errors.append(
            f"Found {invalid_low.sum()} candles where Low is higher than Open or Close."
        )

    negative_vol = df['volume'] < 0
    if negative_vol.any():
        errors.append(f"Found {negative_vol.sum()} rows with negative volume.")

    # Check chronological order
    if not df['timestamp'].is_monotonic_increasing:
        # Check if simply unsorted
        sorted_df = df.sort_values(by='timestamp').reset_index(drop=True)
        # We enforce that the incoming data must be chronological or alert clearly
        errors.append(
            "Candle timestamps are not in strict chronological order. "
            "Please ensure input data is ordered ascending by time."
        )

    if errors:
        return None, errors

    # Clean reset of index
    df = df.reset_index(drop=True)
    return df, []
