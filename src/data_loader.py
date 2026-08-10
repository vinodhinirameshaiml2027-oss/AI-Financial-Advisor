"""
Handles loading of transaction data:
- Bundled sample dataset
- User-uploaded CSV files

Returns a tuple of (dataframe_or_None, error_message_or_None, warnings_list)
so the UI layer can decide how to present the result without ever crashing.
"""
from __future__ import annotations

import io
import os
from typing import List, Optional, Tuple

import pandas as pd

from src.utils import REQUIRED_CSV_COLUMNS, VALID_TYPES

SAMPLE_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_transactions.csv")


def load_sample_data() -> pd.DataFrame:
    """Load the bundled sample transaction dataset. Always returns a valid DataFrame."""
    path = os.path.abspath(SAMPLE_DATA_PATH)
    df = pd.read_csv(path)
    df, _, _ = validate_and_clean(df)
    return df


def read_uploaded_csv(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Read a file-like object (from Streamlit's file_uploader) into a DataFrame.
    Returns (df, error_message). df is None if reading failed.
    """
    try:
        content = uploaded_file.read()
        if isinstance(content, bytes):
            text_stream = io.BytesIO(content)
        else:
            text_stream = io.StringIO(content)
        df = pd.read_csv(text_stream)
    except pd.errors.EmptyDataError:
        return None, "The uploaded file is empty. Please upload a CSV file that contains transaction data."
    except pd.errors.ParserError:
        return None, "The uploaded file could not be parsed as CSV. Please check the file format."
    except UnicodeDecodeError:
        return None, "The uploaded file has an unsupported encoding. Please save it as UTF-8 CSV and try again."
    except Exception as exc:  # noqa: BLE001 - we deliberately want a catch-all for user-facing safety
        return None, f"Could not read the uploaded file. Details: {exc}"

    if df.empty:
        return None, "The uploaded CSV has no rows. Please upload a file with at least one transaction."

    return df, None


def validate_and_clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str], List[str]]:
    """
    Validate that required columns exist, then clean the dataframe:
    - Trim column names
    - Parse dates (invalid dates dropped, with warning)
    - Coerce Amount to numeric (invalid rows dropped, with warning)
    - Normalize Type to 'Income'/'Expense' (unrecognized dropped, with warning)
    - Remove duplicate rows
    - Remove rows with missing critical values

    Returns (cleaned_df, fatal_error_or_None, warnings_list)
    """
    warnings: List[str] = []

    if df is None:
        return pd.DataFrame(columns=REQUIRED_CSV_COLUMNS), "No data provided.", warnings

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    missing_cols = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
    if missing_cols:
        error = (
            f"The CSV is missing required column(s): {', '.join(missing_cols)}. "
            f"Expected columns are: {', '.join(REQUIRED_CSV_COLUMNS)}."
        )
        return pd.DataFrame(columns=REQUIRED_CSV_COLUMNS), error, warnings

    df = df[REQUIRED_CSV_COLUMNS].copy()

    initial_rows = len(df)

    # Clean Category / Type text fields
    df["Category"] = df["Category"].astype(str).str.strip()
    df["Type"] = df["Type"].astype(str).str.strip().str.capitalize()

    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    bad_dates = df["Date"].isna().sum()
    if bad_dates > 0:
        warnings.append(f"{bad_dates} row(s) had invalid or missing dates and were removed.")
    df = df[df["Date"].notna()]

    # Coerce Amount to numeric
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    bad_amounts = df["Amount"].isna().sum()
    if bad_amounts > 0:
        warnings.append(f"{bad_amounts} row(s) had non-numeric amounts and were removed.")
    df = df[df["Amount"].notna()]

    # Remove negative or zero amounts (transactions should be positive magnitudes)
    negative_amounts = (df["Amount"] < 0).sum()
    if negative_amounts > 0:
        warnings.append(
            f"{negative_amounts} row(s) had negative amounts; converted to positive values."
        )
        df["Amount"] = df["Amount"].abs()

    zero_amounts = (df["Amount"] == 0).sum()
    if zero_amounts > 0:
        warnings.append(f"{zero_amounts} row(s) had zero amounts and were removed.")
    df = df[df["Amount"] > 0]

    # Normalize / validate Type
    invalid_type_mask = ~df["Type"].isin(VALID_TYPES)
    invalid_types = invalid_type_mask.sum()
    if invalid_types > 0:
        warnings.append(
            f"{invalid_types} row(s) had an unrecognized Type (expected 'Income' or 'Expense') and were removed."
        )
    df = df[df["Type"].isin(VALID_TYPES)]

    # Missing category -> label as Uncategorized rather than dropping the row
    missing_category = (df["Category"] == "") | (df["Category"].str.lower() == "nan")
    if missing_category.sum() > 0:
        warnings.append(f"{missing_category.sum()} row(s) had a missing category; labeled as 'Uncategorized'.")
        df.loc[missing_category, "Category"] = "Uncategorized"

    # Remove duplicate rows
    before_dedup = len(df)
    df = df.drop_duplicates()
    removed_dupes = before_dedup - len(df)
    if removed_dupes > 0:
        warnings.append(f"{removed_dupes} duplicate row(s) were removed.")

    df = df.sort_values("Date").reset_index(drop=True)

    if df.empty:
        error = (
            "After cleaning, no valid transactions remained. "
            "Please check that your CSV has valid dates, numeric amounts, and Type values of "
            "'Income' or 'Expense'."
        )
        return df, error, warnings

    removed_total = initial_rows - len(df)
    if removed_total > 0 and not warnings:
        warnings.append(f"{removed_total} row(s) were removed during cleaning.")

    return df, None, warnings


def get_sample_csv_bytes() -> bytes:
    """Return the sample CSV as raw bytes for a Streamlit download button."""
    path = os.path.abspath(SAMPLE_DATA_PATH)
    with open(path, "rb") as f:
        return f.read()
