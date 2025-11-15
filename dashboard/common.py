"""
common.py

Shared helpers for the Quantaira dashboard & patient views.

- best_ts_col(df): pick the best timestamp column name
- convert_tz(series, tz_name): convert UTC timestamps -> target timezone
- split_blood_pressure(df): turn combined BP like "120/80" into
  separate rows with metrics 'systolic_bp' and 'diastolic_bp'.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# Timestamp helpers
# ─────────────────────────────────────────────

def best_ts_col(df: pd.DataFrame) -> Optional[str]:
    """
    Try to guess which column is the main timestamp column.

    We first look for common names, then fall back to the first
    datetime-like column if nothing matches.
    """
    if df is None or df.empty:
        return None

    candidates = [
        "timestamp_utc",
        "timestamp",
        "ts",
        "time",
        "event_time",
        "created_at",
        "updated_at",
        "measured_at",
        "recorded_at",
    ]

    cols_lower = {c.lower(): c for c in df.columns}

    for c in candidates:
        if c in cols_lower:
            return cols_lower[c]

    # fallback: first datetime-like column
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            return c

    # last resort: None (caller should handle)
    return None


def convert_tz(ts: Iterable, tz_name: str) -> pd.Series:
    """
    Convert a sequence of timestamps assumed to be in UTC into
    a timezone-aware pandas Series in `tz_name`.

    If conversion fails, returns the original UTC Series.
    """
    # parse as UTC
    s = pd.to_datetime(ts, utc=True, errors="coerce")

    try:
        return s.dt.tz_convert(tz_name)
    except Exception:
        # if tz_name is invalid or something else goes wrong,
        # just return as UTC to avoid crashing the app
        return s


# ─────────────────────────────────────────────
# Blood pressure helpers
# ─────────────────────────────────────────────

def _parse_bp_value(val) -> tuple[Optional[float], Optional[float]]:
    """
    Parse a blood pressure value that looks like '120/80', '120 / 80', etc.

    Returns (systolic, diastolic) as floats or (None, None) if parsing fails.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None, None

    text = str(val).strip()
    if not text:
        return None, None

    # common formats: "120/80", "120 / 80", "120-80"
    for sep in ["/", "-", " "]:
        if sep in text:
            parts = [p for p in text.replace("-", "/").split("/") if p.strip()]
            if len(parts) == 2:
                try:
                    s_val = float(parts[0])
                    d_val = float(parts[1])
                    return s_val, d_val
                except Exception:
                    return None, None

    # if it's already two numbers in a list/tuple
    if isinstance(val, (list, tuple)) and len(val) == 2:
        try:
            s_val = float(val[0])
            d_val = float(val[1])
            return s_val, d_val
        except Exception:
            return None, None

    return None, None


def split_blood_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take a dataframe of vitals and expand any combined blood-pressure
    rows into separate systolic/diastolic rows.

    Expected shape (flexible):

    - 'metric' column: e.g. 'bp', 'BP', 'blood_pressure', 'blood pressure'
    - 'value' column: '120/80', '120 / 80', '120-80', or [120, 80]

    Output:
    - original non-BP rows
    - plus new rows with:
        metric = 'systolic_bp' and 'diastolic_bp'
        value  = numeric systolic / diastolic
      Other columns (timestamp, device_id, patient_id, etc.) are copied.

    If the dataframe doesn't have the expected shape, it is returned unchanged.
    """
    if df is None or df.empty:
        return df

    if "metric" not in df.columns or "value" not in df.columns:
        # nothing to do
        return df

    df = df.copy()

    # identify combined BP rows
    metric_str = df["metric"].astype(str).str.lower()
    bp_mask = metric_str.isin(
        [
            "bp",
            "blood_pressure",
            "blood pressure",
            "bloodpressure",
            "bp_combined",
            "bp (sys/dia)",
        ]
    ) | metric_str.str.contains("blood", na=False) & metric_str.str.contains(
        "press", na=False
    )

    if not bp_mask.any():
        return df

    bp_rows = df.loc[bp_mask]
    other_rows = df.loc[~bp_mask]

    new_rows = []

    for _, row in bp_rows.iterrows():
        s_val, d_val = _parse_bp_value(row["value"])

        # if parsing failed, keep the original row as-is
        if s_val is None or d_val is None:
            new_rows.append(row)
            continue

        base = row.to_dict()

        # systolic
        systolic_row = base.copy()
        systolic_row["metric"] = "systolic_bp"
        systolic_row["value"] = s_val
        new_rows.append(systolic_row)

        # diastolic
        diastolic_row = base.copy()
        diastolic_row["metric"] = "diastolic_bp"
        diastolic_row["value"] = d_val
        new_rows.append(diastolic_row)

    out = pd.concat([other_rows, pd.DataFrame(new_rows)], ignore_index=True)
    return out


__all__ = ["best_ts_col", "convert_tz", "split_blood_pressure"]
