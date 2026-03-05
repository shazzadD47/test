"""
DataFrame transforms for the new (non-legacy) merge flow: clean names,
NA handling, aliases, GROUP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def clean_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names: strip, lower case, replace non-word chars with underscore.
    """
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
    )
    return df


def drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are empty or start with 'unnamed'.
    """
    bad = [c for c in df.columns if c == "" or str(c).lower().startswith("unnamed")]
    return df.drop(columns=bad, errors="ignore")


def na_strings_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace literal 'na' / 'n/a' (case-insensitive) with np.nan.

    Before: df["x"] = ["na", "1", "n/a"]
    After:  df["x"] = [NaN, "1", NaN]
    """
    return df.replace(r"(?i)^(na|n/a)$", np.nan, regex=True)


def normalize_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unify file_name/filename and doi_url/doi so merges don't duplicate columns.
    - If both file_name and filename exist: keep file_name column but prioritize
      filename (from fileName) over file_name (from FILE_NAME); drop filename.
    - If only filename: rename to file_name.
    - Same for doi_url and doi.
    """
    if "file_name" in df.columns and "filename" in df.columns:
        df["file_name"] = df["filename"].where(df["filename"].notna(), df["file_name"])
        df = df.drop(columns=["filename"], errors="ignore")
    elif "file_name" not in df.columns and "filename" in df.columns:
        df = df.rename(columns={"filename": "file_name"})
    if "doi_url" in df.columns and "doi" in df.columns:
        df["doi_url"] = df["doi_url"].where(df["doi_url"].notna(), df["doi"])
        df = df.drop(columns=["doi"], errors="ignore")
    elif "doi_url" not in df.columns and "doi" in df.columns:
        df = df.rename(columns={"doi": "doi_url"})
    return df


def ensure_group_from_arm_trt(df: pd.DataFrame) -> pd.DataFrame:
    """
    URL/export-style tables (e.g. input_files2) may have arm_trt but no group.
    Set group = arm_trt when group is missing so covariate/backfill merges work.
    """
    if "group" in df.columns:
        return df
    if "arm_trt" not in df.columns:
        return df
    df = df.copy()
    df["group"] = df["arm_trt"]
    return df


def std_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the 'group' column: strip, replace nan/None/empty, normalize 1.0 -> 1.

    Before: group = ["  Dupilumab  ", "1.0", " 2 "]
    After:  group = ["Dupilumab", "1", "2"]
    """
    if "group" not in df.columns:
        return df
    g = df["group"].astype("string").str.strip()
    g = g.replace({"nan": pd.NA, "NaN": pd.NA, "None": pd.NA, "": pd.NA})
    g = g.str.replace(r"^(\d+)\.0+$", r"\1", regex=True)
    df["group"] = g
    return df


def has_group_column(df: pd.DataFrame) -> bool:
    """Return True if df has a 'group' column (after lowercase)."""
    cols_lower = [str(c).strip().lower() for c in df.columns]
    return "group" in cols_lower


def get_group_column_name(df: pd.DataFrame) -> str | None:
    """Return the actual column name for 'group' if present (case-insensitive)."""
    for c in df.columns:
        if str(c).strip().lower() == "group":
            return c
    return None


def has_file_name_column(df: pd.DataFrame) -> bool:
    """Return True if df has file_name or filename (after lowercase)."""
    cols_lower = [str(c).strip().lower() for c in df.columns]
    return "file_name" in cols_lower or "filename" in cols_lower
