"""
Quality checks for key 'unit-consistency': time units, ARM_TIME_UNIT
consistency, ARM_TIME >= 0, unit abbreviation/full-form consistency,
ARM_DOSE_UNIT vs AMT_UNIT per row, and AMT_UNIT consistency per GROUP_NAME and per ARM.
"""

from __future__ import annotations

import pandas as pd

from app.v3.endpoints.merging.schemas import QCError

ALLOWED_TIME_UNITS = frozenset(
    s.lower() for s in ("minutes", "hours", "days", "weeks", "months")
)

UNIT_EQUIVALENCE_GROUPS = (
    {"mg", "milligram", "miligram"},
    {"kg", "kilogram"},
    {"mcg", "microgram"},
    {"iu", "international-unit"},
)


def _row_numbers_1based(indices: list[int]) -> str:
    if not indices:
        return ""
    if len(indices) <= 10:
        return ", ".join(str(i + 1) for i in indices)
    return ", ".join(str(i + 1) for i in indices[:10]) + f", ... ({len(indices)} rows)"


def check_arm_time_unit_and_ii_unit_time_units(
    df: pd.DataFrame, qc_name: str
) -> QCError | None:
    """
    ARM_TIME_UNIT and II_UNIT should be time units (minutes, hours, days, weeks,
    months).
    """
    if df.empty:
        return None
    parts = []
    for col in ("ARM_TIME_UNIT", "II_UNIT"):
        if col not in df.columns:
            continue
        vals = df[col].dropna().astype(str).str.strip().str.lower()
        vals = vals[vals != ""]
        if vals.empty:
            continue
        invalid = vals[~vals.isin(ALLOWED_TIME_UNITS)].drop_duplicates()
        if not invalid.empty:
            bad_indices = df.index[
                df[col].astype(str).str.strip().str.lower().isin(invalid)
            ].tolist()
            parts.append(
                f"Column {col}: invalid time unit(s) "
                f"{', '.join(repr(v) for v in invalid.tolist())} "
                f"at row(s) {_row_numbers_1based(bad_indices)} "
                f"(allowed: minutes, hours, days, weeks, months)"
            )
    if not parts:
        return None
    return QCError(
        error_name="Time Units Invalid",
        error_message="\n- ".join(parts),
        error_source="qc",
        qc_name=qc_name,
    )


def check_arm_time_unit_consistent(df: pd.DataFrame, qc_name: str) -> QCError | None:
    """
    ARM_TIME_UNIT should be the same for all rows.
    """
    if df.empty or "ARM_TIME_UNIT" not in df.columns:
        return None
    col = df["ARM_TIME_UNIT"].astype(str).str.strip().str.lower()
    col = col.replace("", pd.NA).dropna()
    if col.empty or col.nunique() <= 1:
        return None
    unique_vals = col.unique().tolist()
    mode_val = col.mode().iloc[0] if not col.empty else None
    bad_mask = col.notna() & (col != mode_val)
    bad_indices = df.index[bad_mask].tolist()
    return QCError(
        error_name="ARM_TIME_UNIT Inconsistent",
        error_message=(
            f"Column ARM_TIME_UNIT must be the same for all rows.\n"
            f"Found multiple values: {', '.join(repr(v) for v in unique_vals)}\n"
            f"Row(s) with differing value: {_row_numbers_1based(bad_indices)}"
        ),
        error_source="qc",
        qc_name=qc_name,
    )


def check_arm_time_non_negative(df: pd.DataFrame, qc_name: str) -> QCError | None:
    """
    ARM_TIME should be greater than or equal to 0.
    """
    if df.empty or "ARM_TIME" not in df.columns:
        return None
    arm_time = pd.to_numeric(df["ARM_TIME"], errors="coerce")
    bad_mask = arm_time.notna() & (arm_time < 0)
    bad_indices = df.index[bad_mask].tolist()
    if not bad_indices:
        return None
    return QCError(
        error_name="ARM_TIME Negative",
        error_message=(
            f"Column ARM_TIME must be >= 0.\n"
            f"Negative or invalid values at row(s): {_row_numbers_1based(bad_indices)}"
        ),
        error_source="qc",
        qc_name=qc_name,
    )


def _unit_mixed_forms_in_series(series: pd.Series) -> list[tuple[str, set[str]]]:
    mixed = []
    vals = series.dropna().astype(str).str.strip().str.lower()
    vals = vals[vals != ""].unique().tolist()
    for group in UNIT_EQUIVALENCE_GROUPS:
        present = [v for v in vals if v in group]
        if len(present) > 1:
            mixed.append((group.copy().pop(), set(present)))
    return mixed


def check_amt_arm_dose_unit_standardized(
    df: pd.DataFrame, qc_name: str
) -> QCError | None:
    """
    AMT_UNIT and ARM_DOSE_UNIT must not mix abbreviation and full form
    (e.g. mg vs milligram, kg vs kilogram, mcg vs microgram, IU vs international-unit).
    """
    if df.empty:
        return None
    parts = []
    for col in ("AMT_UNIT", "ARM_DOSE_UNIT"):
        if col not in df.columns:
            continue
        mixed = _unit_mixed_forms_in_series(df[col])
        if mixed:
            for _canon, forms in mixed:
                bad_mask = df[col].astype(str).str.strip().str.lower().isin(forms)
                bad_indices = df.index[bad_mask].tolist()
                parts.append(
                    f"Column {col}: mixed unit forms {forms} at row(s) "
                    f"{_row_numbers_1based(bad_indices)} "
                    f"(use either abbreviation or full form consistently)"
                )
    if not parts:
        return None
    return QCError(
        error_name="Unit Form Inconsistent",
        error_message="\n- ".join(parts),
        error_source="qc",
        qc_name=qc_name,
    )


def check_arm_dose_unit_amt_unit_same_per_row(
    df: pd.DataFrame, qc_name: str
) -> QCError | None:
    """
    ARM_DOSE_UNIT and AMT_UNIT should be the same for each row.
    """
    if df.empty or "ARM_DOSE_UNIT" not in df.columns or "AMT_UNIT" not in df.columns:
        return None
    a = df["ARM_DOSE_UNIT"].astype(str).str.strip().str.lower()
    b = df["AMT_UNIT"].astype(str).str.strip().str.lower()
    a = a.replace("nan", pd.NA).replace("", pd.NA)
    b = b.replace("nan", pd.NA).replace("", pd.NA)
    both_present = a.notna() & b.notna()
    mismatch = both_present & (a != b)
    bad_indices = df.index[mismatch].tolist()
    if not bad_indices:
        return None
    return QCError(
        error_name="ARM_DOSE_UNIT vs AMT_UNIT Mismatch",
        error_message=(
            f"ARM_DOSE_UNIT and AMT_UNIT must be the same per row.\n"
            f"Mismatch at row(s): {_row_numbers_1based(bad_indices)}"
        ),
        error_source="qc",
        qc_name=qc_name,
    )


def check_amt_unit_same_per_group_name(
    df: pd.DataFrame, qc_name: str
) -> QCError | None:
    """
    Within the same GROUP_NAME, AMT_UNIT should be the same.
    """
    if df.empty or "GROUP_NAME" not in df.columns or "AMT_UNIT" not in df.columns:
        return None
    amt = df["AMT_UNIT"].astype(str).str.strip().str.lower()
    amt = amt.replace("nan", pd.NA).replace("", pd.NA)
    grouped = df.groupby("GROUP_NAME", dropna=False)
    bad_indices = []
    for _name, grp in grouped:
        units = amt.loc[grp.index].dropna().unique()
        if len(units) <= 1:
            continue
        bad_indices.extend(grp.index.tolist())
    if not bad_indices:
        return None
    bad_indices = sorted(set(bad_indices))
    return QCError(
        error_name="AMT_UNIT Inconsistent Within GROUP_NAME",
        error_message=(
            f"Within each GROUP_NAME, AMT_UNIT must be the same.\n"
            f"Violations at row(s): {_row_numbers_1based(bad_indices)}"
        ),
        error_source="qc",
        qc_name=qc_name,
    )


def _amt_unit_same_per_arm(df: pd.DataFrame) -> list[int]:
    group_col = None
    for c in ("ARM", "ARM_TRT"):
        if c in df.columns:
            group_col = c
            break
    if group_col is None or "AMT_UNIT" not in df.columns:
        return []
    amt = df["AMT_UNIT"].astype(str).str.strip().str.lower()
    amt = amt.replace("nan", pd.NA).replace("", pd.NA)
    grouped = df.groupby(group_col, dropna=False)
    bad_indices = []
    for _key, grp in grouped:
        units = amt.loc[grp.index].dropna().unique()
        if len(units) <= 1:
            continue
        bad_indices.extend(grp.index.tolist())
    return sorted(set(bad_indices))


def check_amt_unit_same_per_arm(df: pd.DataFrame, qc_name: str) -> QCError | None:
    """
    Within the same ARM or ARM_TRT, AMT_UNIT should be the same.
    """
    if df.empty:
        return None
    bad_indices = _amt_unit_same_per_arm(df)
    if not bad_indices:
        return None
    return QCError(
        error_name="AMT_UNIT Inconsistent Within ARM/ARM_TRT",
        error_message=(
            f"Within each ARM/ARM_TRT, AMT_UNIT must be the same.\n"
            f"Violations at row(s): {_row_numbers_1based(bad_indices)}"
        ),
        error_source="qc",
        qc_name=qc_name,
    )
