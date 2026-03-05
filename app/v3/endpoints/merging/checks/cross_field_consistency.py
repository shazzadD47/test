"""
Quality checks for key 'cross-field-consistency'.

Rules:
- GROUP_NAME, FILE_NAME, YEAR columns must exist, and at least one of
  DOI_URL or CIT_URL must exist.
- JOURNAL, YEAR, TRIAL_NAME, REGID must not change within a GROUP_NAME.
- FILE_NAME must be the same for all rows.
- ARM_NUMBER, ARM_TRT, ARM_DOSE, ARM_DOSE_UNIT must exist and be non-null
  for all rows.
"""

from __future__ import annotations

import pandas as pd

from app.v3.endpoints.merging.schemas import QCError


def _row_numbers_1based(indices: list[int]) -> str:
    if not indices:
        return ""
    if len(indices) <= 10:
        return ", ".join(str(i + 1) for i in indices)
    return ", ".join(str(i + 1) for i in indices[:10]) + f", ... ({len(indices)} rows)"


def check_required_cross_fields_present(
    df: pd.DataFrame, qc_name: str
) -> QCError | None:
    """
    GROUP_NAME, FILE_NAME, YEAR must be present, and at least one of
    DOI_URL or CIT_URL must be present as columns.
    """
    required = {"GROUP_NAME", "FILE_NAME", "YEAR"}
    missing_required = [c for c in required if c not in df.columns]
    has_doi = "DOI_URL" in df.columns
    has_cit = "CIT_URL" in df.columns
    missing_url_info = not (has_doi or has_cit)
    if not missing_required and not missing_url_info:
        return None

    parts: list[str] = []
    if missing_required:
        parts.append(f"missing required columns: {', '.join(sorted(missing_required))}")
    if missing_url_info:
        parts.append("at least one of DOI_URL or CIT_URL must be present")
    return QCError(
        error_name="Missing Cross-Field Columns",
        error_message="\n- ".join(parts),
        error_source="qc",
        qc_name=qc_name,
    )


def check_study_fields_constant_within_group(
    df: pd.DataFrame, qc_name: str
) -> QCError | None:
    """
    JOURNAL, YEAR, TRIAL_NAME, REGID must not change within GROUP_NAME.
    """
    if df.empty or "GROUP_NAME" not in df.columns:
        return None
    fields = ("JOURNAL", "YEAR", "TRIAL_NAME", "REGID")
    fields_present = [f for f in fields if f in df.columns]
    if not fields_present:
        return None

    bad_indices: list[int] = []
    for _g, grp in df.groupby("GROUP_NAME", dropna=False):
        for field in fields_present:
            series = grp[field]
            # normalize strings; keep NaNs as-is
            if series.dtype == object:
                norm = series.astype(str).str.strip().str.lower().where(series.notna())
            else:
                norm = series
            uniques = norm.dropna().unique()
            if len(uniques) > 1:
                bad_indices.extend(grp.index.tolist())
                break

    if not bad_indices:
        return None
    bad_indices = sorted(set(bad_indices))
    fields_str = ", ".join(fields_present)
    return QCError(
        error_name="Study Fields Vary Within GROUP_NAME",
        error_message=(
            f"Fields {fields_str} must be constant within each GROUP_NAME.\n"
            f"Violations at row(s): {_row_numbers_1based(bad_indices)}"
        ),
        error_source="qc",
        qc_name=qc_name,
    )


def check_filename_same_for_all_rows(df: pd.DataFrame, qc_name: str) -> QCError | None:
    """
    FILE_NAME must be the same for all rows.
    """
    if df.empty or "FILE_NAME" not in df.columns:
        return None
    series = df["FILE_NAME"].astype(str).str.strip().str.lower()
    series = series.replace("", pd.NA).dropna()
    if series.empty or series.nunique() <= 1:
        return None
    unique_vals = series.unique().tolist()
    bad_indices = df.index.tolist()
    return QCError(
        error_name="FILE_NAME Inconsistent",
        error_message=(
            "FILE_NAME must be the same for all rows.\n"
            f"Found values: {', '.join(repr(v) for v in unique_vals)}\n"
            f"At row(s): {_row_numbers_1based(bad_indices)}"
        ),
        error_source="qc",
        qc_name=qc_name,
    )


def check_arm_fields_present_and_non_null(
    df: pd.DataFrame, qc_name: str
) -> QCError | None:
    """
    ARM_NUMBER, ARM_TRT, ARM_DOSE, ARM_DOSE_UNIT must exist and be non-null
    for all rows.
    """
    cols = ("ARM_NUMBER", "ARM_TRT", "ARM_DOSE", "ARM_DOSE_UNIT")
    missing_cols = [c for c in cols if c not in df.columns]
    parts: list[str] = []
    bad_indices: list[int] = []

    if missing_cols:
        parts.append(f"missing required columns: {', '.join(sorted(missing_cols))}")
    present_cols = [c for c in cols if c in df.columns]
    for col in present_cols:
        null_mask = df[col].isna()
        if null_mask.any():
            bad_indices.extend(df.index[null_mask].tolist())
            parts.append(
                f"column {col} has null values at row(s) "
                f"{_row_numbers_1based(df.index[null_mask].tolist())}"
            )

    if not parts:
        return None
    bad_indices = sorted(set(bad_indices))
    msg = "\n- ".join(parts)
    if bad_indices:
        msg += f"\nOverall, violations at row(s): {_row_numbers_1based(bad_indices)}"
    return QCError(
        error_name="ARM Fields Missing Or Null",
        error_message=msg,
        error_source="qc",
        qc_name=qc_name,
    )
