"""
Quality checks for key 'key-integrity-duplicates': duplicate rows,
duplicate DV per (FILE_NAME, GROUP_NAME, ARM_NUMBER, ARM_TIME, DVID), and
text inconsistency (space/case) per column.
"""

from __future__ import annotations

import pandas as pd

from app.v3.endpoints.merging.schemas import QCError

COMPOSITE_KEY_COLUMNS = (
    "FILE_NAME",
    "GROUP_NAME",
    "ARM_NUMBER",
    "ARM_TIME",
    "DVID",
)
DV_COLUMN = "DV"


def _row_numbers_1based(indices: list[int]) -> str:
    if not indices:
        return ""
    if len(indices) <= 10:
        return ", ".join(str(i + 1) for i in indices)
    return ", ".join(str(i + 1) for i in indices[:10]) + f", ... ({len(indices)} rows)"


def check_identical_rows(df: pd.DataFrame, qc_name: str) -> QCError | None:
    """
    No two rows can be identical. Reports one QCError with all duplicate
    row numbers listed in the message.
    """
    if df.empty or len(df) < 2:
        return None
    duplicated = df.duplicated(keep=False)
    if not duplicated.any():
        return None
    dup_indices = df.index[duplicated].tolist()
    row_nums = _row_numbers_1based(dup_indices)
    return QCError(
        error_name="Duplicate Rows",
        error_message=f"Identical rows found.\nRow number(s): {row_nums}",
        error_source="qc",
        qc_name=qc_name,
    )


def check_duplicate_dv_per_composite_key(
    df: pd.DataFrame, qc_name: str
) -> QCError | None:
    """
    For each (FILE_NAME, GROUP_NAME, ARM_NUMBER, ARM_TIME, DVID), DV must be unique.
    Reports one QCError listing all composite keys and row numbers where duplicate
    DV occurs.
    """
    missing = [c for c in COMPOSITE_KEY_COLUMNS + (DV_COLUMN,) if c not in df.columns]
    if missing:
        return None
    grouped = df.groupby(list(COMPOSITE_KEY_COLUMNS), dropna=False)
    parts = []
    for key_vals, grp in grouped:
        if len(COMPOSITE_KEY_COLUMNS) == 1 and not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        dv = grp[DV_COLUMN]
        dup_mask = dv.duplicated(keep=False)
        if not dup_mask.any():
            continue
        dup_indices = grp.index[dup_mask].tolist()
        row_nums = _row_numbers_1based(dup_indices)
        key_str = ", ".join(f"{c}={v}" for c, v in zip(COMPOSITE_KEY_COLUMNS, key_vals))
        vals_str = ", ".join(str(v) for v in dv[dup_mask].unique())
        parts.append(
            f"({key_str}) duplicate DV at row(s) {row_nums} with value(s) {vals_str}"
        )
    if not parts:
        return None
    return QCError(
        error_name="Duplicate DV per Composite Key",
        error_message=(
            "For (FILE_NAME, GROUP_NAME, ARM_NUMBER, ARM_TIME, DVID), "
            "same DV must not repeat:\n- " + "\n- ".join(parts)
        ),
        error_source="qc",
        qc_name=qc_name,
    )


def _normalize_text(s: str) -> str:
    return str(s).strip().lower() if pd.notna(s) and str(s).strip() else ""


def check_text_inconsistency_per_column(
    df: pd.DataFrame, qc_name: str
) -> QCError | None:
    """
    Per column, values that differ only by whitespace or case (e.g. mg, MG, Mg)
    are inconsistent; report column name, row numbers, and the conflicting values.
    """
    if df.empty:
        return None
    parts = []
    for col in df.columns:
        series = df[col].astype(str)
        normalized = series.map(_normalize_text)
        by_norm = series.groupby(normalized, dropna=False)
        for _norm_val, grp in by_norm:
            if _norm_val == "":
                continue
            unique_raw = grp.unique().tolist()
            if len(unique_raw) < 2:
                continue
            indices = grp.index.tolist()
            row_nums = _row_numbers_1based(indices)
            vals_str = ", ".join(repr(v) for v in unique_raw[:5])
            if len(unique_raw) > 5:
                vals_str += f", ... ({len(unique_raw)} variants)"
            parts.append(
                f"Column '{col}' at row(s) {row_nums}: inconsistent values "
                f"(differ by space/case): {vals_str}"
            )
    if not parts:
        return None
    return QCError(
        error_name="Text Inconsistency (Space/Case)",
        error_message="\n- ".join(parts),
        error_source="qc",
        qc_name=qc_name,
    )
