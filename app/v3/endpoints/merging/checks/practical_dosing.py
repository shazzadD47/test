"""
Quality checks for key 'practical-dosing'.

Rules (per group and per dosing rows where applicable):
- If total AMT == 0, there cannot be more than one dosing row (EVENT_TYPE=DOSE).
- For dosing rows with AMT > 0, ARM_DUR must equal SUM(II * (ADDL + 1)).
- For dosing rows, when ARM_TIME increases, AMT must be non-decreasing.
- For dosing rows, II and II_UNIT must be the same within a group.
- ADDL must be in the range 0 <= ADDL <= 729.
- ROUTE must be non-null for all rows and identical across all rows.
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


def _group_column(df: pd.DataFrame) -> str | None:
    for col in ("GROUP_NAME", "GROUP"):
        if col in df.columns:
            return col
    return None


def _dose_mask(df: pd.DataFrame) -> pd.Series:
    if "EVENT_TYPE" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["EVENT_TYPE"].astype(str).str.upper() == "DOSE"


def check_zero_total_amt_multiple_doses(
    df: pd.DataFrame, qc_name: str
) -> QCError | None:
    """
    Within a group, if total AMT == 0, there cannot be more than one dosing row.
    """
    group_col = _group_column(df)
    if df.empty or group_col is None or "AMT" not in df.columns:
        return None
    dose_m = _dose_mask(df)
    if not dose_m.any():
        return None

    bad_indices: list[int] = []
    for _, grp in df[dose_m].groupby(group_col, dropna=False):
        if grp.empty or len(grp) <= 1:
            continue
        amt = pd.to_numeric(grp["AMT"], errors="coerce").fillna(0)
        if amt.sum() == 0:
            bad_indices.extend(grp.index.tolist())

    if not bad_indices:
        return None
    bad_indices = sorted(set(bad_indices))
    return QCError(
        error_name="Zero Total AMT With Multiple Doses",
        error_message=(
            "Within each group, if total AMT is 0 there must be at most one "
            "dosing row (EVENT_TYPE=DOSE).\n"
            f"Violations at row(s): {_row_numbers_1based(bad_indices)}"
        ),
        error_source="qc",
        qc_name=qc_name,
    )


def check_arm_dur_equals_sum_of_ii_addl(
    df: pd.DataFrame, qc_name: str
) -> QCError | None:
    """
    For dosing rows with AMT > 0, ARM_DUR must equal SUM(II * (ADDL + 1)) per group.
    """
    group_col = _group_column(df)
    required = {"AMT", "ARM_DUR", "II", "ADDL"}
    if df.empty or group_col is None or not required.issubset(df.columns):
        return None

    dose_m = _dose_mask(df)
    bad_indices: list[int] = []
    for _, grp in df[dose_m].groupby(group_col, dropna=False):
        if grp.empty:
            continue
        amt = pd.to_numeric(grp["AMT"], errors="coerce")
        pos_mask = amt > 0
        grp_pos = grp[pos_mask]
        if grp_pos.empty:
            continue
        ii = pd.to_numeric(grp_pos["II"], errors="coerce")
        addl = pd.to_numeric(grp_pos["ADDL"], errors="coerce")
        arm_dur = pd.to_numeric(grp_pos["ARM_DUR"], errors="coerce")
        if ii.isna().all() or addl.isna().all() or arm_dur.isna().all():
            continue
        expected = (ii * (addl + 1)).sum()
        mismatch = arm_dur != expected
        if mismatch.any():
            bad_indices.extend(grp_pos.index[mismatch].tolist())

    if not bad_indices:
        return None
    bad_indices = sorted(set(bad_indices))
    return QCError(
        error_name="ARM_DUR vs II and ADDL",
        error_message=(
            "For dosing rows with AMT > 0, ARM_DUR must equal the sum of "
            "II * (ADDL + 1) within each group.\n"
            f"Violations at row(s): {_row_numbers_1based(bad_indices)}"
        ),
        error_source="qc",
        qc_name=qc_name,
    )


def check_amt_non_decreasing_with_time(
    df: pd.DataFrame, qc_name: str
) -> QCError | None:
    """
    For dosing rows within a group, when ARM_TIME increases AMT must be
    non-decreasing.
    """
    group_col = _group_column(df)
    if (
        df.empty
        or group_col is None
        or "ARM_TIME" not in df.columns
        or "AMT" not in df.columns
    ):
        return None

    dose_m = _dose_mask(df)
    if not dose_m.any():
        return None

    bad_indices: list[int] = []
    for _, grp in df[dose_m].groupby(group_col, dropna=False):
        if grp.empty:
            continue
        arm_time = pd.to_numeric(grp["ARM_TIME"], errors="coerce")
        amt = pd.to_numeric(grp["AMT"], errors="coerce")
        valid = arm_time.notna() & amt.notna()
        grp_valid = grp[valid].copy()
        if grp_valid.empty:
            continue
        arm_time = arm_time[valid]
        amt = amt[valid]
        order = arm_time.argsort()
        arm_time_sorted = arm_time.iloc[order].reset_index(drop=True)
        amt_sorted = amt.iloc[order].reset_index(drop=True)
        orig_indices = grp_valid.index[order]
        for i in range(1, len(arm_time_sorted)):
            if (
                arm_time_sorted[i] > arm_time_sorted[i - 1]
                and amt_sorted[i] < amt_sorted[i - 1]
            ):
                bad_indices.append(orig_indices[i])

    if not bad_indices:
        return None
    bad_indices = sorted(set(bad_indices))
    return QCError(
        error_name="Non-monotonic AMT Over Time",
        error_message=(
            "Within each group for dosing rows, when ARM_TIME increases "
            "AMT must be non-decreasing.\n"
            f"Violations at row(s): {_row_numbers_1based(bad_indices)}"
        ),
        error_source="qc",
        qc_name=qc_name,
    )


def check_ii_and_ii_unit_consistent_within_group(
    df: pd.DataFrame, qc_name: str
) -> QCError | None:
    """
    For dosing rows within a group, II and II_UNIT must be the same.
    """
    group_col = _group_column(df)
    if (
        df.empty
        or group_col is None
        or "II" not in df.columns
        or "II_UNIT" not in df.columns
    ):
        return None

    dose_m = _dose_mask(df)
    if not dose_m.any():
        return None

    bad_indices: list[int] = []
    for _, grp in df[dose_m].groupby(group_col, dropna=False):
        if grp.empty:
            continue
        ii = pd.to_numeric(grp["II"], errors="coerce")
        ii_unit = grp["II_UNIT"].astype(str).str.strip().str.lower()
        ii_unit = ii_unit.replace("nan", pd.NA).replace("", pd.NA)
        ii_vals = ii.dropna().unique()
        ii_unit_vals = ii_unit.dropna().unique()
        if len(ii_vals) > 1 or len(ii_unit_vals) > 1:
            bad_indices.extend(grp.index.tolist())

    if not bad_indices:
        return None
    bad_indices = sorted(set(bad_indices))
    return QCError(
        error_name="II and II_UNIT Inconsistent Within Group",
        error_message=(
            "Within each group for dosing rows, II and II_UNIT must be "
            "consistent.\n"
            f"Violations at row(s): {_row_numbers_1based(bad_indices)}"
        ),
        error_source="qc",
        qc_name=qc_name,
    )


def check_addl_range(df: pd.DataFrame, qc_name: str) -> QCError | None:
    """
    ADDL must be in the range 0 <= ADDL <= 729.
    """
    if df.empty or "ADDL" not in df.columns:
        return None
    addl = pd.to_numeric(df["ADDL"], errors="coerce")
    non_null = df["ADDL"].notna()
    bad_mask = non_null & ((addl < 0) | (addl > 729))
    bad_indices = df.index[bad_mask].tolist()
    if not bad_indices:
        return None
    return QCError(
        error_name="ADDL Out of Range",
        error_message=(
            "ADDL must be numeric and in the range 0 <= ADDL <= 729.\n"
            f"Violations at row(s): {_row_numbers_1based(bad_indices)}"
        ),
        error_source="qc",
        qc_name=qc_name,
    )


def check_route_non_null_and_same(df: pd.DataFrame, qc_name: str) -> QCError | None:
    """
    ROUTE must have no missing values and all rows must share the same value.
    """
    if df.empty or "ROUTE" not in df.columns:
        return None
    route = df["ROUTE"].astype(str).str.strip()
    null_mask = route.eq("") | df["ROUTE"].isna()
    non_null = route[~null_mask].str.lower()
    parts: list[str] = []
    if null_mask.any():
        parts.append(
            f"missing/NA ROUTE at row(s) "
            f"{_row_numbers_1based(df.index[null_mask].tolist())}"
        )
    if non_null.empty:
        if not parts:
            return None
    else:
        unique_vals = non_null.unique().tolist()
        if len(unique_vals) > 1:
            bad_indices = df.index[~null_mask].tolist()
            parts.append(
                "ROUTE must be the same for all rows; found values "
                f"{', '.join(repr(v) for v in unique_vals)} at row(s) "
                f"{_row_numbers_1based(bad_indices)}"
            )
    if not parts:
        return None
    return QCError(
        error_name="ROUTE Missing Or Inconsistent",
        error_message="\n- ".join(parts),
        error_source="qc",
        qc_name=qc_name,
    )
