"""
Quality checks for key 'nonmem-structure': EVID/DV, EVID/AMT, MDV vs EVID,
EVID/ENDPOINT, EVID/DV_UNIT consistency, and DV numeric when present.
"""

from __future__ import annotations

import pandas as pd

from app.v3.endpoints.merging.schemas import QCError

EVID_COLUMN = "EVID"


def _row_numbers_1based(indices: list[int]) -> str:
    if not indices:
        return ""
    if len(indices) <= 10:
        return ", ".join(str(i + 1) for i in indices)
    return ", ".join(str(i + 1) for i in indices[:10]) + f", ... ({len(indices)} rows)"


def _is_evid_dose(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip() == "1"


def _is_null(series: pd.Series) -> pd.Series:
    return series.isna()


def check_evid_dv(df: pd.DataFrame, qc_name: str) -> QCError | None:
    """
    If EVID==1 then DV should be null; else DV should not be null.
    """
    if df.empty or EVID_COLUMN not in df.columns or "DV" not in df.columns:
        return None
    evid_dose = _is_evid_dose(df[EVID_COLUMN])
    dv_null = _is_null(df["DV"])
    bad_dose = evid_dose & ~dv_null
    bad_obs = ~evid_dose & dv_null
    bad_indices = df.index[bad_dose | bad_obs].tolist()
    if not bad_indices:
        return None
    row_nums = _row_numbers_1based(bad_indices)
    return QCError(
        error_name="EVID vs DV",
        error_message=(
            f"Column EVID: dose rows (EVID=1) must have null DV; "
            f"observation rows (EVID≠1) must have non-null DV.\n"
            f"Violations at row(s): {row_nums}"
        ),
        error_source="qc",
        qc_name=qc_name,
    )


def check_evid_amt(df: pd.DataFrame, qc_name: str) -> QCError | None:
    """
    If EVID==1 then AMT must be present; else AMT should be null.
    """
    if df.empty or EVID_COLUMN not in df.columns or "AMT" not in df.columns:
        return None
    evid_dose = _is_evid_dose(df[EVID_COLUMN])
    amt_null = _is_null(df["AMT"])
    bad_dose = evid_dose & amt_null
    bad_obs = ~evid_dose & ~amt_null
    bad_indices = df.index[bad_dose | bad_obs].tolist()
    if not bad_indices:
        return None
    row_nums = _row_numbers_1based(bad_indices)
    return QCError(
        error_name="EVID vs AMT",
        error_message=(
            f"Column AMT: dose rows (EVID=1) must have non-null AMT; "
            f"observation rows (EVID≠1) must have null AMT.\n"
            f"Violations at row(s): {row_nums}"
        ),
        error_source="qc",
        qc_name=qc_name,
    )


def check_mdv_evid_pattern(df: pd.DataFrame, qc_name: str) -> QCError | None:
    """
    MDV should match EVID: EVID=1 => MDV=1, EVID≠1 => MDV=0.
    """
    if df.empty or EVID_COLUMN not in df.columns or "MDV" not in df.columns:
        return None
    evid_str = df[EVID_COLUMN].astype(str).str.strip()
    mdv_str = df["MDV"].astype(str).str.strip()
    expect_mdv_1 = evid_str == "1"
    expect_mdv_0 = evid_str != "1"
    bad = (expect_mdv_1 & (mdv_str != "1")) | (expect_mdv_0 & (mdv_str != "0"))
    bad_indices = df.index[bad].tolist()
    if not bad_indices:
        return None
    row_nums = _row_numbers_1based(bad_indices)
    return QCError(
        error_name="MDV vs EVID",
        error_message=(
            f"Column MDV must match EVID: EVID=1 => MDV=1, EVID≠1 => MDV=0.\n"
            f"Violations at row(s): {row_nums}"
        ),
        error_source="qc",
        qc_name=qc_name,
    )


def check_evid_endpoint(df: pd.DataFrame, qc_name: str) -> QCError | None:
    """
    If EVID==1 then ENDPOINT should be null.
    """
    if df.empty or EVID_COLUMN not in df.columns or "ENDPOINT" not in df.columns:
        return None
    evid_dose = _is_evid_dose(df[EVID_COLUMN])
    endpoint_null = _is_null(df["ENDPOINT"])
    bad_dose = evid_dose & ~endpoint_null
    bad_indices = df.index[bad_dose].tolist()
    if not bad_indices:
        return None
    row_nums = _row_numbers_1based(bad_indices)
    return QCError(
        error_name="EVID vs ENDPOINT",
        error_message=(
            "Column ENDPOINT: dose rows (EVID=1) must have null ENDPOINT.\n"
            f"Violations at row(s): {row_nums}"
        ),
        error_source="qc",
        qc_name=qc_name,
    )


def check_evid_dv_unit(df: pd.DataFrame, qc_name: str) -> QCError | None:
    """
    If EVID==1 then DV_UNIT must be null.
    """
    if df.empty or EVID_COLUMN not in df.columns or "DV_UNIT" not in df.columns:
        return None
    evid_dose = _is_evid_dose(df[EVID_COLUMN])
    dv_unit_null = _is_null(df["DV_UNIT"])
    bad_dose = evid_dose & ~dv_unit_null
    bad_indices = df.index[bad_dose].tolist()
    if not bad_indices:
        return None
    row_nums = _row_numbers_1based(bad_indices)
    return QCError(
        error_name="EVID vs DV_UNIT",
        error_message=(
            "Column DV_UNIT: dose rows (EVID=1) must have null DV_UNIT.\n"
            f"Violations at row(s): {row_nums}"
        ),
        error_source="qc",
        qc_name=qc_name,
    )


def check_dv_numeric(df: pd.DataFrame, qc_name: str) -> QCError | None:
    """
    If DV is not null then it must be numeric.
    """
    if df.empty or "DV" not in df.columns:
        return None
    dv = df["DV"]
    numeric = pd.to_numeric(dv, errors="coerce")
    non_numeric_mask = dv.notna() & numeric.isna()
    non_numeric_indices = df.index[non_numeric_mask].tolist()
    if not non_numeric_indices:
        return None
    return QCError(
        error_name="DV Must Be Numeric",
        error_message=(
            "Column DV: when present (not null), values must be numeric.\n"
            f"Non-numeric values at row(s): "
            f"{_row_numbers_1based(non_numeric_indices)}"
        ),
        error_source="qc",
        qc_name=qc_name,
    )
