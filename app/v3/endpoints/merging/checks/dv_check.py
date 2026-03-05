"""
Quality checks for key 'dv-check'.

Rules:
- For DVID in {1,5,6,7,8,14}, DV_UNIT may be NA; for other DVIDs, DV_UNIT must
  be within an allowed set of units.
- For DVID in {1,5,6,7,8,14}, DV_STAT may be NA; for other DVIDs, DV_STAT must
  be within an allowed set of statistics.
- DV_VAR_STAT must not be NA and must be within an allowed set of variability
  descriptors.
- DV ranges must match ENDPOINT:
  * EASI: 0–72
  * IGA: 0–4
  * SNRS: 0–10
  * PNRS: 0–10
  * DLQI: 0–30
  * sdNRS: 0–10
  * EASI50/EASI75/EASI90: positive percentage up to 100
  * Proportion of patients achieving {EASI, IGA, PNRS, SNRS, DLQI}: positive
    percentage up to 100
  * CFB {EASI, IGA, PNRS, SNRS, DLQI}: numeric with no fixed bounds (can be
    negative or positive).
"""

from __future__ import annotations

import pandas as pd

from app.v3.endpoints.merging.schemas import QCError

DVIDS_ALLOWING_NULL_DV_UNIT_STAT: set[int] = {1, 5, 6, 7, 8, 14}

ALLOWED_DV_UNITS: set[str] = {
    "nmol/l",
    "kg",
    "percentage",
    "mmol/l",
    "mg/dl",
    "ng/ml",
    "mmol/mol",
    "kg/m2",
}

ALLOWED_DV_STATS: set[str] = {
    "geometric mean",
    "mean cfb",
    "mean",
    "percentage",
    "median",
    "ls mean cfb",
    "ls mean",
    "cfb",
    "proportion of patients",
}

ALLOWED_DV_VAR_STATS: set[str] = {
    "95% ci",
    "se",
    "sd",
    "iqr",
    "sem",
    "90% ci",
}


def _row_numbers_1based(indices: list[int]) -> str:
    if not indices:
        return ""
    if len(indices) <= 10:
        return ", ".join(str(i + 1) for i in indices)
    return ", ".join(str(i + 1) for i in indices[:10]) + f", ... ({len(indices)} rows)"


def _numeric_dvid(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def check_dv_unit_vs_dvid(df: pd.DataFrame, qc_name: str) -> QCError | None:
    """
    Validate DV_UNIT based on DVID.
    """
    if df.empty or "DVID" not in df.columns or "DV_UNIT" not in df.columns:
        return None
    dvid = _numeric_dvid(df["DVID"])
    unit = df["DV_UNIT"].astype(str)
    unit_blank = df["DV_UNIT"].isna() | unit.str.strip().eq("")
    unit_norm = unit.str.strip().str.lower()

    allowed_mask = unit_norm.isin(ALLOWED_DV_UNITS)
    bad_indices: list[int] = []

    for idx, (dv, u_blank, u_allowed) in enumerate(zip(dvid, unit_blank, allowed_mask)):
        if pd.isna(dv):
            continue
        dv_int = int(dv)
        if dv_int in DVIDS_ALLOWING_NULL_DV_UNIT_STAT:
            # NA allowed, any non-empty value also allowed (no restriction here)
            continue
        # For other DVIDs: DV_UNIT must be in allowed set (non-blank and allowed)
        if u_blank or not u_allowed:
            bad_indices.append(df.index[idx])

    if not bad_indices:
        return None
    return QCError(
        error_name="DV_UNIT Invalid For DVID",
        error_message=(
            "For DVID not in {1, 5, 6, 7, 8, 14}, DV_UNIT must be one of "
            f"{', '.join(sorted(ALLOWED_DV_UNITS))}.\n"
            f"Violations at row(s): {_row_numbers_1based(bad_indices)}"
        ),
        error_source="qc",
        qc_name=qc_name,
    )


def check_dv_stat_vs_dvid(df: pd.DataFrame, qc_name: str) -> QCError | None:
    """
    Validate DV_STAT based on DVID.
    """
    if df.empty or "DVID" not in df.columns or "DV_STAT" not in df.columns:
        return None
    dvid = _numeric_dvid(df["DVID"])
    stat = df["DV_STAT"].astype(str)
    stat_blank = df["DV_STAT"].isna() | stat.str.strip().eq("")
    stat_norm = stat.str.strip().str.lower()

    allowed_mask = stat_norm.isin(ALLOWED_DV_STATS)
    bad_indices: list[int] = []

    for idx, (dv, s_blank, s_allowed) in enumerate(zip(dvid, stat_blank, allowed_mask)):
        if pd.isna(dv):
            continue
        dv_int = int(dv)
        if dv_int in DVIDS_ALLOWING_NULL_DV_UNIT_STAT:
            # NA allowed, any non-empty value also allowed (no restriction here)
            continue
        # For other DVIDs: DV_STAT must be in allowed set (non-blank and allowed)
        if s_blank or not s_allowed:
            bad_indices.append(df.index[idx])

    if not bad_indices:
        return None
    return QCError(
        error_name="DV_STAT Invalid For DVID",
        error_message=(
            "For DVID not in {1, 5, 6, 7, 8, 14}, DV_STAT must be one of "
            f"{', '.join(sorted(ALLOWED_DV_STATS))}.\n"
            f"Violations at row(s): {_row_numbers_1based(bad_indices)}"
        ),
        error_source="qc",
        qc_name=qc_name,
    )


def check_dv_var_stat(df: pd.DataFrame, qc_name: str) -> QCError | None:
    """
    DV_VAR_STAT must not be NA and must be within the allowed set.
    """
    if df.empty or "DV_VAR_STAT" not in df.columns:
        return None
    stat = df["DV_VAR_STAT"].astype(str)
    blank = df["DV_VAR_STAT"].isna() | stat.str.strip().eq("")
    norm = stat.str.strip().str.lower()
    allowed = norm.isin(ALLOWED_DV_VAR_STATS)

    bad_indices: list[int] = []
    # Blank is always an error
    bad_indices.extend(df.index[blank].tolist())
    # Non-blank but not allowed
    bad_indices.extend(df.index[~blank & ~allowed].tolist())
    bad_indices = sorted(set(bad_indices))
    if not bad_indices:
        return None
    return QCError(
        error_name="DV_VAR_STAT Invalid",
        error_message=(
            "DV_VAR_STAT must be one of "
            f"{', '.join(sorted(ALLOWED_DV_VAR_STATS))} and not NA.\n"
            f"Violations at row(s): {_row_numbers_1based(bad_indices)}"
        ),
        error_source="qc",
        qc_name=qc_name,
    )


ENDPOINT_RANGE_RULES: dict[str, str] = {
    "EASI": "EASI_RANGE",
    "IGA": "IGA_RANGE",
    "SNRS": "SNRS_RANGE",
    "PNRS": "PNRS_RANGE",
    "DLQI": "DLQI_RANGE",
    "sdNRS": "SDNRS_RANGE",
    "EASI50": "PERCENT_RANGE",
    "EASI75": "PERCENT_RANGE",
    "EASI90": "PERCENT_RANGE",
    "Proportion of Patients achieving EASI": "PERCENT_RANGE",
    "Proportion of Patients achieving IGA": "PERCENT_RANGE",
    "Proportion of Patients achieving PNRS": "PERCENT_RANGE",
    "Proportion of Patients achieving SNRS": "PERCENT_RANGE",
    "Proportion of Patients achieving DLQI": "PERCENT_RANGE",
    "CFB EASI": "CFB_RANGE",
    "CFB IGA": "CFB_RANGE",
    "CFB PNRS": "CFB_RANGE",
    "CFB SNRS": "CFB_RANGE",
    "CFB DLQI": "CFB_RANGE",
}


def check_dv_ranges(df: pd.DataFrame, qc_name: str) -> QCError | None:
    """
    Enforce DV ranges based on ENDPOINT.
    """
    if df.empty or "DV" not in df.columns or "ENDPOINT" not in df.columns:
        return None
    dv = pd.to_numeric(df["DV"], errors="coerce")
    ep = df["ENDPOINT"].astype(str).str.strip()

    bad_indices: list[int] = []

    range_bounds: dict[str, tuple[float, float] | None] = {
        "EASI_RANGE": (0, 72),
        "IGA_RANGE": (0, 4),
        "SNRS_RANGE": (0, 10),
        "PNRS_RANGE": (0, 10),
        "DLQI_RANGE": (0, 30),
        "SDNRS_RANGE": (0, 10),
        "PERCENT_RANGE": (0, 100),
        "CFB_RANGE": None,
    }

    for idx, (v, endpoint) in enumerate(zip(dv, ep)):
        if pd.isna(v):
            continue
        rule = ENDPOINT_RANGE_RULES.get(endpoint)
        if rule is None:
            continue
        bounds = range_bounds.get(rule)
        if bounds is None:
            # CFB_RANGE: any numeric (negative or positive) is allowed
            continue
        lower, upper = bounds
        if rule == "PERCENT_RANGE":
            out_of_range = not (lower < v <= upper)
        else:
            out_of_range = not (lower <= v <= upper)
        if out_of_range:
            bad_indices.append(df.index[idx])

    if not bad_indices:
        return None
    return QCError(
        error_name="DV Out of Range For ENDPOINT",
        error_message=(
            "DV values must fall within the expected range for each ENDPOINT "
            f"(e.g. EASI 0–72, IGA 0–4, percentages 0–100, etc.).\n"
            f"Violations at row(s): {_row_numbers_1based(bad_indices)}"
        ),
        error_source="qc",
        qc_name=qc_name,
    )


def check_negative_dv_by_dvid(df: pd.DataFrame, qc_name: str) -> QCError | None:
    """
    Negative DV values should be floored to zero for DVID in
    {1, 5, 6, 7, 8, 14}, but since we cannot modify the CSV here we report
    an error for any negative DV across all DVIDs.
    """
    if df.empty or "DV" not in df.columns or "DVID" not in df.columns:
        return None
    dv = pd.to_numeric(df["DV"], errors="coerce")

    bad_indices: list[int] = []
    for idx, v in enumerate(dv):
        if pd.isna(v):
            continue
        if v < 0:
            bad_indices.append(df.index[idx])

    if not bad_indices:
        return None
    return QCError(
        error_name="Negative DV Values",
        error_message=(
            "DV values must be non-negative. For DVID in {1, 5, 6, 7, 8, 14} "
            "they should be floored to zero.\n"
            f"Negative DV found at row(s): {_row_numbers_1based(bad_indices)}"
        ),
        error_source="qc",
        qc_name=qc_name,
    )
