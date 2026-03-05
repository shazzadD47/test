"""
Quality checks for key 'error-bars' (error bar and baseline consistency).

Assumptions:
- VAR/LCI/UCI may appear as DV_VAR, DV_LCI, DV_UCI or as VAR, LCI, UCI, or both.
  When both exist, their values should be identical cell-wise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.v3.endpoints.merging.schemas import QCError


def _row_numbers_1based(indices: list[int]) -> str:
    if not indices:
        return ""
    if len(indices) <= 10:
        return ", ".join(str(i + 1) for i in indices)
    return ", ".join(str(i + 1) for i in indices[:10]) + f", ... ({len(indices)} rows)"


def _blank_mask(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        s = series.astype(str)
        return series.isna() | s.str.strip().eq("")
    return series.isna()


def _effective_pair(
    df: pd.DataFrame, base: str
) -> tuple[pd.Series | None, pd.Series, list[int]]:
    """
    Return an effective series for a base field (VAR/LCI/UCI) by combining
    DV_{base} and base. Also returns a mask where the effective value is
    considered present (non-blank) and any row indices where both exist but
    disagree.
    """
    dv_col = f"DV_{base}"
    base_col = base
    has_dv = dv_col in df.columns
    has_base = base_col in df.columns

    if not has_dv and not has_base:
        return None, pd.Series(False, index=df.index), []

    if has_dv and has_base:
        dv = df[dv_col]
        base_s = df[base_col]
        # numeric compare where both numeric, else string compare
        dv_num = pd.to_numeric(dv, errors="coerce")
        base_num = pd.to_numeric(base_s, errors="coerce")
        both_num = dv_num.notna() & base_num.notna()
        mismatch_mask = pd.Series(False, index=df.index)
        mismatch_num = both_num & (np.abs(dv_num - base_num) > 1e-9)
        mismatch_mask |= mismatch_num
        dv_str = dv.astype(str).str.strip()
        base_str = base_s.astype(str).str.strip()
        mismatch_str = ~both_num & (dv_str != base_str)
        mismatch_mask |= mismatch_str
        mismatch_indices = df.index[mismatch_mask].tolist()
        effective = dv
    elif has_dv:
        effective = df[dv_col]
        mismatch_indices = []
    else:
        effective = df[base_col]
        mismatch_indices = []

    present_mask = ~_blank_mask(effective)
    return effective, present_mask, mismatch_indices


def check_prefixed_unprefixed_consistency(
    df: pd.DataFrame, qc_name: str
) -> QCError | None:
    """
    For VAR/LCI/UCI, when both DV_* and non-prefixed columns exist, values
    must match cell-wise.
    """
    bases = ("VAR", "LCI", "UCI")
    mismatch_indices: list[int] = []
    mismatch_fields: list[str] = []

    for base in bases:
        _, _, mism = _effective_pair(df, base)
        if mism:
            mismatch_indices.extend(mism)
            mismatch_fields.append(base)

    if not mismatch_indices:
        return None
    mismatch_indices = sorted(set(mismatch_indices))
    return QCError(
        error_name="DV_* vs Non-prefixed Mismatch",
        error_message=(
            "For error-bar fields, DV_VAR/VAR, DV_LCI/LCI, and DV_UCI/UCI must "
            "match when both are present.\n"
            f"Mismatches found for: {', '.join(mismatch_fields)}\n"
            f"At row(s): {_row_numbers_1based(mismatch_indices)}"
        ),
        error_source="qc",
        qc_name=qc_name,
    )


def check_stat_blank_implies_fields_blank(
    df: pd.DataFrame, qc_name: str
) -> QCError | None:
    """
    If DV_VAR_STAT is blank/NA, then DV_LCI/LCI, DV_UCI/UCI, DV_VAR/VAR
    must all be blank.
    """
    if "DV_VAR_STAT" not in df.columns:
        return None
    stat = df["DV_VAR_STAT"]
    stat_blank = _blank_mask(stat)

    eff_var, var_present, _ = _effective_pair(df, "VAR")
    eff_lci, lci_present, _ = _effective_pair(df, "LCI")
    eff_uci, uci_present, _ = _effective_pair(df, "UCI")

    present_when_stat_blank = stat_blank & (
        (eff_var is not None and var_present)
        | (eff_lci is not None and lci_present)
        | (eff_uci is not None and uci_present)
    )
    bad_indices = df.index[present_when_stat_blank].tolist()
    if not bad_indices:
        return None
    return QCError(
        error_name="DV_VAR_STAT Blank But Error Bars Present",
        error_message=(
            "DV_VAR_STAT is blank/NA, so DV_LCI/LCI, DV_UCI/UCI, and DV_VAR/VAR "
            f"must also be blank.\n"
            f"Violations at row(s): {_row_numbers_1based(bad_indices)}"
        ),
        error_source="qc",
        qc_name=qc_name,
    )


def check_ci_iqr_requires_bounds_and_vice_versa(
    df: pd.DataFrame, qc_name: str
) -> QCError | None:
    """
    If DV_VAR_STAT contains CI/IQR, LCI/UCI must be present; if LCI/UCI are
    present, DV_VAR_STAT must contain CI or IQR.
    """
    if "DV_VAR_STAT" not in df.columns:
        return None
    stat = df["DV_VAR_STAT"].astype(str).str.lower()
    blank = _blank_mask(df["DV_VAR_STAT"])
    is_ci_iqr = ~blank & (stat.str.contains("ci") | stat.str.contains("iqr"))

    _, lci_present, _ = _effective_pair(df, "LCI")
    _, uci_present, _ = _effective_pair(df, "UCI")
    bounds_present = lci_present & uci_present

    # CI/IQR but missing bounds
    missing_bounds = is_ci_iqr & ~bounds_present
    # Bounds present but DV_VAR_STAT not CI/IQR
    bounds_without_stat = bounds_present & ~is_ci_iqr

    bad_indices = df.index[missing_bounds | bounds_without_stat].tolist()
    if not bad_indices:
        return None
    return QCError(
        error_name="CI/IQR and Bounds Inconsistent",
        error_message=(
            "When DV_VAR_STAT indicates CI/IQR, both DV_LCI/LCI and DV_UCI/UCI "
            "must be present, and when both bounds are present DV_VAR_STAT must "
            f"contain CI or IQR.\n"
            f"Violations at row(s): {_row_numbers_1based(bad_indices)}"
        ),
        error_source="qc",
        qc_name=qc_name,
    )


def check_other_stat_implies_var_only(df: pd.DataFrame, qc_name: str) -> QCError | None:
    """
    If DV_VAR_STAT is non-blank and does not contain CI or IQR, DV_VAR/VAR
    must be present and LCI/UCI must be blank.
    """
    if "DV_VAR_STAT" not in df.columns:
        return None
    stat = df["DV_VAR_STAT"].astype(str).str.lower()
    blank = _blank_mask(df["DV_VAR_STAT"])
    is_ci_iqr = ~blank & (stat.str.contains("ci") | stat.str.contains("iqr"))
    other_stat = ~blank & ~is_ci_iqr

    _, var_present, _ = _effective_pair(df, "VAR")
    _, lci_present, _ = _effective_pair(df, "LCI")
    _, uci_present, _ = _effective_pair(df, "UCI")

    bad_mask = (other_stat & ~var_present) | (other_stat & (lci_present | uci_present))
    bad_indices = df.index[bad_mask].tolist()
    if not bad_indices:
        return None
    return QCError(
        error_name="DV_VAR_STAT Other Value Inconsistent",
        error_message=(
            "When DV_VAR_STAT does not contain CI or IQR but has another value, "
            "DV_VAR/VAR must be present and DV_LCI/LCI and DV_UCI/UCI must be "
            f"blank.\n"
            f"Violations at row(s): {_row_numbers_1based(bad_indices)}"
        ),
        error_source="qc",
        qc_name=qc_name,
    )


def check_var_non_negative(df: pd.DataFrame, qc_name: str) -> QCError | None:
    """
    DV_VAR/VAR must not be negative.
    """
    eff_var, _, _ = _effective_pair(df, "VAR")
    if eff_var is None:
        return None
    var_num = pd.to_numeric(eff_var, errors="coerce")
    bad_mask = var_num < 0
    bad_indices = df.index[bad_mask].tolist()
    if not bad_indices:
        return None
    return QCError(
        error_name="DV_VAR Negative",
        error_message=(
            "DV_VAR/VAR must not be negative.\n"
            f"Violations at row(s): {_row_numbers_1based(bad_indices)}"
        ),
        error_source="qc",
        qc_name=qc_name,
    )


def check_male_female_percentages(df: pd.DataFrame, qc_name: str) -> QCError | None:
    """
    MALE_PERCENTAGE + FEMALE_PERCENTAGE must be 100 within ±1 tolerance.
    """
    if "MALE_PERCENTAGE" not in df.columns or "FEMALE_PERCENTAGE" not in df.columns:
        return None
    male = pd.to_numeric(df["MALE_PERCENTAGE"], errors="coerce")
    female = pd.to_numeric(df["FEMALE_PERCENTAGE"], errors="coerce")
    valid = male.notna() & female.notna()
    if not valid.any():
        return None
    total = male + female
    bad_mask = valid & (np.abs(total - 100) > 1)
    bad_indices = df.index[bad_mask].tolist()
    if not bad_indices:
        return None
    return QCError(
        error_name="Male/Female Percentages Invalid",
        error_message=(
            "MALE_PERCENTAGE + FEMALE_PERCENTAGE must be 100 (±1 tolerance).\n"
            f"Violations at row(s): {_row_numbers_1based(bad_indices)}"
        ),
        error_source="qc",
        qc_name=qc_name,
    )


def _find_unit_column(df: pd.DataFrame, label: str) -> str | None:
    label_l = label.lower()
    for col in df.columns:
        col_l = col.lower()
        if label_l in col_l and "unit" in col_l:
            return col
    return None


def check_baseline_units(df: pd.DataFrame, qc_name: str) -> QCError | None:
    """
    AGE unit must be years, BMI unit kg/m2, BW unit kg. Columns are matched
    by containing AGE/BMI/BW and UNIT in their names.
    """
    parts: list[str] = []
    bad_indices: list[int] = []

    age_unit_col = _find_unit_column(df, "AGE")
    bmi_unit_col = _find_unit_column(df, "BMI")
    bw_unit_col = _find_unit_column(df, "BW")

    if age_unit_col is not None:
        vals = df[age_unit_col].astype(str).str.strip().str.lower()
        invalid = ~vals.isin({"year", "years"})
        if invalid.any():
            idx = df.index[invalid].tolist()
            bad_indices.extend(idx)
            parts.append(
                f"{age_unit_col} must be in years; invalid values at row(s) "
                f"{_row_numbers_1based(idx)}"
            )

    if bmi_unit_col is not None:
        vals = df[bmi_unit_col].astype(str).str.strip().str.lower()
        invalid = ~vals.isin({"kg/m2"})
        if invalid.any():
            idx = df.index[invalid].tolist()
            bad_indices.extend(idx)
            parts.append(
                f"{bmi_unit_col} must be kg/m2; invalid values at row(s) "
                f"{_row_numbers_1based(idx)}"
            )

    if bw_unit_col is not None:
        vals = df[bw_unit_col].astype(str).str.strip().str.lower()
        invalid = ~vals.isin({"kg"})
        if invalid.any():
            idx = df.index[invalid].tolist()
            bad_indices.extend(idx)
            parts.append(
                f"{bw_unit_col} must be kg; invalid values at row(s) "
                f"{_row_numbers_1based(idx)}"
            )

    if not parts:
        return None
    bad_indices = sorted(set(bad_indices))
    return QCError(
        error_name="Baseline Units Invalid",
        error_message="\n- ".join(parts),
        error_source="qc",
        qc_name=qc_name,
    )


def _first_existing(df: pd.DataFrame, names: tuple[str, str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def check_baseline_ranges(df: pd.DataFrame, qc_name: str) -> QCError | None:
    """
    Ranges: AGE/AGE_BL in [0, 120], BMI/BMI_BL in [5, 100], BW/BW_BL in [1, 150].
    """
    parts: list[str] = []
    bad_indices: list[int] = []

    age_col = _first_existing(df, ("AGE_BL", "AGE"))
    bmi_col = _first_existing(df, ("BMI_BL", "BMI"))
    bw_col = _first_existing(df, ("BW_BL", "BW"))

    if age_col is not None:
        vals = pd.to_numeric(df[age_col], errors="coerce")
        bad = vals.notna() & ((vals < 0) | (vals > 120))
        if bad.any():
            idx = df.index[bad].tolist()
            bad_indices.extend(idx)
            parts.append(
                f"{age_col} must be in range 0–120; violations at row(s) "
                f"{_row_numbers_1based(idx)}"
            )

    if bmi_col is not None:
        vals = pd.to_numeric(df[bmi_col], errors="coerce")
        bad = vals.notna() & ((vals < 5) | (vals > 100))
        if bad.any():
            idx = df.index[bad].tolist()
            bad_indices.extend(idx)
            parts.append(
                f"{bmi_col} must be in range 5–100; violations at row(s) "
                f"{_row_numbers_1based(idx)}"
            )

    if bw_col is not None:
        vals = pd.to_numeric(df[bw_col], errors="coerce")
        bad = vals.notna() & ((vals < 1) | (vals > 150))
        if bad.any():
            idx = df.index[bad].tolist()
            bad_indices.extend(idx)
            parts.append(
                f"{bw_col} must be in range 1–150; violations at row(s) "
                f"{_row_numbers_1based(idx)}"
            )

    if not parts:
        return None
    bad_indices = sorted(set(bad_indices))
    return QCError(
        error_name="Baseline Ranges Invalid",
        error_message="\n- ".join(parts),
        error_source="qc",
        qc_name=qc_name,
    )
