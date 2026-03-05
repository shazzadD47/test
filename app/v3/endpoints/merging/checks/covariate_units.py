"""
Quality checks for key 'covariate-units'.

Validates that covariate columns contain only their expected allowed values.
"""

from __future__ import annotations

import pandas as pd

from app.v3.endpoints.merging.schemas import QCError

COVARIATE_ALLOWED_VALUES: dict[str, set[str]] = {
    "AGE_BL_UNIT": {"years"},
    "AGE_BL_VAR_STAT": {"SD", "IQR", "range", "CV", "95%CI"},
    "AGE_BL_STAT": {"mean", "median", "geometric mean"},
    "MALE_PERCENT_BL_UNIT": {"percentage"},
    "FEMALE_PERCENT_BL_UNIT": {"percentage"},
    "BMI_BL_UNIT": {"kg/m2"},
    "BMI_BL_STAT": {"mean", "median", "geometric mean"},
    "BMI_BL_VAR_STAT": {"SD", "IQR", "range", "CV", "95%CI"},
    "BW_BL_UNIT": {"kg"},
    "BW_BL_STAT": {"mean", "median", "geometric mean"},
    "BW_BL_VAR_STAT": {"SD", "IQR", "range", "CV", "95%CI"},
    "HT_BL_UNIT": {"cm"},
    "HT_BL_STAT": {"mean", "median", "geometric mean"},
    "HT_BL_VAR_STAT": {"SD", "IQR", "range", "CV", "95%CI"},
    "ETHN_HISPANIC_PERC": {"percentage"},
    "RACE_ASIAN_PERC": {"percentage"},
    "RACE_WHITE_PERC": {"percentage"},
    "RACE_BLACK_AA_PERC": {"percentage"},
    "RACE_OTHER_PERC": {"percentage"},
    "RACE_NON_REPORT_PERC": {"percentage"},
    "TCI": {"percentage"},
    "TCS": {"percentage"},
    "TCI_OR_TCS_BL": {"percentage"},
    "TCI_AND_TCS_BL": {"percentage"},
    "AD_DUR_BL_UNIT": {"years"},
    "AD_DUR_STAT": {"mean", "median", "geometric mean"},
    "AD_DUR_VAR_STAT": {"SD", "IQR", "range", "CV", "95%CI"},
    "AGE_OF_ONSET_SUB18": {"percentage"},
    "AGE_OF_ONSET_SUPRA18": {"percentage"},
    "EASI_BL_STAT": {"mean", "median", "geometric mean"},
    "EASI_VAR_STAT": {"SD", "IQR", "range", "CV", "95%CI"},
    "PNRS_BL_STAT": {"mean", "median", "geometric mean"},
    "PNRS_VAR_STAT": {"SD", "IQR", "range", "CV", "95%CI"},
    "SNRS_BL_STAT": {"mean", "median", "geometric mean"},
    "SNRA_VAR_STAT": {"SD", "IQR", "range", "CV", "95%CI"},
    "IGA_BL_STAT": {"mean", "median", "geometric mean"},
    "IGA_VAR_STAT": {"SD", "IQR", "range", "CV", "95%CI"},
    "AFF_BSA_BL_UNIT": {"percentage"},
    "AFF_BSA_BL_STAT": {"mean", "median", "geometric mean"},
    "AFF_BSA_VAR_STAT": {"SD", "IQR", "range", "CV", "95%CI"},
    "SCORAD_BL_STAT": {"mean", "median", "geometric mean"},
    "SCORAD_VAR_STAT": {"SD", "IQR", "range", "CV", "95%CI"},
    "POEM_BL_STAT": {"mean", "median", "geometric mean"},
    "POEM_VAR_STAT": {"SD", "IQR", "range", "CV", "95%CI"},
    "DLQI_BL_STAT": {"mean", "median", "geometric mean"},
    "DLQI_VAR_STAT": {"SD", "IQR", "range", "CV", "95%CI"},
    "ALLERGIC_AD": {"percentage"},
    "NONALLERGIC_AD": {"percentage"},
    "IGE_BL_UNIT": {"U/mL"},
    "IGE_BL_STAT": {"mean", "median", "geometric mean"},
    "IGE_VAR_STAT": {"SD", "IQR", "range", "CV", "95%CI"},
}


def _row_numbers_1based(indices: list[int]) -> str:
    if not indices:
        return ""
    if len(indices) <= 10:
        return ", ".join(str(i + 1) for i in indices)
    return ", ".join(str(i + 1) for i in indices[:10]) + f", ... ({len(indices)} rows)"


def check_covariate_units(df: pd.DataFrame, qc_name: str) -> QCError | None:
    """Validate that covariate columns contain only their allowed values."""
    if df.empty:
        return None

    mismatches: list[str] = []

    for col, allowed in COVARIATE_ALLOWED_VALUES.items():
        if col not in df.columns:
            continue

        values = df[col].astype(str)
        blank = df[col].isna() | values.str.strip().eq("")
        stripped = values.str.strip()

        bad_mask = ~blank & ~stripped.isin(allowed)
        if not bad_mask.any():
            continue

        bad_indices = df.index[bad_mask].tolist()
        bad_vals = sorted(stripped[bad_mask].unique())

        allowed_lower = {v.lower() for v in allowed}
        case_mismatched = [v for v in bad_vals if v.lower() in allowed_lower]
        fully_invalid = [v for v in bad_vals if v.lower() not in allowed_lower]

        parts: list[str] = []
        if case_mismatched:
            parts.append(f"case mismatch {case_mismatched} (expected exact case)")
        if fully_invalid:
            parts.append(f"invalid {fully_invalid}")

        mismatches.append(
            f"{col}: {', '.join(parts)} at row(s) "
            f"{_row_numbers_1based(bad_indices)} "
            f"(allowed: {', '.join(sorted(allowed))})"
        )

    if not mismatches:
        return None

    return QCError(
        error_name="Covariate Value Mismatch",
        error_message=(
            "Covariate columns contain values outside their allowed sets:\n- "
            + "\n- ".join(mismatches)
        ),
        error_source="qc",
        qc_name=qc_name,
    )


def check_covariate_missing_values(df: pd.DataFrame, qc_name: str) -> QCError | None:
    """Flag rows where covariate columns have missing, NA, or NULL values."""
    if df.empty:
        return None

    missing_reports: list[str] = []

    for col in COVARIATE_ALLOWED_VALUES:
        if col not in df.columns:
            continue

        values = df[col].astype(str)
        blank_mask = (
            df[col].isna()
            | values.str.strip().eq("")
            | values.str.lower().isin({"na", "null", "none", "nan"})
        )
        if not blank_mask.any():
            continue

        blank_indices = df.index[blank_mask].tolist()
        missing_reports.append(
            f"{col}: missing/NA at row(s) {_row_numbers_1based(blank_indices)}"
        )

    if not missing_reports:
        return None

    return QCError(
        error_name="Covariate Missing Values",
        error_message=(
            "Covariate columns contain missing or NA/NULL values:\n- "
            + "\n- ".join(missing_reports)
        ),
        error_source="qc",
        qc_name=qc_name,
    )
