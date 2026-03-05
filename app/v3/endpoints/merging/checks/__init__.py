"""
QC check registry: maps QC keys to lists of check functions.

Each check function has signature (df: pd.DataFrame, qc_name: str) -> QCError | None.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from app.v3.endpoints.merging.schemas import QCError

QC_CHECK = Callable[[pd.DataFrame, str], QCError | None]

KEY_INTEGRITY_DUPLICATES = "key-integrity-duplicates"
KEY_NONMEM_STRUCTURE = "nonmem-structure"
KEY_UNIT_CONSISTENCY = "unit-consistency"
KEY_PRACTICAL_DOSING = "practical-dosing"
KEY_CROSS_FIELD_CONSISTENCY = "cross-field-consistency"
KEY_ERROR_BARS = "error-bars"
KEY_DVID_ENDPOINT_MATCH = "dvid-endpoint-match"
KEY_DV_CHECK = "dv-check"
KEY_COVARIATE_UNITS = "covariate-units"

_REGISTRY: dict[str, list[QC_CHECK]] = {}


def get_checks_for_key(key: str) -> list[QC_CHECK]:
    """Return the list of check functions for a QC key, or empty if unknown."""
    return _REGISTRY.get(key, [])


def _register_key_integrity_duplicates() -> None:
    from app.v3.endpoints.merging.checks.key_integrity_duplicates import (
        check_duplicate_dv_per_composite_key,
        check_identical_rows,
        check_text_inconsistency_per_column,
    )

    _REGISTRY[KEY_INTEGRITY_DUPLICATES] = [
        check_identical_rows,
        check_duplicate_dv_per_composite_key,
        check_text_inconsistency_per_column,
    ]


def _register_nonmem_structure() -> None:
    from app.v3.endpoints.merging.checks.nonmem_structure import (
        check_dv_numeric,
        check_evid_amt,
        check_evid_dv,
        check_evid_dv_unit,
        check_evid_endpoint,
        check_mdv_evid_pattern,
    )

    _REGISTRY[KEY_NONMEM_STRUCTURE] = [
        check_evid_dv,
        check_evid_amt,
        check_mdv_evid_pattern,
        check_evid_endpoint,
        check_evid_dv_unit,
        check_dv_numeric,
    ]


def _register_unit_consistency() -> None:
    from app.v3.endpoints.merging.checks.unit_consistency import (
        check_amt_arm_dose_unit_standardized,
        check_amt_unit_same_per_arm,
        check_amt_unit_same_per_group_name,
        check_arm_dose_unit_amt_unit_same_per_row,
        check_arm_time_non_negative,
        check_arm_time_unit_and_ii_unit_time_units,
        check_arm_time_unit_consistent,
    )

    _REGISTRY[KEY_UNIT_CONSISTENCY] = [
        check_arm_time_unit_and_ii_unit_time_units,
        check_arm_time_unit_consistent,
        check_arm_time_non_negative,
        check_amt_arm_dose_unit_standardized,
        check_arm_dose_unit_amt_unit_same_per_row,
        check_amt_unit_same_per_group_name,
        check_amt_unit_same_per_arm,
    ]


def _register_practical_dosing() -> None:
    from app.v3.endpoints.merging.checks.practical_dosing import (
        check_addl_range,
        check_amt_non_decreasing_with_time,
        check_arm_dur_equals_sum_of_ii_addl,
        check_ii_and_ii_unit_consistent_within_group,
        check_route_non_null_and_same,
        check_zero_total_amt_multiple_doses,
    )

    _REGISTRY[KEY_PRACTICAL_DOSING] = [
        check_zero_total_amt_multiple_doses,
        check_arm_dur_equals_sum_of_ii_addl,
        check_amt_non_decreasing_with_time,
        check_ii_and_ii_unit_consistent_within_group,
        check_addl_range,
        check_route_non_null_and_same,
    ]


def _register_cross_field_consistency() -> None:
    from app.v3.endpoints.merging.checks.cross_field_consistency import (
        check_arm_fields_present_and_non_null,
        check_filename_same_for_all_rows,
        check_required_cross_fields_present,
        check_study_fields_constant_within_group,
    )

    _REGISTRY[KEY_CROSS_FIELD_CONSISTENCY] = [
        check_required_cross_fields_present,
        check_study_fields_constant_within_group,
        check_filename_same_for_all_rows,
        check_arm_fields_present_and_non_null,
    ]


def _register_error_bars() -> None:
    from app.v3.endpoints.merging.checks.error_bars import (
        check_baseline_ranges,
        check_baseline_units,
        check_ci_iqr_requires_bounds_and_vice_versa,
        check_male_female_percentages,
        check_other_stat_implies_var_only,
        check_prefixed_unprefixed_consistency,
        check_stat_blank_implies_fields_blank,
        check_var_non_negative,
    )

    _REGISTRY[KEY_ERROR_BARS] = [
        check_prefixed_unprefixed_consistency,
        check_stat_blank_implies_fields_blank,
        check_ci_iqr_requires_bounds_and_vice_versa,
        check_other_stat_implies_var_only,
        check_var_non_negative,
        check_male_female_percentages,
        check_baseline_units,
        check_baseline_ranges,
    ]


def _register_dvid_endpoint_match() -> None:
    from app.v3.endpoints.merging.checks.dvid_endpoint_match import (
        check_dvid_endpoint_match,
    )

    _REGISTRY[KEY_DVID_ENDPOINT_MATCH] = [
        check_dvid_endpoint_match,
    ]


def _register_dv_check() -> None:
    from app.v3.endpoints.merging.checks.dv_check import (
        check_dv_ranges,
        check_dv_stat_vs_dvid,
        check_dv_unit_vs_dvid,
        check_dv_var_stat,
        check_negative_dv_by_dvid,
    )

    _REGISTRY[KEY_DV_CHECK] = [
        check_dv_unit_vs_dvid,
        check_dv_stat_vs_dvid,
        check_dv_var_stat,
        check_dv_ranges,
        check_negative_dv_by_dvid,
    ]


def _register_covariate_units() -> None:
    from app.v3.endpoints.merging.checks.covariate_units import (
        check_covariate_missing_values,
        check_covariate_units,
    )

    _REGISTRY[KEY_COVARIATE_UNITS] = [
        check_covariate_units,
        check_covariate_missing_values,
    ]


_register_key_integrity_duplicates()
_register_nonmem_structure()
_register_unit_consistency()
_register_practical_dosing()
_register_cross_field_consistency()
_register_error_bars()
_register_dvid_endpoint_match()
_register_dv_check()
_register_covariate_units()
