"""
Quality checks for key 'dvid-endpoint-match'.

Rule:
- For each row, DVID must map to the correct ENDPOINT string according to
  a fixed mapping. If mismatch, report an error listing row numbers.
"""

from __future__ import annotations

import pandas as pd

from app.v3.endpoints.merging.schemas import QCError

DVID_ENDPOINT_MAP: dict[int, str] = {
    1: "EASI",
    2: "EASI50",
    3: "EASI75",
    4: "EASI90",
    5: "IGA",
    6: "SNRS",
    7: "PNRS",
    8: "DLQI",
    9: "CFB EASI",
    10: "CFB IGA",
    11: "CFB SNRS",
    12: "CFB PNRS",
    13: "CFB DLQI",
    14: "sdNRS",
    15: "Proportion of Patients achieving EASI",
    16: "Proportion of Patients achieving IGA",
    17: "Proportion of Patients achieving PNRS",
    18: "Proportion of Patients achieving SNRS",
    19: "Proportion of Patients achieving DLQI",
}


def _row_numbers_1based(indices: list[int]) -> str:
    if not indices:
        return ""
    if len(indices) <= 10:
        return ", ".join(str(i + 1) for i in indices)
    return ", ".join(str(i + 1) for i in indices[:10]) + f", ... ({len(indices)} rows)"


def check_dvid_endpoint_match(df: pd.DataFrame, qc_name: str) -> QCError | None:
    """
    Ensure DVID and ENDPOINT values match the predefined mapping.
    """
    if df.empty or "DVID" not in df.columns or "ENDPOINT" not in df.columns:
        return None

    dvid_numeric = pd.to_numeric(df["DVID"], errors="coerce")
    endpoint = df["ENDPOINT"].astype(str)

    bad_indices: list[int] = []
    for idx, (dvid, ep) in enumerate(zip(dvid_numeric, endpoint)):
        if pd.isna(dvid):
            continue
        dvid_int = int(dvid)
        expected = DVID_ENDPOINT_MAP.get(dvid_int)
        if expected is None:
            continue
        # compare trimmed strings exactly; do not case-normalize to avoid
        # masking important differences
        if ep.strip() != expected:
            bad_indices.append(df.index[idx])

    if not bad_indices:
        return None

    return QCError(
        error_name="DVID and ENDPOINT Mismatch",
        error_message=(
            "DVID and ENDPOINT must match the predefined mapping.\n"
            f"Violations at row(s): {_row_numbers_1based(bad_indices)}"
        ),
        error_source="qc",
        qc_name=qc_name,
    )
