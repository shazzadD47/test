"""
Load and classify tables for the V1 merge flow.

Each table is downloaded from table_url, parsed as CSV, and grouped by type.
There are exactly five types: plot, dosing, covariate, observation_table,
paper_labels. Classification order (first match wins):
  1. table_type = plot -> tables_by_type['plot']
  2. table_type = dosing -> tables_by_type['dosing']
  3. table_type = covariate -> tables_by_type['covariate']
  4. inferred from table_name = observation_table -> tables_by_type['observation_table']
  5. everything else -> tables_by_type['paper_labels']
"""

from __future__ import annotations

from io import StringIO

import pandas as pd

from app.utils.download import download_file_from_url
from app.v3.endpoints.merging.constants import TableNames
from app.v3.endpoints.merging.logging import logger
from app.v3.endpoints.merging.schemas import QCError, TablesByType, merge_error


def _normalize_table_type(table_type: str | None) -> str | None:
    """
    Map payload table_type to internal key: Observation->plot, Dosing->dosing,
    Covariate->covariate. None or empty -> None (caller infers from table_name).
    """
    if table_type is None or (isinstance(table_type, str) and table_type.strip() == ""):
        return None
    s = str(table_type).strip().lower()
    if s == TableNames.OBSERVATION.value:
        return TableNames.OBSERVATION.value
    if s == TableNames.DOSING.value:
        return TableNames.DOSING.value
    if s == TableNames.COVARIATE.value:
        return TableNames.COVARIATE.value
    return None


def _infer_observation_table_from_name(table_name: str) -> bool:
    """
    When table_type is null/empty, infer from table_name. Names containing
    both 'observation' and 'table' -> observation_table. All other names
    are classified as paper_labels by the caller.
    """
    if not table_name or not isinstance(table_name, str):
        return False
    n = table_name.strip().lower()
    n = " ".join(n.split()).replace(" ", "_").replace("-", "_")
    while "__" in n:
        n = n.replace("__", "_")
    return "observation" in n and "table" in n


def load_and_parse_tables(table_dict: dict) -> tuple[TablesByType, list[QCError]]:
    """
    Download each table from table_url, parse CSV, and group by type.

    Classification order: plot, dosing, covariate, observation_table (inferred
    from name), then everything else -> paper_labels. No null type.

    Before (input):
        table_dict["tables"] = [
            {"table_name": "R-Observation", "table_type": "Observation",
             "table_url": "https://...", "table_structure": [...]},
            {"table_name": "AD_Paper Label", "table_type": null, "table_url": "..."},
        ]

    After (output):
        tables_by_type = {
            "plot": [df_obs],
            "dosing": [],
            "covariate": [],
            "observation_table": [],
            "paper_labels": [df_paper, ...],  # all remaining tables
        }
        errors = [QCError(...), ...]  # if any
    """
    tables: list[dict] = table_dict.get("tables") or []
    errors: list[QCError] = []
    tables_by_type: TablesByType = {
        TableNames.OBSERVATION.value: [],
        TableNames.DOSING.value: [],
        TableNames.COVARIATE.value: [],
        "observation_table": [],
        "paper_labels": [],
    }

    for table_info in tables:
        table_name = str(table_info.get("table_name") or "unknown")
        table_type_raw = table_info.get("table_type")
        table_url = table_info.get("table_url") or ""

        if not table_url or not str(table_url).strip():
            errors.append(
                merge_error(
                    f"Missing table_url: {table_name}",
                    (
                        f"Table '{table_name}' has no table_url. "
                        "Each table must provide a table_url for download."
                    ),
                )
            )
            continue

        # Download from table_url
        _, file_content = download_file_from_url(table_url)
        if file_content is None:
            errors.append(
                merge_error(
                    f"Download failed: {table_name}",
                    f"Failed to download table '{table_name}' from URL.",
                )
            )
            continue

        # Decode UTF-8 and parse CSV
        try:
            decoded = file_content.decode("utf-8")
        except UnicodeDecodeError as e:
            errors.append(
                merge_error(
                    f"Decode error: {table_name}",
                    (
                        f"Table '{table_name}' could not be decoded as UTF-8. "
                        f"Error: {e}"
                    ),
                )
            )
            continue

        try:
            df = pd.read_csv(StringIO(decoded))
        except Exception as e:
            errors.append(
                merge_error(
                    f"Parse error: {table_name}",
                    (
                        f"Table '{table_name}' could not be parsed as CSV. "
                        f"Check content and format. Error: {e}"
                    ),
                )
            )
            continue

        # Skip empty tables
        if df.empty and len(df.columns) == 0:
            errors.append(
                merge_error(
                    f"Empty table: {table_name}",
                    f"Table '{table_name}' is empty after parse; skipping.",
                )
            )
            continue

        if "index" in df.columns:
            df = df.drop(columns=["index"], errors="ignore")
        else:
            df = df.reset_index(drop=True)

        # Classify: preset type (plot/dosing/covariate) or infer observation_table,
        # else paper_labels (five types only; no null).
        internal_type: str | None = _normalize_table_type(table_type_raw)
        if internal_type == TableNames.OBSERVATION.value:
            tables_by_type[TableNames.OBSERVATION.value].append(df)
        elif internal_type == TableNames.DOSING.value:
            tables_by_type[TableNames.DOSING.value].append(df)
        elif internal_type == TableNames.COVARIATE.value:
            tables_by_type[TableNames.COVARIATE.value].append(df)
        elif _infer_observation_table_from_name(table_name):
            tables_by_type["observation_table"].append(df)
            internal_type = "observation_table"
        else:
            tables_by_type["paper_labels"].append(df)
            internal_type = "paper_labels"

        logger.info(f"Loaded table {table_name} (type={internal_type}) rows={len(df)}")

    return tables_by_type, errors
