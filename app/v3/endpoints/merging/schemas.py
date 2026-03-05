from typing import Literal, TypedDict

import pandas as pd
from pydantic import BaseModel, Field

from app.v3.endpoints.merging.constants import TableNames

QCErrorSource = Literal["qc", "merge"]


class QCError(BaseModel):
    error_name: str
    error_message: str
    error_source: QCErrorSource = Field(
        default="merge", description="Source: qc or merge"
    )
    qc_name: str | None = Field(
        default=None, description="Name of the QC check that produced the error"
    )


def merge_error(error_name: str, error_message: str) -> QCError:
    """Build a QCError for merge-flow errors (source=merge, no qc_name)."""
    return QCError(
        error_name=error_name,
        error_message=error_message,
        error_source="merge",
    )


class TablesByType(TypedDict, total=False):
    """
    Tables grouped by type after load (load_and_parse_tables).
    Five types only: plot, dosing, covariate, observation_table, paper_labels.
    Each key holds a list of DataFrames. Everything not plot/dosing/covariate
    or inferred observation_table goes to paper_labels.
    """

    plot: list[pd.DataFrame]
    dosing: list[pd.DataFrame]
    covariate: list[pd.DataFrame]
    observation_table: list[pd.DataFrame]
    paper_labels: list[pd.DataFrame]


class MergeResponse(BaseModel):
    final_df: str
    errors: list[QCError] | None
    status: Literal["success", "failed"]
    metadata: dict | None = None


class TableFieldInfo(BaseModel):
    name: str = Field(..., description="The name of the field")
    description: str = Field(..., description="The description of the field")
    d_type: Literal["string", "float", "integer", "list", "dict", "boolean"] = Field(
        ..., description="The data type of the field"
    )


class TableInfo(BaseModel):
    table_name: str = Field(..., description="The name of the table to be checked")
    table_url: str | None = Field(None, description="The url of the table")
    table_structure: list[TableFieldInfo] = Field(
        ..., description="The structure of the table"
    )
    table_type: (
        Literal[
            TableNames.OBSERVATION.value,
            TableNames.COVARIATE.value,
            TableNames.DOSING.value,
            "",
            None,
        ]
        | None
    ) = Field(None, description="The type of the table")


class MergeRequest(BaseModel):
    """Request payload for the merge-tables API."""

    project_id: str
    flag_id: str
    tables: list[TableInfo]
    version: Literal["v0", "v1"] = Field(
        default="v1",
        description=(
            "Merge flow version: v0 uses merge+QC, other values use "
            "generalized merge"
        ),
    )


class QualityControlRequest(BaseModel):
    file_id: str = Field(
        ..., description="ID of the file whose CSV to run quality checks on"
    )
    qc_list: list[str] = Field(..., description="List of QC check names to run")
    token: str | None = Field(
        default=None,
        description="Token to use when fetching CSV from backend (Bearer auth)",
    )


class QualityControlResponse(BaseModel):
    errors: list[QCError] = Field(
        default_factory=list,
        description="List of quality control errors, if any",
    )


class StandardizeSchema(BaseModel):
    table_values: dict
    table_structure: list[TableFieldInfo]
