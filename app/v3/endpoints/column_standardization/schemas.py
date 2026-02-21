from pydantic import BaseModel, Field


class ColumnStandardizationRequest(BaseModel):
    table_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Unique identifier for the table to standardize column",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    column_name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="the column that need to standardize",
        examples=["InVivoPK_Species"],
    )
    column_description: str | None = Field(
        None,
        min_length=1,
        max_length=1024,
        description="Description of the column's content to aid standardization.",
        examples=[
            "The species of the animal used in the in-vivo pharmacokinetic study."
        ],
    )
    usr_instruction: str | None = Field(
        None,
        min_length=1,
        max_length=2048,
        description="A user prompt detailing how the column should be standardized.",
        examples=[
            "Standardize all species names to their common names \
            (e.g., 'Cynomolgus monkey' instead of 'Cynomolgus' or 'Cyno')."
        ],
    )


class ColumnStandardizationTaskResponse(BaseModel):
    message: str = Field(..., description="Status message about the task")
    task_id: str = Field(..., description="ID of the background task")
    table_id: str = Field(..., description="ID of the table being processed")
    column_name: str = Field(..., description="Column of the table being processed")
    column_description: str | None = Field(
        None, description="description of the column data"
    )
    usr_instruction: str | None = Field(
        None, description="user prompt detailing how the column should be standardized."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Column standardization task has been started in the bg",
                "task_id": "a1b2c3d4-e5f6-7890",
                "table_id": "12345",
                "column_name": "InVivoPK_Species",
            }
        }


class ColumnChange(BaseModel):
    prev: str = Field(..., description="Previous column value")
    current: str = Field(..., description="Standardized column value")


class ExtractionMetadata(BaseModel):
    flagId: str = Field(..., description="Flag ID")
    extractionId: str = Field(..., description="Extraction ID")
    extractionType: str = Field(..., description="Type of extraction (e.g., plot)")


class ProcessedExtraction(BaseModel):
    metadata: ExtractionMetadata = Field(..., description="Extraction metadata")
    changed_parameter: dict[str, list[ColumnChange]] = Field(
        ..., description="Parameters with column changes, grouped by parameter name"
    )


class StandardizationSummary(BaseModel):
    total_extractions: int = Field(
        ..., description="Total number of extractions processed"
    )
    total_standardized_extractions: int = Field(
        ..., description="Number of extractions with changes"
    )
    total_column_changes: int = Field(
        0, description="Total number of column changes made"
    )


class CostMetadata(BaseModel):
    """Cost metadata structure that matches the decorator output."""

    total_cost: float = Field(0.0, description="Total cost of all LLM operations")
    llm_cost_details: dict = Field(
        default_factory=dict, description="Detailed cost breakdown by model"
    )


class ColumnStandardizationCompletedData(BaseModel):
    table_id: str = Field(..., description="ID of the processed table")
    task_id: str = Field(..., description="ID of the background task")
    status: str = Field(..., description="Task completion status")
    processed_data: list[ProcessedExtraction] = Field(
        ..., description="List of processed extractions"
    )
    summary: StandardizationSummary = Field(
        ..., description="Summary of the standardization process"
    )
    cost_metadata: CostMetadata | None = Field(
        None, description="Cost information for LLM operations"
    )


class ColumnStandardizationFailedData(BaseModel):
    table_id: str = Field(..., description="ID of the processed table")
    column_name: str = Field(..., description="Column of the table being processed")
    column_description: str | None = Field(
        None, description="description of the column data"
    )
    usr_instruction: str | None = Field(
        None, description="user prompt detailing how the column should be standardized."
    )
    task_id: str = Field(..., description="ID of the background task")
    status: str = Field(..., description="Task failure status")
    error: str = Field(..., description="Error message")


class ColumnStandardizationOutput(BaseModel):
    """Output schema for column standardization with \
    default empty dict to prevent validation errors."""

    column_mappings: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of original column to standardized column",
    )

    def to_dict(self) -> dict[str, str]:
        return self.column_mappings
