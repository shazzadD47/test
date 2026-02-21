from pydantic import BaseModel, Field


class UnitStandardizationRequest(BaseModel):
    table_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Unique identifier for the table to standardize units",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )


class UnitStandardizationTaskResponse(BaseModel):
    message: str = Field(..., description="Status message about the task")
    task_id: str = Field(..., description="ID of the background task")
    table_id: str = Field(..., description="ID of the table being processed")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Unit standardization task has been started in the bg",
                "task_id": "a1b2c3d4-e5f6-7890",
                "table_id": "12345",
            }
        }


class UnitChange(BaseModel):
    prev: str = Field(..., description="Previous unit value")
    current: str = Field(..., description="Standardized unit value")


class ExtractionMetadata(BaseModel):
    flagId: str = Field(..., description="Flag ID")
    extractionId: str = Field(..., description="Extraction ID")
    extractionType: str = Field(..., description="Type of extraction (e.g., plot)")


class ProcessedExtraction(BaseModel):
    metadata: ExtractionMetadata = Field(..., description="Extraction metadata")
    changed_parameter: dict[str, list[UnitChange]] = Field(
        ..., description="Parameters with unit changes, grouped by parameter name"
    )


class StandardizationSummary(BaseModel):
    total_extractions: int = Field(
        ..., description="Total number of extractions processed"
    )
    total_standardized_extractions: int = Field(
        ..., description="Number of extractions with changes"
    )
    total_unit_changes: int = Field(0, description="Total number of unit changes made")


class CostMetadata(BaseModel):
    """Cost metadata structure that matches the decorator output."""

    total_cost: float = Field(0.0, description="Total cost of all LLM operations")
    llm_cost_details: dict = Field(
        default_factory=dict, description="Detailed cost breakdown by model"
    )


class UnitStandardizationCompletedData(BaseModel):
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


class UnitStandardizationFailedData(BaseModel):
    table_id: str = Field(..., description="ID of the processed table")
    task_id: str = Field(..., description="ID of the background task")
    status: str = Field(..., description="Task failure status")
    error: str = Field(..., description="Error message")


class UnitStandardizationOutput(BaseModel):
    unit_mappings: dict[str, str] = Field(
        ..., description="Mapping of original units to standardized units"
    )

    def to_dict(self) -> dict[str, str]:
        return self.unit_mappings
