from pydantic import BaseModel, Field, field_validator


class TagDefinition(BaseModel):
    name: str = Field(..., description="Display name of the tag")
    description: str = Field(..., description="What to look for to mark relevant")


class TagExtractionRequest(BaseModel):
    column_name: str = Field(..., description="Name of the column to store tag results")
    tags: list[TagDefinition] = Field(
        ..., min_length=1, max_length=50, description="List of tags to evaluate"
    )
    flag_ids: list[str] = Field(..., description="Paper flag_ids to process")
    meta_data: dict = Field(..., description="Metadata to echo back and send to RMQ")

    @field_validator("tags")
    @classmethod
    def validate_unique_tag_names(
        cls, tags: list[TagDefinition]
    ) -> list[TagDefinition]:
        names = [tag.name for tag in tags]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValueError(
                f"Tag names must be unique. Duplicates found: {set(duplicates)}"
            )
        return tags


class TagResult(BaseModel):
    name: str = Field(..., description="Name of the tag that was evaluated")
    reasoning: str = Field(
        ..., description="Explanation of why the tag was or wasn't marked as relevant"
    )
    is_relevant: bool = Field(
        ..., description="Whether the tag topic is substantively addressed in the paper"
    )
    relevance_score: int = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "Confidence score (0-100): 0-25=not present, 26-50=mentioned, "
            "51-75=discussed, 76-100=central focus"
        ),
    )


class CostMetadata(BaseModel):
    """Cost metadata structure that matches the decorator output."""

    total_cost: float = Field(0.0, description="Total cost of all LLM operations")
    llm_cost_details: dict = Field(
        default_factory=dict, description="Detailed cost breakdown by model"
    )


class TagExtractionResponse(BaseModel):
    flag_id: str = Field(
        ..., description="Unique identifier of the paper that was processed"
    )
    tags: list[TagResult] = Field(
        ..., description="List of tag evaluation results for this paper"
    )
    cost_metadata: CostMetadata | None = Field(
        None, description="Cost information for LLM operations used in processing"
    )
    meta_data: dict = Field(
        ..., description="Original metadata that was passed in the request"
    )
