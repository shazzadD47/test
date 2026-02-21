from typing import Literal

from pydantic import BaseModel, Field


class PaperLabelsTableField(BaseModel):
    name: str = Field(..., description="The name of the field")
    description: str = Field(..., description="The description of the field")
    d_type: Literal["string", "float", "integer", "list", "dict"] = Field(
        ..., description="The data type of the field"
    )
    c_type: str = Field(..., description="The source type of the field")


class DynamicPaperLabelsRequest(BaseModel):
    paper_id: str = Field(..., description="The ID of the paper")
    project_id: str | None = Field(None, description="The ID of the project")
    table_structure: list[PaperLabelsTableField] = Field(
        ..., description="The table structure to extract information"
    )
