from typing import Literal

from pydantic import BaseModel, Field


class LabelRelationship(BaseModel):
    related_label: str | None = Field(
        None, description="The label that this label has relationship with"
    )
    description: str | None = Field(None, description="how two labels are related")


class TableField(BaseModel):
    name: str = Field(..., description="The name of the field")
    description: str = Field(..., description="The description of the field")
    d_type: Literal["string", "float", "integer", "list", "dict"] = Field(
        ..., description="The data type of the field"
    )
    c_type: str = Field(..., description="The source type of the field")
    relationships: list[LabelRelationship] | None = Field(
        None, description="The relationships of the field"
    )
    generate: bool = Field(False, description="Whether to generate the field")


class IterativeAutofillPayload(BaseModel):
    paper_id: str = Field(..., description="The ID of the paper")
    project_id: str | None = Field(None, description="The ID of the project")
    table_structure: list[TableField] = Field(
        ..., description="The table structure to extract information"
    )
    prev_response: list[dict] | None = Field(
        None, description="The response from the previous iteration"
    )


class IterativeAutofillRequest(BaseModel):
    payload: IterativeAutofillPayload
    metadata: dict
