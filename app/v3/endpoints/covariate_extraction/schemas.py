from typing import Literal

from pydantic import BaseModel, Field

from app.constants import VALID_C_TYPES, VALID_D_TYPES


class MetaAnalysisTableField(BaseModel):
    name: str = Field(..., description="The name of the field")
    description: str = Field(..., description="The description of the field")
    d_type: Literal[VALID_D_TYPES] = Field(  # type: ignore[valid-type]
        ..., description="The data type of the field"
    )
    c_type: Literal[VALID_C_TYPES] | None = Field(  # type: ignore[valid-type]
        None, description="The source type of the field"
    )
    literal_options: list[str] | None = Field(
        None, description="The options for the literal type field"
    )


class CovariateAutofillPayload(BaseModel):
    project_id: str = Field(..., description="The ID of the project")
    paper_id: str = Field(..., description="The ID of the paper")
    image_url: list[str] | str | None = Field(
        None, description="The URL of the image to be analyzed"
    )
    table_definition: list[MetaAnalysisTableField] = Field(
        ..., description="The table structure to extract information"
    )


class CovariateAutofillRequest(BaseModel):
    payload: CovariateAutofillPayload
    metadata: dict


class ImageType(BaseModel):
    image_type: Literal["Table", "Text"]


class CovariateMapping(BaseModel):
    mapping: dict = Field(..., description="mapping of covariates")


class AdversePlotDetails(BaseModel):
    arms: list[str] = Field(..., description="The arms presented in the table")
    final_endpoints: list[str] = Field(
        ...,
        description="""The final endpoints presented in the table which includes
        the endpoint and the sub-endpoints if found""",
    )
