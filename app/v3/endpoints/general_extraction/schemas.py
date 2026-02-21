import operator
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
    validator,
)
from typing_extensions import TypedDict

from app.constants import VALID_C_TYPES, VALID_D_TYPES


class PlotBounds(BaseModel):
    top_left_x: int = Field(..., description="The x coordinate of the top left corner")
    top_left_y: int = Field(..., description="The y coordinate of the top left corner")
    bottom_right_x: int = Field(
        ..., description="The x coordinate of the bottom right corner"
    )
    bottom_right_y: int = Field(
        ..., description="The y coordinate of the bottom right corner"
    )

    @field_validator("*", mode="before")
    def check_non_negative(cls, value, info):
        if info.field_name != "page_number":
            if value is None:
                raise ValueError("Coordinates cannot be None")
            elif value < 0:
                raise ValueError("Coordinates cannot be negative")
        return value


class ImageMetadata(BaseModel):
    figure_url: str = Field(..., description="The url of the figure")
    bounding_box: PlotBounds | None = Field(
        None, description="The bounding box of the figure"
    )
    page_number: int | None = Field(None, description="The page number of the figure")

    @validator("figure_url", pre=True, always=True)
    def validate_figure_url(cls, v):
        if isinstance(v, str):
            url = HttpUrl(url=v)
            return str(url)
        validated_urls = []
        for each_url in v:
            url = HttpUrl(url=each_url)
            validated_urls.append(str(url))
        return validated_urls


class FigureMetadata(BaseModel):
    figure_url: str = Field(..., description="The url of the figure")
    bounding_box: PlotBounds | None = Field(
        None, description="The bounding box of the figure"
    )
    page_number: int | None = Field(None, description="The page number of the figure")
    legends: list[ImageMetadata] | None = Field(
        None, description="The legends of the figure"
    )

    @validator("figure_url", pre=True, always=True)
    def validate_figure_url(cls, v):
        if isinstance(v, str):
            url = HttpUrl(url=v)
            return str(url)
        validated_urls = []
        for each_url in v:
            url = HttpUrl(url=each_url)
            validated_urls.append(str(url))
        return validated_urls


class BaseDataSchema(BaseModel):
    type: Literal["text", "chart", "image", "table", "equation"] = Field(
        ..., description="The type of the input data"
    )
    name: str | None = Field(None, description="The name of the input data")
    description: str | None = Field(
        None, description="The description of the input data"
    )
    data: str | list[FigureMetadata] | None = Field(
        None, description="The metadata for an image or content of the input data"
    )

    @model_validator(mode="after")
    def validate_data(self) -> Self:
        if self.type != "text" and self.data is not None:
            for item in self.data:
                if not isinstance(item, FigureMetadata):
                    raise ValueError(
                        f"Invalid data type for input data of type {self.type}"
                    )
        elif (
            self.type == "text"
            and self.data is not None
            and not isinstance(self.data, str)
        ):
            self.data = str(self.data)
        return self


class TableField(BaseModel):
    name: str = Field(..., description="The name of the field")
    description: str = Field(..., description="The description of the field")
    d_type: Literal[VALID_D_TYPES] = Field(  # type: ignore[valid-type]
        ..., description="The data type of the field"
    )
    c_type: Literal[VALID_C_TYPES] = Field(  # type: ignore[valid-type]
        ..., description="The source type of the field"
    )
    literal_options: list[str] | None = Field(
        None, description="The options for the literal type field"
    )


class GeneralExtractionRequest(BaseModel):
    project_id: str = Field(..., description="The project id of the project")
    flag_id: str = Field(..., description="The flag id of the resource")
    user_id: str | None = Field(None, description="The user id of the user")
    extraction_id: str | None = Field(
        None, description="The extraction id of the extraction"
    )
    inputs: list[BaseDataSchema] | None = Field(
        None, description="inputs, e.g chart, image, table, etc."
    )
    custom_instruction: str | None = Field(
        None, description="Optional custom instruction for the agent"
    )
    table_structure: list[TableField] = Field(
        ..., description="The table structure to extract information"
    )
    metadata: dict | None = Field(None, description="extra metadata")


def validate_payload(
    payload: GeneralExtractionRequest,
) -> GeneralExtractionRequest:
    if len(payload.table_structure) == 0:
        raise ValueError("Table structure cannot be empty")

    return payload


def update_ctype(
    table_structure: list[dict],
) -> list[dict]:
    for field in table_structure:
        if field["c_type"] == "General":
            field["c_type"] = "general"
        elif field["c_type"] == "Root":
            field["c_type"] = "root"
    return table_structure


class AgentState(TypedDict):
    messages: Annotated[list, operator.add, "Store inputs here for resue in nodes."]
    # workflow_input should contain everything in GeneralExtractionRequest
    workflow_input: Annotated[dict | None, "The input for the workflow"]
    # table structure with summarized context
    table_structure: Annotated[
        dict | None, "The table structure with summarized context"
    ]
    total_rows: Annotated[int | None, "The total number of rows in the final table"]
    file_details: Annotated[dict | None, "The details of the file"]
