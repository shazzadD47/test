from typing import Annotated, Literal

from fastapi import Query
from pydantic import BaseModel, Field, field_validator


class ListImageUrlsRequest(BaseModel):
    folder_name: str = Query(
        ...,
        description="The name of the folder containing the images.",
        min_length=4,
        max_length=12,
    )


class MinerUMetadata(BaseModel):
    chunk_id: int
    parent_id: int
    page_id: int
    heading: str


class MinerULegendChunk(BaseModel):
    legend_bbox: list[float] | None = Field(None, min_length=4, max_length=4)
    legend_path: str | None = None


class BaseChunk(BaseModel):
    value: str
    type: Literal[
        "table",
        "image",
        "narrative_text",
        "equation",
        "notebook_cell_code",
        "notebook_cell_markdown",
    ]
    bbox: list[float] | None = Field(None, min_length=4, max_length=4)
    metadata: MinerUMetadata

    @field_validator("bbox")
    def validate_bbox_values(cls, v):
        if v is None:
            return v
        if not all(0 <= x <= 1 for x in v):
            raise ValueError("All bbox values must be between 0 and 1")
        return v


class MinerUTableChunk(BaseChunk):
    type: Literal["table"]
    table_number: str
    table_caption: str
    table_body: str
    table_footnote: str
    img_path: str = Field(..., min_length=1)
    img_s3_uri: str = Field(..., min_length=1)


class MinerUSubFigureChunk(BaseChunk):
    type: Literal["image"] | None = None
    figure_number: str
    img_caption: str
    img_footnote: str
    img_path: str = Field(..., min_length=1)
    img_s3_uri: str = Field(..., min_length=1)
    legends: list[MinerULegendChunk] | None = None


class MinerUFigureChunk(BaseChunk):
    type: Literal["image"]
    figure_number: str
    img_caption: str
    img_footnote: str
    img_path: str = Field(..., min_length=1)
    img_s3_uri: str = Field(..., min_length=1)
    legends: list[MinerULegendChunk] | None = None
    subfigures: list[MinerUSubFigureChunk]


class MineruNotebookMetadata(MinerUMetadata):
    cell_id: int
    page_id: int | None = None


class MinerUNotebookChunk(BaseChunk):
    type: Literal["notebook_cell_code", "notebook_cell_markdown"]
    metadata: MineruNotebookMetadata


class MinerUNarrativeTextChunk(BaseChunk):
    type: Literal["narrative_text"]


class MinerUEquationChunk(BaseChunk):
    type: Literal["equation"]


class MineruWebhookRequest(BaseModel):
    chunks: list[
        MinerUTableChunk
        | MinerUFigureChunk
        | MinerUNarrativeTextChunk
        | MinerUEquationChunk
        | MinerUNotebookChunk
    ]
    project_id: str | None = None
    user_id: str | None = None
    file_id: str | None = None
    file_location: str | None = None


class PDFSummarizationPayload(BaseModel):
    flag_id: str
    project_id: str
    supplementary_id: str | None = None
    status: Literal["SUCCESS", "FAILED"]
    summary_text: str | None = None
    title: str | None = None
    title_verified: bool | None = None
    file_id: str | None = None


class ImageInfo(BaseModel):
    image_number: Annotated[int, Field(ge=1, description="Page number, 1-based index")]
    image_url: str | None = None
    status: Literal["SUCCESS", "FAILED"]


class ConvertPDFToImagePayload(BaseModel):
    flag_id: str
    project_id: str
    supplementary_id: str | None = None
    status: Literal["SUCCESS", "FAILED"]
    images: list[ImageInfo]
    file_id: str | None = None


class AnnotationItem(BaseModel):
    id: str
    x: int
    y: int
    width: int
    height: int
    pageNo: int
    imgSrc: str
    type: Literal["plot", "table"]
    caption: str | None
    description: str | None
    footnote: str | None
    number: str | None
    legends: list[MinerULegendChunk] | None = None
    chartType: str | None = None


class AnnotationPayload(BaseModel):
    annotations: list[AnnotationItem]


class MinerUOutputStatusPayload(BaseModel):
    flag_id: str
    file_id: str | None = None
    supplementary_id: str | None = None
    status: Literal[
        "1ST_PASS_COMPLETED", "2ND_PASS_COMPLETED", "FALLBACK_COMPLETED", "FAILED"
    ]
    response_type: Literal["initial", "final", "fallback", "failed"]
    message: str | None = None
    annotations: list[AnnotationItem] | None = None


class DetectLegends(BaseModel):
    answer: str = Field(
        ...,
        description="""Detects if second image have the required legend area.
        Only write yes or no""",
    )
    explanation: str = Field(
        ...,
        description="""Explain concisely in 1 line how you followed the instructions
        to find valid legend names.""",
    )


class AreaCompare(BaseModel):
    area_number_to_choose: int = Field(..., description="The area number to choose")
    does_second_image_need_any_legend_area: str
    explanation: str = Field(
        ...,
        description="""Explain concisely in 1 line how you followed the instructions
        to find valid legend names.""",
    )
