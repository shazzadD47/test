from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, HttpUrl, field_validator

from app.constants import VALID_C_TYPES, VALID_D_TYPES
from app.v3.endpoints.plot_digitizer.constants import AxisMinMax


class PlotBounds(BaseModel):
    top_left_x: int = Field(..., description="The x coordinate of the top left corner")
    top_left_y: int = Field(..., description="The y coordinate of the top left corner")
    bottom_right_x: int = Field(
        ..., description="The x coordinate of the bottom right corner"
    )
    bottom_right_y: int = Field(
        ..., description="The y coordinate of the bottom right corner"
    )
    page_number: int | None = Field(None, description="The page number of the plot")

    @field_validator("*", mode="before")
    def check_non_negative(cls, value, info):
        if info.field_name != "page_number":
            if value is None:
                raise ValueError("Coordinates cannot be None")
            elif value < 0:
                raise ValueError("Coordinates cannot be negative")
        return value


class PlotDigitizerTableField(BaseModel):
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


class DynamicPlotDigitizerPayload(BaseModel):
    figure_url: str = Field(..., description="The URL of the figure to be analyzed")
    legend_urls: list[str] | None = Field(
        None, description="The URLs of the legends to be analyzed"
    )
    paper_id: str = Field(..., description="The ID of the paper")
    project_id: str | None = Field(None, description="The ID of the project")
    table_structure: list[PlotDigitizerTableField] = Field(
        ..., description="The table structure to extract information"
    )
    bounding_box: PlotBounds | None = Field(
        None,
        description="The bounding box coordinates of the plot",
    )
    page_number: int | None = Field(None, description="The page number of the plot")
    bounding_box_legends: list[PlotBounds] | None = Field(
        None,
        description="The list of bounding box coordinates of the cropped legends",
    )
    run_autofill: bool = Field(True, description="Whether to run autofill or not")
    run_digitization: bool = Field(
        True, description="Whether to run digitization or not"
    )
    line_names_to_extract: list[dict] | None = Field(
        None,
        description="The list of line names (with their ids) to be autofilled",
    )

    @field_validator("figure_url", mode="before")
    def validate_figure_url(cls, v):
        if isinstance(v, str):
            url = HttpUrl(url=v)
            return str(url)
        validated_urls = []
        for each_url in v:
            url = HttpUrl(url=each_url)
            validated_urls.append(str(url))
        return validated_urls


class DynamicPlotDigitizerRequest(BaseModel):
    payload: DynamicPlotDigitizerPayload
    metadata: dict


class DataPoints(BaseModel):
    x: int = Field(..., description="x coordinate value of the datapoint")
    y: int = Field(..., description="y coordinate value of the datapoint")


class DataPointsAutofil(BaseModel):
    x: int = Field(..., description="x coordinate value of the datapoint")
    y: int = Field(..., description="y coordinate value of the datapoint")
    topBarPixelDistance: int = Field(..., description="Distance from the top bar")
    bottomBarPixelDistance: int = Field(..., description="Distance from the bottom bar")
    deviationPixelDistance: int = Field(..., description="Deviation distance")


class LineFormerOutput(BaseModel):
    line_num: int = Field(..., description="Line serial number in the image")
    line_points: list[DataPoints] = Field(..., description="Points of the dataline")


class Florence2Output(BaseModel):
    line_name: str = Field(..., description="Line name")
    data_points: list[tuple[int, int]] = Field(
        ..., description="Correspoinding points (x,y)"
    )


class Florence2OutputErrorBar(BaseModel):
    line_name: str = Field(..., description="Line name")
    data_points: list[tuple[int, int, int, int, int]] = Field(
        ...,
        description="Correspoinding points with error bar (x,y,topBarPixelDistance,bottomBarPixelDistance,deviationPixelDistance)",  # noqa E501
    )


class Florence2DataInfo(BaseModel):
    data_points: list[dict[str, list[tuple[int, int]]]] = Field(
        ...,
        description="All lines name and corresponding points detected by florence 2",
    )


class MappedLegendPatch(BaseModel):
    label_bbox: list[int, int, int, int] = Field(
        ..., description="bounding box of the label"
    )
    patch_bbox: list[int, int, int, int] = Field(
        ..., description="bounding box of the patch"
    )
    patch_image: np.ndarray = Field(..., description="patch image")
    label_image: np.ndarray = Field(..., description="label image")
    lable_text: str = Field(..., description="label name")
    data_points: list[tuple[int, int]]

    class Config:
        arbitrary_types_allowed = True


class ChartDeteExtraction(BaseModel):
    legend_label_bbox: list[list[int, int, int, int]]
    legend_patch_bbox: list[list[int, int, int, int]]
    line_count_mismatch: bool


class AxisMinMaxPoint(BaseModel):
    label: Literal["xmin", "xmax", "ymin", "ymax"]
    x: int
    y: int


class AxesMinMaxOutput(BaseModel):
    points: list[AxisMinMaxPoint] = Field(..., example=AxisMinMax.example_out)

    @classmethod
    def from_tuples(
        cls,
        ymin: tuple[int, int],
        ymax: tuple[int, int],
        xmin: tuple[int, int],
        xmax: tuple[int, int],
        scale_percent: int,
    ):
        return cls(
            points=[
                {
                    "label": "ymin",
                    "x": round(ymin[0] / scale_percent),
                    "y": round(ymin[1] / scale_percent),
                },
                {
                    "label": "ymax",
                    "x": round(ymax[0] / scale_percent),
                    "y": round(ymax[1] / scale_percent),
                },
                {
                    "label": "xmin",
                    "x": round(xmin[0] / scale_percent),
                    "y": round(xmin[1] / scale_percent),
                },
                {
                    "label": "xmax",
                    "x": round(xmax[0] / scale_percent),
                    "y": round(xmax[1] / scale_percent),
                },
            ]
        )


class LegendSubstitutionMap(BaseModel):
    input_name: str
    matched_name: str


class PatchLabelGridImageOut(BaseModel):
    output_image: np.ndarray
    do_pad: bool
    pad_width: int | None

    class Config:
        arbitrary_types_allowed = True


class ResolutionCheckOutput(BaseModel):
    is_resized_image: bool
    new_height: int
    new_width: int
    height: int
    width: int
    object_key: str
    image_url: str


class LLMLegendMapGenInput(BaseModel):
    ai_model_legends: list[str] = Field(
        ..., description="The list of legend names detected by the AI model"
    )
    autofill_legends: list[str] = Field(
        ..., description="The list of legend names detected by the autofill pipeline"
    )


class LLMLegendMapGenOutput(BaseModel):
    ai_model_legend: str = Field(
        ...,
        description="The legend name detected by the AI model matched to the autofill legend",  # noqa E501
    )
    autofill_legend: str = Field(
        ...,
        description="The legend name detected by the autofill pipeline matched to the ai model legend",  # noqa E501
    )


class LLMLegendMapGenOutputList(BaseModel):
    matched_list: list[LLMLegendMapGenOutput] = Field(
        ...,
        description="The list of matched legend names. Total number of matched legend names should be equal to the total number of legend names detected by the autofill pipeline",  # noqa E501
    )


class AxisValue(BaseModel):
    min_val: float = Field(..., description="Minimum numeric value of the axis")
    max_val: float = Field(..., description="Maximum numeric value of the axis")


class GeminiAxisValuesOutput(BaseModel):
    x_axis: AxisValue = Field(
        ..., description="Extracted minimum and maximum values of the x-axis"
    )
    y_axis: AxisValue = Field(
        ..., description="Extracted minimum and maximum values of the y-axis"
    )
