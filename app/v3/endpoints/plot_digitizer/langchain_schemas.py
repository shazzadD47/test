from typing import Annotated, Literal

from pydantic import BaseModel, Field


class PlotLegends(BaseModel):
    legends: list[str] = Field(..., description="The legends in the plot")
    explanation: str = Field(
        ...,
        description=(
            "Explain concisely how you followed the instructions"
            " to find valid legend names."
        ),
    )


class PlotTrtArms(BaseModel):
    trial_arms: list[str] = Field(
        ..., description="The treatment/trial arms in the paper."
    )
    arms_found: bool = Field(
        ..., description="Whether the trial arms are found in the paper."
    )


class PlotTrtArmsFigure(BaseModel):
    trial_arms: list[str] = Field(
        ..., description="The treatment/trial arms in the plot image."
    )
    arms_found: bool = Field(
        ..., description="Whether the trial arms are found in the plot image."
    )


class NumberOfLines(BaseModel):
    number_of_lines: int


class StringMatchingOutput(BaseModel):
    matched_string: str = Field(
        ..., description="The most similar string from Input_B."
    )


class FieldModel(BaseModel):
    value: Annotated[str, Field(description="The value of the field.")]
    found: Annotated[
        bool, Field(description="Whether the value is found in the figure.")
    ]

    class Config:
        arbitrary_types_allowed = True


class AxisDetails(BaseModel):
    x_min: int | float | str | None = Field(
        ..., description="The minimum value of the x axis"
    )
    x_max: int | float | str | None = Field(
        ..., description="The maximum value of the x axis"
    )
    y_min: int | float | str | None = Field(
        ..., description="The minimum value of the y axis"
    )
    y_max: int | float | str | None = Field(
        ..., description="The maximum value of the y axis"
    )
    x_label: str = Field(..., description="The label of the x axis")
    y_label: str = Field(..., description="The label of the y axis")
    x_tick_values: list[int | float | str] = Field(
        ..., description="The x tick values. can be int/float/str"
    )
    y_tick_values: list[int | float | str] = Field(
        ..., description="The y tick values. can be int/float/str"
    )
    x_interval: int | float | str | None = Field(
        ..., description="The interval between tick values of the x axis"
    )
    y_interval: int | float | str | None = Field(
        ..., description="The interval between tick values of the y axis"
    )
    x_unit: str = Field(..., description="The unit of the x axis")
    y_unit: str = Field(..., description="The unit of the y axis")
    x_is_log: bool = Field(..., description="Whether the x axis is in log scale or not")
    y_is_log: bool = Field(..., description="Whether the y axis is in log scale or not")
    x_is_categorical: bool = Field(
        ..., description="Whether the x axis is categorical or not"
    )
    y_is_categorical: bool = Field(
        ..., description="Whether the y axis is categorical or not"
    )
    chart_type: Literal[
        "line", "spider-plot", "bar", "box", "scatter", "kaplan-meier-curve", "other"
    ] = Field(
        ...,
        description=(
            "The type of the chart can be "
            "line/spider-plot/bar/box/scatter/kaplan-meier-curve/other"
        ),
    )


class LineLabels(BaseModel):
    line_name: str = Field(
        ...,
        description="""
    Individual name of each line/bar/box/other entity in the figure.
    """,
        c_type="general",
    )


class PlotDetails(BaseModel):
    title: str = Field(..., description="The title of the plot")
    caption: str = Field(..., description="The caption of the plot")
    figure_number: str = Field(..., description="The figure number")
    plot_axis_data: AxisDetails = Field(..., description="The details of the plot axis")
