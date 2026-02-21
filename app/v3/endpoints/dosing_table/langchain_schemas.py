from typing import Literal

from pydantic import BaseModel, Field


class BaseTableRow(BaseModel):
    GROUP: str = Field(
        ..., title="group", description="The treatment group/arm like XYZ 10 mg"
    )
    ROUTE: str = Field(..., title="route", description="The route of administration")
    ARM_TIME_UNIT: Literal["day", "hour", "week", "month", "year"] = Field(
        ...,
        title="arm time unit",
        description="The unit of the time of the dosing of the arm",
    )
    AMT: int | float | None = Field(
        ..., title="amount", description="The amount of the doses"
    )
    AMT_UNIT: str = Field(..., title="amount unit", description="The unit of the doses")
    II: int | float | None = Field(
        ..., title="interval", description="The inter-dose interval"
    )
    II_UNIT: Literal["day", "hour", "week", "month", "year"] = Field(
        ..., title="interval unit", description="The unit of the inter-dose interval"
    )
    STD_TRT: str | None = Field(
        ...,
        title="standard treatment name",
        description=(
            "The standard name of the treatment like"
            " Placebo, semaglutide, tirzepatide, etc."
        ),
    )


class TableRow(BaseTableRow):
    ARM_START_TIME: int | float | None = Field(
        ...,
        title="arm start time",
        description="The starting time of a dose of the arm like day 1.",
    )
    ARM_END_TIME: int | float | None = Field(
        ...,
        title="arm end time",
        description="The ending time of a dose of the arm like day 7.",
    )
    ADDL: int | float | None = Field(
        ..., title="additional", description="The additional doses count"
    )


class Table(BaseModel):
    rows: list[TableRow] = Field(
        ..., title="fields", description="The fields of the table"
    )
