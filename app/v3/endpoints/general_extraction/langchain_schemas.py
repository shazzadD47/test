from pydantic import BaseModel, Field


class QueryList(BaseModel):
    queries: list[str] = Field(..., description="A list of queries about the label")


class AnswerList(BaseModel):
    answers: list[str] = Field(..., description="List of answers of the label.")


class CitationInfo(BaseModel):
    flag_id: str = Field(
        ...,
        description="Flag ID of the file",
    )
    page_no: int | None = Field(
        None,
        description="Page number of the file",
    )
    content: str = Field(..., description="Provide the exact content of the citation.")


class NumericalAnswer(BaseModel):
    values: list[float] = Field(
        ...,
        description="Values of the answer",
    )
    unit: str = Field(
        ...,
        description="Unit of the values of the answer",
    )
    citations: list[CitationInfo] = Field(
        ...,
        description="List of citations of the values of the answer",
    )


class RootNumericalAnswer(BaseModel):
    value: float | None = Field(
        ...,
        description="Value of the label",
    )
    unit: str = Field(
        ...,
        description="Unit of the value",
    )
    citations: list[CitationInfo] = Field(
        ...,
        description="List of citations of the source of the value",
    )
