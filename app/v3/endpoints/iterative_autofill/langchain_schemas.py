from pydantic import BaseModel, Field


class AnswerModel(BaseModel):
    answer: str = Field(
        ...,
        description="The answer to the question. Answers must be cited.",
    )
    answer_found: bool = Field(
        ...,
        description="Whether the answer was found in the contexts.",
    )
    chunk_numbers: list[int] = Field(
        ...,
        description="The chunk numbers used to derive the answer.",
    )
