from pydantic import BaseModel, Field


class DataFileChoice(BaseModel):
    file_path: str | list[str] | None = Field(
        description="The path to the data file to use for the query."
    )
    reason: str = Field(description="The reason for choosing the data file.")
