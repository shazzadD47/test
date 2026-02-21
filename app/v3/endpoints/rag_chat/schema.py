from pydantic import BaseModel


class SourceInfo(BaseModel):
    page_content: str
    page: int | None
    flag_id: str | None
    title: str | None


class ContextInfo(BaseModel):
    context: list[SourceInfo]


class TokenInfo(BaseModel):
    input_tokens: int
    usd_input_cost: float
    total_generated_tokens: int
    usd_output_cost: float
