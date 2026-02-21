from pydantic import BaseModel


class CaseStudyRAGRequest(BaseModel):
    message: str
    case_study_id: str | None = None
    project_id: str | None = None


class TokenInfo(BaseModel):
    input_tokens: int
    usd_input_cost: float
    total_generated_tokens: int
    usd_output_cost: float
