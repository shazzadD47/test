from pydantic_settings import BaseSettings

from app.v3.endpoints.auto_suggestions.constants import (
    AISelectionInputTypes,
)


class AutoFigureSuggestionsConfig(BaseSettings):
    KEYWORD_GEN_MODEL_NAME: str = "gemini-2.5-pro"
    SUGGESTION_GEN_MODEL_NAME: str = "gemini-2.5-pro"
    MANDATORY_KEYWORD_CONTRIBUTION: float = 0.8
    SELECTION_TYPE: list[str] = [
        AISelectionInputTypes.IMAGE,
        AISelectionInputTypes.CHART,
        AISelectionInputTypes.TABLE,
    ]

    class Config:
        env_prefix = "AUTO_FIGURE_SUGGESTION_"


settings = AutoFigureSuggestionsConfig()
