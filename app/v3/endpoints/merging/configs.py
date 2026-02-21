from pydantic_settings import BaseSettings


class Config(BaseSettings):
    LLM_MODEL: str = "claude-sonnet-4-5"
    MAX_OUTPUT_TOKENS: int = 32768
    TEMPERATURE: float = 0.2


settings = Config()
