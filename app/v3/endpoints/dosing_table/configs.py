from pydantic_settings import BaseSettings


class Configs(BaseSettings):
    REASONING_LLM: str = "o3"
    MAX_RETRIES: int = 3
    DEFAULT_RETRY_DELAY: int = 10
    CONTEXT_GENERATOR_LLM: str = "gemini-2.5-flash"

    class Config:
        env_prefix = "DOSING_"


settings = Configs()
