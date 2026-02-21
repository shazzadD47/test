from pydantic_settings import BaseSettings


class Configs(BaseSettings):
    ANALYZER_LLM: str = "gemini-2.5-flash"
    QUERY_GENERATOR_LLM: str = "gemini-2.5-flash"
    CONTEXT_GENERATOR_LLM: str = "gemini-3-pro-preview"
    CONTEXT_GENERATOR_FALLBACK_LLM: str = "gpt-5.2"
    CONTEXT_GENERATOR_FOR_ROOT_LLM: str = "gemini-3-pro-preview"
    CONTEXT_GENERATOR_FOR_ROOT_FALLBACK_LLM: str = "gpt-5.2"
    TEMPERATURE: float = 0.2
    MAX_ROWS_PER_ITERATION: int = 20
    BATCH_SIZE: int = 100
    ROOT_ANSWERS_BATCH_SIZE: int = 20
    QUERY_BATCH_SIZE: int = 20
    MAX_PARALLEL_LLM_CALLS: int = 24
    MAX_RETRIES: int = 3
    DEFAULT_RETRY_DELAY: int = 10

    class Config:
        env_prefix = "GENERAL_EXTRACTION_"


settings = Configs()
