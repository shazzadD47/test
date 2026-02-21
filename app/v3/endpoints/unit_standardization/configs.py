from pydantic_settings import BaseSettings


class Config(BaseSettings):
    # Unit Standardization Configuration
    MODEL_NAME: str = "gpt-4.1-mini"
    TEMPERATURE: float = 0.0
    BATCH_SIZE: int = 16
    MAX_RETRIES: int = 3

    BACKEND_UNIT_STD_URL: str = "v3/extractions/{table_id}/unit-standardization"

    CACHE_PREFIX: str = "unit_std:"
    CACHE_EXPIRATION: int = 60 * 60 * 24 * 1  # 1 days

    MAX_CONCURRENCY: int = 3

    class Config:
        env_prefix = "UNIT_STANDARDIZATION_"


settings = Config()
