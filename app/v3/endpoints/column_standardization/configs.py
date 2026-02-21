from pydantic_settings import BaseSettings


class Config(BaseSettings):
    MODEL_NAME: str = "gpt-4.1-mini"
    TEMPERATURE: float = 0.0
    BATCH_SIZE: int = 16
    MAX_RETRIES: int = 3

    BACKEND_COLUMN_STD_URL: str = (
        "v3/extractions/{table_id}/column-standardization/{col_name}"
    )

    CACHE_PREFIX: str = "column_std:"
    CACHE_EXPIRATION: int = 60 * 60 * 24 * 1  # 1 days

    MAX_CONCURRENCY: int = 3

    class Config:
        env_prefix = "COLUMN_STANDARDIZATION_"


settings = Config()
