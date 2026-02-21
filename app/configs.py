from pathlib import Path

from dotenv import load_dotenv
from pydantic import RedisDsn, field_validator
from pydantic_settings import BaseSettings

load_dotenv(override=True)


class Config(BaseSettings):
    ENV: str = "development"

    API_SECRET_KEY: str
    API_TIME_OUT: int = 600  # 10 minutes
    PDF_DOI_TIME_OUT: int = 40
    LLM_INVOKE_TIMEOUT: int = 1200  # 20 minutes timeout for LLM and chain invocations
    LLM_BATCH_INVOKE_TIMEOUT: int = (
        3600  # 60 minutes timeout for LLM and chain invocations
    )

    BACKEND_BASE_URL: str
    BACKEND_SECRET: str
    BACKEND_KEY: str
    BACKEND_FILE_URL_ENDPOINT: str = "v1/knowledge-files/file-url/{flag_id}"
    BACKEND_FILE_URL_WITH_SUPPLEMENTS_ENDPOINT: str = (
        "v1/knowledge-files/file-with-supplements/{flag_id}"
    )
    BACKEND_PROJECT_DETAILS_ENDPOINT: str = "v1/projects/{project_id}/details"

    @field_validator("BACKEND_BASE_URL")
    @classmethod
    def remove_trailing_slash(cls, v: str) -> str:
        """Remove trailing slash if present."""
        return v.rstrip("/")

    @field_validator("BACKEND_FILE_URL_ENDPOINT", "BACKEND_PROJECT_DETAILS_ENDPOINT")
    @classmethod
    def remove_leading_slash(cls, v: str) -> str:
        """Remove leading slash if present."""
        return v.lstrip("/")

    OPENAI_API_KEY: str
    OPENAI_EMBEDDING_MODEL: str
    GPT_4_TEXT_MODEL: str
    GPT_3_MODEL: str
    GPT_4O_INPUT_TOKEN: float = 0.000005
    GPT_4O_OUTPUT_TOKEN: float = 0.000015
    COST_PER_1000_TOKENS: float = 0.00013
    OPENAI_EMBEDDING_DIMENSIONS: int = 2048
    EMBEDDING_CHUNK_SIZE: int = 256
    EMBEDDING_CHUNK_OVERLAP: int = 64
    DEEP_SEARCH_PDF_CHUNK_SIZE: int = 1000
    DEEP_SEARCH_PDF_CHUNK_OVERLAP: int = 200

    ANTHROPIC_API_KEY: str
    CLAUDE_MODEL_ID: str

    GOOGLE_API_KEY: str

    CSV_API_KEY: str

    PDF_CACHE_DIR: Path = Path("data/cache/pdfs")

    DB_URL: str

    QDRANT_LOCATION: str
    QDRANT_API_KEY: str
    QDRANT_REST_PORT: int = 6333
    QDRANT_GRPC_PORT: int = 6334
    QDRANT_PREFER_GRPC: bool = True
    QDRANT_EMBEDDING_COLLECTION: str = "embeddings"

    S3_SPACES_SECRET_KEY: str
    S3_SPACES_ACCESS_KEY: str
    S3_SPACES_BUCKET: str
    S3_SPACES_ENDPOINT_URL: str
    S3_SPACES_PUBLIC_BASE_URL: str
    S3_SPACES_REGION: str = "us-east-2"

    MATHPIX_APP_ID: str
    MATHPIX_APP_KEY: str

    LOG_LEVEL: str = "INFO"
    LOGGING_FORMAT: str = "%(asctime)s [%(levelname)-8s] [%(name)s] %(message)s"

    SENTRY_DSN: str | None = None

    PDF_PAGE_IMAGE_RESOLUTION: int = 300

    CONCURRENT_THREADS: int = 10

    CELERY_BROKER_URL: RedisDsn
    CELERY_RESULT_BACKEND: RedisDsn
    CELERY_TASK_QUEUE: str = "delineate-ai-queue"

    REDIS_URL: RedisDsn
    CACHE_DAY: int

    SAVE_DIR: str = "data/internal/tables"

    RMQ_URL: str
    RMQ_QUEUE: str

    LANGFUSE_SECRET_KEY: str | None = None
    LANGFUSE_PUBLIC_KEY: str | None = None
    LANGFUSE_HOST: str | None = "https://us.cloud.langfuse.com"

    JWT_S2S_PUBLIC_KEY: str
    JWT_USER_AUTH_PUBLIC_KEY: str

    TAG_EXTRACTION_TEMPARAUTURE: float = 0.1

    NOTEBOOK_AGENT_BASE_URL: str


settings = Config()
