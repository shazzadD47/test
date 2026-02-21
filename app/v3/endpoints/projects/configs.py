from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Config(BaseSettings):
    S3_MAX_RETRIES: int = 3
    S3_RETRY_BASE_DELAY: int = 1
    S3_MAX_FILE_SIZE: int = 50 * 1024 * 1024
    S3_RATE_LIMIT_PAUSE: float = 0.03  # 30ms pause between operations
    S3_MAX_CONCURRENT_OPERATIONS: int = 3

    class Config:
        env_prefix = "PROJECTS_"


settings = Config()
