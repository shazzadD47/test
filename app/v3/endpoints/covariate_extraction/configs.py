from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Config(BaseSettings):
    LLM_NAME: str = "gpt-4.1"
    CONTEXT_EXTRACTOR_LLM: str = "gemini-2.5-flash"
    BATCH_SIZE: int = 32
    TRIAL_ARM_LLM: str = "gpt-4.1"
    MAX_RETRIES: int = 3
    DEFAULT_RETRY_DELAY: int = 10
    BATCH_SIZE: int = 32
    CONTEXT_QA_BATCH_SIZE: int = 10
    QUERY_REPHRASE_BATCH_SIZE: int = 64
    TOP_K: int = 20

    class Config:
        env_prefix = "COVARIATE_EXTRACTION_"


settings = Config()
