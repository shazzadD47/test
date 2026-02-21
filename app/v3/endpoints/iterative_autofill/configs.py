from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Config(BaseSettings):
    OPENAI_MODEL_NAME: str = "gpt-4.1"
    OPENAI_MAX_TOKENS: int = 16384
    CLAUDE_MODEL_ID: str = "claude-3-7-sonnet-latest"
    CLAUDE_MAX_TOKENS: int = 8192
    BATCH_SIZE: int = 32
    QUERY_REPHRASE_BATCH_SIZE: int = 64
    CONTEXT_QA_BATCH_SIZE: int = 10
    LOOP_LIMIT: int = 100

    class Config:
        env_prefix = "ITERATIVE_AUTOFILL_"


settings = Config()
