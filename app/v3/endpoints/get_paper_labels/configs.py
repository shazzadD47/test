from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Config(BaseSettings):
    LLM_NAME: str = "gpt-4.1"

    class Config:
        env_prefix = "PAPER_LABELS_"


settings = Config()
