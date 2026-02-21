from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Config(BaseSettings):
    LLM_NAME: str = "gpt-4o"

    class Config:
        env_prefix = "RAG_CHAT_"


settings = Config()
